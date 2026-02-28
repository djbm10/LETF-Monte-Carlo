import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from letf import config as cfg
from letf.utils import save_cache, load_cache, nearest_psd_matrix, compute_high_vol_probability, infer_regime_from_vix, calculate_daily_borrow_cost

# Try to import arch library for professional GARCH estimation
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def calibrate_regime_model_volatility(df):
    """
    Fit volatility-driven regime model with probabilistic stress inference.

    Uses VIX/realized-vol/term-structure signals for regime assignment,
    then estimates regime parameters, transition matrix, and spell durations.
    """
    cached = load_cache(cfg.REGIME_MODEL_CACHE)
    if cached is not None:
        print("[OK] Using cached regime model")
        return cached

    print(f"\n{'='*80}")
    print("CALIBRATING REGIME MODEL FROM VOLATILITY (CORRECT APPROACH)")
    print(f"{'='*80}\n")
    print(f"  Fitting {cfg.N_REGIMES}-regime model to VIX levels...")

    vix_series = df['VIX'].values
    realized_vol = df['SPY_Ret'].rolling(20, min_periods=5).std().bfill().fillna(0) * np.sqrt(252)
    term_spread = (df['TNX'] - df['IRX']).values if 'TNX' in df.columns and 'IRX' in df.columns else None

    regimes = infer_regime_from_vix(vix_series=vix_series, realized_vol=realized_vol.values, term_spread=term_spread)
    p_high_vol = compute_high_vol_probability(vix_series=vix_series, realized_vol=realized_vol.values, term_spread=term_spread)
    print(f"\n  Regime assignment: probabilistic stress score + hysteresis")

    regime_params = {}
    for regime_id in range(cfg.N_REGIMES):
        mask = regimes == regime_id
        regime_returns = df['SPY_Ret'].values[mask]
        daily_mean = regime_returns.mean() if mask.sum() > 0 else 0.0
        daily_std = regime_returns.std() if mask.sum() > 0 else 0.01
        regime_params[regime_id] = {
            'daily_mean': daily_mean,
            'daily_std': daily_std,
            'annual_mean': daily_mean * 252,
            'annual_vol': daily_std * np.sqrt(252),
            'frequency': mask.sum() / len(regimes),
            'avg_vix': float(np.nanmean(vix_series[mask])) if mask.sum() > 0 else 20.0
        }

    transitions = np.zeros((cfg.N_REGIMES, cfg.N_REGIMES))
    for i in range(len(regimes) - 1):
        transitions[int(regimes[i]), int(regimes[i + 1])] += 1

    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = transitions / row_sums

    # hard guard against numerical corruption
    transition_matrix = np.nan_to_num(transition_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(cfg.N_REGIMES):
        rs = transition_matrix[i].sum()
        if rs <= 0:
            transition_matrix[i, i] = 1.0
        else:
            transition_matrix[i] = transition_matrix[i] / rs

    for i in range(cfg.N_REGIMES):
        persistence = transition_matrix[i, i]
        regime_params[i]['avg_duration_days'] = 1.0 / (1.0 - persistence) if persistence < 1.0 else np.inf

    duration_samples = {i: [] for i in range(cfg.N_REGIMES)}
    if len(regimes) > 0:
        run_regime = int(regimes[0])
        run_length = 1
        for r in regimes[1:]:
            r = int(r)
            if r == run_regime:
                run_length += 1
            else:
                duration_samples[run_regime].append(run_length)
                run_regime = r
                run_length = 1
        duration_samples[run_regime].append(run_length)

    for i in range(cfg.N_REGIMES):
        samples = duration_samples[i] if len(duration_samples[i]) > 0 else [int(max(1, cfg.MIN_REGIME_DURATION[i]))]
        regime_params[i]['duration_median_days'] = float(np.median(samples))
        regime_params[i]['duration_p90_days'] = float(np.percentile(samples, 90))

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / steady_state.sum()

    print(f"\n[OK] Volatility Regime Model Calibrated:")
    print(f"{'='*80}")
    for i in range(cfg.N_REGIMES):
        params = regime_params[i]
        print(f"{cfg.REGIME_NAMES[i]:10s}:")
        print(f"  Annual Return: {params['annual_mean']*100:+6.2f}% (drift is constant!)")
        print(f"  Annual Vol:    {params['annual_vol']*100:5.2f}%")
        print(f"  Avg VIX:       {params['avg_vix']:.2f}")
        print(f"  Frequency:     {params['frequency']*100:5.2f}% (steady: {steady_state[i]*100:.2f}%)")
        print(f"  Avg Duration:  {params['avg_duration_days']:.0f} days")
        print(f"  Spell Length:  med={params['duration_median_days']:.0f}d p90={params['duration_p90_days']:.0f}d")

    print(f"\nTransition Matrix:")
    print(f"        Low Vol  High Vol")
    for i in range(cfg.N_REGIMES):
        row_str = f"{cfg.REGIME_NAMES[i]:10s}"
        for j in range(cfg.N_REGIMES):
            row_str += f"  {transition_matrix[i,j]:5.3f}"
        print(row_str)

    expected_return = sum(steady_state[i] * regime_params[i]['annual_mean'] for i in range(cfg.N_REGIMES))
    print(f"\n  Expected SPY Return: {expected_return*100:.2f}%")
    print(f"  (Note: Similar across regimes - only vol changes!)")

    vix_dynamics = calibrate_vix_dynamics(df, regimes)
    result = {
        'regime_params': regime_params,
        'transition_matrix': transition_matrix,
        'steady_state': steady_state,
        'expected_return': expected_return,
        'regimes_historical': regimes,
        'duration_samples': duration_samples,
        'regime_probability_high': p_high_vol,
        'vix_dynamics': vix_dynamics
    }

    save_cache(result, cfg.REGIME_MODEL_CACHE)
    return result


# ============================================================================
# FIX #6: TIME-VARYING CORRELATIONS (SPIKE IN CRISIS)
# ============================================================================

def calibrate_correlations_time_varying(df, regime_model):
    """
    FIX #6: Correlations are TIME-VARYING and spike to 0.95+ in high vol regime.

    This captures diversification failure in crisis.
    """
    cached = load_cache(cfg.CORRELATION_CACHE)
    if cached is not None:
        print("[OK] Using cached correlations")
        return cached

    print(f"\n{'='*80}")
    print("CALIBRATING TIME-VARYING CORRELATION MATRICES")
    print(f"{'='*80}\n")

    regimes_historical = regime_model.get('regimes_historical', None)

    if regimes_historical is None or len(regimes_historical) != len(df):
        print("  [WARN] No historical regimes - using defaults")
        return get_default_correlations_time_varying()

    df_regimes = df.copy()
    df_regimes['Regime'] = regimes_historical[:len(df)]

    correlation_data = {}

    for regime in range(cfg.N_REGIMES):
        regime_mask = df_regimes['Regime'] == regime
        regime_df = df_regimes[regime_mask]

        if len(regime_df) < 60:
            print(f"  [WARN] {cfg.REGIME_NAMES[regime]}: Insufficient data ({len(regime_df)} days)")
            correlation_data[regime] = None
            continue

        corr_cols = []
        if 'QQQ_Ret' in regime_df.columns:
            corr_cols.append('QQQ_Ret')
        if 'SPY_Ret' in regime_df.columns:
            corr_cols.append('SPY_Ret')
        if 'TLT_Ret' in regime_df.columns:
            corr_cols.append('TLT_Ret')

        if len(corr_cols) >= 2:
            corr_matrix = regime_df[corr_cols].corr()
            correlation_data[regime] = {
                'matrix': corr_matrix,
                'assets': corr_cols,
                'n_obs': len(regime_df)
            }

            print(f"  {cfg.REGIME_NAMES[regime]:10s} ({len(regime_df):4d} days):")
            if 'QQQ_Ret' in corr_cols and 'SPY_Ret' in corr_cols:
                corr_val = corr_matrix.loc['QQQ_Ret', 'SPY_Ret']
                print(f"    QQQ-SPY:  {corr_val:.3f}")
            if 'TLT_Ret' in corr_cols and 'SPY_Ret' in corr_cols:
                corr_val = corr_matrix.loc['TLT_Ret', 'SPY_Ret']
                print(f"    TLT-SPY:  {corr_val:.3f}")
        else:
            correlation_data[regime] = None

    print(f"\n  Building full correlation matrices with time-varying dynamics...")
    print(f"  KEY INSIGHT: Equity correlations spike to 0.95+ in high vol (crisis)")

    full_correlations = {}

    for regime in range(cfg.N_REGIMES):
        data = correlation_data.get(regime)

        if data is None:
            full_correlations[regime] = get_default_correlation_for_regime_time_varying(regime)
            continue

        corr_matrix = data['matrix']

        if 'QQQ_Ret' in data['assets'] and 'SPY_Ret' in data['assets']:
            qqq_spy_corr = corr_matrix.loc['QQQ_Ret', 'SPY_Ret']
        else:
            qqq_spy_corr = 0.85 if regime == 0 else 0.95  # Spike in crisis

        if 'TLT_Ret' in data['assets'] and 'SPY_Ret' in data['assets']:
            tlt_spy_corr = corr_matrix.loc['TLT_Ret', 'SPY_Ret']
        else:
            tlt_spy_corr = -0.20 if regime == 0 else -0.05  # Flight-to-quality weakens

        # FIX: In high vol, equity correlations spike (diversification fails)
        if regime == 1:  # High vol
            qqq_spy_corr = max(qqq_spy_corr, 0.95)  # Force high correlation

        # Build full matrix: TQQQ, UPRO, SSO, TMF, SPY
        full_corr = np.array([
            [1.000, qqq_spy_corr, qqq_spy_corr, tlt_spy_corr, qqq_spy_corr],  # TQQQ
            [qqq_spy_corr, 1.000, 0.980, tlt_spy_corr, 0.980],  # UPRO
            [qqq_spy_corr, 0.980, 1.000, tlt_spy_corr, 0.980],  # SSO
            [tlt_spy_corr, tlt_spy_corr, tlt_spy_corr, 1.000, tlt_spy_corr],  # TMF
            [qqq_spy_corr, 0.980, 0.980, tlt_spy_corr, 1.000]   # SPY
        ])

        full_corr = nearest_psd_matrix(full_corr)
        full_correlations[regime] = full_corr

        print(f"    {cfg.REGIME_NAMES[regime]:10s}: QQQ-SPY={qqq_spy_corr:.3f}, TLT-SPY={tlt_spy_corr:.3f}")

    print(f"\n[OK] Time-varying correlation matrices calibrated")
    print(f"  -> Diversification FAILS in high vol (all equities move together)")

    save_cache(full_correlations, cfg.CORRELATION_CACHE)
    return full_correlations


def get_default_correlation_for_regime_time_varying(regime):
    """Default time-varying correlations"""
    if regime == 0:  # Low vol
        corr = np.array([
            [1.000, 0.850, 0.850, -0.200, 0.850],
            [0.850, 1.000, 0.980, -0.200, 0.980],
            [0.850, 0.980, 1.000, -0.200, 0.980],
            [-0.200, -0.200, -0.200, 1.000, -0.200],
            [0.850, 0.980, 0.980, -0.200, 1.000]
        ])
    else:  # High vol - CORRELATIONS SPIKE
        corr = np.array([
            [1.000, 0.950, 0.950, -0.050, 0.950],
            [0.950, 1.000, 0.985, -0.050, 0.985],
            [0.950, 0.985, 1.000, -0.050, 0.985],
            [-0.050, -0.050, -0.050, 1.000, -0.050],
            [0.950, 0.985, 0.985, -0.050, 1.000]
        ])

    return nearest_psd_matrix(corr)


def get_default_correlations_time_varying():
    """Return default correlations for all regimes"""
    return {regime: get_default_correlation_for_regime_time_varying(regime) for regime in range(cfg.N_REGIMES)}


def calibrate_vix_dynamics(df: pd.DataFrame, regimes: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Calibrate regime-conditional VIX dynamics from historical data.

    Estimates persistence, innovation scale, and jump sensitivity to equity shocks
    by regime, then stores diagnostics (skew/kurtosis).
    """
    vix = df['VIX'].astype(float).values
    spy = df['SPY_Ret'].astype(float).values

    dynamics = {}
    for regime in range(cfg.N_REGIMES):
        idx = np.where(regimes == regime)[0]
        if len(idx) < 80:
            dynamics[regime] = {
                'phi': 0.90,
                'noise_std': 1.25,
                'jump_threshold_sigma': 2.0,
                'jump_scale': 6.0,
                'target_vix': 15.0 if regime == 0 else 35.0,
                'residual_skew': 0.0,
                'residual_kurtosis': 3.0
            }
            continue

        vix_reg = vix[idx]
        spy_reg = spy[idx]
        target_vix = float(np.nanmedian(vix_reg))

        vix_prev = vix_reg[:-1]
        vix_next = vix_reg[1:]
        valid = np.isfinite(vix_prev) & np.isfinite(vix_next)
        if valid.sum() < 30:
            phi = 0.90
            noise_std = 1.25
            residual = np.zeros(10)
        else:
            x = vix_prev[valid] - target_vix
            y = vix_next[valid] - target_vix
            denom = np.dot(x, x)
            phi = 0.90 if denom <= 0 else float(np.dot(x, y) / denom)
            phi = float(np.clip(phi, 0.70, 0.985))
            residual = y - phi * x
            noise_std = float(np.nanstd(residual))
            noise_std = float(np.clip(noise_std, 0.5, 4.0))

        shock_sigma = np.nanstd(spy_reg)
        shock_sigma = shock_sigma if shock_sigma > 0 else 0.01
        shock_z = np.abs(spy_reg) / shock_sigma
        jump_threshold = float(np.nanpercentile(shock_z, 90))
        jump_threshold = float(np.clip(jump_threshold, 1.5, 3.5))

        vix_diff = np.diff(vix_reg)
        shock_excess = np.maximum(0, shock_z[1:] - jump_threshold)
        valid_jump = np.isfinite(vix_diff) & np.isfinite(shock_excess)
        if valid_jump.sum() > 20 and np.any(shock_excess[valid_jump] > 0):
            xj = shock_excess[valid_jump]
            yj = np.maximum(0, vix_diff[valid_jump])
            jump_scale = float(np.dot(xj, yj) / (np.dot(xj, xj) + 1e-8))
        else:
            jump_scale = 6.0 if regime == 0 else 9.0
        jump_scale = float(np.clip(jump_scale, 2.0, 15.0))

        dynamics[regime] = {
            'phi': phi,
            'noise_std': noise_std,
            'jump_threshold_sigma': jump_threshold,
            'jump_scale': jump_scale,
            'target_vix': target_vix,
            'residual_skew': float(stats.skew(residual, nan_policy='omit')) if len(residual) > 3 else 0.0,
            'residual_kurtosis': float(stats.kurtosis(residual, fisher=False, nan_policy='omit')) if len(residual) > 3 else 3.0
        }

    return dynamics


def calibrate_joint_return_model(df: pd.DataFrame, regimes: np.ndarray) -> Dict:
    """
    Calibrate regime-conditional multivariate Student-t return model.

    Assets modeled jointly: SPY, QQQ, TLT.
    """
    cached = load_cache(cfg.JOINT_RETURN_MODEL_CACHE)
    if cached is not None:
        return cached

    assets = ['SPY_Ret', 'QQQ_Ret', 'TLT_Ret']
    model = {'assets': assets, 'regimes': {}}

    for regime in range(cfg.N_REGIMES):
        mask = regimes == regime
        reg_df = df.loc[mask, assets].dropna()

        if len(reg_df) < 80:
            # Fallback conservative defaults
            mu = np.array([0.08/252, 0.10/252, 0.03/252], dtype=float)
            vol = np.array([0.16, 0.24, 0.12], dtype=float) if regime == 0 else np.array([0.28, 0.42, 0.16], dtype=float)
            corr = np.array([
                [1.0, 0.90 if regime == 0 else 0.96, -0.20 if regime == 0 else -0.05],
                [0.90 if regime == 0 else 0.96, 1.0, -0.18 if regime == 0 else -0.03],
                [-0.20 if regime == 0 else -0.05, -0.18 if regime == 0 else -0.03, 1.0]
            ])
            cov = np.outer(vol / np.sqrt(252), vol / np.sqrt(252)) * corr
            nu = 5.0 if regime == 0 else 4.0
            garch_alpha = 0.06 if regime == 0 else 0.09
            garch_beta = 0.90 if regime == 0 else 0.86
            dcc_a = 0.02 if regime == 0 else 0.04
            dcc_b = 0.95 if regime == 0 else 0.90
        else:
            arr = reg_df.values
            mu = np.nanmean(arr, axis=0)
            cov = np.cov(arr, rowvar=False)
            cov = nearest_psd_matrix(cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))) * np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

            # Tail heaviness estimate from average excess kurtosis.
            k = np.nanmean([stats.kurtosis(reg_df[c], fisher=False, nan_policy='omit') for c in assets])
            if np.isfinite(k) and k > 3.05:
                nu = float(np.clip(4 + 6 / (k - 3 + 1e-6), 3.2, 12.0))
            else:
                nu = 8.0

            # GARCH parameter estimation: Use arch library if available (professional-grade)
            # Otherwise fall back to autocorrelation-based proxy
            if ARCH_AVAILABLE and len(reg_df) >= 200:
                # Use arch library for professional GARCH(1,1) estimation
                garch_params_list = []
                for asset in assets:
                    try:
                        returns_pct = 100 * reg_df[asset].dropna()  # Scale to percentage
                        # Fit GARCH(1,1) with Student-t innovations
                        am = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='StudentsT')
                        res = am.fit(disp='off', show_warning=False, options={'maxiter': 500})

                        garch_params_list.append({
                            'alpha': float(res.params['alpha[1]']),
                            'beta': float(res.params['beta[1]']),
                            'nu': float(res.params.get('nu', nu)) if 'nu' in res.params else nu
                        })
                    except Exception:
                        # Fallback to proxy estimation if arch fails
                        garch_params_list.append(None)

                # Average parameters if all assets succeeded
                if all(p is not None for p in garch_params_list):
                    garch_alpha = float(np.mean([p['alpha'] for p in garch_params_list]))
                    garch_beta = float(np.mean([p['beta'] for p in garch_params_list]))
                    nu = float(np.mean([p['nu'] for p in garch_params_list]))  # Update nu from arch
                else:
                    # Fallback to autocorrelation-based proxy
                    abs_ret = np.abs(arr - mu)
                    acf1 = np.nanmean([
                        np.corrcoef(abs_ret[:-1, j], abs_ret[1:, j])[0, 1]
                        for j in range(abs_ret.shape[1])
                        if abs_ret.shape[0] > 2
                    ])
                    if not np.isfinite(acf1):
                        acf1 = 0.25
                    garch_alpha = float(np.clip(0.05 + 0.10 * max(acf1, 0), 0.04, 0.15))
                    garch_beta = float(np.clip(0.98 - garch_alpha, 0.78, 0.94))
            else:
                # Fallback: Vol clustering persistence proxies (regime-specific)
                abs_ret = np.abs(arr - mu)
                acf1 = np.nanmean([
                    np.corrcoef(abs_ret[:-1, j], abs_ret[1:, j])[0, 1]
                    for j in range(abs_ret.shape[1])
                    if abs_ret.shape[0] > 2
                ])
                if not np.isfinite(acf1):
                    acf1 = 0.25
                garch_alpha = float(np.clip(0.05 + 0.10 * max(acf1, 0), 0.04, 0.15))
                garch_beta = float(np.clip(0.98 - garch_alpha, 0.78, 0.94))

            # DCC parameters (correlation dynamics)
            abs_ret = np.abs(arr - mu)
            acf1 = np.nanmean([
                np.corrcoef(abs_ret[:-1, j], abs_ret[1:, j])[0, 1]
                for j in range(abs_ret.shape[1])
                if abs_ret.shape[0] > 2
            ]) if 'abs_ret' in locals() else 0.25
            if not np.isfinite(acf1):
                acf1 = 0.25
            dcc_a = float(np.clip(0.015 + 0.04 * max(acf1, 0), 0.01, 0.08))
            dcc_b = float(np.clip(0.97 - dcc_a, 0.84, 0.97))

        model['regimes'][regime] = {
            'mu': mu,
            'cov': cov,
            'nu': nu,
            'garch_alpha': garch_alpha,
            'garch_beta': garch_beta,
            'dcc_a': dcc_a,
            'dcc_b': dcc_b
        }

    save_cache(model, cfg.JOINT_RETURN_MODEL_CACHE)
    return model


def simulate_joint_returns_t(n_days: int, regime_path: np.ndarray, joint_model: Dict,
                             rng: np.random.Generator, antithetic: bool = False) -> Dict[str, np.ndarray]:
    """Simulate regime-conditional multivariate Student-t returns with DCC/GARCH-lite dynamics.

    Args:
        n_days: Number of days to simulate
        regime_path: Regime assignments for each day
        joint_model: Calibrated joint model parameters
        rng: Random number generator
        antithetic: If True, negate normal components for 30-50% variance reduction
    """
    assets = joint_model['assets']
    out = {a: np.zeros(n_days) for a in assets}

    if n_days == 0:
        return out

    n_assets = len(assets)
    prev_regime = int(regime_path[0])
    p0 = joint_model['regimes'][prev_regime]
    cov0 = np.asarray(p0['cov'], dtype=float)
    long_var = np.clip(np.diag(cov0), 1e-8, None)
    h = long_var.copy()
    R_bar = cov0 / np.outer(np.sqrt(np.diag(cov0)), np.sqrt(np.diag(cov0)))
    R_bar = nearest_psd_matrix(R_bar)
    Q = R_bar.copy()
    prev_z = np.zeros(n_assets)

    for t in range(n_days):
        regime = int(regime_path[t])
        p = joint_model['regimes'][regime]
        mu = np.asarray(p['mu'], dtype=float)
        cov = np.asarray(p['cov'], dtype=float)
        nu = float(p['nu'])
        alpha = float(p.get('garch_alpha', 0.06))
        beta = float(p.get('garch_beta', 0.90))
        dcc_a = float(p.get('dcc_a', 0.02))
        dcc_b = float(p.get('dcc_b', 0.95))

        # Student-t variance correction: Var(t_v) = S * v/(v-2), so the
        # scale matrix S must be cov * (v-2)/v to match historical variance.
        t_var_scale = (nu - 2.0) / nu if nu > 2.0 else 0.5
        reg_long_var = np.clip(np.diag(cov) * t_var_scale, 1e-8, None)

        # GARCH stationarity with t-innovations: alpha*v/(v-2) + beta < 1.
        # If violated, cap beta to maintain stationary dynamics.
        effective_alpha = alpha * nu / (nu - 2.0) if nu > 2.0 else alpha * 2.0
        if effective_alpha + beta >= 1.0:
            beta = max(0.70, 0.98 - effective_alpha)
        reg_Rbar = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        reg_Rbar = nearest_psd_matrix(reg_Rbar)

        if t == 0:
            h = reg_long_var.copy()
            Q = reg_Rbar.copy()
        elif regime != prev_regime:
            # Smooth regime transition: blend toward new regime params over ~10 days
            # Hard resets create correlation/variance discontinuities that add spurious
            # vol drag, especially harmful for leveraged products.
            blend = 0.10
            h = blend * reg_long_var + (1 - blend) * h
            Q = blend * reg_Rbar + (1 - blend) * Q
        else:
            # Univariate GARCH-like variance update per asset
            h = (1 - alpha - beta) * reg_long_var + alpha * (prev_z ** 2) * h + beta * h
            h = np.clip(h, 1e-10, None)

            # DCC-like correlation update
            Q = (1 - dcc_a - dcc_b) * reg_Rbar + dcc_a * np.outer(prev_z, prev_z) + dcc_b * Q

        # Q, R_t, and cov_t are all PSD by construction:
        #   Q: weighted sum of PSD matrices (R_bar, outer(z,z), old Q)
        #   R_t: congruence transform of PSD Q
        #   cov_t: Schur product of rank-1 PSD and R_t (Schur product theorem)
        # nearest_psd_matrix MUST NOT be called on cov_t -- its diagonal
        # normalization destroys variance information and explodes GARCH.
        d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
        R_t = Q / np.outer(d, d)
        cov_t = np.outer(np.sqrt(h), np.sqrt(h)) * R_t

        # Multivariate t: x = mu + z / sqrt(u/nu), z~N(0,cov), u~ChiSq(nu)
        z = rng.multivariate_normal(mean=np.zeros(len(mu)), cov=cov_t)
        # Antithetic variates: negate normal component for variance reduction
        if antithetic:
            z = -z
        u = rng.chisquare(df=nu)
        scale = np.sqrt(nu / max(u, 1e-12))
        x = mu + z * scale
        x_clipped = np.clip(x, -0.95, 4.0)

        # Standardized residual for next step dynamics
        prev_z = (x_clipped - mu) / np.sqrt(np.clip(h, 1e-10, None))
        prev_regime = regime

        for i, a in enumerate(assets):
            # Keep single-day returns within plausible numerical bounds.
            out[a][t] = float(x_clipped[i])

    # Apply moment matching for numerical stability (if enabled)
    if cfg.USE_MOMENT_MATCHING:
        for a in assets:
            returns = out[a]
            # Adjust returns to match theoretical mean (reduces drift in long simulations)
            # This is especially important for 30-year horizons
            theoretical_mean = np.mean([joint_model['regimes'][r]['mu'][assets.index(a)]
                                       for r in range(len(joint_model['regimes']))])
            actual_mean = np.mean(returns)
            # Apply small correction to eliminate systematic drift
            out[a] = returns + (theoretical_mean - actual_mean) * 0.1  # 10% correction factor

    return out


def calibrate_funding_spread_model(df: pd.DataFrame, bypass_cache: bool = False) -> Dict[str, float]:
    """Calibrate borrow spread loadings from historical proxies and realized LETF gap data."""
    cached = None if bypass_cache else load_cache(cfg.FUNDING_MODEL_CACHE)
    if cached is not None:
        return cached

    n = len(df)
    if n == 0:
        model = {
            'base': 0.0050,
            'beta_vix': 0.00035,
            'beta_inv_curve': 0.0014,
            'beta_liquidity': 0.0010,
            'beta_credit': 0.0014,
            'min_spread': 0.0030,
            'max_spread': 0.0450
        }
        if not bypass_cache:
            save_cache(model, cfg.FUNDING_MODEL_CACHE)
        return model

    vix = df['VIX'].ffill().bfill().fillna(20.0).to_numpy(dtype=float)
    irx = df.get('IRX', pd.Series(4.5, index=df.index)).ffill().bfill().fillna(4.5).to_numpy(dtype=float)

    if 'TNX' in df.columns:
        tnx = df['TNX'].ffill().bfill().fillna(irx + 1.0).to_numpy(dtype=float)
    else:
        tnx = irx + 1.0

    term_spread = tnx - irx
    stress = np.maximum(vix - 20.0, 0.0)
    inv_curve = np.maximum(-term_spread, 0.0)
    rv = (df['SPY_Ret'].rolling(20, min_periods=5).std().bfill().fillna(0.15 / np.sqrt(252)).to_numpy(dtype=float) * np.sqrt(252))
    liquidity_proxy = np.maximum(vix - 18.0, 0.0) / 25.0 + np.maximum(rv - 0.18, 0.0)
    credit_proxy = np.maximum(-term_spread, 0.0)

    # Build implied spread target from observed LETF returns where available.
    implied_candidates = []
    for asset in ['TQQQ', 'UPRO', 'SSO']:
        ret_col = f'{asset}_Real_Ret'
        if ret_col not in df.columns:
            continue
        lev = cfg.ASSETS[asset]['leverage']
        if lev <= 1.0:
            continue

        idx = df['QQQ_Ret'] if (asset == 'TQQQ' and 'QQQ_Ret' in df.columns) else df['SPY_Ret']
        real_ret = df[ret_col].to_numpy(dtype=float)
        idx_ret = idx.to_numpy(dtype=float)
        expense_daily = cfg.ASSETS[asset]['expense_ratio'] / 252.0

        implied = ((lev * idx_ret - expense_daily - real_ret) * 252.0 / (lev - 1.0)) - (irx / 100.0)
        implied_candidates.append(implied)

    if implied_candidates:
        stacked = np.vstack(implied_candidates)
        target = np.nanmedian(stacked, axis=0)
    else:
        target = 0.0045 + 0.00035 * stress + 0.0012 * inv_curve

    finite_target = target[np.isfinite(target)]
    if finite_target.size > 20:
        lo, hi = np.nanpercentile(finite_target, [1.0, 99.0])
        target = np.clip(target, lo, hi)
    target = np.nan_to_num(target, nan=float(np.nanmedian(finite_target) if finite_target.size else 0.0060))

    X = np.column_stack([
        np.ones(n),
        stress,
        inv_curve,
        np.clip(liquidity_proxy, 0.0, 3.0),
        np.clip(credit_proxy, 0.0, 3.0)
    ])
    mask = np.isfinite(target) & np.all(np.isfinite(X), axis=1)

    if mask.sum() < 120:
        beta = np.array([0.0045, 0.00035, 0.0014, 0.0010, 0.0014])
    else:
        # Ridge-regularized least squares for numerical stability.
        X_fit = X[mask]
        y_fit = target[mask]
        reg = np.diag([1e-6, 1e-4, 1e-4, 1e-4, 1e-4])
        lhs = X_fit.T @ X_fit + reg
        rhs = X_fit.T @ y_fit
        beta = np.linalg.solve(lhs, rhs)
        beta[1:] = np.clip(beta[1:], 0.0, None)

    predicted = X @ beta
    pred_finite = predicted[np.isfinite(predicted)]
    if pred_finite.size > 10:
        min_spread = float(max(np.nanpercentile(pred_finite, 1.0), 0.0025))
        max_spread = float(min(np.nanpercentile(pred_finite, 99.5), 0.0300))
        if max_spread <= min_spread:
            max_spread = min_spread + 0.005
    else:
        min_spread, max_spread = 0.0030, 0.0300

    model = {
        'base': float(max(beta[0], 0.0015)),
        'beta_vix': float(beta[1]),
        'beta_inv_curve': float(beta[2]),
        'beta_liquidity': float(beta[3]),
        'beta_credit': float(beta[4]),
        'min_spread': min_spread,
        'max_spread': max_spread
    }

    if not bypass_cache:
        save_cache(model, cfg.FUNDING_MODEL_CACHE)
    return model


def calibrate_stress_state_model(df: pd.DataFrame, regimes: np.ndarray) -> Dict:
    """
    Calibrate latent stress channels used by institutional_v1:
    - liquidity stress
    - credit stress
    - crisis jump intensity
    """
    cached = load_cache(cfg.STRESS_STATE_CACHE)
    if cached is not None:
        return cached

    # Build simple proxies from available columns.
    vix = df['VIX'].ffill().bfill().fillna(20.0).values
    rv = (df['SPY_Ret'].rolling(20, min_periods=5).std().bfill().fillna(0.15 / np.sqrt(252)).values * np.sqrt(252))

    if 'TNX' in df.columns and 'IRX' in df.columns:
        credit_proxy = np.maximum(-(df['TNX'] - df['IRX']).fillna(0.0).values, 0.0)
    else:
        credit_proxy = np.maximum(vix - 20.0, 0.0) / 20.0

    liquidity_proxy = np.maximum(vix - 18.0, 0.0) / 25.0 + np.maximum(rv - 0.18, 0.0)

    model = {'regimes': {}}
    for regime in range(cfg.N_REGIMES):
        mask = regimes == regime
        if mask.sum() < 60:
            model['regimes'][regime] = {
                'liq_mu': 0.10 if regime == 0 else 0.35,
                'liq_phi': 0.90,
                'liq_sigma': 0.08,
                'credit_mu': 0.05 if regime == 0 else 0.25,
                'credit_phi': 0.88,
                'credit_sigma': 0.07,
                'jump_base_prob': 0.0002 if regime == 0 else 0.001,
                'jump_scale': 0.0005 if regime == 0 else 0.002
            }
            continue

        liq = liquidity_proxy[mask]
        cred = credit_proxy[mask]

        liq_mu = float(np.nanmedian(liq))
        cred_mu = float(np.nanmedian(cred))

        def ar1_params(series, default_phi=0.9, default_sigma=0.08):
            s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().values
            if len(s) < 20:
                return default_phi, default_sigma
            x = s[:-1] - np.nanmedian(s)
            y = s[1:] - np.nanmedian(s)
            denom = np.dot(x, x)
            phi = default_phi if denom <= 0 else float(np.dot(x, y) / denom)
            phi = float(np.clip(phi, 0.50, 0.98))
            sigma = float(np.nanstd(y - phi * x))
            sigma = float(np.clip(sigma, 0.01, 0.30))
            return phi, sigma

        liq_phi, liq_sigma = ar1_params(liq, default_phi=0.90, default_sigma=0.08)
        cred_phi, cred_sigma = ar1_params(cred, default_phi=0.88, default_sigma=0.07)

        # Jump parameters calibrated jointly with Student-t(df=5) + GARCH:
        # Since fat tails are already represented by the return distribution,
        # jumps model only distinct structural events (flash crashes, circuit breakers).
        # Reduced intensity to avoid double-counting tail risk.
        jump_base_prob = float(np.clip(0.0002 + 0.003 * np.nanmean(np.maximum(rv[mask] - 0.25, 0.0)), 0.0002, 0.003))
        jump_scale = float(np.clip(0.0005 + 0.002 * np.nanmean(np.maximum(rv[mask] - 0.25, 0.0)), 0.0005, 0.0025))

        model['regimes'][regime] = {
            'liq_mu': liq_mu,
            'liq_phi': liq_phi,
            'liq_sigma': liq_sigma,
            'credit_mu': cred_mu,
            'credit_phi': cred_phi,
            'credit_sigma': cred_sigma,
            'jump_base_prob': jump_base_prob,
            'jump_scale': jump_scale
        }

    save_cache(model, cfg.STRESS_STATE_CACHE)
    return model


def simulate_latent_stress_state(n_days: int, regime_path: np.ndarray,
                                 stress_model: Dict, vix_series: np.ndarray,
                                 rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Simulate latent liquidity/credit stress channels and jump process."""
    liquidity = np.zeros(n_days)
    credit = np.zeros(n_days)
    jump = np.zeros(n_days)

    if n_days == 0:
        return {'liquidity': liquidity, 'credit': credit, 'jump': jump}

    first_reg = int(regime_path[0])
    p0 = stress_model['regimes'].get(first_reg, {})
    liquidity[0] = float(p0.get('liq_mu', 0.1))
    credit[0] = float(p0.get('credit_mu', 0.05))

    for t in range(1, n_days):
        reg = int(regime_path[t])
        p = stress_model['regimes'].get(reg, {})

        liq_mu = p.get('liq_mu', 0.1)
        liq_phi = p.get('liq_phi', 0.9)
        liq_sigma = p.get('liq_sigma', 0.08)
        credit_mu = p.get('credit_mu', 0.05)
        credit_phi = p.get('credit_phi', 0.88)
        credit_sigma = p.get('credit_sigma', 0.07)

        liquidity[t] = liq_mu + liq_phi * (liquidity[t-1] - liq_mu) + rng.normal(0, liq_sigma)
        credit[t] = credit_mu + credit_phi * (credit[t-1] - credit_mu) + rng.normal(0, credit_sigma)

        liquidity[t] = float(np.clip(liquidity[t], 0.0, 3.0))
        credit[t] = float(np.clip(credit[t], 0.0, 3.0))

        base_prob = p.get('jump_base_prob', 0.001)
        jump_scale = p.get('jump_scale', 0.005)
        vix_amp = max((vix_series[t] - 25.0) / 30.0, 0.0)
        # Capped at 1% daily probability (was 3%); VIX and liquidity loadings
        # halved to avoid over-representing jumps when Student-t + GARCH
        # already produce fat-tailed moves.
        jump_prob = float(np.clip(base_prob + 0.008 * vix_amp + 0.003 * liquidity[t], 0.0, 0.01))
        if rng.random() < jump_prob:
            jump[t] = abs(rng.standard_t(df=5)) * jump_scale

    return {'liquidity': liquidity, 'credit': credit, 'jump': jump}


def predict_borrow_spread_series(df: pd.DataFrame, funding_model: Dict[str, float],
                                 stress_state: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Predict annual borrow spread (decimal) from stress covariates."""
    vix = df['VIX'].ffill().bfill().fillna(20.0).values
    stress = np.maximum(vix - 20.0, 0.0)

    inv_curve = np.zeros(len(df))
    if 'TNX' in df.columns and 'IRX' in df.columns:
        inv_curve = np.maximum(-(df['TNX'] - df['IRX']).fillna(0.0).values, 0.0)

    spread = (
        funding_model['base']
        + funding_model['beta_vix'] * stress
        + funding_model['beta_inv_curve'] * inv_curve
    )

    if stress_state is not None:
        liq = np.asarray(stress_state.get('liquidity', np.zeros(len(spread))), dtype=float)
        cred = np.asarray(stress_state.get('credit', np.zeros(len(spread))), dtype=float)
        spread += (
            funding_model.get('beta_liquidity', 0.0012) * np.clip(liq, 0, 3)
            + funding_model.get('beta_credit', 0.0018) * np.clip(cred, 0, 3)
        )

    return np.clip(spread, funding_model['min_spread'], funding_model['max_spread'])


def calibrate_tracking_residual_model(df: pd.DataFrame,
                                      funding_model: Optional[Dict[str, float]] = None,
                                      bypass_cache: bool = False) -> Dict:
    """
    Calibrate ETF tracking residual dynamics from observed post-inception returns.
    """
    cached = None if bypass_cache else load_cache(cfg.TRACKING_RESIDUAL_CACHE)
    if cached is not None:
        return cached

    model = {}
    for asset in ['TQQQ', 'UPRO', 'SSO']:
        ret_col = f'{asset}_Real_Ret'
        if ret_col not in df.columns:
            continue

        real = df[ret_col]
        if asset == 'TQQQ':
            idx = df.get('QQQ_Ret', df['SPY_Ret'])
        else:
            idx = df['SPY_Ret']

        leverage = cfg.ASSETS[asset]['leverage']
        rf = df.get('IRX', pd.Series(4.5, index=df.index)).fillna(4.5).values / 100.0

        if funding_model is not None:
            spread_df = pd.DataFrame({'VIX': df['VIX'].values}, index=df.index)
            if 'IRX' in df.columns:
                spread_df['IRX'] = df['IRX'].values
            if 'TNX' in df.columns:
                spread_df['TNX'] = df['TNX'].values

            # Historical stress channels are optional at calibration-time;
            # if unavailable, the model still uses VIX/curve-linked spread dynamics.
            spread_series = predict_borrow_spread_series(spread_df, funding_model, stress_state=None)
        else:
            spread_series = np.full(len(df), 0.0075)

        financing_daily = (leverage - 1.0) * (rf + spread_series) / 252.0
        expense_daily = cfg.ASSETS[asset]['expense_ratio'] / 252.0

        expected = leverage * idx.values - financing_daily - expense_daily
        residual = (real.values - expected)
        mask = np.isfinite(residual) & np.isfinite(df['VIX'].values)

        if mask.sum() < 120:
            model[asset] = {
                'rho': 0.25,
                'base_scale': cfg.ASSETS[asset]['tracking_error_base'],
                'downside_mult': 1.25,
                'df': cfg.ASSETS[asset]['tracking_error_df'],
                'clip_limit': 0.15
            }
            continue

        r = residual[mask]
        r_prev = r[:-1]
        r_next = r[1:]
        denom = np.dot(r_prev, r_prev)
        rho = 0.25 if denom <= 0 else float(np.dot(r_prev, r_next) / denom)
        rho = float(np.clip(rho, 0.0, 0.7))

        innov = r_next - rho * r_prev
        scale = float(np.nanstd(innov))
        scale = float(np.clip(scale, cfg.ASSETS[asset]['tracking_error_base'] * 0.5,
                              cfg.ASSETS[asset]['tracking_error_base'] * 8.0))

        downside = np.nanmean(np.abs(innov[innov < 0])) if np.any(innov < 0) else scale
        upside = np.nanmean(np.abs(innov[innov >= 0])) if np.any(innov >= 0) else scale
        downside_mult = float(np.clip((downside / max(upside, 1e-9)), 1.0, 2.0))

        clip_limit = float(np.nanpercentile(np.abs(innov), 99.5) * 1.35) if len(innov) > 30 else 0.15
        clip_limit = float(np.clip(clip_limit, 0.08, 0.35))

        model[asset] = {
            'rho': rho,
            'base_scale': scale,
            'downside_mult': downside_mult,
            'df': cfg.ASSETS[asset]['tracking_error_df'],
            'clip_limit': clip_limit
        }

    if not bypass_cache:
        save_cache(model, cfg.TRACKING_RESIDUAL_CACHE)
    return model
