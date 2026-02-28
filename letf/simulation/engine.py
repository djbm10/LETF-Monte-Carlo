import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from typing import Dict, List, Optional
from letf import config as cfg
from letf.utils import nearest_psd_matrix, calculate_daily_borrow_cost
from letf.calibration import simulate_joint_returns_t, simulate_latent_stress_state, predict_borrow_spread_series
from letf.simulation.random_start import apply_random_start_conditions

# Try to import Numba for JIT compilation (10-20x speedup on hot paths)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    # Create conditional decorator: applies @jit if Numba available, otherwise no-op
    def conditional_jit(*args, **kwargs):
        def decorator(func):
            return jit(*args, **kwargs)(func)
        return decorator
except ImportError:
    NUMBA_AVAILABLE = False
    # No-op decorator if Numba not available
    def conditional_jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Mock prange (fallback to range)
    prange = range


def generate_fat_tailed_returns(n_days: int, regime_path: np.ndarray,
                                regime_params: Dict,
                                bootstrap_sampler=None,
                                vix_dynamics: Dict[int, Dict[str, float]] = None,
                                joint_return_model: Dict = None,
                                sim_engine_mode: str = 'legacy_hybrid',
                                seed: int = None,
                                antithetic: bool = False) -> Dict[str, np.ndarray]:
    """Generate joint returns using institutional (multivariate-t) or legacy engines.

    Args:
        antithetic: If True, use antithetic variates for variance reduction (30-50%)
    """
    rng = np.random.default_rng(seed)

    if sim_engine_mode == 'institutional_v1' and joint_return_model is not None:
        joint = simulate_joint_returns_t(n_days=n_days, regime_path=regime_path, joint_model=joint_return_model, rng=rng, antithetic=antithetic)
        spy_returns = joint['SPY_Ret']
        qqq_returns = joint['QQQ_Ret']
        tlt_returns = joint['TLT_Ret']
    elif bootstrap_sampler is not None and cfg.USE_BLOCK_BOOTSTRAP:
        sampled = bootstrap_sampler.sample_returns(n_days=n_days, regime_path=regime_path, rng=rng)
        spy_boot = sampled['SPY_Ret']
        qqq_boot = sampled['QQQ_Ret']
        tlt_boot = sampled['TLT_Ret']

        # Blend historical blocks with correlated Student-t innovations
        # to preserve serial structure while allowing non-historical scenarios.
        spy_noise = np.zeros(n_days)
        qqq_noise = np.zeros(n_days)
        tlt_noise = np.zeros(n_days)
        corr_low = nearest_psd_matrix(np.array([[1.0, 0.88, -0.18], [0.88, 1.0, -0.12], [-0.18, -0.12, 1.0]]))
        corr_high = nearest_psd_matrix(np.array([[1.0, 0.94, -0.42], [0.94, 1.0, -0.30], [-0.42, -0.30, 1.0]]))
        chol_low = np.linalg.cholesky(corr_low)
        chol_high = np.linalg.cholesky(corr_high)
        noise_w = float(np.clip(1.0 - cfg.BOOTSTRAP_WEIGHT, 0.0, 1.0))

        for t in range(n_days):
            reg = int(regime_path[t])
            z = rng.standard_t(df=cfg.STUDENT_T_DF, size=3)
            x = (chol_low if reg == 0 else chol_high) @ z
            spy_std = regime_params[reg].get('daily_std', 0.01)
            qqq_std = 1.35 * spy_std
            tlt_std = 0.55 * spy_std
            spy_noise[t] = noise_w * spy_std * 0.55 * x[0]
            qqq_noise[t] = noise_w * qqq_std * 0.45 * x[1]
            tlt_noise[t] = noise_w * tlt_std * 0.35 * x[2]

        spy_returns = cfg.BOOTSTRAP_WEIGHT * spy_boot + (1.0 - cfg.BOOTSTRAP_WEIGHT) * (spy_boot + spy_noise)
        qqq_returns = cfg.BOOTSTRAP_WEIGHT * qqq_boot + (1.0 - cfg.BOOTSTRAP_WEIGHT) * (qqq_boot + qqq_noise)
        tlt_returns = cfg.BOOTSTRAP_WEIGHT * tlt_boot + (1.0 - cfg.BOOTSTRAP_WEIGHT) * (tlt_boot + tlt_noise)
    else:
        spy_returns = np.zeros(n_days)
        qqq_returns = np.zeros(n_days)
        tlt_returns = np.zeros(n_days)
        corr_low = np.array([[1.0, 0.85, -0.15], [0.85, 1.0, -0.10], [-0.15, -0.10, 1.0]])
        corr_high = np.array([[1.0, 0.92, -0.45], [0.92, 1.0, -0.30], [-0.45, -0.30, 1.0]])
        chol_low = np.linalg.cholesky(nearest_psd_matrix(corr_low))
        chol_high = np.linalg.cholesky(nearest_psd_matrix(corr_high))
        for t in range(n_days):
            reg = int(regime_path[t])
            z = rng.standard_t(df=cfg.STUDENT_T_DF, size=3)
            x = (chol_low if reg == 0 else chol_high) @ z
            spy_std = regime_params[reg].get('daily_std', 0.01)
            spy_mu = regime_params[reg].get('daily_mean', 0.0003)
            spy_returns[t] = spy_mu + spy_std * x[0]
            qqq_returns[t] = 1.15 * spy_returns[t] + 0.006 * x[1]
            tlt_returns[t] = -0.12 * spy_returns[t] + 0.004 * x[2]

    # regime-linked VIX path for downstream frictions
    vix = np.zeros(n_days)
    vix_base = {0: 15.0, 1: 35.0}
    if n_days > 0:
        vix[0] = vix_base[int(regime_path[0])]
    for t in range(1, n_days):
        reg = int(regime_path[t])
        p = (vix_dynamics or {}).get(reg, {})
        phi = p.get('phi', 0.88)
        target = p.get('target_vix', vix_base[reg])
        noise_std = p.get('noise_std', 1.2)
        jump_threshold = p.get('jump_threshold_sigma', 2.0)
        jump_scale = p.get('jump_scale', 8.0)
        denom = max(regime_params[reg].get('daily_std', 0.01), 1e-4)
        equity_shock = max(-spy_returns[t], 0.0) / denom  # VIX only spikes on negative equity moves
        vix_jump = jump_scale * max(0.0, equity_shock - jump_threshold)
        vix[t] = max(10.0, phi * vix[t-1] + (1 - phi) * target + vix_jump + rng.normal(0, noise_std))

    irx = np.zeros(n_days)
    for reg in range(cfg.N_REGIMES):
        m = regime_path == reg
        if m.sum() > 0:
            base = 3.5 if reg == 0 else 1.5
            irx[m] = base + rng.normal(0, 0.5, m.sum())
    irx = np.clip(irx, 0.0, 15.0)

    return {'SPY_Ret': spy_returns, 'QQQ_Ret': qqq_returns, 'TLT_Ret': tlt_returns, 'VIX': vix, 'IRX': irx}


@conditional_jit(nopython=True, cache=True)
def compute_letf_return_correct(underlying_return, leverage, realized_vol_daily,
                                expense_ratio, daily_borrow_cost=0):
    """
    CORRECT volatility drag formula for daily-rebalanced LETFs.
    [Numba JIT-compiled for 10-20x speedup]

    For daily rebalancing, the LETF return is simply:
    R_letf = L * R_underlying - daily_expenses - daily_borrow_cost

    The "volatility drag" (-0.5*L*(L-1)*sigma^2) emerges naturally from
    GEOMETRIC COMPOUNDING over time, not from subtracting a drag term
    each day.

    IMPORTANT: Both expense_ratio and daily_borrow_cost handling:
      - expense_ratio is ANNUAL (e.g., 0.0086 for 0.86%), divided by 252 here
      - daily_borrow_cost is ALREADY DAILY (from calculate_daily_borrow_cost),
        so it is NOT divided by 252 again
    """
    gross_return = leverage * underlying_return

    # Net return (before tracking error)
    # expense_ratio is annual -> divide by 252
    # daily_borrow_cost is already daily -> use directly
    net_return = gross_return - expense_ratio/252 - daily_borrow_cost

    return net_return


def generate_tracking_error_ar1(n_days, regime_path, vix_series, underlying_returns,
                               base_te, df_param, seed=None, rng=None,
                               rho=0.3, downside_asymmetry=1.05,
                               liquidity_series=None, clip_limit=None):
    """Tracking residual process calibrated with AR(1), fat tails, downside/liquidity asymmetry.
    [Partially vectorized for better performance]
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Multipliers scaled for EXECUTION NOISE ONLY (Option A: explicit friction model
    # handles borrow costs + expenses; TE represents bid-ask slippage and rebalance timing).
    # All multipliers are near-unity to avoid re-introducing friction through the TE channel.
    vix_multipliers = np.clip((vix_series / 20.0) ** 0.5, 0.5, 1.5)
    regime_multipliers = np.where(regime_path == 0, 1.0, 1.15)

    # Liquidity effects on funding are already modeled via predict_borrow_spread_series.
    # Only apply minimal execution-noise widening from liquidity stress.
    liq_mults = np.ones(n_days)
    if liquidity_series is not None:
        liq_mults += 0.05 * np.clip(liquidity_series[:n_days], 0.0, 3.0)

    downside_scales = np.where(underlying_returns < 0, downside_asymmetry, 0.98)
    move_multipliers = (1.0 + 0.3 * np.abs(underlying_returns)) * downside_scales

    # AR(1) loop (cannot be vectorized due to sequential dependency)
    te_series = np.zeros(n_days)
    for i in range(1, n_days):
        innovation = student_t.rvs(df=df_param, random_state=rng)
        innovation *= base_te * vix_multipliers[i] * regime_multipliers[i] * liq_mults[i]
        te_series[i] = rho * te_series[i-1] + np.sqrt(max(1 - rho**2, 1e-6)) * innovation * move_multipliers[i]

    # Demean: the asymmetric scaling (downside_asymmetry > upside) causes non-zero
    # drift that compounds over long horizons. Remove slowly-adapting bias via EMA
    # (half-life ~126 days) to force zero long-run mean while preserving serial structure.
    alpha = 2.0 / (126 + 1)
    bias = 0.0
    for i in range(n_days):
        bias = alpha * te_series[i] + (1 - alpha) * bias
        te_series[i] -= bias

    if clip_limit is None or clip_limit <= 0:
        return te_series

    cap = float(clip_limit)
    # Smoothly saturate extremes while preserving tail ordering.
    return cap * np.tanh(te_series / cap)


def validate_simulation_layers(sim_df: pd.DataFrame) -> Dict[str, float]:
    """Basic layer integrity checks for generated simulation paths."""
    checks = {
        'rows': float(len(sim_df)),
        'nan_returns': float(sim_df.filter(regex='_Ret$').isna().sum().sum()) if len(sim_df) > 0 else 0.0,
        'nan_prices': float(sim_df.filter(regex='_Price$').isna().sum().sum()) if len(sim_df) > 0 else 0.0,
        'nonfinite_returns': float((~np.isfinite(sim_df.filter(regex='_Ret$').to_numpy())).sum()) if len(sim_df) > 0 else 0.0,
        'nonfinite_prices': float((~np.isfinite(sim_df.filter(regex='_Price$').to_numpy())).sum()) if len(sim_df) > 0 else 0.0,
        'min_price': float(sim_df.filter(regex='_Price$').min().min()) if len(sim_df) > 0 else float('nan'),
        'min_vix': float(sim_df['VIX'].min()) if 'VIX' in sim_df.columns and len(sim_df) > 0 else float('nan'),
        'max_vix': float(sim_df['VIX'].max()) if 'VIX' in sim_df.columns and len(sim_df) > 0 else float('nan')
    }
    checks['is_valid'] = bool(
        checks['rows'] > 0
        and checks['nan_returns'] == 0
        and checks['nan_prices'] == 0
        and checks['nonfinite_returns'] == 0
        and checks['nonfinite_prices'] == 0
        and np.isfinite(checks['min_price'])
        and checks['min_price'] > 0.0
        and np.isfinite(checks['min_vix'])
        and np.isfinite(checks['max_vix'])
        and checks['min_vix'] >= 5.0
        and checks['max_vix'] <= 120.0
    )
    return checks


def build_simulation_metadata(sim_id: int, regime_path: np.ndarray,
                              start_conditions: Dict, stress_state: Optional[Dict],
                              layer_checks: Dict[str, float]) -> Dict:
    """Attach reproducible simulation metadata for auditing/diagnostics."""
    regime_counts = {int(r): int((regime_path == r).sum()) for r in np.unique(regime_path)}
    meta = {
        'model_version': cfg.SIM_ENGINE_MODE,
        'sim_id': int(sim_id),
        'regime_counts': regime_counts,
        'start_method': start_conditions.get('start_method'),
        'layer_checks': layer_checks
    }
    if stress_state is not None:
        meta['stress_summary'] = {
            'liq_mean': float(np.nanmean(stress_state.get('liquidity', np.array([0.0])))),
            'credit_mean': float(np.nanmean(stress_state.get('credit', np.array([0.0])))),
            'jump_days': int(np.sum((stress_state.get('jump', np.array([])) > 0).astype(int)))
        }
    return meta


def simulate_regime_path_semi_markov(total_days: int,
                                     start_regime: int,
                                     transition_matrix: np.ndarray,
                                     duration_samples: Optional[Dict[int, List[int]]],
                                     rng: np.random.Generator) -> np.ndarray:
    """Generate a semi-Markov discrete regime path with explicit dwell-time draws."""
    if total_days <= 0:
        return np.zeros(0, dtype=int)

    tm = np.asarray(transition_matrix, dtype=float).copy()
    tm = np.nan_to_num(tm, nan=0.0, posinf=0.0, neginf=0.0)
    tm[tm < 0] = 0.0
    for i in range(tm.shape[0]):
        rs = tm[i].sum()
        if rs <= 0:
            tm[i, i] = 1.0
        else:
            tm[i] = tm[i] / rs

    regime_path_full = np.zeros(total_days, dtype=int)
    current_regime = int(start_regime)
    t = 0

    while t < total_days:
        if duration_samples and current_regime in duration_samples and len(duration_samples[current_regime]) > 0:
            spell = int(rng.choice(duration_samples[current_regime]))
        else:
            p_stay = np.clip(tm[current_regime, current_regime], 0.80, 0.995)
            spell = max(1, int(rng.geometric(1 - p_stay)))

        end_t = min(t + spell, total_days)
        regime_path_full[t:end_t] = current_regime
        t = end_t

        if t < total_days:
            row = tm[current_regime].copy()
            # Semi-Markov transition at spell end: pick next state from OFF-diagonal mass.
            row[current_regime] = 0.0
            rs = row.sum()
            if rs <= 0:
                # Degenerate row fallback: remain in current regime.
                next_regime = current_regime
            else:
                row = row / rs
                next_regime = int(rng.choice(np.arange(cfg.N_REGIMES), p=row))
            current_regime = next_regime

    return regime_path_full


def map_underlying_series_for_asset(asset: str,
                                    beta: float,
                                    spy_returns: np.ndarray,
                                    qqq_returns: np.ndarray,
                                    tlt_returns: np.ndarray) -> np.ndarray:
    """Layer A: choose underlying index stream (no ETF frictions applied here)."""
    if asset in ['TQQQ', 'QQQ']:
        return qqq_returns * beta
    if asset in ['UPRO', 'SSO', 'SPY']:
        return spy_returns * beta
    if asset == 'TMF':
        return tlt_returns
    return spy_returns


def compute_daily_financing_series(leverage: float,
                                   risk_free_annual: np.ndarray,
                                   vix: np.ndarray,
                                   funding_model: Optional[Dict[str, float]],
                                   fallback_spread: float,
                                   stress_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Layer C: state-linked financing cost (daily), separated from return dynamics."""
    n = len(risk_free_annual)
    if cfg.SIM_ENGINE_MODE == 'institutional_v1' and funding_model is not None:
        irx_level = risk_free_annual * 100
        if stress_state is not None and 'credit' in stress_state:
            cred = np.clip(np.asarray(stress_state['credit'], dtype=float), 0.0, 3.0)
            term_spread = 1.25 - 1.1 * cred
        else:
            term_spread = np.where(vix > 30, -0.25, 1.10)
        tnx_level = irx_level + term_spread

        spread_df = pd.DataFrame({
            'VIX': vix,
            'TNX': tnx_level,
            'IRX': irx_level
        })
        spread_series = predict_borrow_spread_series(spread_df, funding_model, stress_state=stress_state)
    else:
        spread_series = np.full(n, fallback_spread)

    return np.array([
        calculate_daily_borrow_cost(leverage, risk_free_annual[t], spread_series[t])
        for t in range(n)
    ])


def _stable_asset_seed(sim_id: int, asset: str) -> int:
    """Collision-resistant deterministic seed component from full ticker string."""
    h = 0
    for i, ch in enumerate(asset):
        h += (i + 1) * ord(ch)
    return int(sim_id + 7919 * h)


def simulate_etf_returns_from_layers(asset: str,
                                     config: Dict,
                                     underlying: np.ndarray,
                                     regime_path: np.ndarray,
                                     vix: np.ndarray,
                                     risk_free_annual: np.ndarray,
                                     sim_id: int,
                                     funding_model: Optional[Dict[str, float]],
                                     tracking_residual_model: Optional[Dict],
                                     stress_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Layer B/C/D composition: leverage math, financing drag, calibrated ETF residual."""
    leverage = config['leverage']
    expense_ratio = config['expense_ratio']

    daily_borrow = compute_daily_financing_series(
        leverage=leverage,
        risk_free_annual=risk_free_annual,
        vix=vix,
        funding_model=funding_model,
        fallback_spread=config.get('borrow_spread', 0.0),
        stress_state=stress_state
    )

    leveraged_before_te = np.array([
        compute_letf_return_correct(underlying[t], leverage, 0, expense_ratio, daily_borrow[t])
        for t in range(len(underlying))
    ])

    te_params = (tracking_residual_model or {}).get(asset, {})
    # Use config-level tracking_error_base (execution noise only, ~2 bps).
    # The calibrated base_scale (~16 bps) embeds friction already captured
    # by explicit borrow cost + expense ratio, causing double-counting.
    te_scale = config['tracking_error_base']
    # Cap downside_mult: calibrated values can reach 2.0 because the residual
    # absorbs friction asymmetry.  Under Option A (explicit friction + minimal TE),
    # only genuine execution-noise asymmetry should remain (≤1.10).
    te_downside = min(te_params.get('downside_mult', 1.05), 1.10)

    # Liquidity effects on funding costs are already modeled via
    # predict_borrow_spread_series; passing liquidity to TE would double-count.
    liquidity = None

    tracking_errors = generate_tracking_error_ar1(
        len(underlying),
        regime_path,
        vix,
        underlying,
        te_scale,
        te_params.get('df', config['tracking_error_df']),
        rng=np.random.default_rng(_stable_asset_seed(sim_id, asset)),
        rho=te_params.get('rho', 0.3),
        downside_asymmetry=te_downside,
        liquidity_series=liquidity,
        clip_limit=te_params.get('clip_limit', None)
    )

    # Apply tracking error ADDITIVELY (not multiplicatively!)
    # Tracking error is measured in basis points of return drag, so we simply add it
    etf_ret = leveraged_before_te + tracking_errors
    # Prevent economically impossible daily loss below -100%.
    return np.clip(etf_ret, -0.999, 10.0)


def simulate_single_path_fixed(args):
    """Single Monte Carlo path with institutional layer separation and semi-Markov states.
    [Enhanced with comprehensive error handling and validation]
    """
    from letf.strategy import run_strategy_fixed

    # Unpack args - handle both old format (5 items) and new format (6 items with antithetic)
    if len(args) == 6:
        sim_id, sim_years, regime_model, correlation_matrices, strategies, antithetic = args
    else:
        sim_id, sim_years, regime_model, correlation_matrices, strategies = args
        antithetic = False  # Default for backward compatibility

    rng = np.random.default_rng(sim_id + 50000)
    sim_days = int(sim_years * 252)
    regime_params = regime_model['regime_params']
    transition_matrix = regime_model['transition_matrix']
    duration_samples = regime_model.get('duration_samples', None)
    vix_dynamics = regime_model.get('vix_dynamics', None)
    joint_return_model = regime_model.get('joint_return_model', None)
    funding_model = regime_model.get('funding_model', None)
    tracking_residual_model = regime_model.get('tracking_residual_model', None)
    stress_state_model = regime_model.get('stress_state_model', None)

    start_conditions = apply_random_start_conditions(
        sim_id,
        sim_days,
        regime_model,
        regime_model.get('historical_df_for_anchors', None)
    )
    start_regime = start_conditions['start_regime']
    initial_vix = start_conditions['initial_vix']
    buffer_days = start_conditions['buffer_days']
    start_offset = start_conditions['start_offset']
    total_days = sim_days + buffer_days

    regime_path_full = simulate_regime_path_semi_markov(
        total_days=total_days,
        start_regime=start_regime,
        transition_matrix=transition_matrix,
        duration_samples=duration_samples,
        rng=rng
    )
    regime_path = regime_path_full[start_offset:start_offset + sim_days]

    bootstrap_sampler = regime_model.get('bootstrap_sampler', None)
    generated = generate_fat_tailed_returns(
        total_days,
        regime_path_full,
        regime_params,
        bootstrap_sampler,
        vix_dynamics=vix_dynamics,
        joint_return_model=joint_return_model,
        sim_engine_mode=cfg.SIM_ENGINE_MODE,
        seed=sim_id + 50000,
        antithetic=antithetic
    )

    spy_returns = generated['SPY_Ret'][start_offset:start_offset + sim_days]
    qqq_returns_raw = generated['QQQ_Ret'][start_offset:start_offset + sim_days]
    tlt_returns_raw = generated['TLT_Ret'][start_offset:start_offset + sim_days]

    # Validation: Check for NaN/Inf in generated returns
    if not np.all(np.isfinite(spy_returns)):
        raise ValueError(f"Sim {sim_id}: Non-finite SPY returns detected. Check GARCH/DCC parameters.")
    if not np.all(np.isfinite(qqq_returns_raw)):
        raise ValueError(f"Sim {sim_id}: Non-finite QQQ returns detected. Check GARCH/DCC parameters.")
    if not np.all(np.isfinite(tlt_returns_raw)):
        raise ValueError(f"Sim {sim_id}: Non-finite TLT returns detected. Check GARCH/DCC parameters.")

    generated_vix = generated.get('VIX', None)
    if generated_vix is not None:
        vix = generated_vix[start_offset:start_offset + sim_days]
        # Validation: VIX must be positive and finite
        if not np.all(np.isfinite(vix)) or np.any(vix <= 0):
            raise ValueError(f"Sim {sim_id}: Invalid VIX values detected. VIX must be positive and finite.")
    else:
        vix = np.full(sim_days, initial_vix)

    stress_state = None
    if cfg.SIM_ENGINE_MODE == 'institutional_v1' and stress_state_model is not None:
        stress_state = simulate_latent_stress_state(sim_days, regime_path, stress_state_model, vix_series=vix, rng=rng)
        jump = stress_state.get('jump', np.zeros(sim_days))
        # Make jumps symmetric: randomly positive or negative (50/50)
        # The Student-t model already captures fat tails; jumps should add
        # tail volatility, not one-directional drag
        jump_signs = rng.choice([-1.0, 1.0], size=sim_days)
        signed_jump = jump * jump_signs
        spy_returns = np.clip(spy_returns - signed_jump, -0.95, 3.0)
        qqq_returns_raw = np.clip(qqq_returns_raw - 1.12 * signed_jump, -0.95, 4.0)

    irx_boot = generated.get('IRX', None)
    if irx_boot is not None:
        risk_free_annual = np.clip(irx_boot[start_offset:start_offset + sim_days], 0.0, 20.0) / 100.0
    else:
        risk_free_annual = np.where(regime_path == 0, 0.045, 0.015)

    asset_returns = {}
    for asset, config in cfg.LETF_CONFIGS.items():
        beta = config.get('beta_to_spy', 1.0)
        underlying = map_underlying_series_for_asset(asset, beta, spy_returns, qqq_returns_raw, tlt_returns_raw)
        asset_returns[asset] = simulate_etf_returns_from_layers(
            asset=asset,
            config=config,
            underlying=underlying,
            regime_path=regime_path,
            vix=vix,
            risk_free_annual=risk_free_annual,
            sim_id=sim_id,
            funding_model=funding_model,
            tracking_residual_model=tracking_residual_model,
            stress_state=stress_state
        )

    assets_order = ['TQQQ', 'UPRO', 'SPY', 'QQQ', 'TMF', 'SSO']
    sim_df = pd.DataFrame({f'{k}_Ret': v for k, v in asset_returns.items()})

    cash_ret = np.zeros(sim_days)
    for regime in range(cfg.N_REGIMES):
        m = regime_path == regime
        cash_ret[m] = cfg.CASH_RATE_BY_REGIME[regime] / 252
    sim_df['Cash_Ret'] = cash_ret

    for asset in assets_order:
        sim_df[f'{asset}_Price'] = (1 + sim_df[f'{asset}_Ret'].fillna(0)).cumprod() * 100
    sim_df['TLT_Ret'] = tlt_returns_raw
    sim_df['TLT_Price'] = (1 + sim_df['TLT_Ret'].fillna(0)).cumprod() * 100
    sim_df['VIX'] = vix

    layer_checks = validate_simulation_layers(sim_df)
    if not layer_checks.get('is_valid', False):
        # Hard safety pass: bound returns and rebuild prices to avoid invalid path propagation.
        for c in [col for col in sim_df.columns if col.endswith('_Ret')]:
            sim_df[c] = np.clip(sim_df[c].astype(float), -0.999, 10.0)
        for asset in assets_order:
            sim_df[f'{asset}_Price'] = (1 + sim_df[f'{asset}_Ret'].fillna(0)).cumprod() * 100
        sim_df['TLT_Ret'] = np.clip(tlt_returns_raw, -0.999, 10.0)
        sim_df['TLT_Price'] = (1 + sim_df['TLT_Ret'].fillna(0)).cumprod() * 100
        layer_checks = validate_simulation_layers(sim_df)
        layer_checks['repaired_after_validation'] = True

    metadata = build_simulation_metadata(sim_id, regime_path, start_conditions, stress_state, layer_checks)

    from letf.trade import TradeJournal, TAXABLE_IDS

    path_results = {}
    for sid in strategies:
        try:
            # Create TradeJournal for taxable strategies (needed for tax calc)
            journal = TradeJournal() if sid in TAXABLE_IDS else None

            equity_curve, trades = run_strategy_fixed(
                sim_df, sid, regime_path, correlation_matrices,
                apply_costs=True, trade_journal=journal
            )
            final_wealth = float(equity_curve.iloc[-1]) if len(equity_curve) else 0.0

            # Compute max drawdown from equity curve
            max_dd = 0.0
            if len(equity_curve) > 1:
                ec = equity_curve.values
                running_max = np.maximum.accumulate(ec)
                drawdowns = (ec - running_max) / np.where(running_max > 0, running_max, 1.0)
                max_dd = float(abs(drawdowns.min()))

            # Compute trades per year
            num_trades = int(trades)
            trades_per_year = num_trades / sim_years if sim_years > 0 else 0.0

            result = {
                'Final_Wealth': final_wealth,
                'Num_Trades': num_trades,
                'Trades_Per_Year': trades_per_year,
                'Max_DD': max_dd,
                'Regime_Path': regime_path.tolist(),
                'Metadata': metadata,
            }

            # Include trade list for taxable strategies (needed for tax engine)
            if journal is not None:
                result['Trade_List'] = journal.get_full_trades()

            path_results[sid] = result
        except Exception as e:
            import traceback
            print(f"  [ERR] Sim {sim_id} strategy {sid}: {e}")
            traceback.print_exc()
            path_results[sid] = {
                'Final_Wealth': 0,
                'Num_Trades': 0,
                'Trades_Per_Year': 0.0,
                'Max_DD': 0.0,
                'Error': str(e),
                'Regime_Path': regime_path.tolist()
            }

    return path_results
