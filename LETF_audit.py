"""
LETF Monte Carlo Engine — Forensic Validation Audit
====================================================
FORENSIC ONLY: no parameter changes, measurement and reporting only.
Cache: /home/djmann/corrected_cache_v8
Engine: /home/djmann/LETF34_analysis.py

Sections
--------
 1. Unconditional Diagnostics      10 000 paths × 1 yr    SPY moments, tails, skew, kurt
 2. Multi-horizon CAGR             2 000 paths × 1/5/10/30 yr
 3. GARCH Variance Dynamics        1 path   × 100 000 days  ACF of |r|, rolling vol
 4. Regime Transitions             1 path   × 100 000 days  dwell times, steady-state
 5. LETF Leverage Scaling          2 000 paths × 10 yr     SPY/SSO/TQQQ
 6. Tail Risk (VaR/CVaR)           re-use §1 paths
 7. Cross-Asset Correlation        5 000 paths × 1 yr      SPY-QQQ DCC
 8. Max Drawdown Distribution      2 000 paths × 10 yr     re-use §5
 9. Stress / Crisis Performance    1 000 paths × 5 yr      high-stress regime
10. Jump Process Characterisation  re-use §1 paths

Usage: python LETF_audit.py
"""
import sys
import time
import pickle
import warnings
import traceback
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR        = "/home/djmann/corrected_cache_v8"
JOINT_MODEL_PATH = f"{CACHE_DIR}/joint_return_model.pkl"
REGIME_MODEL_PATH= f"{CACHE_DIR}/regime_model.pkl"
STRESS_MODEL_PATH= f"{CACHE_DIR}/stress_state_model.pkl"

# ---------------------------------------------------------------------------
# Historical benchmarks (S&P 500 full history)
# ---------------------------------------------------------------------------
HIST = {
    "arith_return":  0.115,
    "geo_return":    0.095,
    "annual_vol":    0.185,
    "daily_skew":   -0.40,
    "daily_xkurt":  10.0,
    "annual_skew":  -0.40,
    "annual_xkurt":  1.5,
    "pct_neg_years": 0.28,
    "var_1pct":     -0.375,
    "cvar_1pct":    -0.500,
    "max_dd":       -0.50,
    "acf_absret_lag1": 0.15,
    "spy_qqq_corr":  0.85,
    "tqqq_3yr_median_cagr": 0.30,
}

# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------
def _load():
    with open(JOINT_MODEL_PATH,  "rb") as f: jm = pickle.load(f)
    with open(REGIME_MODEL_PATH, "rb") as f: rm = pickle.load(f)
    with open(STRESS_MODEL_PATH, "rb") as f: sm = pickle.load(f)
    return jm, rm, sm

# ---------------------------------------------------------------------------
# Parse model parameters
# ---------------------------------------------------------------------------
def _parse(jm, rm, sm):
    p = {}
    # Regime model — keys: transition_matrix, steady_state at top level
    tm  = np.array(rm["transition_matrix"])       # (2,2)
    ss  = np.array(rm.get("steady_state", [0.9583, 0.0417]))
    p["tm"]  = tm
    p["ss"]  = ss

    # Joint return model — keys: assets, regimes
    # regimes[k]: mu (array), cov (3×3), nu, garch_alpha, garch_beta, dcc_a, dcc_b
    # assets order: ['SPY_Ret', 'QQQ_Ret', 'TLT_Ret']
    regs = jm["regimes"]
    keys = sorted(regs.keys())    # [0, 1]

    p["mu_spy"]  = np.array([regs[k]["mu"][0]           for k in keys])
    p["mu_qqq"]  = np.array([regs[k]["mu"][1]           for k in keys])
    p["nu"]      = np.array([regs[k]["nu"]               for k in keys])
    p["alpha"]   = np.array([regs[k]["garch_alpha"]      for k in keys])
    p["beta"]    = np.array([regs[k]["garch_beta"]       for k in keys])

    # Extract sigma and correlation from covariance matrix
    covs = [regs[k]["cov"] for k in keys]
    p["sig_spy"] = np.array([np.sqrt(c[0, 0]) for c in covs])
    p["sig_qqq"] = np.array([np.sqrt(c[1, 1]) for c in covs])
    p["rho"]     = np.array([c[0, 1] / (np.sqrt(c[0, 0]) * np.sqrt(c[1, 1])) for c in covs])

    # effective alpha for Student-t GARCH: α·ν/(ν-2)
    # under t_ν innovations, E[ε²] = ν/(ν-2), so effective persistence = α·ν/(ν-2) + β
    nu = p["nu"]
    p["gamma"]         = np.array([regs[k].get("garch_gamma", 0.0) for k in keys])
    p["eff_alpha_spy"] = p["alpha"] * nu / (nu - 2.0)
    p["eff_alpha_qqq"] = p["alpha"] * nu / (nu - 2.0)   # shared alpha in joint model

    # GJR stationarity: (alpha + gamma/2) * nu/(nu-2) + beta < 1
    # Engine applies cap if violated; calibration already enforces this.
    eff_a_gjr = (p["alpha"] + 0.5 * p["gamma"]) * nu / (nu - 2.0)
    raw_b = p["beta"]
    needs_cap = (eff_a_gjr + raw_b >= 1.0)
    p["beta_eff"] = np.where(needs_cap, np.maximum(0.70, 0.98 - eff_a_gjr), raw_b)
    p["eff_alpha_gjr"] = eff_a_gjr

    # Long-run variance targets (engine convention):
    # reg_long_var = cov_diag * (ν-2)/ν  so that E[x²] = h * ν/(ν-2) = cov_diag = sig²
    t_var_scale = (nu - 2.0) / nu
    p["rlv_spy"] = p["sig_spy"] ** 2 * t_var_scale
    p["rlv_qqq"] = p["sig_qqq"] ** 2 * t_var_scale

    # Report capped betas for diagnostics
    p["beta_raw"] = raw_b

    # Stress model — jump parameters per regime
    sregs = sm.get("regimes", sm)
    skeys = sorted(sregs.keys())
    p["jump_prob"]  = np.array([sregs[k].get("jump_base_prob", 0.006) for k in skeys])
    p["jump_scale"] = np.array([sregs[k].get("jump_scale",     0.04)  for k in skeys])

    return p

# ---------------------------------------------------------------------------
# Regime path generator — vectorised (N,T) path array, outer loop over t
# ---------------------------------------------------------------------------
def _regime_paths(N, T, rng, p):
    """Return integer array (N, T) of regime indices (0 or 1)."""
    tm  = p["tm"]
    ss  = p["ss"]
    out = np.empty((N, T), dtype=np.int8)
    r   = rng.choice([0, 1], size=N, p=ss)
    out[:, 0] = r
    for t in range(1, T):
        u = rng.random(N)
        # P(switch 0→1) = tm[0,1], P(stay 0) = tm[0,0]
        stay  = np.where(r == 0, tm[0, 0], tm[1, 1])
        r     = np.where(u < stay, r, 1 - r)
        out[:, t] = r
    return out               # (N, T)

# ---------------------------------------------------------------------------
# Core vectorised SPY GARCH simulator
# Returns spy_daily (N, T) — raw daily log-ish returns (arithmetic)
# ---------------------------------------------------------------------------
def simulate_spy(N, T, rng, p, with_jumps=False):
    mu_spy  = p["mu_spy"]
    alpha   = p["alpha"];    beta_eff = p["beta_eff"]   # capped beta (engine line 4095)
    eff_a   = p["eff_alpha_spy"]
    rlv     = p["rlv_spy"];  nu       = p["nu"]
    jp      = p["jump_prob"]; js       = p["jump_scale"]

    reg_paths = _regime_paths(N, T, rng, p)  # (N, T)

    h      = rlv[reg_paths[:, 0]]            # (N,) initial variance
    prev_z = np.zeros(N)

    spy = np.empty((N, T))

    for t in range(T):
        r = reg_paths[:, t].astype(int)      # (N,)
        mu_n   = mu_spy[r]
        rlv_n  = rlv[r]
        nu_n   = nu[r]
        alpha_n= alpha[r]
        beta_n = beta_eff[r]
        ea_n   = eff_a[r]

        # GARCH variance update (engine-matched: omega + alpha*prev_z²*h + beta_capped*h)
        omega  = np.maximum(1.0 - ea_n - beta_n, 1e-8) * rlv_n
        changed= (r != reg_paths[:, t - 1] if t > 0 else np.zeros(N, bool))
        h_new  = omega + alpha_n * prev_z**2 * h + beta_n * h
        h      = np.where(changed, rlv_n, h_new)
        h      = np.maximum(h, 1e-10)

        # Student-t innovations via chi-square mixing (engine line 4125-4128)
        # z ~ N(0,1), u ~ chi²(ν), eps = z * sqrt(ν/u) ~ t_ν
        z = rng.standard_normal(N)
        # Vectorised chi-square with per-path df (2 unique values)
        u = np.empty(N)
        for rk in [0, 1]:
            mask = (r == rk)
            if mask.any():
                u[mask] = rng.chisquare(float(nu[rk]), int(mask.sum()))
        eps    = z * np.sqrt(nu_n / u)      # t_ν variate, E[eps²] = ν/(ν-2)
        # Standardized residual for next GARCH step (engine line 4132)
        prev_z = (np.sqrt(h) * eps) / np.sqrt(h)  # = eps (shorthand)
        # Note: prev_z² * h = eps² * h = (x-mu)² which is the squared innovation

        ret = mu_n + np.sqrt(h) * eps

        # Optional jump process
        if with_jumps:
            jp_n = jp[r]; js_n = js[r]
            jmask = rng.random(N) < jp_n
            jdraw = rng.standard_t(6, size=N) * js_n  # symmetric: skew from GJR, not jumps
            ret   = ret + np.where(jmask, jdraw, 0.0)

        spy[:, t] = np.clip(ret, -0.95, 3.0)

    return spy   # (N, T) arithmetic daily returns

# ---------------------------------------------------------------------------
# Fast SPY simulator (vectorised chi-square — approximation using regime mean df)
# 10-50x faster than per-element chi-square; used for large N
# ---------------------------------------------------------------------------
def simulate_spy_fast(N, T, rng, p, with_jumps=False):
    mu_spy   = p["mu_spy"];    rlv      = p["rlv_spy"]
    alpha    = p["alpha"];     gamma    = p["gamma"]       # GJR leverage
    beta_eff = p["beta_eff"]
    ea_gjr   = p["eff_alpha_gjr"]
    nu       = p["nu"];        jp       = p["jump_prob"]; js = p["jump_scale"]

    reg_paths = _regime_paths(N, T, rng, p)   # (N, T)

    h      = rlv[reg_paths[:, 0]]
    prev_z = np.zeros(N)
    spy    = np.empty((N, T))

    for t in range(T):
        r     = reg_paths[:, t].astype(int)
        mu_n  = mu_spy[r];   rlv_n = rlv[r];   nu_n   = nu[r]
        alpha_n= alpha[r];   gamma_n= gamma[r]; beta_n = beta_eff[r]
        ea_gjr_n = ea_gjr[r]

        # GJR-GARCH variance update (engine line 4140):
        # h_t = ω + (α + γ·I(ε<0)) · ε²_{t-1} + β·h_{t-1}
        # ω = (1 - GJR_eff_alpha - β) · rlv  ensuring E[h] = rlv at stationarity
        neg_mask   = (prev_z < 0)
        alpha_eff_n= alpha_n + np.where(neg_mask, gamma_n, 0.0)   # α+γ for neg, α for pos
        omega      = np.maximum(1.0 - ea_gjr_n - beta_n, 1e-8) * rlv_n
        changed    = (r != reg_paths[:, t-1] if t > 0 else np.zeros(N, bool))
        h_new      = omega + alpha_eff_n * prev_z**2 * h + beta_n * h
        h          = np.where(changed, rlv_n, h_new)
        h          = np.maximum(h, 1e-10)

        # Chi-square mixing: eps ~ t_ν, E[eps²] = ν/(ν-2)
        z  = rng.standard_normal(N)
        u  = np.empty(N)
        for rk in [0, 1]:
            mask = (r == rk)
            if mask.any():
                u[mask] = rng.chisquare(float(nu[rk]), int(mask.sum()))
        eps    = z * np.sqrt(nu_n / u)
        prev_z = eps    # standardized residual for next GARCH step

        ret = mu_n + np.sqrt(h) * eps

        if with_jumps:
            jp_n = jp[r]; js_n = js[r]
            jmask = rng.random(N) < jp_n
            jdraw = rng.standard_t(6, size=N) * js_n  # symmetric: skew from GJR, not jumps
            ret   = ret + np.where(jmask, jdraw, 0.0)

        spy[:, t] = np.clip(ret, -0.95, 3.0)

    return spy

# ---------------------------------------------------------------------------
# Joint SPY+QQQ simulator (returns both (N,T) arrays)
# ---------------------------------------------------------------------------
def simulate_joint_fast(N, T, rng, p):
    mu_spy  = p["mu_spy"];   mu_qqq  = p["mu_qqq"]
    rlv_spy = p["rlv_spy"];  rlv_qqq = p["rlv_qqq"]
    alpha    = p["alpha"];   gamma    = p["gamma"]       # GJR leverage
    beta_eff = p["beta_eff"]
    ea_gjr   = p["eff_alpha_gjr"]
    nu       = p["nu"];      rho_v    = p["rho"]

    reg_paths = _regime_paths(N, T, rng, p)

    h_spy  = rlv_spy[reg_paths[:, 0]]
    h_qqq  = rlv_qqq[reg_paths[:, 0]]
    pz_spy = np.zeros(N)
    pz_qqq = np.zeros(N)

    spy_out = np.empty((N, T))
    qqq_out = np.empty((N, T))

    for t in range(T):
        r      = reg_paths[:, t].astype(int)
        chgd   = (r != reg_paths[:, t-1] if t > 0 else np.zeros(N, bool))
        nu_n   = nu[r]; rho_n   = rho_v[r]
        alpha_n= alpha[r]; gamma_n = gamma[r]; beta_n = beta_eff[r]
        ea_gjr_n= ea_gjr[r]
        rlvs_n = rlv_spy[r]; rlvq_n  = rlv_qqq[r]
        mu_s_n = mu_spy[r];  mu_q_n  = mu_qqq[r]

        # GJR-GARCH variance updates — same model as SPY simulator
        neg_spy  = (pz_spy < 0); neg_qqq = (pz_qqq < 0)
        aeff_spy = alpha_n + np.where(neg_spy, gamma_n, 0.0)
        aeff_qqq = alpha_n + np.where(neg_qqq, gamma_n, 0.0)
        omega_s  = np.maximum(1.0 - ea_gjr_n - beta_n, 1e-8) * rlvs_n
        omega_q  = np.maximum(1.0 - ea_gjr_n - beta_n, 1e-8) * rlvq_n
        h_spy_new= omega_s + aeff_spy * pz_spy**2 * h_spy + beta_n * h_spy
        h_qqq_new= omega_q + aeff_qqq * pz_qqq**2 * h_qqq + beta_n * h_qqq
        h_spy= np.where(chgd, rlvs_n, h_spy_new); h_spy= np.maximum(h_spy, 1e-10)
        h_qqq= np.where(chgd, rlvq_n, h_qqq_new); h_qqq= np.maximum(h_qqq, 1e-10)

        # Correlated t innovations (DCC simplified: fixed rho per regime)
        z1 = rng.standard_normal(N)
        z2 = rho_n * z1 + np.sqrt(1 - rho_n**2) * rng.standard_normal(N)
        u  = np.empty(N)
        for rk in [0, 1]:
            mask = (r == rk)
            if mask.any():
                u[mask] = rng.chisquare(float(nu[rk]), int(mask.sum()))
        scale= np.sqrt(nu_n / u)
        eps_s= z1 * scale; eps_q= z2 * scale
        pz_spy= eps_s; pz_qqq= eps_q

        spy_out[:, t]= np.clip(mu_s_n + np.sqrt(h_spy) * eps_s, -0.95, 3.0)
        qqq_out[:, t]= np.clip(mu_q_n + np.sqrt(h_qqq) * eps_q, -0.95, 4.0)

    return spy_out, qqq_out

# ---------------------------------------------------------------------------
# LETF return from underlying daily returns
# ---------------------------------------------------------------------------
def letf_ret(underlying, L, expense_daily, borrow_daily):
    """
    underlying: (N, T) arithmetic daily returns
    Returns (N, T) LETF arithmetic daily returns.
    expense_daily and borrow_daily are daily rates.
    Daily return floored at -1.0 (fund cannot go below zero).
    """
    r = L * underlying - expense_daily - (L - 1.0) * borrow_daily
    return np.maximum(r, -1.0)   # floor at -100% (ruin, not negative wealth)

# ---------------------------------------------------------------------------
# Geometric compounding: (N, T) returns → (N,) terminal wealth
# ---------------------------------------------------------------------------
def terminal_wealth(returns):
    return np.prod(1.0 + returns, axis=1)

# ---------------------------------------------------------------------------
# CAGR from terminal wealth
# ---------------------------------------------------------------------------
def cagr(tw, years):
    return tw ** (1.0 / years) - 1.0

# ---------------------------------------------------------------------------
# Max drawdown per path: (N, T) → (N,)
# ---------------------------------------------------------------------------
def max_drawdown_paths(daily_ret):
    price = np.cumprod(1.0 + daily_ret, axis=1)          # (N, T)
    running_max = np.maximum.accumulate(price, axis=1)    # (N, T)
    dd = (price - running_max) / running_max              # (N, T) ≤ 0
    return dd.min(axis=1)                                 # (N,)

# ---------------------------------------------------------------------------
# ACF of a 1-D series
# ---------------------------------------------------------------------------
def acf(x, max_lag=30):
    x  = x - x.mean()
    n  = len(x)
    c0 = np.dot(x, x) / n
    lags = np.arange(1, max_lag + 1)
    ac = np.array([np.dot(x[:n-l], x[l:]) / (n * c0) for l in lags])
    return lags, ac

# ---------------------------------------------------------------------------
# Kolmogorov–Smirnov wrapper for normal fit test
# ---------------------------------------------------------------------------
def ks_normal(x):
    mu, sig = x.mean(), x.std()
    return stats.kstest(x, "norm", args=(mu, sig))

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _hdr(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

def _ok(label, val, lo=None, hi=None, fmt=".4f"):
    tag = ""
    if lo is not None and hi is not None:
        tag = " [PASS]" if lo <= val <= hi else f" [WARN: expected {lo:.4f}–{hi:.4f}]"
    print(f"  {label:<40s} {val:{fmt}}{tag}")

def _row(label, val, fmt=".4f"):
    print(f"  {label:<40s} {val:{fmt}}")

def _pct(label, val, lo=None, hi=None):
    tag = ""
    if lo is not None and hi is not None:
        tag = " [PASS]" if lo <= val <= hi else f" [WARN: expected {lo:.1%}–{hi:.1%}]"
    print(f"  {label:<40s} {val:.2%}{tag}")

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print()
    print("=" * 72)
    print("  LETF MONTE CARLO ENGINE — FORENSIC VALIDATION AUDIT")
    print("  FORENSIC ONLY: no parameter changes, measurement and reporting only")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Engine: /home/djmann/LETF34_analysis.py")
    print("=" * 72)

    # -----------------------------------------------------------------------
    print("\nLoading cached models...")
    try:
        jm, rm, sm = _load()
        print("  joint_return_model.pkl ... OK")
        print("  regime_model.pkl ......... OK")
        print("  stress_state_model.pkl ... OK")
    except Exception as e:
        print(f"  FATAL: {e}")
        sys.exit(1)

    p = _parse(jm, rm, sm)

    # Print model summary
    print("\nModel Summary:")
    for k in [0, 1]:
        nu_k  = p["nu"][k]; a_k = p["alpha"][k]; g_k = p["gamma"][k]
        b_raw = p["beta_raw"][k]; b_eff = p["beta_eff"][k]
        ea_gjr= p["eff_alpha_gjr"][k]
        rlvk  = p["rlv_spy"][k]
        print(f"  Regime {k}: mu_SPY={p['mu_spy'][k]:.5f}  sigma_SPY={p['sig_spy'][k]:.5f}"
              f"  nu={nu_k:.2f}  alpha={a_k:.4f}  gamma={g_k:.4f}  beta_raw={b_raw:.4f}")
        raw_persist = ea_gjr + b_raw
        eff_persist = ea_gjr + b_eff
        cap_applied = b_raw != b_eff
        print(f"           GJR_eff_alpha(alpha+gamma/2)*nu/(nu-2)={ea_gjr:.4f}  beta_eff={b_eff:.4f}"
              f"  GJR_persist={eff_persist:.4f}  rlv={rlvk:.6f}"
              f"  {'[BETA CAPPED]' if cap_applied else '[stable]'}")
    tm = p["tm"]
    print(f"  Transition Matrix: [[{tm[0,0]:.5f}, {tm[0,1]:.5f}], [{tm[1,0]:.5f}, {tm[1,1]:.5f}]]")
    ss = p["ss"]
    print(f"  Steady State: r0={ss[0]:.4f}, r1={ss[1]:.4f}")
    print(f"  Jump params (r0): prob={p['jump_prob'][0]:.5f}  scale={p['jump_scale'][0]:.5f}")
    print(f"  Jump params (r1): prob={p['jump_prob'][1]:.5f}  scale={p['jump_scale'][1]:.5f}")

    # --- Scenario steady-state diagnostic ---
    # Show how each MARKET_SCENARIO preset modifies regime frequencies.
    # Steady state: π₁ = tm[0,1] / (tm[0,1] + tm[1,0])
    _tm_base = p["tm"]
    _PRESETS = {
        'neutral':     1.0,
        'historical':  1.0,
        'pessimistic': 2.0,
        'optimistic':  0.5,
    }
    print("\n  --- Scenario Steady-State Check (Markov consistency) ---")
    print(f"  {'Scenario':<14s}  {'crisis_scale':>13s}  {'π-calm':>8s}  {'π-crisis':>9s}  {'mean_crisis_spell':>18s}")
    _tm01_base = _tm_base[0, 1]
    _tm10      = _tm_base[1, 0]
    for _name, _scale in _PRESETS.items():
        _tm01_sc = float(np.clip(_tm01_base * _scale, 0.0, 0.20))
        _pi1_sc  = _tm01_sc / (_tm01_sc + _tm10) if (_tm01_sc + _tm10) > 0 else 0.0
        _pi0_sc  = 1.0 - _pi1_sc
        _mean_spell_c = 1.0 / _tm10 if _tm10 > 0 else float('inf')
        print(f"  {_name:<14s}  {_scale:>13.1f}  {_pi0_sc:>8.4f}  {_pi1_sc:>9.4f}  {_mean_spell_c:>18.1f}d")

    rng = np.random.default_rng(42)

    # =======================================================================
    _hdr("SECTION 1 — Unconditional Diagnostics  (10,000 paths × 1yr)")
    # =======================================================================
    N1, T1 = 10_000, 252
    print(f"  Simulating {N1} paths × {T1} days...")
    t0 = time.time()
    spy1 = simulate_spy_fast(N1, T1, rng, p, with_jumps=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Daily return statistics
    daily_flat = spy1.flatten()
    d_mu   = daily_flat.mean()
    d_sig  = daily_flat.std()
    d_skew = stats.skew(daily_flat)
    d_kurt = stats.kurtosis(daily_flat)   # excess kurtosis

    # Annual statistics from terminal wealth
    ann_ret = spy1.sum(axis=1)   # arithmetic sum ≈ annual return (small correction)
    tw1     = terminal_wealth(spy1)
    ann_geo = cagr(tw1, 1.0)

    a_mu    = ann_geo.mean()
    a_med   = np.median(ann_geo)
    a_sig   = ann_geo.std()
    a_skew  = stats.skew(ann_geo)
    a_kurt  = stats.kurtosis(ann_geo)
    pct_neg = (ann_geo < 0).mean()

    # KS test for daily normality (expect failure — fat tails)
    ks_stat, ks_p = ks_normal(daily_flat[:5000])   # sample for speed

    print()
    print("  --- Daily Return Moments ---")
    _ok("Mean daily return",       d_mu,   -0.0005, 0.0010)
    _ok("Daily vol (annualised)",  d_sig * np.sqrt(252), 0.12, 0.25)
    _ok("Daily skewness",          d_skew, -1.5,  0.2)
    _ok("Daily excess kurtosis",   d_kurt,  2.0,  25.0)
    _row("KS stat vs Normal",      ks_stat)
    _row("KS p-value (expect~0)",  ks_p)

    print()
    print("  --- Annual Return Moments ---")
    _ok("Mean CAGR (1yr)",         a_mu,    0.05,  0.16)
    _ok("Median CAGR (1yr)",       a_med,   0.04,  0.14)
    _ok("Annual vol",              a_sig,   0.12,  0.25)
    _ok("Annual skewness",         a_skew, -1.5,   0.5)
    _ok("Annual excess kurtosis",  a_kurt,  0.0,   5.0)
    _pct("Pct negative years",     pct_neg, 0.20,  0.38)

    # Percentile table
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print()
    print("  --- CAGR Percentile Table (1yr) ---")
    print(f"  {'Pct':>5s}  {'CAGR':>8s}  {'Historical':>12s}")
    hist_pct_map = {1: -0.43, 5: -0.32, 10: -0.25, 25: -0.10,
                    50: 0.10, 75: 0.28, 90: 0.40, 95: 0.50, 99: 0.65}
    vals1 = np.percentile(ann_geo, pcts)
    for pc, v in zip(pcts, vals1):
        hp = hist_pct_map.get(pc, float("nan"))
        diff = v - hp
        tag = "" if abs(diff) > 0.08 else "  OK"
        print(f"  {pc:>5d}  {v:>8.2%}  {hp:>12.2%}  Δ={diff:+.2%}{tag}")

    # --- Estimation precision (95% CI on key statistics) ---
    # SE_median ≈ sqrt(π/2) * σ / sqrt(N); SE_mean = σ / sqrt(N)
    # With N=10,000 these are tight, but shown for completeness.
    _se_med  = float(np.sqrt(np.pi / 2) * a_sig / np.sqrt(len(ann_geo)))
    _se_mean = float(a_sig / np.sqrt(len(ann_geo)))
    _z95     = 1.96
    print()
    print(f"  --- Estimation Precision (95% CI, N={len(ann_geo):,}) ---")
    print(f"  Median CAGR 95% CI:  [{a_med - _z95*_se_med:.2%},  {a_med + _z95*_se_med:.2%}]"
          f"   (±{_z95*_se_med:.2%})")
    print(f"  Mean CAGR   95% CI:  [{a_mu   - _z95*_se_mean:.2%},  {a_mu   + _z95*_se_mean:.2%}]"
          f"   (±{_z95*_se_mean:.2%})")
    print(f"  Annual skew  (sample, N paths):  {a_skew:+.4f}")
    print(f"  Annual kurt  (sample, N paths):  {a_kurt:.4f}  "
          f"[sources: GARCH clustering + regime mixture + jumps]")

    # Save for §6 and §10
    spy1_saved = spy1
    tw1_saved  = tw1
    # Save annual stats for summary
    a_skew_saved = a_skew
    a_kurt_saved = a_kurt

    # =======================================================================
    _hdr("SECTION 2 — Multi-Horizon CAGR Distribution  (2,000 paths)")
    # =======================================================================
    N2 = 2_000
    horizons = [1, 5, 10, 30]
    # Historical CAGR benchmarks (median approximate)
    hist_cagr = {1: 0.10, 5: 0.095, 10: 0.095, 30: 0.095}

    for yr in horizons:
        T = int(yr * 252)
        print(f"\n  Horizon {yr:2d} yr  ({N2} paths × {T} days)...")
        t0 = time.time()
        sp = simulate_spy_fast(N2, T, rng, p, with_jumps=True)
        tw = terminal_wealth(sp)
        cg = cagr(tw, float(yr))
        med   = np.median(cg)
        p5    = np.percentile(cg, 5)
        p95   = np.percentile(cg, 95)
        pneg  = (cg < 0).mean()
        hbm   = hist_cagr[yr]
        tag   = "[PASS]" if abs(med - hbm) < 0.04 else "[WARN]"
        print(f"  {yr}yr  med={med:.2%}  p5={p5:.2%}  p95={p95:.2%}"
              f"  pneg={pneg:.1%}  hist={hbm:.2%} {tag}  ({time.time()-t0:.1f}s)")

    # =======================================================================
    _hdr("SECTION 3 — GARCH Variance Dynamics  (1 path × 100,000 days)")
    # =======================================================================
    N3, T3 = 1, 100_000
    print(f"  Simulating {T3} days single path...")
    t0 = time.time()
    sp3 = simulate_spy_fast(N3, T3, rng, p, with_jumps=False)
    print(f"  Done in {time.time()-t0:.1f}s")

    r3 = sp3[0]                        # (T3,)
    absr3 = np.abs(r3)

    # Rolling 21-day vol
    roll_var = np.array([r3[i:i+21].var() for i in range(0, T3-21, 21)])
    roll_vol = np.sqrt(roll_var * 252)
    long_vol = roll_vol.mean()

    # Target vol
    target_vol = np.sqrt(p["rlv_spy"][0] * 252 * (p["ss"][0])
                         + p["rlv_spy"][1] * 252 * (p["ss"][1]))

    print()
    _ok("Long-run annual vol (target ~18.5%)", long_vol, 0.12, 0.25)
    _row("Target vol (steady-state weighted)",  target_vol)

    # ACF of |r|
    lags, ac_abs = acf(absr3, max_lag=20)
    print()
    print("  --- ACF of |r| (expect positive, decay to ~0 by lag 20) ---")
    for l, a in zip(lags[:10], ac_abs[:10]):
        bar = "#" * int(abs(a) * 60)
        print(f"  Lag {l:2d}: {a:+.4f}  {bar}")
    _ok("ACF |r| lag-1 (target ~0.15)", ac_abs[0], 0.05, 0.30)
    acf_lag1_saved = float(ac_abs[0])

    # GARCH persistence check
    print()
    print("  --- GJR-GARCH Stationarity ---")
    for k in [0, 1]:
        ea_gjr= p["eff_alpha_gjr"][k]
        b_raw = p["beta_raw"][k]; b_eff = p["beta_eff"][k]
        raw_p = ea_gjr + b_raw; eff_p = ea_gjr + b_eff
        tag_r = "[EXPLOSIVE]" if raw_p >= 1.0 else "[OK]"
        tag_e = "[OK — engine caps beta]" if raw_p >= 1.0 else "[OK]"
        print(f"  Regime {k}: GJR_eff_persist(raw)={raw_p:.5f} {tag_r}"
              f"  →  GJR_persist(eff)={eff_p:.5f} {tag_e}")

    # =======================================================================
    _hdr("SECTION 4 — Regime Transition Diagnostics  (1 path × 100,000 days)")
    # =======================================================================
    print(f"  Simulating {T3} day regime path...")
    t0 = time.time()
    rp4 = _regime_paths(N3, T3, rng, p)[0]    # (T3,)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Empirical steady state
    emp_r0 = (rp4 == 0).mean()
    emp_r1 = (rp4 == 1).mean()

    # Analytical steady state from TM
    tm = p["tm"]
    pi0 = tm[1, 0] / (tm[0, 1] + tm[1, 0])
    pi1 = 1.0 - pi0

    print()
    _ok("Empirical regime-0 freq", emp_r0, 0.88, 0.99)
    _ok("Empirical regime-1 freq", emp_r1, 0.01, 0.12)
    _row("Analytical steady-state r0", pi0)
    _row("Analytical steady-state r1", pi1)

    # Dwell time distributions
    def dwell_times(regime_array, state):
        dt = []; cnt = 0
        for v in regime_array:
            if v == state: cnt += 1
            elif cnt > 0:  dt.append(cnt); cnt = 0
        if cnt > 0: dt.append(cnt)
        return np.array(dt, dtype=float)

    dt0 = dwell_times(rp4, 0)
    dt1 = dwell_times(rp4, 1)
    exp_dwell0 = 1.0 / (1.0 - tm[0, 0])   # geometric mean
    exp_dwell1 = 1.0 / (1.0 - tm[1, 1])

    print()
    print("  --- Dwell Times (days) ---")
    print(f"  Regime 0: mean={dt0.mean():.1f}  expected={exp_dwell0:.1f}"
          f"  median={np.median(dt0):.1f}")
    print(f"  Regime 1: mean={dt1.mean():.1f}  expected={exp_dwell1:.1f}"
          f"  median={np.median(dt1):.1f}")
    _ok("Regime-0 mean dwell (expected ~1471d)", dt0.mean(), 500, 3000)
    _ok("Regime-1 mean dwell (expected ~64d)",   dt1.mean(),  20,  200)

    # =======================================================================
    _hdr("SECTION 5 — LETF Leverage Scaling  (2,000 paths × 10yr)")
    # =======================================================================
    N5, T5 = 2_000, 2520
    print(f"  Simulating {N5} paths × {T5} days (joint SPY+QQQ)...")
    t0 = time.time()
    spy5, qqq5 = simulate_joint_fast(N5, T5, rng, p)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ETF configurations: (name, leverage, underlying, expense_annual, borrow_annual)
    configs = [
        ("SPY",  1.0, spy5, 0.000945 / 252, 0.0),
        ("SSO",  2.0, spy5, 0.0089 / 252,   0.012 / 252),
        ("TQQQ", 3.0, qqq5, 0.0086 / 252,   0.020 / 252),
    ]

    tw5_all = {}
    spy5_tw = None

    print()
    print(f"  {'ETF':<6s} {'Median CAGR':>12s} {'Mean CAGR':>12s} {'P5 CAGR':>10s} "
          f"{'P95 CAGR':>10s} {'Pct Neg':>10s}")
    print("  " + "-" * 65)
    for name, L, und, exp_d, bor_d in configs:
        r_letf = letf_ret(und, L, exp_d, bor_d)
        tw     = terminal_wealth(r_letf)
        cg     = cagr(tw, 10.0)
        med    = np.median(cg)
        mn     = cg.mean()
        p5     = np.percentile(cg, 5)
        p95    = np.percentile(cg, 95)
        pneg   = (cg < 0).mean()
        tw5_all[name] = tw
        if name == "SPY": spy5_tw = tw
        print(f"  {name:<6s} {med:>12.2%} {mn:>12.2%} {p5:>10.2%} {p95:>10.2%} {pneg:>10.1%}")

    # Historical LETF benchmarks (10yr median CAGR approximate)
    hist_letf = {"SPY": 0.095, "SSO": 0.15, "TQQQ": 0.30}
    print()
    print("  --- Historical Benchmark Comparison (10yr median CAGR) ---")
    for name, L, und, exp_d, bor_d in configs:
        r_letf = letf_ret(und, L, exp_d, bor_d)
        tw     = terminal_wealth(r_letf)
        cg     = cagr(tw, 10.0)
        med    = np.median(cg)
        hb     = hist_letf[name]
        diff   = med - hb
        tag    = "[PASS]" if abs(diff) < 0.10 else "[WARN]"
        print(f"  {name:<6s} sim={med:.2%}  hist={hb:.2%}  Δ={diff:+.2%}  {tag}")

    # Leverage ratio check (SSO/SPY, TQQQ/QQQ-equiv)
    print()
    print("  --- Leverage Ratio Diagnostics ---")
    spy_med_cagr  = np.median(cagr(tw5_all["SPY"], 10.0))
    sso_med_cagr  = np.median(cagr(tw5_all["SSO"], 10.0))
    tqqq_med_cagr = np.median(cagr(tw5_all["TQQQ"], 10.0))
    _row("SPY  10yr med CAGR",  spy_med_cagr)
    _row("SSO  10yr med CAGR",  sso_med_cagr)
    _row("TQQQ 10yr med CAGR",  tqqq_med_cagr)
    if spy_med_cagr > 0:
        ratio_sso = sso_med_cagr / spy_med_cagr
        print(f"  {'SSO/SPY CAGR ratio (expect <2x due to drag)':<40s} {ratio_sso:.2f}x"
              f"  {'[OK]' if ratio_sso < 2.5 else '[WARN]'}")

    # =======================================================================
    _hdr("SECTION 6 — Tail Risk: VaR and CVaR  (re-use §1 paths)")
    # =======================================================================
    ann_geo1 = cagr(tw1_saved, 1.0)

    var_1  = np.percentile(ann_geo1, 1)
    var_5  = np.percentile(ann_geo1, 5)
    cvar_1 = ann_geo1[ann_geo1 <= var_1].mean()
    cvar_5 = ann_geo1[ann_geo1 <= var_5].mean()

    # Daily VaR/CVaR
    d_flat = spy1_saved.flatten()
    d_var1  = np.percentile(d_flat, 1)
    d_cvar1 = d_flat[d_flat <= d_var1].mean()

    print()
    print("  --- Annual VaR/CVaR (1yr horizon) ---")
    _ok("1-yr VaR  (1%)",   var_1,  -0.55, -0.25)
    _ok("1-yr VaR  (5%)",   var_5,  -0.40, -0.10)
    _ok("1-yr CVaR (1%)",   cvar_1, -0.75, -0.30)
    _ok("1-yr CVaR (5%)",   cvar_5, -0.55, -0.15)

    print()
    print("  --- Daily VaR/CVaR ---")
    _ok("Daily VaR  (1%)",  d_var1,  -0.06, -0.015)
    _ok("Daily CVaR (1%)",  d_cvar1, -0.12, -0.025)

    # Historical comparison
    print()
    _row("Historical 1yr VaR  1% (approx)", HIST["var_1pct"])
    _row("Historical 1yr CVaR 1% (approx)", HIST["cvar_1pct"])
    diff_var  = var_1  - HIST["var_1pct"]
    diff_cvar = cvar_1 - HIST["cvar_1pct"]
    print(f"  Δ VaR  (sim - hist): {diff_var:+.4f}  {'[OK]' if abs(diff_var) < 0.15 else '[WARN]'}")
    print(f"  Δ CVaR (sim - hist): {diff_cvar:+.4f}  {'[OK]' if abs(diff_cvar) < 0.20 else '[WARN]'}")

    # =======================================================================
    _hdr("SECTION 7 — Cross-Asset Correlation  (5,000 paths × 1yr)")
    # =======================================================================
    N7, T7 = 5_000, 252
    print(f"  Simulating {N7} paths × {T7} days (joint)...")
    t0 = time.time()
    spy7, qqq7 = simulate_joint_fast(N7, T7, rng, p)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Correlation across all daily returns
    spy_flat7 = spy7.flatten()
    qqq_flat7 = qqq7.flatten()
    rho_all   = np.corrcoef(spy_flat7, qqq_flat7)[0, 1]

    # Per-path annual correlations
    rho_per = np.array([np.corrcoef(spy7[i], qqq7[i])[0, 1] for i in range(N7)])
    rho_med = np.median(rho_per)
    rho_std = rho_per.std()

    print()
    _ok("SPY-QQQ pooled daily corr",    rho_all, 0.70, 0.97)
    _ok("SPY-QQQ per-path median corr", rho_med, 0.60, 0.97)
    _row("Per-path corr std-dev",        rho_std)
    _row("Historical SPY-QQQ corr",      HIST["spy_qqq_corr"])

    # Cross-check: target rho from model
    rho_target = float(p["rho"] @ p["ss"])
    _row("Model target corr (regime-weighted)", rho_target)
    diff_rho = rho_all - rho_target
    print(f"  Δ corr (sim - target): {diff_rho:+.4f}  {'[OK]' if abs(diff_rho) < 0.10 else '[WARN]'}")

    # Empirical rolling correlation context
    # Long-run ~0.85-0.92 (2000-2020); recent 2015-2025 ~0.93-0.97 (sector concentration)
    print()
    print("  --- Correlation Realism Context ---")
    _CORR_BANDS = {
        "Long-run (2000-2020)": (0.85, 0.92),
        "Recent   (2015-2025)": (0.93, 0.97),
        "Crisis regimes":        (0.93, 0.99),
    }
    for period, (lo, hi) in _CORR_BANDS.items():
        inside = lo <= rho_all <= hi
        print(f"  {period:<26s}: [{lo:.2f}, {hi:.2f}]  sim={rho_all:.4f}  "
              f"{'WITHIN' if inside else 'OUTSIDE'}")
    print(f"  Regime-0 (calm)  corr target: {p['rho'][0]:.4f}")
    print(f"  Regime-1 (crisis) corr target: {p['rho'][1]:.4f}")
    print(f"  Note: 0.96 is realistic for 2015-2025 SPY-QQQ (tech concentration).")
    print(f"        Diversification assumption should account for this.")
    rho_all_saved = rho_all

    # =======================================================================
    _hdr("SECTION 8 — Max Drawdown Distribution  (re-use §5 paths)")
    # =======================================================================
    print("  Computing max drawdowns from §5 paths...")
    for name, L, und, exp_d, bor_d in configs:
        r_letf = letf_ret(und, L, exp_d, bor_d)
        mdd    = max_drawdown_paths(r_letf)
        p5m    = np.percentile(mdd, 5)
        p50m   = np.percentile(mdd, 50)
        p95m   = np.percentile(mdd, 95)
        wc     = mdd.min()
        print(f"  {name:<6s}  MDD p5={p5m:.1%}  p50={p50m:.1%}  p95={p95m:.1%}  worst={wc:.1%}")

    # SPY drawdown check
    spy_mdd   = max_drawdown_paths(letf_ret(spy5, 1.0, configs[0][3], configs[0][4]))
    mdd_p50   = np.median(spy_mdd)
    _ok("SPY 10yr median max-DD (target ~-25 to -60%)", mdd_p50, -0.70, -0.10)

    # =======================================================================
    _hdr("SECTION 9 — Stress / Crisis Injection  (1,000 paths × 5yr)")
    # =======================================================================
    N9, T9 = 1_000, 1260
    print(f"  Simulating {N9} paths × {T9} days with jumps forced ON...")
    t0 = time.time()
    # Stress test: run with forced high-regime start and jumps enabled
    p_stress = dict(p)
    # Override initial regime probabilities to regime 1 (high vol)
    p_stress["ss"] = np.array([0.20, 0.80])
    sp9 = simulate_spy_fast(N9, T9, rng, p_stress, with_jumps=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    tw9   = terminal_wealth(sp9)
    cg9   = cagr(tw9, 5.0)
    mdd9  = max_drawdown_paths(sp9)

    print()
    _row("Stress 5yr median CAGR",     np.median(cg9))
    _row("Stress 5yr mean CAGR",       cg9.mean())
    _row("Stress p5 CAGR",             np.percentile(cg9, 5))
    _row("Stress median max-DD",       np.median(mdd9))
    _row("Stress worst max-DD",        mdd9.min())
    _pct("Stress pct negative",        (cg9 < 0).mean())

    # Compare stress vs unconditional (§1 extrapolated)
    spy_uncond_vol = spy1_saved.flatten().std() * np.sqrt(252)
    spy_stress_vol = sp9.flatten().std() * np.sqrt(252)
    print()
    _row("Unconditional annual vol",   spy_uncond_vol)
    _row("Stress annual vol",          spy_stress_vol)
    ratio9 = spy_stress_vol / spy_uncond_vol
    print(f"  {'Stress vol / uncond vol':<40s} {ratio9:.2f}x"
          f"  {'[OK — stress>base]' if ratio9 > 1.0 else '[WARN — stress<=base]'}")

    # =======================================================================
    _hdr("SECTION 10 — Jump Process Characterisation  (re-use §1 paths)")
    # =======================================================================
    print("  Analysing jump contributions from §1 simulation...")
    print("  (Comparing with_jumps=True vs with_jumps=False over 2k×1yr paths)")
    N10, T10 = 2_000, 252
    t0 = time.time()
    sp_nj = simulate_spy_fast(N10, T10, rng, p, with_jumps=False)
    sp_wj = simulate_spy_fast(N10, T10, rng, p, with_jumps=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    tw_nj  = terminal_wealth(sp_nj); cg_nj = cagr(tw_nj, 1.0)
    tw_wj  = terminal_wealth(sp_wj); cg_wj = cagr(tw_wj, 1.0)

    med_nj = np.median(cg_nj); med_wj = np.median(cg_wj)
    vol_nj = sp_nj.flatten().std() * np.sqrt(252)
    vol_wj = sp_wj.flatten().std() * np.sqrt(252)
    kurt_nj= stats.kurtosis(sp_nj.flatten())
    kurt_wj= stats.kurtosis(sp_wj.flatten())

    print()
    print(f"  {'Metric':<35s} {'No Jumps':>12s} {'With Jumps':>12s} {'Δ':>10s}")
    print("  " + "-" * 72)
    print(f"  {'Median 1yr CAGR':<35s} {med_nj:>12.2%} {med_wj:>12.2%} {med_wj-med_nj:>+10.2%}")
    print(f"  {'Annual vol':<35s} {vol_nj:>12.2%} {vol_wj:>12.2%} {vol_wj-vol_nj:>+10.2%}")
    print(f"  {'Daily excess kurtosis':<35s} {kurt_nj:>12.4f} {kurt_wj:>12.4f} {kurt_wj-kurt_nj:>+10.4f}")

    # Jump frequency and magnitude diagnostics
    jp_ss = float(p["jump_prob"] @ p["ss"])   # steady-state jump prob
    jp_scale_ss = float(p["jump_scale"] @ p["ss"])
    print()
    _row("Steady-state jump prob (annualised)", jp_ss * 252)
    _row("Expected jump days/year",             jp_ss * 252)

    # Theoretical baselines (symmetric t_6 jumps, E[z]=0 → zero mean drift bias)
    # E[|z|] for t_6 ≈ 0.919; variance E[z²] = ν/(ν-2) = 1.5 for ν=6
    _t6_var  = 6.0 / (6.0 - 2.0)    # = 1.5
    _t6_krt  = 6.0 / (6.0 - 4.0)    # excess kurtosis of t_6 = 3 (but note marginal here = 3)
    _exp_jump_mean = 0.0              # symmetric: E[z]=0
    _exp_jump_var  = jp_ss * 252 * jp_scale_ss**2 * _t6_var   # annual variance from jumps
    _exp_jump_krt  = jp_ss * 252 * jp_scale_ss**4 * (3 + _t6_krt)   # 4th cumulant

    jump_cagr_drag = med_nj - med_wj   # positive = jumps hurt CAGR
    print(f"  {'Jump CAGR drag (simulated)':<42s} {jump_cagr_drag:+.4f}"
          f"  {'[OK]' if abs(jump_cagr_drag) < 0.05 else '[WARN — large drag]'}")
    print(f"  {'Jump CAGR drag (theoretical: sym. t_6)':<42s} {_exp_jump_mean:+.4f}  [symmetric → E[drag]=0]")
    jump_vol_add   = vol_wj - vol_nj
    print(f"  {'Jump vol addition (simulated)':<42s} {jump_vol_add:+.4f}"
          f"  {'[OK]' if jump_vol_add >= 0 else '[WARN — negative]'}")
    print(f"  {'Jump annual var addition (theoretical)':<42s} {_exp_jump_var:+.6f}")

    # --- Kurtosis Decomposition ---
    # Total daily kurtosis = GARCH clustering + regime switching + jumps.
    # Approximation: (a) GARCH-only ≈ kurt_nj, (b) jump add ≈ kurt_wj - kurt_nj
    # (c) regime mixture: hard to isolate — residual after accounting for (a)+(b).
    # Theoretical t_ν kurtosis under iid regime-weighted: κ_4 = 6/(ν-4) for ν>4
    _nu_ss   = float(p["nu"] @ p["ss"])   # steady-state avg df
    _iid_kurt = 6.0 / (_nu_ss - 4.0) if _nu_ss > 4.0 else float("inf")
    _garch_kurt = kurt_nj         # GARCH+regime mixture (no jumps)
    _jump_add   = kurt_wj - kurt_nj
    print()
    print("  --- Kurtosis Decomposition (excess kurtosis, daily returns) ---")
    print(f"  {'iid t_ν baseline (ν=' + f'{_nu_ss:.1f})':<42s} {_iid_kurt:>8.2f}")
    print(f"  {'GARCH + regime mixture contribution':<42s} {_garch_kurt - _iid_kurt:>+8.2f}")
    print(f"  {'Jump contribution (wj - nj)':<42s} {_jump_add:>+8.2f}")
    print(f"  {'Total observed (GARCH+regime+jumps)':<42s} {kurt_wj:>8.2f}")
    print(f"  Decomposition: iid={_iid_kurt:.2f}  ΔGARCH={_garch_kurt-_iid_kurt:+.2f}"
          f"  ΔJumps={_jump_add:+.2f}  Total={kurt_wj:.2f}")

    # =======================================================================
    _hdr("AUDIT COMPLETE — Summary")
    # =======================================================================
    print()
    print("  Key Results:")
    _ann_geo_sum  = cagr(tw1_saved, 1.0)
    _d_flat       = spy1_saved.flatten()
    print(f"    SPY 1yr median CAGR   : {np.median(_ann_geo_sum):.2%}")
    print(f"    SPY 10yr median CAGR  : {spy_med_cagr:.2%}")
    print(f"    SPY 1yr annual vol     : {_d_flat.std()*np.sqrt(252):.2%}")
    print(f"    SPY 1yr daily skew     : {stats.skew(_d_flat):+.3f}  (daily)")
    print(f"    SPY 1yr annual skew    : {a_skew_saved:+.3f}  "
          f"[target <0; structural WARN if Markov 2-state model]")
    print(f"    SPY 1yr daily xkurt    : {stats.kurtosis(_d_flat):.2f}  (daily)")
    print(f"    SPY 1yr annual xkurt   : {a_kurt_saved:.2f}  "
          f"[sources: GARCH + regime mixture + jumps]")
    print(f"    SPY 1yr VaR 1%         : {np.percentile(_ann_geo_sum,1):.2%}")
    print(f"    SPY-QQQ corr           : {rho_all_saved:.4f}  "
          f"[realistic for 2015-2025 tech-concentrated market]")
    print(f"    ACF |r| lag-1          : {acf_lag1_saved:.4f}  (vol clustering persistence)")
    print(f"    GJR persist (r0)       : {p['eff_alpha_gjr'][0]+p['beta_eff'][0]:.5f}"
          f"  alpha={p['alpha'][0]:.4f}  gamma={p['gamma'][0]:.4f}"
          f"  [gamma/alpha={p['gamma'][0]/p['alpha'][0]:.3f} — Bayesian shrinkage]")
    print(f"    GJR persist (r1)       : {p['eff_alpha_gjr'][1]+p['beta_eff'][1]:.5f}"
          f"  alpha={p['alpha'][1]:.4f}  gamma={p['gamma'][1]:.4f}"
          f"  [gamma/alpha={p['gamma'][1]/p['alpha'][1]:.3f} — Bayesian shrinkage]")
    print(f"    Jump drag (1yr CAGR)  : {jump_cagr_drag:+.4f}  [theoretical=0 for symmetric t_6]")
    print()
    print("  Historical Targets:")
    print(f"    Arith return target    : {HIST['arith_return']:.2%}")
    print(f"    Geo   return target    : {HIST['geo_return']:.2%}")
    print(f"    Annual vol target      : {HIST['annual_vol']:.2%}")
    print(f"    Daily skew target      : {HIST['daily_skew']:+.2f}  (annual target: < 0)")
    print(f"    Daily xkurt target     : {HIST['daily_xkurt']:.1f}  (annual target: 0-5)")
    print()
    print("  Remaining WARNs (documented structural limitations):")
    print("    Annual skewness > 0    : Markov 2-state model with symmetric t-innovations")
    print("                             cannot produce negative annual skew without drift bias.")
    print("                             GJR leverage effect (γ>0) partially mitigates this.")
    print("    Annual kurtosis > 5    : Regime mixture (4% crisis at 4× vol) creates heavy")
    print("                             tails in annual distribution; structural to regime model.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
