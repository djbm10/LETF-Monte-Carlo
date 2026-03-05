"""
Baseline model competition: M0, M1, M2
Fit on TRAIN (pre-2015), evaluate OOS on VALID (2015-2024).
Assets: UPRO (3x SPY), SSO (2x SPY), TQQQ (3x QQQ)

M0: r = L*idx - (L-1)*rf - fee                         [no fit]
M1: r = L*idx - (L-1)*(alpha_f + beta_f*rf) - fee      [fit alpha_f, beta_f via OLS]
M2: r = L*idx - (L-1)*(rf + s) + gamma*rf - fee        [fit s, gamma via OLS + clip to bounds]
    s in [-2%, +6%]/yr annualised;  gamma in [0, 1.2]

All metrics reported on VALID (OOS) window.
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats, optimize

os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

TRAIN_END  = '2014-12-31'
VALID_START = '2015-01-01'
VALID_END   = '2024-12-31'

# ─── helpers ──────────────────────────────────────────────────────────────────

def _cagr(rets: np.ndarray) -> float:
    n = len(rets)
    if n < 2:
        return np.nan
    w = float(np.prod(1.0 + rets))
    return float(w ** (252.0 / n) - 1.0)

def _max_drawdown(rets: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + rets)
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth - peak) / peak
    return float(np.min(dd))   # most negative value

def _metrics(actual: np.ndarray, predicted: np.ndarray, label: str) -> dict:
    resid = actual - predicted
    n = len(actual)
    corr_v = float(np.corrcoef(actual, predicted)[0, 1]) if n > 2 else np.nan
    te_ann = float(np.std(resid) * np.sqrt(252))
    cagr_a = _cagr(actual)
    cagr_p = _cagr(predicted)
    cagr_d = cagr_p - cagr_a           # positive = model overpredicts
    W_a    = float(np.prod(1.0 + actual))
    W_p    = float(np.prod(1.0 + predicted))
    tw_err = float(abs(W_p / W_a - 1.0)) if W_a != 0 else np.nan
    mdd_a  = _max_drawdown(actual)
    mdd_p  = _max_drawdown(predicted)
    mdd_d  = float(abs(mdd_p - mdd_a))
    mean_r = float(np.mean(resid) * 252 * 100)
    ac1    = float(pd.Series(resid).autocorr(lag=1))
    return {
        'label': label,
        'n': n,
        'mean_resid_pct_yr': mean_r,
        'corr': corr_v,
        'TE_ann_pct': te_ann * 100,
        'CAGR_diff_pct': cagr_d * 100,   # signed: + = overpredicts
        'tw_rel_err_pct': tw_err * 100,
        'MaxDD_diff_pct': mdd_d * 100,
        'lag1_autocorr': ac1,
        'cagr_actual_pct': cagr_a * 100,
        'cagr_pred_pct':   cagr_p * 100,
    }

def _print_metrics(m: dict) -> None:
    print(f"    mean resid   : {m['mean_resid_pct_yr']:+.3f}%/yr")
    print(f"    corr         : {m['corr']:.6f}")
    print(f"    TE_ann       : {m['TE_ann_pct']:.4f}%")
    print(f"    CAGR_diff    : {m['CAGR_diff_pct']:+.3f}%/yr  "
          f"(pred={m['cagr_pred_pct']:.2f}%  actual={m['cagr_actual_pct']:.2f}%)")
    print(f"    TW_rel_err   : {m['tw_rel_err_pct']:.2f}%")
    print(f"    MaxDD_diff   : {m['MaxDD_diff_pct']:.3f}%")
    print(f"    lag-1 AC     : {m['lag1_autocorr']:+.4f}")

# ─── load data ────────────────────────────────────────────────────────────────
print("=" * 90)
print("Loading data...")
df = eng.load_cache(eng.DATA_CACHE)
assert df is not None
df = eng._attach_real_only_driver_columns(df)
rf_full, rf_src = eng._load_daily_ff_rf_aligned(df)
rf_full = np.resize(rf_full, len(df))
print(f"  Rows: {len(df):,}  RF source: {rf_src}")

ASSETS_TO_TEST = [
    ('UPRO', 3.0),
    ('SSO',  2.0),
    ('TQQQ', 3.0),
]

all_results = {}   # asset → { 'train': {M0,M1,M2}, 'valid': {M0,M1,M2}, 'params': ... }

for asset, lev in ASSETS_TO_TEST:
    print("\n" + "=" * 90)
    print(f"ASSET: {asset}  L={lev:.0f}x")
    print("=" * 90)

    real_col = f'{asset}_Real_Ret'
    syn_col  = f'{asset}_IsSynthetic'
    driver_col = eng._underlying_return_column_for_asset(asset, df=df, prefer_real_only=True)
    fee_daily  = float(eng.ASSETS[asset]['expense_ratio']) / 252.0

    if real_col not in df.columns or driver_col not in df.columns:
        print(f"  SKIP: missing {real_col} or {driver_col}")
        continue

    syn_flag = df.get(syn_col, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    actual_s = pd.to_numeric(df[real_col], errors='coerce')
    driver_s = pd.to_numeric(df[driver_col], errors='coerce')

    # Real-only rows across the full history
    valid_rows = actual_s.notna() & driver_s.notna() & (~syn_flag)
    dates = pd.DatetimeIndex(df.index)

    # Split
    train_mask = valid_rows & (dates <= pd.Timestamp(TRAIN_END))
    valid_mask = valid_rows & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

    if train_mask.sum() < 252 or valid_mask.sum() < 252:
        print(f"  SKIP: insufficient data  train={train_mask.sum()} valid={valid_mask.sum()}")
        continue

    idx_t  = driver_s[train_mask].to_numpy()
    act_t  = actual_s[train_mask].to_numpy()
    rf_t   = rf_full[train_mask]

    idx_v  = driver_s[valid_mask].to_numpy()
    act_v  = actual_s[valid_mask].to_numpy()
    rf_v   = rf_full[valid_mask]

    print(f"  Driver : {driver_col}")
    print(f"  Fee    : {fee_daily*1e4:.4f} bps/day  ({fee_daily*252*100:.4f}%/yr)")
    print(f"  TRAIN  : {df[train_mask].index[0].date()} → {df[train_mask].index[-1].date()}  N={train_mask.sum():,}")
    print(f"  VALID  : {df[valid_mask].index[0].date()} → {df[valid_mask].index[-1].date()}  N={valid_mask.sum():,}")

    # ── M0 ────────────────────────────────────────────────────────────────────
    m0_pred_t = lev * idx_t - (lev - 1.0) * rf_t  - fee_daily
    m0_pred_v = lev * idx_v - (lev - 1.0) * rf_v  - fee_daily
    m0_resid_t = act_t - m0_pred_t    # = LETF_actual - theoretical (positive = LETF outperforms)

    # ── M1: r = L*idx - (L-1)*(alpha_f + beta_f*rf) - fee ───────────────────
    # Residual re: actual: act = L*idx - (L-1)*(alpha_f + beta_f*rf) - fee + eps
    # Let y = act - L*idx + fee = -(L-1)*(alpha_f + beta_f*rf) + eps
    # OLS: y = c0 + c1*rf  → alpha_f = -c0/(L-1),  beta_f = -c1/(L-1)
    y_t     = act_t - lev * idx_t + fee_daily
    X_t     = np.column_stack([np.ones(len(y_t)), rf_t])
    coef_m1, _, _, _ = np.linalg.lstsq(X_t, y_t, rcond=None)
    c0_m1, c1_m1 = float(coef_m1[0]), float(coef_m1[1])
    alpha_f = -c0_m1 / (lev - 1.0)
    beta_f  = -c1_m1 / (lev - 1.0)

    m1_pred_t = lev * idx_t - (lev - 1.0) * (alpha_f + beta_f * rf_t) - fee_daily
    m1_pred_v = lev * idx_v - (lev - 1.0) * (alpha_f + beta_f * rf_v) - fee_daily

    # ── M2: r = L*idx - (L-1)*(rf+s) + gamma*rf - fee ────────────────────────
    # M0_resid = act - M0_pred = gamma*rf - (L-1)*s + eps
    # OLS of m0_resid on [rf, 1]:
    #   coeff of rf  → gamma
    #   coeff of 1   → -(L-1)*s  →  s = -intercept/(L-1)
    # Constraints: s_daily ∈ [-0.02/252, +0.06/252], gamma ∈ [0, 1.2]
    S_LO, S_HI = -0.02 / 252.0, 0.06 / 252.0    # daily spread bounds
    G_LO, G_HI = 0.0,           1.2              # gamma bounds

    X2_t   = np.column_stack([rf_t, np.ones(len(m0_resid_t))])
    coef_m2_unc, _, _, _ = np.linalg.lstsq(X2_t, m0_resid_t, rcond=None)
    gamma_unc  = float(coef_m2_unc[0])
    intcpt_unc = float(coef_m2_unc[1])
    s_unc      = -intcpt_unc / (lev - 1.0)

    # Check if unconstrained solution satisfies bounds
    gamma_fit = float(np.clip(gamma_unc, G_LO, G_HI))
    s_fit     = float(np.clip(s_unc,     S_LO, S_HI))
    constrained = (gamma_fit != gamma_unc or s_fit != s_unc)

    if constrained:
        # Re-optimise with bounds using scipy
        def _m2_sse(params):
            g, s_d = params
            pred = lev * idx_t - (lev - 1.0) * (rf_t + s_d) + g * rf_t - fee_daily
            resid = act_t - pred
            return float(np.sum(resid ** 2))
        res = optimize.minimize(
            _m2_sse, x0=[gamma_fit, s_fit],
            bounds=[(G_LO, G_HI), (S_LO, S_HI)],
            method='L-BFGS-B'
        )
        gamma_fit = float(res.x[0])
        s_fit     = float(res.x[1])

    m2_pred_t = lev * idx_t - (lev - 1.0) * (rf_t + s_fit) + gamma_fit * rf_t - fee_daily
    m2_pred_v = lev * idx_v - (lev - 1.0) * (rf_v + s_fit) + gamma_fit * rf_v - fee_daily

    # ── Fitted parameter summary ──────────────────────────────────────────────
    print(f"\n  Fitted parameters:")
    print(f"  M1:  alpha_f = {alpha_f*252*100:+.4f}%/yr  ({alpha_f*1e4:+.5f} bps/day)")
    print(f"       beta_f  = {beta_f:+.6f}  (rf sensitivity; 1.0 = M0 baseline)")
    print(f"  M2:  s       = {s_fit*252*100:+.4f}%/yr  ({s_fit*1e4:+.6f} bps/day)  "
          f"{'[clipped]' if constrained else '[unconstrained]'}")
    print(f"       gamma   = {gamma_fit:+.6f}  (collateral income fraction of rf)")
    if constrained:
        print(f"       [unconstrained: s={s_unc*252*100:+.4f}%/yr  gamma={gamma_unc:+.6f}]")

    # Net effective borrow formula for M2:
    # borrow_net = (L-1)*(rf+s) - gamma*rf = (L-1-gamma)*rf + (L-1)*s
    net_rf_coeff = (lev - 1.0 - gamma_fit)
    net_s_contrib = (lev - 1.0) * s_fit * 252 * 100
    print(f"\n  M2 net borrow: {net_rf_coeff:.4f}*rf + {net_s_contrib:+.4f}%/yr_spread")
    print(f"       (M0 uses {lev-1:.1f}*rf + 0%/yr_spread)")

    # ── TRAIN metrics (fit quality check) ─────────────────────────────────────
    print(f"\n  --- TRAIN metrics ---")
    for label, pred in [('M0', m0_pred_t), ('M1', m1_pred_t), ('M2', m2_pred_t)]:
        m = _metrics(act_t, pred, label)
        print(f"\n  [{label}]  (train, N={m['n']:,})")
        _print_metrics(m)

    # ── OOS / VALID metrics ───────────────────────────────────────────────────
    print(f"\n  --- OOS VALID metrics (2015–2024) ---")
    valid_models = {}
    for label, pred in [('M0', m0_pred_v), ('M1', m1_pred_v), ('M2', m2_pred_v)]:
        m = _metrics(act_v, pred, label)
        valid_models[label] = m
        print(f"\n  [{label}]  (OOS, N={m['n']:,})")
        _print_metrics(m)

    # ── Year-by-year OOS residual mean for M0, M1, M2 ────────────────────────
    print(f"\n  --- Year-by-year OOS residual mean (%/yr) ---")
    yy_dates = pd.DatetimeIndex(df[valid_mask].index)
    yy_yr = yy_dates.year
    years = sorted(set(yy_yr.tolist()))
    preds = {'M0': m0_pred_v, 'M1': m1_pred_v, 'M2': m2_pred_v}
    print(f"  {'Year':>5}  {'N':>5}  {'M0 resid':>10}  {'M1 resid':>10}  {'M2 resid':>10}  {'UPRO actual CAGR':>18}")
    for yr in years:
        mask_yr = (yy_yr == yr)
        act_yr  = act_v[mask_yr]
        if len(act_yr) < 10: continue
        cagr_yr = _cagr(act_yr) * 100
        row = f"  {yr:>5}  {mask_yr.sum():>5}"
        for mdl in ['M0', 'M1', 'M2']:
            pred_yr = preds[mdl][mask_yr]
            r_yr = float(np.mean(act_yr - pred_yr) * 252 * 100)
            row += f"  {r_yr:>+10.3f}%"
        row += f"  {cagr_yr:>+18.2f}%"
        print(row)

    all_results[asset] = {
        'valid_metrics': valid_models,
        'params': {
            'M1': {'alpha_f': alpha_f, 'beta_f': beta_f},
            'M2': {'s': s_fit, 'gamma': gamma_fit, 'constrained': constrained,
                   's_unc': s_unc, 'gamma_unc': gamma_unc},
        },
        'lev': lev,
        'driver_col': driver_col,
    }

# ─── Final comparison table ───────────────────────────────────────────────────
print("\n" + "=" * 90)
print("OOS SUMMARY TABLE  (2015–2024)")
print("=" * 90)

metrics_keys = [
    ('mean_resid_pct_yr', 'mean_resid(%/yr)', '{:+.3f}%'),
    ('corr',              'corr',              '{:.6f}'),
    ('TE_ann_pct',        'TE_ann(%)',         '{:.4f}%'),
    ('CAGR_diff_pct',     'CAGR_diff(%/yr)',   '{:+.3f}%'),
    ('tw_rel_err_pct',    'TW_rel_err(%)',     '{:.2f}%'),
    ('MaxDD_diff_pct',    'MaxDD_diff(%)',     '{:.3f}%'),
    ('lag1_autocorr',     'lag1_AC',           '{:+.4f}'),
]

hdr = f"  {'Metric':<22}"
for asset, _ in ASSETS_TO_TEST:
    if asset not in all_results:
        continue
    for mdl in ['M0', 'M1', 'M2']:
        hdr += f"  {asset+'/'+mdl:>12}"
print(hdr)
print("  " + "-" * (22 + 14 * 3 * len(all_results)))

for key, label, fmt in metrics_keys:
    row = f"  {label:<22}"
    for asset, _ in ASSETS_TO_TEST:
        if asset not in all_results:
            continue
        for mdl in ['M0', 'M1', 'M2']:
            v = all_results[asset]['valid_metrics'][mdl].get(key, np.nan)
            row += f"  {fmt.format(v):>12}"
    print(row)

# ─── Parameter summary ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("FITTED PARAMETERS SUMMARY")
print("=" * 90)
print(f"  {'Asset':>6}  {'M1 alpha_f %/yr':>18}  {'M1 beta_f':>12}  "
      f"{'M2 s %/yr':>12}  {'M2 gamma':>10}  {'M2 net_rf_mult':>16}  {'constrained':>12}")
for asset, lev in ASSETS_TO_TEST:
    if asset not in all_results:
        continue
    p = all_results[asset]['params']
    m1 = p['M1']
    m2 = p['M2']
    net_rf = lev - 1.0 - m2['gamma']
    print(f"  {asset:>6}  {m1['alpha_f']*252*100:>18.4f}%  {m1['beta_f']:>12.6f}  "
          f"  {m2['s']*252*100:>10.4f}%  {m2['gamma']:>10.6f}  {net_rf:>16.6f}  "
          f"{'yes' if m2['constrained'] else 'no':>12}")

print("\n" + "=" * 90)
print("KEY: mean_resid = actual - predicted  (positive → model underpredicts = LETF outperforms)")
print("     CAGR_diff  = pred_CAGR - actual_CAGR  (positive → model overpredicts CAGR)")
print("     TW_rel_err = |W_pred/W_actual - 1|")
print("     lag1_AC    = autocorrelation of daily residuals at lag 1 (M0 target: -0.44)")
print("=" * 90)
