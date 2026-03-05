"""
Regime-conditional intercept_alpha model for LETF reconstruction.

Model: r = L*idx - (L-1)*rf - fee + intercept_alpha(regime)

Regime classification: 63d rolling realised vol of driver returns,
  thresholds calibrated on TRAIN-only data (no lookahead).

  low   : rolling_vol < P60(train)
  high  : P60 <= rolling_vol < P90(train)
  crash : rolling_vol >= P90(train)

intercept_alpha per regime: trimmed mean (10%) of M0 residuals in that regime.
TRAIN: pre-2015  |  VALID (OOS): 2015-2024
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

# ─── constants ────────────────────────────────────────────────────────────────
TRAIN_END   = '2014-12-31'
VALID_START = '2015-01-01'
VALID_END   = '2024-12-31'

VOL_WINDOW   = 63        # trading days for rolling vol
P_MID        = 60.0      # low/high boundary (percentile)
P_HI         = 90.0      # high/crash boundary (percentile)
TRIM_FRAC    = 0.10      # each tail trimmed (10% total)
MIN_VOL_OBS  = 20        # minimum days to compute rolling vol

REGIME_IDX   = {0: 'low', 1: 'high', 2: 'crash'}
REGIME_COLS  = {0: 'R_low', 1: 'R_high', 2: 'R_crash'}

# ─── helper: metrics ──────────────────────────────────────────────────────────
def _cagr(rets: np.ndarray) -> float:
    n = len(rets)
    if n < 2: return np.nan
    return float(np.prod(1.0 + rets) ** (252.0 / n) - 1.0)

def _max_dd(rets: np.ndarray) -> float:
    w = np.cumprod(1.0 + rets)
    return float(np.min((w - np.maximum.accumulate(w)) / np.maximum.accumulate(w)))

def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    resid  = actual - predicted
    n      = len(actual)
    corr   = float(np.corrcoef(actual, predicted)[0, 1]) if n > 2 else np.nan
    te_ann = float(np.std(resid) * np.sqrt(252))
    ca, cp = _cagr(actual), _cagr(predicted)
    W_a    = float(np.prod(1.0 + actual))
    W_p    = float(np.prod(1.0 + predicted))
    tw_err = float(abs(W_p / W_a - 1.0)) if W_a else np.nan
    mdd_d  = float(abs(_max_dd(predicted) - _max_dd(actual)))
    ac1    = float(pd.Series(resid).autocorr(lag=1))
    return dict(
        n=n,
        mean_resid_pct   = float(np.mean(resid) * 252 * 100),
        corr             = corr,
        TE_ann_pct       = te_ann * 100,
        CAGR_diff_pct    = (cp - ca) * 100,
        TW_rel_err_pct   = tw_err * 100,
        MaxDD_diff_pct   = mdd_d * 100,
        lag1_AC          = ac1,
        cagr_actual_pct  = ca * 100 if ca is not np.nan else np.nan,
        cagr_pred_pct    = cp * 100 if cp is not np.nan else np.nan,
    )

def print_metrics(m: dict, label: str) -> None:
    tw_flag   = "✓" if m['TW_rel_err_pct']  < 10.0  else "✗"
    cagr_flag = "✓" if abs(m['CAGR_diff_pct']) < 0.5  else "✗"
    print(f"  [{label}]  N={m['n']:,}")
    print(f"    mean resid  : {m['mean_resid_pct']:+.3f}%/yr")
    print(f"    corr        : {m['corr']:.6f}")
    print(f"    TE_ann      : {m['TE_ann_pct']:.4f}%")
    print(f"    CAGR_diff   : {m['CAGR_diff_pct']:+.3f}%/yr  "
          f"(pred={m['cagr_pred_pct']:.2f}%  actual={m['cagr_actual_pct']:.2f}%)  {cagr_flag}")
    print(f"    TW_rel_err  : {m['TW_rel_err_pct']:.2f}%  {tw_flag}")
    print(f"    MaxDD_diff  : {m['MaxDD_diff_pct']:.3f}%")
    print(f"    lag-1 AC    : {m['lag1_AC']:+.4f}")

def trimmed_mean(x: np.ndarray, frac: float = TRIM_FRAC) -> float:
    n = len(x)
    if n < 4: return float(np.median(x))
    k = max(1, int(np.floor(n * frac / 2)))
    s = np.sort(x)
    return float(np.mean(s[k: n - k]))

# ─── load data ────────────────────────────────────────────────────────────────
print("=" * 90)
print("Loading data...")
df   = eng.load_cache(eng.DATA_CACHE)
assert df is not None
df   = eng._attach_real_only_driver_columns(df)
rf_s, rf_src = eng._load_daily_ff_rf_aligned(df)
rf_arr = np.resize(rf_s, len(df))
dates = pd.DatetimeIndex(df.index)
print(f"  Rows: {len(df):,}   RF: {rf_src}")

ASSETS_TO_TEST = [('UPRO', 3.0), ('SSO', 2.0), ('TQQQ', 3.0)]
all_results = {}

for asset, lev in ASSETS_TO_TEST:
    print("\n" + "=" * 90)
    print(f"ASSET: {asset}   L={lev:.0f}x")
    print("=" * 90)

    real_col   = f'{asset}_Real_Ret'
    syn_col    = f'{asset}_IsSynthetic'
    driver_col = eng._underlying_return_column_for_asset(asset, df=df, prefer_real_only=True)
    fee_daily  = float(eng.ASSETS[asset]['expense_ratio']) / 252.0

    if real_col not in df.columns or driver_col not in df.columns:
        print(f"  SKIP — missing columns"); continue

    is_syn  = df.get(syn_col, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    act_s   = pd.to_numeric(df[real_col],   errors='coerce')
    drv_s   = pd.to_numeric(df[driver_col], errors='coerce')
    real_ok = act_s.notna() & drv_s.notna() & (~is_syn)

    # ── Step 1: 63d rolling vol on driver (full history, causal window) ───────
    # Compute on the full driver series (SPY_Ret, QQQ_Ret have long history)
    # This is causal: rolling window ends at row t, looks back 63 days.
    drv_full = pd.to_numeric(df[driver_col].fillna(np.nan), errors='coerce')
    rvol_full = (drv_full
                 .rolling(window=VOL_WINDOW, min_periods=MIN_VOL_OBS)
                 .std() * np.sqrt(252))

    # ── Step 2: calibrate thresholds on TRAIN rows only ───────────────────────
    train_real = real_ok & (dates <= pd.Timestamp(TRAIN_END))
    valid_real = real_ok & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

    if train_real.sum() < 252 or valid_real.sum() < 252:
        print(f"  SKIP — insufficient data"); continue

    rvol_train = rvol_full[train_real].dropna()
    thresh_mid = float(np.percentile(rvol_train, P_MID))   # P60
    thresh_hi  = float(np.percentile(rvol_train, P_HI))    # P90

    print(f"  Driver     : {driver_col}")
    print(f"  TRAIN range: {df[train_real].index[0].date()} → {df[train_real].index[-1].date()}  "
          f"N={train_real.sum():,}")
    print(f"  VALID range: {df[valid_real].index[0].date()} → {df[valid_real].index[-1].date()}  "
          f"N={valid_real.sum():,}")
    print(f"\n  Vol thresholds (from TRAIN rolling 63d vol):")
    print(f"    P60 = {thresh_mid*100:.2f}%/yr  ← low/high boundary")
    print(f"    P90 = {thresh_hi *100:.2f}%/yr  ← high/crash boundary")

    # ── Classify regimes (0=low, 1=high, 2=crash) for ALL rows ───────────────
    vol_arr  = rvol_full.to_numpy()
    regime_all = np.zeros(len(df), dtype=int)
    regime_all[vol_arr >= thresh_mid] = 1
    regime_all[vol_arr >= thresh_hi]  = 2
    regime_all[~np.isfinite(vol_arr)] = 0   # NaN vol → low fallback

    # ── Step 3: fit intercept_alpha per regime on TRAIN ───────────────────────
    idx_t  = drv_s[train_real].to_numpy()
    act_t  = act_s[train_real].to_numpy()
    rf_t   = rf_arr[train_real]
    reg_t  = regime_all[train_real]

    m0_resid_t = act_t - (lev * idx_t - (lev - 1.0) * rf_t - fee_daily)

    intercept_alpha = {}    # regime_idx → daily intercept (decimal return)
    print(f"\n  Regime distribution and fitted intercept_alpha (TRAIN):")
    print(f"  {'Regime':<8}  {'N':>5}  {'%days':>7}  "
          f"{'trim_mean %/yr':>16}  {'median %/yr':>14}  {'std %/yr':>12}  {'alpha bps/d':>12}")
    for r in [0, 1, 2]:
        mask_r = (reg_t == r)
        resid_r = m0_resid_t[mask_r]
        n_r = int(mask_r.sum())
        pct_r = 100.0 * n_r / len(reg_t)
        if n_r < 5:
            alpha_val = float(np.median(m0_resid_t)) if len(m0_resid_t) > 0 else 0.0
            print(f"  {REGIME_IDX[r]:<8}  {n_r:>5}  {pct_r:>7.1f}%  "
                  f"  [too few — falling back to overall median]")
        else:
            alpha_val = trimmed_mean(resid_r, TRIM_FRAC)
            tm_yr  = float(alpha_val * 252 * 100)
            med_yr = float(np.median(resid_r) * 252 * 100)
            std_yr = float(np.std(resid_r) * 252**0.5 * 100)
            alpha_bps = float(alpha_val * 1e4)
            print(f"  {REGIME_IDX[r]:<8}  {n_r:>5}  {pct_r:>7.1f}%  "
                  f"  {tm_yr:>+14.3f}%  {med_yr:>+12.3f}%  {std_yr:>10.3f}%  {alpha_bps:>+10.4f}")
        intercept_alpha[r] = alpha_val

    # ── Step 4: apply intercept_alpha in VALID (OOS, no lookahead) ────────────
    idx_v  = drv_s[valid_real].to_numpy()
    act_v  = act_s[valid_real].to_numpy()
    rf_v   = rf_arr[valid_real]
    reg_v  = regime_all[valid_real]

    m0_pred_v   = lev * idx_v - (lev - 1.0) * rf_v - fee_daily
    alpha_v     = np.array([intercept_alpha[r] for r in reg_v])
    mreg_pred_v = m0_pred_v + alpha_v

    # Regime distribution in VALID
    print(f"\n  Regime distribution in VALID:")
    for r in [0, 1, 2]:
        nr_v = int((reg_v == r).sum())
        print(f"    {REGIME_IDX[r]:<8}: {nr_v:>5} days  ({100.0*nr_v/len(reg_v):.1f}%)"
              f"  intercept_alpha={intercept_alpha[r]*1e4:+.4f} bps/d"
              f"  ({intercept_alpha[r]*252*100:+.3f}%/yr)")

    # ── Step 5: OOS metrics ───────────────────────────────────────────────────
    m_m0   = metrics(act_v, m0_pred_v)
    m_reg  = metrics(act_v, mreg_pred_v)

    print(f"\n  --- OOS VALID metrics ---")
    print_metrics(m_m0,  'M0_baseline     ')
    print()
    print_metrics(m_reg, 'M_regime_intercept')

    # ── Year-by-year OOS breakdown ────────────────────────────────────────────
    yy = pd.DatetimeIndex(df[valid_real].index).year
    print(f"\n  Year-by-year OOS residual mean (%/yr)  [+= model underpredicts]")
    print(f"  {'Year':>5}  {'N':>4}  {'regime%':>8}  {'M0 resid':>10}  {'Mreg resid':>12}  {'delta':>8}  {'actual CAGR':>13}")
    for yr in sorted(set(yy.tolist())):
        mask_yr  = (yy == yr)
        if mask_yr.sum() < 10: continue
        a_yr     = act_v[mask_yr]
        p0_yr    = m0_pred_v[mask_yr]
        pr_yr    = mreg_pred_v[mask_yr]
        r0       = float(np.mean(a_yr - p0_yr)  * 252 * 100)
        rr       = float(np.mean(a_yr - pr_yr)  * 252 * 100)
        cagr_yr  = _cagr(a_yr) * 100
        # dominant regime this year
        reg_yr   = reg_v[mask_yr]
        dom_r    = REGIME_IDX[int(stats.mode(reg_yr, keepdims=False).mode)]
        print(f"  {yr:>5}  {mask_yr.sum():>4}  {dom_r:>8}  "
              f"{r0:>+10.3f}%  {rr:>+12.3f}%  {rr-r0:>+8.3f}%  {cagr_yr:>+13.2f}%")

    all_results[asset] = {
        'M0':          m_m0,
        'M_regime':    m_reg,
        'intercept_alpha': intercept_alpha,
        'thresholds': (thresh_mid, thresh_hi),
        'lev': lev,
        'driver_col': driver_col,
    }

# ─── Final OOS summary table ──────────────────────────────────────────────────
print("\n" + "=" * 90)
print("OOS SUMMARY TABLE  (2015–2024)")
print("=" * 90)

rows = [
    ('mean_resid_pct',  'mean_resid(%/yr)',  '{:+.3f}%'),
    ('CAGR_diff_pct',   'CAGR_diff(%/yr)',   '{:+.3f}%'),
    ('TW_rel_err_pct',  'TW_rel_err(%)',     '{:.2f}%'),
    ('TE_ann_pct',      'TE_ann(%)',         '{:.4f}%'),
    ('MaxDD_diff_pct',  'MaxDD_diff(%)',     '{:.3f}%'),
    ('lag1_AC',         'lag-1 AC',          '{:+.4f}'),
]

col_w = 16
hdr = f"  {'Metric':<22}"
for asset, _ in ASSETS_TO_TEST:
    if asset not in all_results:
        continue
    for lbl in ['M0', 'M_regime']:
        hdr += f"  {(asset+'/'+lbl):>{col_w}}"
print(hdr)
print("  " + "-" * (22 + (col_w + 2) * 2 * len(all_results)))

for key, label, fmt in rows:
    row = f"  {label:<22}"
    for asset, _ in ASSETS_TO_TEST:
        if asset not in all_results:
            continue
        for mdl in ['M0', 'M_regime']:
            v = all_results[asset][mdl].get(key, np.nan)
            row += f"  {fmt.format(v):>{col_w}}"
    print(row)

# Gates
print(f"\n  Gate: TW_rel_err < 10%  |  |CAGR_diff| < 0.5%/yr")
print(f"  {'':22}", end="")
for asset, _ in ASSETS_TO_TEST:
    if asset not in all_results:
        continue
    for mdl in ['M0', 'M_regime']:
        tw  = all_results[asset][mdl]['TW_rel_err_pct']
        cd  = abs(all_results[asset][mdl]['CAGR_diff_pct'])
        ok  = (tw < 10.0) and (cd < 0.5)
        print(f"  {'PASS' if ok else 'FAIL':>{col_w}}", end="")
print()

# ─── Intercept alpha parameter table ─────────────────────────────────────────
print("\n" + "=" * 90)
print("FITTED INTERCEPT_ALPHA PARAMETERS  (from TRAIN, applied OOS)")
print("=" * 90)
print(f"  {'Asset':>6}  {'P60 vol':>8}  {'P90 vol':>8}  "
      f"{'α_low %/yr':>12}  {'α_high %/yr':>14}  {'α_crash %/yr':>15}")
for asset, _ in ASSETS_TO_TEST:
    if asset not in all_results:
        continue
    res = all_results[asset]
    v60, v90 = res['thresholds']
    ia = res['intercept_alpha']
    print(f"  {asset:>6}  {v60*100:>8.2f}%  {v90*100:>8.2f}%  "
          f"  {ia[0]*252*100:>+10.3f}%  {ia[1]*252*100:>+12.3f}%  {ia[2]*252*100:>+13.3f}%")

print("\n" + "=" * 90)
print("KEY:")
print("  intercept_alpha : regime-dependent constant added to M0 prediction (bps/day)")
print("  mean_resid      : actual − predicted  (+= model underpredicts)")
print("  CAGR_diff       : pred_CAGR − actual_CAGR  (+= overpredicts)")
print("  Thresholds calibrated on TRAIN only (no lookahead)")
print("=" * 90)
