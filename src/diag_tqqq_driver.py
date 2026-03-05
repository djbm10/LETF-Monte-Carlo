"""
TQQQ driver audit and regime-intercept failure root-cause analysis.

Parallel to diag_baseline_audit.py but focused on:
  1. QQQ_Ret_RealOnly driver source and basis (vs TQQQ_Real_Ret)
  2. TQQQ TRAIN M0 residual year-by-year — identify which years drive high alpha
  3. Per-year breakdown WITHIN the high-vol regime (TRAIN only)
  4. QLD (2x QQQ) comparison — same driver, less leverage
  5. Test: late-TRAIN-only intercept_alpha (2012-2014 vs 2010-2014)
  6. Test: regime-intercept with shrinkage toward overall mean

TRAIN: pre-2015  |  VALID (OOS): 2015-2024
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

# ── constants ──────────────────────────────────────────────────────────────────
TRAIN_END   = '2014-12-31'
VALID_START = '2015-01-01'
VALID_END   = '2024-12-31'
VOL_WINDOW  = 63
P_MID       = 60.0
P_HI        = 90.0
TRIM_FRAC   = 0.10

# ── load data ──────────────────────────────────────────────────────────────────
print("=" * 90)
print("Loading data...")
df = eng.load_cache(eng.DATA_CACHE)
assert df is not None
df = eng._attach_real_only_driver_columns(df)
rf_s, rf_src = eng._load_daily_ff_rf_aligned(df)
rf_arr = np.resize(rf_s, len(df))
dates  = pd.DatetimeIndex(df.index)
print(f"  Rows: {len(df):,}   RF: {rf_src}   {df.index[0].date()} → {df.index[-1].date()}")

# ── helper: trimmed mean ───────────────────────────────────────────────────────
def trimmed_mean(x, frac=TRIM_FRAC):
    n = len(x)
    if n < 4: return float(np.median(x))
    k = max(1, int(np.floor(n * frac / 2)))
    s = np.sort(x)
    return float(np.mean(s[k: n - k]))

def _cagr(rets):
    n = len(rets)
    if n < 2: return np.nan
    return float(np.prod(1.0 + rets) ** (252.0 / n) - 1.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: QQQ driver source and basis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 1: QQQ driver source and basis check")
print("=" * 90)

# Identify driver column for TQQQ
tqqq_driver = eng._underlying_return_column_for_asset('TQQQ', df=df, prefer_real_only=True)
qqq_driver  = eng._underlying_return_column_for_asset('QLD',  df=df, prefer_real_only=True)
print(f"  TQQQ driver_col : {tqqq_driver}")
print(f"  QLD  driver_col : {qqq_driver}")

# Check all QQQ-related columns
qqq_related = [c for c in df.columns if 'QQQ' in c or 'qqq' in c.lower() or 'IXIC' in c]
print(f"\n  QQQ-related columns in df: {qqq_related}")

# Compare QQQ_Ret_RealOnly vs QQQ_Ret
for colA, colB in [('QQQ_Ret_RealOnly', 'QQQ_Ret'), ('QQQ_Ret', 'SPY_Ret')]:
    if colA in df.columns and colB in df.columns:
        a = pd.to_numeric(df[colA], errors='coerce').dropna()
        b = pd.to_numeric(df[colB], errors='coerce').dropna()
        # align
        idx = a.index.intersection(b.index)
        a, b = a[idx], b[idx]
        if len(a) > 2:
            corr  = float(np.corrcoef(a, b)[0, 1])
            diff  = (a - b).abs()
            print(f"\n  {colA} vs {colB}: N={len(a):,}  corr={corr:.8f}  max|diff|={diff.max():.2e}  "
                  f"mean(A-B)/yr={float((a - b).mean())*252*100:+.4f}%")

# Source tag for QQQ_Ret_RealOnly if available
qro_tag = getattr(df, '_qqq_ret_source', None)
print(f"\n  _qqq_ret_source attribute : {qro_tag}")

# Check if SPY_Real_Price / QQQ_Real_Price in df
for price_col in ['SPY_Real_Price', 'QQQ_Real_Price', 'TQQQ_Real_Price']:
    if price_col in df.columns:
        has_real = df[price_col].notna().sum()
        print(f"  {price_col}: {has_real:,} non-NaN rows")
    else:
        print(f"  {price_col}: NOT in df")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: TQQQ TRAIN M0 residual year-by-year
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 2: TQQQ TRAIN M0 residual year-by-year (2010-2014)")
print("=" * 90)
print("  TQQQ model: r = 3*QQQ_Ret_RealOnly - 2*rf_daily - expense_daily")

lev_tqqq = 3.0
fee_tqqq = float(eng.ASSETS['TQQQ']['expense_ratio']) / 252.0

real_col_t = 'TQQQ_Real_Ret'
syn_col_t  = 'TQQQ_IsSynthetic'
driver_t   = tqqq_driver

is_syn_t = df.get(syn_col_t, pd.Series(True, index=df.index)).fillna(True).astype(bool)
act_t_s  = pd.to_numeric(df[real_col_t],  errors='coerce') if real_col_t in df.columns else pd.Series(np.nan, index=df.index)
drv_t_s  = pd.to_numeric(df[driver_t],    errors='coerce')
real_ok_t = act_t_s.notna() & drv_t_s.notna() & (~is_syn_t)

train_real_t = real_ok_t & (dates <= pd.Timestamp(TRAIN_END))
valid_real_t = real_ok_t & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

act_train  = act_t_s[train_real_t].to_numpy()
drv_train  = drv_t_s[train_real_t].to_numpy()
rf_train   = rf_arr[train_real_t]
yrs_train  = pd.DatetimeIndex(df[train_real_t].index).year

m0_resid_train = act_train - (lev_tqqq * drv_train - (lev_tqqq - 1.0) * rf_train - fee_tqqq)

print(f"\n  TRAIN N={train_real_t.sum():,}  VALID N={valid_real_t.sum():,}")
print(f"  Overall TRAIN M0 residual mean   : {np.mean(m0_resid_train)*252*100:+.3f}%/yr")
print(f"  Overall TRAIN M0 residual tm10%  : {trimmed_mean(m0_resid_train)*252*100:+.3f}%/yr")
print(f"  Overall TRAIN M0 residual median : {np.median(m0_resid_train)*252*100:+.3f}%/yr")
print(f"  Overall TRAIN M0 residual std    : {np.std(m0_resid_train)*np.sqrt(252)*100:.3f}%/yr (TE)")

print(f"\n  {'Year':>5}  {'N':>5}  {'rf_mean%/yr':>12}  {'M0_resid%/yr':>14}  "
      f"{'TQQQ actual%/yr':>17}  {'M0 pred%/yr':>13}  {'drv×3 %/yr':>13}")
for yr in sorted(set(yrs_train.tolist())):
    mask = (yrs_train == yr)
    a = act_train[mask]
    d = drv_train[mask]
    r = rf_train[mask]
    resid = a - (lev_tqqq * d - (lev_tqqq - 1.0) * r - fee_tqqq)
    m0_pred = lev_tqqq * d - (lev_tqqq - 1.0) * r - fee_tqqq
    print(f"  {yr:>5}  {int(mask.sum()):>5}  {np.mean(r)*252*100:>12.3f}%  "
          f"  {np.mean(resid)*252*100:>+12.3f}%  "
          f"{_cagr(a)*100:>+15.2f}%  {_cagr(m0_pred)*100:>+11.2f}%  {np.mean(d)*252*3*100:>+11.3f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: Regime classification for TQQQ TRAIN, residual per regime per year
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 3: TQQQ TRAIN residual per REGIME and per YEAR within regime")
print("=" * 90)

# Rolling vol on driver (full history, causal)
drv_full_t = pd.to_numeric(df[driver_t].fillna(np.nan), errors='coerce')
rvol_full_t = drv_full_t.rolling(window=VOL_WINDOW, min_periods=20).std() * np.sqrt(252)

# Thresholds from TRAIN real days
rvol_train_t = rvol_full_t[train_real_t].dropna()
thresh_mid_t = float(np.percentile(rvol_train_t, P_MID))
thresh_hi_t  = float(np.percentile(rvol_train_t, P_HI))

print(f"  Vol thresholds (TQQQ TRAIN rolling 63d vol of {driver_t}):")
print(f"    P60 = {thresh_mid_t*100:.2f}%/yr  ← low/high boundary")
print(f"    P90 = {thresh_hi_t *100:.2f}%/yr  ← high/crash boundary")

vol_arr_t  = rvol_full_t.to_numpy()
regime_all_t = np.zeros(len(df), dtype=int)
regime_all_t[vol_arr_t >= thresh_mid_t] = 1
regime_all_t[vol_arr_t >= thresh_hi_t]  = 2
regime_all_t[~np.isfinite(vol_arr_t)] = 0

reg_train_t = regime_all_t[train_real_t]
RNAME = {0: 'low', 1: 'high', 2: 'crash'}

# Per-regime overall intercept_alpha
print(f"\n  Overall TRAIN intercept_alpha per regime:")
intercept_alpha_v1 = {}
for r in [0, 1, 2]:
    mask_r = (reg_train_t == r)
    resid_r = m0_resid_train[mask_r]
    n_r = int(mask_r.sum())
    if n_r < 5:
        ia = trimmed_mean(m0_resid_train)
        print(f"    {RNAME[r]:<8}: N={n_r:>4}  α=overall (too few)  {ia*252*100:+.3f}%/yr")
    else:
        ia = trimmed_mean(resid_r)
        print(f"    {RNAME[r]:<8}: N={n_r:>4} ({100.*n_r/len(reg_train_t):.1f}%)  "
              f"trim_mean={ia*252*100:+.3f}%/yr  "
              f"median={np.median(resid_r)*252*100:+.3f}%/yr  "
              f"std={np.std(resid_r)*np.sqrt(252)*100:.2f}%/yr")
    intercept_alpha_v1[r] = ia

# Per-regime per-year breakdown
for r in [0, 1, 2]:
    mask_r = (reg_train_t == r)
    if mask_r.sum() < 2: continue
    yrs_r   = yrs_train[mask_r]
    resid_r = m0_resid_train[mask_r]
    drv_r   = drv_train[mask_r]
    rf_r    = rf_train[mask_r]

    print(f"\n  HIGH-DETAIL: regime={RNAME[r]} ({mask_r.sum()} days)")
    print(f"  {'Year':>5}  {'N':>4}  {'rf%/yr':>8}  {'resid%/yr':>11}  "
          f"{'drv3x%/yr':>11}  {'rvol_mean':>12}")
    for yr in sorted(set(yrs_r.tolist())):
        mask_yr = (yrs_r == yr)
        res_yr  = resid_r[mask_yr]
        # mean rolling vol for these days
        rvol_yr = rvol_full_t[train_real_t][mask_r].to_numpy()[mask_yr]
        print(f"  {yr:>5}  {int(mask_yr.sum()):>4}  "
              f"{np.mean(rf_r[mask_yr])*252*100:>8.3f}%  "
              f"{np.mean(res_yr)*252*100:>+11.3f}%  "
              f"{np.mean(drv_r[mask_yr])*252*3*100:>+11.3f}%  "
              f"{np.nanmean(rvol_yr)*100:>10.2f}%/yr")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: TQQQ OOS M0 residual year-by-year (to understand OOS true level)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 4: TQQQ OOS M0 residual year-by-year (2015-2024)")
print("=" * 90)

act_valid  = act_t_s[valid_real_t].to_numpy()
drv_valid  = drv_t_s[valid_real_t].to_numpy()
rf_valid   = rf_arr[valid_real_t]
reg_valid  = regime_all_t[valid_real_t]
yrs_valid  = pd.DatetimeIndex(df[valid_real_t].index).year

m0_pred_valid  = lev_tqqq * drv_valid - (lev_tqqq - 1.0) * rf_valid - fee_tqqq
m0_resid_valid = act_valid - m0_pred_valid

print(f"\n  Overall OOS M0 residual mean   : {np.mean(m0_resid_valid)*252*100:+.3f}%/yr")
print(f"  Overall OOS M0 residual tm10%  : {trimmed_mean(m0_resid_valid)*252*100:+.3f}%/yr")
print(f"  Overall OOS M0 residual median : {np.median(m0_resid_valid)*252*100:+.3f}%/yr")
print(f"  Overall OOS M0 residual std    : {np.std(m0_resid_valid)*np.sqrt(252)*100:.3f}%/yr")
print(f"  lag-1 autocorr                 : {float(pd.Series(m0_resid_valid).autocorr(1)):+.4f}")

print(f"\n  {'Year':>5}  {'N':>5}  {'rf%/yr':>8}  {'M0_resid%/yr':>14}  "
      f"{'TQQQ_cagr':>11}  {'M0_cagr':>9}  {'dom_regime':>12}")
for yr in sorted(set(yrs_valid.tolist())):
    mask = (yrs_valid == yr)
    a = act_valid[mask]
    p = m0_pred_valid[mask]
    r = rf_valid[mask]
    resid = a - p
    dom_r = RNAME[int(stats.mode(reg_valid[mask], keepdims=False).mode)]
    print(f"  {yr:>5}  {int(mask.sum()):>5}  {np.mean(r)*252*100:>8.3f}%  "
          f"  {np.mean(resid)*252*100:>+12.3f}%  "
          f"{_cagr(a)*100:>+9.2f}%  {_cagr(p)*100:>+7.2f}%  {dom_r:>12s}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: QLD (2x QQQ) baseline residual as sanity check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 5: QLD (2x QQQ) baseline residual — longer history, same driver")
print("=" * 90)

lev_qld = 2.0
fee_qld = float(eng.ASSETS.get('QLD', {}).get('expense_ratio', 0.0095)) / 252.0
driver_qld = qqq_driver

real_col_qld = 'QLD_Real_Ret'
syn_col_qld  = 'QLD_IsSynthetic'

if real_col_qld in df.columns:
    is_syn_qld = df.get(syn_col_qld, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    act_qld_s  = pd.to_numeric(df[real_col_qld], errors='coerce')
    drv_qld_s  = pd.to_numeric(df[driver_qld],   errors='coerce')
    real_ok_qld = act_qld_s.notna() & drv_qld_s.notna() & (~is_syn_qld)

    train_qld = real_ok_qld & (dates <= pd.Timestamp(TRAIN_END))
    valid_qld = real_ok_qld & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

    act_qtrain = act_qld_s[train_qld].to_numpy()
    drv_qtrain = drv_qld_s[train_qld].to_numpy()
    rf_qtrain  = rf_arr[train_qld]
    m0_qld_tr  = act_qtrain - (lev_qld * drv_qtrain - (lev_qld - 1.0) * rf_qtrain - fee_qld)

    act_qval   = act_qld_s[valid_qld].to_numpy()
    drv_qval   = drv_qld_s[valid_qld].to_numpy()
    rf_qval    = rf_arr[valid_qld]
    m0_qld_v   = act_qval - (lev_qld * drv_qval - (lev_qld - 1.0) * rf_qval - fee_qld)

    yrs_qtrain = pd.DatetimeIndex(df[train_qld].index).year
    print(f"  QLD TRAIN N={train_qld.sum():,}  VALID N={valid_qld.sum():,}  fee_daily={fee_qld*1e4:.4f} bps")
    print(f"  TRAIN range: {df[train_qld].index[0].date()} → {df[train_qld].index[-1].date()}")
    print(f"  TRAIN M0 residual mean   : {np.mean(m0_qld_tr)*252*100:+.3f}%/yr")
    print(f"  TRAIN M0 residual tm10%  : {trimmed_mean(m0_qld_tr)*252*100:+.3f}%/yr")
    print(f"  VALID M0 residual mean   : {np.mean(m0_qld_v)*252*100:+.3f}%/yr")

    print(f"\n  QLD year-by-year (TRAIN):")
    print(f"  {'Year':>5}  {'N':>4}  {'M0 resid %/yr':>15}  {'QLD CAGR':>11}  {'M0 CAGR':>10}")
    for yr in sorted(set(yrs_qtrain.tolist())):
        mask = (yrs_qtrain == yr)
        a = act_qtrain[mask]
        d = drv_qtrain[mask]
        r = rf_qtrain[mask]
        resid = a - (lev_qld * d - (lev_qld - 1.0) * r - fee_qld)
        m0p = lev_qld * d - (lev_qld - 1.0) * r - fee_qld
        print(f"  {yr:>5}  {int(mask.sum()):>4}  {np.mean(resid)*252*100:>+13.3f}%  "
              f"{_cagr(a)*100:>+9.2f}%  {_cagr(m0p)*100:>+8.2f}%")
else:
    print("  QLD_Real_Ret not in df — QLD check skipped")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: Test late-TRAIN-only intercept_alpha for TQQQ (2012-2014)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 6: Test — late-TRAIN-only intercept_alpha for TQQQ (2012-2014 only)")
print("=" * 90)

LATE_TRAIN_START = '2012-01-01'
late_train_t = train_real_t & (dates >= pd.Timestamp(LATE_TRAIN_START))
print(f"  Late-train window: {LATE_TRAIN_START} → {TRAIN_END}  N={late_train_t.sum():,}")

act_lt  = act_t_s[late_train_t].to_numpy()
drv_lt  = drv_t_s[late_train_t].to_numpy()
rf_lt   = rf_arr[late_train_t]
reg_lt  = regime_all_t[late_train_t]
m0_lt   = act_lt - (lev_tqqq * drv_lt - (lev_tqqq - 1.0) * rf_lt - fee_tqqq)

intercept_alpha_late = {}
print(f"\n  Late-TRAIN per-regime intercept_alpha:")
for r in [0, 1, 2]:
    mask_r = (reg_lt == r)
    n_r = int(mask_r.sum())
    resid_r = m0_lt[mask_r]
    if n_r < 5:
        ia = trimmed_mean(m0_lt)
        print(f"    {RNAME[r]:<8}: N={n_r:>4}  α=overall {ia*252*100:+.3f}%/yr (too few)")
    else:
        ia = trimmed_mean(resid_r)
        print(f"    {RNAME[r]:<8}: N={n_r:>4} ({100.*n_r/len(reg_lt):.1f}%)  "
              f"trim_mean={ia*252*100:+.3f}%/yr  "
              f"median={np.median(resid_r)*252*100:+.3f}%/yr")
    intercept_alpha_late[r] = ia

# OOS performance with late-train alpha
alpha_v_late = np.array([intercept_alpha_late[r] for r in reg_valid])
mreg_pred_late = m0_pred_valid + alpha_v_late

W_act  = float(np.prod(1.0 + act_valid))
W_late = float(np.prod(1.0 + mreg_pred_late))
cagr_a = _cagr(act_valid); cagr_late = _cagr(mreg_pred_late)
tw_late = abs(W_late / W_act - 1.0) * 100.0
cd_late = (cagr_late - cagr_a) * 100.0

print(f"\n  OOS results with late-TRAIN alpha (2012-2014 calibration):")
print(f"    CAGR_diff : {cd_late:+.3f}%/yr  ({'✓' if abs(cd_late)<0.5 else '✗'} gate: <0.5%)")
print(f"    TW_rel_err: {tw_late:.2f}%    ({'✓' if tw_late<10 else '✗'} gate: <10%)")
print(f"    Mean resid: {np.mean(act_valid-mreg_pred_late)*252*100:+.3f}%/yr")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: Test — shrinkage toward overall trimmed mean
# Shrink per-regime alpha toward grand mean using inverse-sample-size weighting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 7: Test — shrinkage toward overall mean (TQQQ, full TRAIN)")
print("=" * 90)
print("  Formula: alpha_r_shrunk = (n_r * alpha_r_raw + N_PRIOR * alpha_overall) / (n_r + N_PRIOR)")

alpha_overall_tqqq = trimmed_mean(m0_resid_train)
print(f"  alpha_overall (TRAIN tm10%) = {alpha_overall_tqqq*252*100:+.3f}%/yr")

def _test_shrinkage(N_PRIOR: int) -> tuple:
    """Return (CAGR_diff, TW_rel_err) for TQQQ OOS with given shrinkage strength."""
    shrunk = {}
    for r in [0, 1, 2]:
        mask_r = (reg_train_t == r)
        n_r = int(mask_r.sum())
        resid_r = m0_resid_train[mask_r]
        alpha_raw = trimmed_mean(resid_r) if n_r >= 5 else alpha_overall_tqqq
        alpha_s = (n_r * alpha_raw + N_PRIOR * alpha_overall_tqqq) / (n_r + N_PRIOR)
        shrunk[r] = alpha_s
    alpha_v = np.array([shrunk[r] for r in reg_valid])
    pred_v  = m0_pred_valid + alpha_v
    W_a = float(np.prod(1.0 + act_valid))
    W_p = float(np.prod(1.0 + pred_v))
    cagr_d = (_cagr(pred_v) - _cagr(act_valid)) * 100.0
    tw_e   = abs(W_p / W_a - 1.0) * 100.0
    # show applied alpha weights
    wtd_alpha = float(np.mean(alpha_v)) * 252 * 100
    return cagr_d, tw_e, wtd_alpha, shrunk

print(f"\n  {'N_PRIOR':>8}  {'CAGR_diff':>11}  {'TW_rel_err':>11}  {'Wtd_alpha%/yr':>15}  "
      f"{'α_low%/yr':>12}  {'α_high%/yr':>12}  {'α_crash%/yr':>13}")
for n_prior in [0, 126, 252, 504, 1008, 2016]:
    cd, tw, wa, shrunk = _test_shrinkage(n_prior)
    ok = ("✓✓" if abs(cd) < 0.5 and tw < 10.0 else
          "✓✗" if abs(cd) < 0.5 else
          "✗✓" if tw < 10.0 else "✗✗")
    print(f"  {n_prior:>8}  {cd:>+11.3f}%  {tw:>9.2f}%  "
          f"  {wa:>+13.3f}%  "
          f"{shrunk[0]*252*100:>+10.3f}%  {shrunk[1]*252*100:>+10.3f}%  "
          f"{shrunk[2]*252*100:>+11.3f}%  {ok}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: Cross-asset summary — shrinkage at optimal N_PRIOR for each
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 8: Cross-asset summary — regime-intercept with shrinkage")
print("  (using FULL TRAIN for all assets; N_PRIOR swept per-asset to find ~0 CAGR_diff)")
print("=" * 90)

ASSETS_TEST = [('UPRO', 3.0), ('SSO', 2.0), ('TQQQ', 3.0), ('QLD', 2.0)]

for asset, lev in ASSETS_TEST:
    real_col   = f'{asset}_Real_Ret'
    syn_col    = f'{asset}_IsSynthetic'
    drv_col    = eng._underlying_return_column_for_asset(asset, df=df, prefer_real_only=True)
    fee_d      = float(eng.ASSETS.get(asset, {}).get('expense_ratio', 0.01)) / 252.0

    if real_col not in df.columns or drv_col not in df.columns:
        continue

    is_syn  = df.get(syn_col, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    act_s   = pd.to_numeric(df[real_col],  errors='coerce')
    drv_s   = pd.to_numeric(df[drv_col],   errors='coerce')
    real_ok = act_s.notna() & drv_s.notna() & (~is_syn)

    train_m = real_ok & (dates <= pd.Timestamp(TRAIN_END))
    valid_m = real_ok & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))
    if train_m.sum() < 252 or valid_m.sum() < 252:
        continue

    # Rolling vol + regime
    drv_full = pd.to_numeric(df[drv_col].fillna(np.nan), errors='coerce')
    rvol     = drv_full.rolling(window=VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
    rvol_tr  = rvol[train_m].dropna()
    thr_mid  = float(np.percentile(rvol_tr, P_MID))
    thr_hi   = float(np.percentile(rvol_tr, P_HI))
    vol_a    = rvol.to_numpy()
    reg_a    = np.zeros(len(df), dtype=int)
    reg_a[vol_a >= thr_mid] = 1
    reg_a[vol_a >= thr_hi]  = 2
    reg_a[~np.isfinite(vol_a)] = 0

    # TRAIN data
    act_tr   = act_s[train_m].to_numpy()
    drv_tr   = drv_s[train_m].to_numpy()
    rf_tr    = rf_arr[train_m]
    reg_tr   = reg_a[train_m]
    m0_r_tr  = act_tr - (lev * drv_tr - (lev - 1.0) * rf_tr - fee_d)
    alpha_ov = trimmed_mean(m0_r_tr)

    # VALID data
    act_v    = act_s[valid_m].to_numpy()
    drv_v    = drv_s[valid_m].to_numpy()
    rf_v     = rf_arr[valid_m]
    reg_v    = reg_a[valid_m]
    m0_pred  = lev * drv_v - (lev - 1.0) * rf_v - fee_d

    # Sweep N_PRIOR
    print(f"\n  {'─'*70}")
    print(f"  Asset: {asset}  L={lev:.0f}x  TRAIN_N={train_m.sum():,}  VALID_N={valid_m.sum():,}")
    print(f"  alpha_overall={alpha_ov*252*100:+.3f}%/yr  driver={drv_col}")
    print(f"  {'N_PRIOR':>8}  {'CAGR_diff':>11}  {'TW_rel_err':>11}  {'Wtd_alpha':>11}")

    for n_prior in [0, 252, 504, 1008, 2016]:
        shrunk = {}
        for r in [0, 1, 2]:
            mask_r = (reg_tr == r)
            n_r    = int(mask_r.sum())
            alpha_raw = trimmed_mean(m0_r_tr[mask_r]) if n_r >= 5 else alpha_ov
            shrunk[r] = (n_r * alpha_raw + n_prior * alpha_ov) / (n_r + n_prior)
        alpha_v = np.array([shrunk[r] for r in reg_v])
        pred_v  = m0_pred + alpha_v
        W_a = float(np.prod(1.0 + act_v))
        W_p = float(np.prod(1.0 + pred_v))
        cd  = (_cagr(pred_v) - _cagr(act_v)) * 100.0
        tw  = abs(W_p / W_a - 1.0) * 100.0
        wa  = float(np.mean(alpha_v)) * 252 * 100
        ok  = ("✓✓" if abs(cd) < 0.5 and tw < 10.0 else
               "✓✗" if abs(cd) < 0.5 else
               "✗✓" if tw < 10.0 else "✗✗")
        print(f"  {n_prior:>8}  {cd:>+11.3f}%  {tw:>9.2f}%  {wa:>+9.3f}%  {ok}")

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)
