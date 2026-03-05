"""
Baseline Reconstruction Audit — UPRO (and cross-check SSO, TQQQ)

Verifies:
  1. SPY_Ret_RealOnly == SPY_Ret  (origin, correlation, identity)
  2. SPY and UPRO prices share the same date index and pct_change basis
  3. Whether the 1-day lag between SPY and UPRO is an issue
  4. Theoretical return (NO alpha, NO spread): T = 3*SPY - borrow - expense
  5. Residual = UPRO_actual - T : mean, median, std, autocorr, year-by-year
  6. Cross-check TQQQ and SSO to see whether residual is asset-specific or systematic
  7. Compare SPY_Ret (auto_adjust Close) vs ^GSPC (price return) to confirm basis
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 90)
print("STEP 0 — Load cached data")
print("=" * 90)
df = eng.load_cache(eng.DATA_CACHE)
assert df is not None, "historical_data.pkl missing"
df = eng._attach_real_only_driver_columns(df)
print(f"  Rows : {len(df):,}   Range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Columns relevant: {[c for c in df.columns if any(x in c for x in ['SPY','UPRO','SSO','TQQQ','QQQ'])][:18]}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 1 — Verify SPY_Ret_RealOnly == SPY_Ret (identity and origin)")
print("=" * 90)

# Origin path of SPY_Ret_RealOnly in cached df
spy_ronly_src = df.get('SPY_Ret_RealOnly_Source', pd.Series('(no source col)', index=df.index)).iloc[0]
print(f"  SPY_Ret_RealOnly_Source tag  : {spy_ronly_src}")

spy_ret    = pd.to_numeric(df['SPY_Ret'],          errors='coerce')
spy_ronly  = pd.to_numeric(df['SPY_Ret_RealOnly'],  errors='coerce')
spy_rprice = pd.to_numeric(df.get('SPY_Real_Price', pd.Series(np.nan, index=df.index)), errors='coerce')

both_valid = spy_ret.notna() & spy_ronly.notna()
corr_sr = float(np.corrcoef(spy_ret[both_valid], spy_ronly[both_valid])[0, 1])
max_diff = float(np.nanmax(np.abs(spy_ret[both_valid] - spy_ronly[both_valid])))
print(f"  corr(SPY_Ret, SPY_Ret_RealOnly) = {corr_sr:.10f}")
print(f"  max |SPY_Ret - SPY_Ret_RealOnly| = {max_diff:.2e}   (expect 0.0 if identical)")

# Also check whether SPY_Ret reconstructed from SPY_Real_Price.pct_change matches
if spy_rprice.notna().sum() > 252:
    spy_from_price = spy_rprice.pct_change()
    both2 = spy_ret.notna() & spy_from_price.notna()
    corr2 = float(np.corrcoef(spy_ret[both2], spy_from_price[both2])[0, 1])
    diff2 = float(np.nanmax(np.abs(spy_ret[both2] - spy_from_price[both2])))
    print(f"  corr(SPY_Ret, SPY_Real_Price.pct_change) = {corr2:.10f}  max_diff={diff2:.2e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 2 — Alignment: same date index and no 1-day shift between SPY and UPRO")
print("=" * 90)

# Restrict to real UPRO data
upro_syn   = df.get('UPRO_IsSynthetic', pd.Series(True, index=df.index)).fillna(True).astype(bool)
upro_real  = pd.to_numeric(df.get('UPRO_Real_Ret', pd.Series(np.nan, index=df.index)), errors='coerce')
upro_rpx   = pd.to_numeric(df.get('UPRO_Real_Price', pd.Series(np.nan, index=df.index)), errors='coerce')

real_mask  = upro_real.notna() & (~upro_syn) & spy_ret.notna()
sub = df.loc[real_mask].copy()
print(f"  Real UPRO rows: {len(sub):,}   {sub.index[0].date()} → {sub.index[-1].date()}")

# Check for any dates where SPY is NaN but UPRO is not (or vice versa)
spy_nan_upro_ok = (~spy_ret.loc[real_mask].isna()).sum()  # SPY always ok when UPRO is ok
upro_nan_spy_ok = (~upro_real.loc[real_mask].isna()).sum()
print(f"  Days where both SPY and UPRO non-NaN: {int(real_mask.sum()):,}")
print(f"  SPY non-NaN in real_mask: {spy_nan_upro_ok:,}  (expect = {int(real_mask.sum()):,})")
print(f"  UPRO non-NaN in real_mask: {upro_nan_spy_ok:,} (expect = {int(real_mask.sum()):,})")

# 1-day lag test: correlate UPRO_t vs SPY_t, UPRO_t vs SPY_{t-1}, UPRO_t vs SPY_{t+1}
spy_sub   = spy_ret.loc[real_mask].to_numpy()
upro_sub  = upro_real.loc[real_mask].to_numpy()
c_same    = float(np.corrcoef(upro_sub, spy_sub)[0, 1])
c_lag1    = float(np.corrcoef(upro_sub[1:], spy_sub[:-1])[0, 1])   # UPRO lags SPY by 1d
c_lead1   = float(np.corrcoef(upro_sub[:-1], spy_sub[1:])[0, 1])   # UPRO leads SPY by 1d
print(f"\n  Lag test (should be ~0.997 same-day; lags should be much lower):")
print(f"  corr(UPRO_t,   SPY_t)   = {c_same:.6f}   ← same-day")
print(f"  corr(UPRO_t,   SPY_t-1) = {c_lag1:.6f}   ← UPRO lags SPY by 1d")
print(f"  corr(UPRO_t,   SPY_t+1) = {c_lead1:.6f}   ← UPRO leads SPY by 1d")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 3 — Return basis check: SPY total return vs price return")
print("=" * 90)

# Load RF
rf_full, rf_src = eng._load_daily_ff_rf_aligned(df)
print(f"  RF source: {rf_src}")

# Compute trailing 12m dividend yield proxy: (SPY_total_ret - SPY_price_ret)
# We approximate SPY price return by assuming ^GSPC is the price-return index.
# If df has ^GSPC data, compare. Otherwise use SPY_Ret directly.
gspc_col = [c for c in df.columns if 'GSPC' in c or ('SPY' in c and 'Price' in c and 'Real' not in c)]
print(f"  Potential price-return columns: {gspc_col}")

# If SPY_Real_Price exists, also compare SPY_Ret with pct_change of Close (if accessible)
# Just print mean SPY_Ret for real period — compare to known SPY total return CAGR
real_spy_ann = float(spy_ret.loc[real_mask].mean() * 252 * 100)
real_upro_ann = float(upro_real.loc[real_mask].mean() * 252 * 100)
lev_spy_ann = 3.0 * real_spy_ann
print(f"\n  Arithmetic mean annualised (real UPRO period {sub.index[0].date()}→{sub.index[-1].date()}):")
print(f"  SPY_Ret mean          = {real_spy_ann:+.3f}%/yr")
print(f"  3 × SPY_Ret mean      = {lev_spy_ann:+.3f}%/yr  (expected to exceed UPRO by ~borrow+expense)")
print(f"  UPRO_Real_Ret mean    = {real_upro_ann:+.3f}%/yr")
print(f"  UPRO − 3×SPY (arith)  = {real_upro_ann - lev_spy_ann:+.3f}%/yr  (sign: neg = UPRO beats formula)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 4 — Theoretical return: T = 3*SPY - borrow - expense  (no alpha, no spread)")
print("=" * 90)

lev = 3.0
exp_daily = eng.ASSETS['UPRO']['expense_ratio'] / 252.0
print(f"  Leverage      = {lev:.1f}x")
print(f"  Expense/day   = {exp_daily*1e4:.4f} bps  ({eng.ASSETS['UPRO']['expense_ratio']*100:.4f}%/yr)")

spy_sub_v   = spy_ret.loc[real_mask].to_numpy()
upro_sub_v  = upro_real.loc[real_mask].to_numpy()
rf_sub_v    = np.resize(rf_full, len(df))[real_mask]

borrow      = (lev - 1.0) * rf_sub_v
theoretical = lev * spy_sub_v - borrow - exp_daily

residual    = upro_sub_v - theoretical

print(f"\n  Borrow cost (daily mean)  = {np.mean(borrow)*1e4:.4f} bps/day  ({np.mean(borrow)*252*100:.3f}%/yr)")
print(f"\n  === Theoretical vs actual (NO alpha, NO spread) ===")
print(f"  Theoretical mean          = {np.mean(theoretical)*1e4:+.4f} bps/day  ({np.mean(theoretical)*252*100:+.3f}%/yr)")
print(f"  Actual UPRO mean          = {np.mean(upro_sub_v)*1e4:+.4f} bps/day  ({np.mean(upro_sub_v)*252*100:+.3f}%/yr)")
print(f"  Mean difference (act−T)   = {np.mean(residual)*1e4:+.4f} bps/day  ({np.mean(residual)*252*100:+.3f}%/yr)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 5 — Residual analysis: residual = UPRO_actual - theoretical")
print("=" * 90)

resid_ann = residual * 252 * 100  # annualised in %
print(f"  N days                      : {len(residual):,}")
print(f"  Mean (bps/day)              : {np.mean(residual)*1e4:+.4f}")
print(f"  Mean (%/yr annualised)      : {np.mean(residual)*252*100:+.4f}")
print(f"  Median (bps/day)            : {np.median(residual)*1e4:+.4f}")
print(f"  Median (%/yr annualised)    : {np.median(residual)*252*100:+.4f}")
print(f"  Std (bps/day)               : {np.std(residual)*1e4:.4f}")
print(f"  Std (%/yr annualised TE)    : {np.std(residual)*252**0.5*100:.4f}")
print(f"  Skewness                    : {float(stats.skew(residual)):.4f}")
print(f"  Kurtosis (excess)           : {float(stats.kurtosis(residual)):.4f}")

# Autocorrelation at lag 1 and 5
ac1 = float(pd.Series(residual).autocorr(lag=1))
ac5 = float(pd.Series(residual).autocorr(lag=5))
ac21 = float(pd.Series(residual).autocorr(lag=21))
print(f"  Autocorr lag-1              : {ac1:+.4f}")
print(f"  Autocorr lag-5              : {ac5:+.4f}")
print(f"  Autocorr lag-21 (~1 month)  : {ac21:+.4f}")

# Is mean residual significantly different from zero?
t_stat, p_val = stats.ttest_1samp(residual, 0.0)
print(f"\n  t-test: mean == 0:  t={t_stat:.4f}  p={p_val:.4e}  {'REJECT (systematic bias)' if p_val < 0.01 else 'FAIL TO REJECT (consistent with zero mean)'}")

# Distribution: what fraction of the residual is explained by the mean vs noise
signal_frac = abs(np.mean(residual)) / np.std(residual)
print(f"  |mean| / std  (signal-to-noise) : {signal_frac:.4f}   "
      f"({'large: systematic bias dominates' if signal_frac > 0.1 else 'small: mean is noise-level'} )")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 5b — Year-by-year residual breakdown")
print("=" * 90)

sub2 = sub.copy()
sub2['residual'] = residual
sub2['theoretical'] = theoretical
sub2['year'] = pd.DatetimeIndex(sub2.index).year
sub2['spy'] = spy_sub_v
sub2['upro'] = upro_sub_v
sub2['borrow'] = borrow

print(f"  {'Year':>5}  {'N':>5}  {'SPY %/yr':>10}  {'T %/yr':>10}  {'UPRO %/yr':>10}  {'resid %/yr':>12}  {'resid bps/d':>12}  {'resid std bps/d':>16}")
for yr, grp in sub2.groupby('year'):
    if len(grp) < 20: continue
    spy_yr   = float(grp['spy'].mean() * 252 * 100)
    t_yr     = float(grp['theoretical'].mean() * 252 * 100)
    upro_yr  = float(grp['upro'].mean() * 252 * 100)
    r_yr_ann = float(grp['residual'].mean() * 252 * 100)
    r_bpsd   = float(grp['residual'].mean() * 1e4)
    r_std    = float(grp['residual'].std() * 1e4)
    print(f"  {yr:>5}  {len(grp):>5}  {spy_yr:>10.2f}%  {t_yr:>10.2f}%  {upro_yr:>10.2f}%  {r_yr_ann:>12.3f}%  {r_bpsd:>12.3f}  {r_std:>16.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 6 — Cross-check: SSO (2x SPY) and TQQQ (3x QQQ)")
print("=" * 90)

for asset, driver_key, asset_lev in [('SSO', 'SPY_Ret', 2.0), ('TQQQ', 'QQQ_Ret_RealOnly', 3.0)]:
    real_ret_col = f'{asset}_Real_Ret'
    syn_col_a    = f'{asset}_IsSynthetic'
    if real_ret_col not in df.columns:
        print(f"  {asset}: missing {real_ret_col}, skipped")
        continue
    driver_col_a = eng._underlying_return_column_for_asset(asset, df=df, prefer_real_only=True)
    if driver_col_a not in df.columns:
        print(f"  {asset}: missing driver {driver_col_a}, skipped")
        continue
    syn_a   = df.get(syn_col_a, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    ret_a   = pd.to_numeric(df[real_ret_col], errors='coerce')
    drv_a   = pd.to_numeric(df[driver_col_a], errors='coerce')
    mask_a  = ret_a.notna() & drv_a.notna() & (~syn_a)
    if mask_a.sum() < 252:
        print(f"  {asset}: too few real days ({mask_a.sum()}), skipped")
        continue
    exp_a   = eng.ASSETS[asset]['expense_ratio'] / 252.0
    drv_v   = drv_a[mask_a].to_numpy()
    ret_v   = ret_a[mask_a].to_numpy()
    rf_v    = np.resize(rf_full, len(df))[mask_a]
    borrow_a   = (asset_lev - 1.0) * rf_v
    theoretical_a = asset_lev * drv_v - borrow_a - exp_a
    residual_a    = ret_v - theoretical_a
    t_a, p_a = stats.ttest_1samp(residual_a, 0.0)
    print(f"\n  {asset} (driver={driver_col_a}, L={asset_lev:.0f}x, N={mask_a.sum():,}  "
          f"{df[mask_a].index[0].date()}→{df[mask_a].index[-1].date()})")
    print(f"    mean residual      : {np.mean(residual_a)*1e4:+.4f} bps/day  ({np.mean(residual_a)*252*100:+.3f}%/yr)")
    print(f"    median residual    : {np.median(residual_a)*1e4:+.4f} bps/day  ({np.median(residual_a)*252*100:+.3f}%/yr)")
    print(f"    std residual       : {np.std(residual_a)*1e4:.4f} bps/day  (TE={np.std(residual_a)*252**0.5*100:.3f}%/yr)")
    print(f"    t-test mean==0     : t={t_a:.3f}  p={p_a:.2e}")
    print(f"    |mean|/std         : {abs(np.mean(residual_a))/np.std(residual_a):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 7 — Per-day formula consistency: show 10 sample rows")
print("=" * 90)
rng = np.random.default_rng(42)
sample_idx = sorted(rng.choice(len(spy_sub_v), size=10, replace=False).tolist())
print(f"  {'Date':<12}  {'SPY%':>8}  {'3×SPY%':>8}  {'borrow%':>9}  {'exp%':>8}  {'T%':>8}  {'UPRO%':>8}  {'resid%':>9}  {'resid bps':>10}")
dates = pd.DatetimeIndex(sub.index)
for i in sample_idx:
    d   = str(dates[i].date())
    s   = spy_sub_v[i]
    b   = borrow[i]
    t   = theoretical[i]
    u   = upro_sub_v[i]
    r   = residual[i]
    print(f"  {d:<12}  {s*100:>+8.4f}  {3*s*100:>+8.4f}  {b*100:>+9.5f}  {exp_daily*100:>8.5f}  "
          f"{t*100:>+8.4f}  {u*100:>+8.4f}  {r*100:>+9.5f}  {r*1e4:>+10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 8 — UPRO price construction: verify from raw cached prices")
print("=" * 90)

# Reconstruct UPRO return from UPRO_Real_Price and compare to UPRO_Real_Ret
upro_px = pd.to_numeric(df.get('UPRO_Real_Price', pd.Series(np.nan, index=df.index)), errors='coerce')
spy_px2 = pd.to_numeric(df.get('SPY_Real_Price', pd.Series(np.nan, index=df.index)), errors='coerce')

if upro_px.notna().sum() > 252:
    upro_from_px = upro_px.pct_change()
    both_px = upro_real.notna() & upro_from_px.notna() & (~upro_syn)
    diff_px = (upro_real - upro_from_px)[both_px]
    print(f"  UPRO_Real_Ret vs UPRO_Real_Price.pct_change():  max|diff|={float(diff_px.abs().max()):.2e}  (expect ~0)")
    if spy_px2.notna().sum() > 252:
        spy_from_px2 = spy_px2.pct_change()
        both_spx = spy_ret.notna() & spy_from_px2.notna()
        diff_spx = (spy_ret - spy_from_px2)[both_spx]
        print(f"  SPY_Ret vs SPY_Real_Price.pct_change():         max|diff|={float(diff_spx.abs().max()):.2e}  (expect ~0)")

    # Check if SPY_Real_Price and UPRO_Real_Price share same non-NaN dates
    upro_px_dates = set(upro_px[upro_px.notna()].index)
    spy_px_dates  = set(spy_px2[spy_px2.notna()].index)
    shared = upro_px_dates & spy_px_dates
    only_upro = upro_px_dates - spy_px_dates
    only_spy  = spy_px_dates  - upro_px_dates
    print(f"\n  Price dates overlap: {len(shared):,} shared | {len(only_upro):,} UPRO-only | {len(only_spy):,} SPY-only")
    if only_upro:
        print(f"  UPRO-only dates (first 5): {sorted(only_upro)[:5]}")
    if only_spy:
        print(f"  SPY-only dates (first 3, last 3): {sorted(only_spy)[:3]} ... {sorted(only_spy)[-3:]}")
else:
    print("  UPRO_Real_Price not in cached df — cannot verify from prices directly")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("DIAGNOSTIC SUMMARY")
print("=" * 90)
mean_resid_bps = np.mean(residual) * 1e4
mean_resid_yr  = np.mean(residual) * 252 * 100
std_resid_te   = np.std(residual) * 252**0.5 * 100

print(f"""
  UPRO baseline residual  (no alpha, no spread):
    mean  = {mean_resid_bps:+.4f} bps/day  = {mean_resid_yr:+.4f}%/yr
    std   = {std_resid_te:.4f}%/yr  (tracking error)

  Interpretation:
    If mean ≈ 0%/yr   → SPY total-return is the correct driver; spread is a small financing cost
    If mean ≈ -5%/yr  → structural bias; likely return-basis mismatch (price-return vs total-return driver)
    If mean ≈ +X%/yr  → UPRO pays X%/yr in financing; driver + basis correct

  driver_alpha_daily in current model: {eng.ASSETS['UPRO'].get('expense_ratio',0.0)/252:.6f} bps/day
    (driver_alpha compensates for mean residual; is correct ONLY if residual is truly ~constant)
""")
