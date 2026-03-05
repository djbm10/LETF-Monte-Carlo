"""
TQQQ driver root-cause confirmation and correct-driver test.

Section 1: Confirm QQQ_Ret_RealOnly source (is it NASDAQ proxy or actual QQQ?)
Section 2: Test M0 with QQQ_Ret as driver — does residual shrink?
Section 3: Regime-intercept model using QQQ_Ret driver for TQQQ/QLD
Section 4: Cross-asset summary — all four assets, QQQ_Ret for TQQQ/QLD
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

TRAIN_END   = '2014-12-31'
VALID_START = '2015-01-01'
VALID_END   = '2024-12-31'
VOL_WINDOW  = 63
P_MID, P_HI = 60.0, 90.0
TRIM_FRAC   = 0.10

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

def gate(cagr_d, tw):
    return ("✓✓" if abs(cagr_d) < 0.5 and tw < 10.0 else
            "✓✗" if abs(cagr_d) < 0.5 else
            "✗✓" if tw < 10.0 else "✗✗")

def oos_metrics(act_v, pred_v):
    resid = act_v - pred_v
    W_a = float(np.prod(1.0 + act_v))
    W_p = float(np.prod(1.0 + pred_v))
    cd  = (_cagr(pred_v) - _cagr(act_v)) * 100.0
    tw  = abs(W_p / W_a - 1.0) * 100.0
    mr  = float(np.mean(resid)) * 252 * 100
    te  = float(np.std(resid)) * np.sqrt(252) * 100
    return dict(cd=cd, tw=tw, mr=mr, te=te)

# ─── Load data ─────────────────────────────────────────────────────────────────
print("=" * 90)
print("Loading data...")
df = eng.load_cache(eng.DATA_CACHE)
assert df is not None
df = eng._attach_real_only_driver_columns(df)
rf_s, rf_src = eng._load_daily_ff_rf_aligned(df)
rf_arr = np.resize(rf_s, len(df))
dates  = pd.DatetimeIndex(df.index)
print(f"  Rows: {len(df):,}   RF: {rf_src}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: Confirm QQQ_Ret_RealOnly source
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 1: QQQ_Ret_RealOnly source confirmation")
print("=" * 90)

src_col = 'QQQ_Ret_RealOnly_Source'
if src_col in df.columns:
    src_vals = df[src_col].dropna()
    print(f"  QQQ_Ret_RealOnly_Source unique values: {src_vals.unique().tolist()}")
    print(f"  QQQ_Ret_RealOnly_Source value_counts:\n{src_vals.value_counts().to_string()}")
else:
    print(f"  {src_col} not in df")

# Check NASDAQ_Ret_RealOnly presence
nas_related = [c for c in df.columns if 'NASDAQ' in c or 'HiTec' in c or 'ff_hitech' in c.lower()]
print(f"\n  NASDAQ/HiTec related columns: {nas_related}")

# Direct comparison: QQQ_Ret_RealOnly vs NASDAQ_Ret_RealOnly
if 'NASDAQ_Ret_RealOnly' in df.columns:
    a = pd.to_numeric(df['QQQ_Ret_RealOnly'],    errors='coerce')
    b = pd.to_numeric(df['NASDAQ_Ret_RealOnly'],  errors='coerce')
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) > 2:
        aa, bb = a[idx], b[idx]
        corr_nas = float(np.corrcoef(aa, bb)[0, 1])
        diff_nas = float((aa - bb).abs().max())
        print(f"\n  QQQ_Ret_RealOnly vs NASDAQ_Ret_RealOnly: N={len(idx):,}  "
              f"corr={corr_nas:.8f}  max|diff|={diff_nas:.2e}  "
              f"mean(A-B)/yr={float((aa-bb).mean())*252*100:+.4f}%")
        if diff_nas < 1e-10:
            print("  >>> CONFIRMED: QQQ_Ret_RealOnly IS IDENTICAL to NASDAQ_Ret_RealOnly <<<")
            print("  >>> The cache-healing fallback swapped the QQQ driver to the NASDAQ proxy! <<<")
        elif corr_nas > 0.999:
            print("  >>> Extremely high corr → QQQ_Ret_RealOnly ≈ NASDAQ_Ret_RealOnly <<<")

# Compare QQQ_Ret_RealOnly vs QQQ_Ret on the TQQQ REAL period only
tqqq_real_mask = df.get('TQQQ_IsSynthetic', pd.Series(True, index=df.index)).fillna(True) == False
print(f"\n  TQQQ real days (non-synthetic): {tqqq_real_mask.sum():,}")
if tqqq_real_mask.any():
    a_real  = pd.to_numeric(df.loc[tqqq_real_mask, 'QQQ_Ret_RealOnly'], errors='coerce')
    b_real  = pd.to_numeric(df.loc[tqqq_real_mask, 'QQQ_Ret'],          errors='coerce')
    c_real  = pd.to_numeric(df.loc[tqqq_real_mask, 'TQQQ_Real_Ret'],    errors='coerce') if 'TQQQ_Real_Ret' in df.columns else None

    idx_r = a_real.dropna().index.intersection(b_real.dropna().index)
    if len(idx_r) > 2:
        ar, br = a_real[idx_r], b_real[idx_r]
        corr_r = float(np.corrcoef(ar, br)[0, 1])
        print(f"  On TQQQ real days: corr(QQQ_Ret_RealOnly, QQQ_Ret)={corr_r:.6f}  "
              f"mean(A-B)/yr={float((ar-br).mean())*252*100:+.4f}%  N={len(idx_r):,}")
        if c_real is not None:
            # Correlations with TQQQ
            idx_t = idx_r.intersection(c_real.dropna().index)
            if len(idx_t) > 2:
                tqqq_r = c_real[idx_t]
                corr_tqqq_qqqonly = float(np.corrcoef(ar[idx_t], tqqq_r)[0, 1])
                corr_tqqq_qqq    = float(np.corrcoef(br[idx_t], tqqq_r)[0, 1])
                print(f"  corr(QQQ_Ret_RealOnly, TQQQ_Real_Ret) = {corr_tqqq_qqqonly:.6f}")
                print(f"  corr(QQQ_Ret,          TQQQ_Real_Ret) = {corr_tqqq_qqq:.6f}")
                print(f"  → Driver with higher corr to TQQQ: "
                      f"{'QQQ_Ret_RealOnly' if corr_tqqq_qqqonly > corr_tqqq_qqq else 'QQQ_Ret'}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: TQQQ M0 residual — QQQ_Ret driver vs QQQ_Ret_RealOnly driver
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 2: TQQQ M0 residual — compare two driver choices")
print("=" * 90)

lev = 3.0
fee_daily = float(eng.ASSETS['TQQQ']['expense_ratio']) / 252.0
real_col  = 'TQQQ_Real_Ret'
syn_col   = 'TQQQ_IsSynthetic'

is_syn  = df.get(syn_col, pd.Series(True, index=df.index)).fillna(True).astype(bool)
act_s   = pd.to_numeric(df[real_col], errors='coerce') if real_col in df.columns else pd.Series(np.nan, index=df.index)

for drv_col, drv_label in [
    ('QQQ_Ret_RealOnly', 'QQQ_Ret_RealOnly (current engine driver)'),
    ('QQQ_Ret',          'QQQ_Ret (actual QQQ total return)'),
]:
    if drv_col not in df.columns:
        print(f"\n  {drv_label}: column not in df, skipping")
        continue

    drv_s   = pd.to_numeric(df[drv_col], errors='coerce')
    real_ok = act_s.notna() & drv_s.notna() & (~is_syn)

    train_m = real_ok & (dates <= pd.Timestamp(TRAIN_END))
    valid_m = real_ok & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

    act_t = act_s[train_m].to_numpy(); drv_t = drv_s[train_m].to_numpy(); rf_t = rf_arr[train_m]
    act_v = act_s[valid_m].to_numpy(); drv_v = drv_s[valid_m].to_numpy(); rf_v = rf_arr[valid_m]

    resid_t = act_t - (lev * drv_t - (lev - 1.0) * rf_t - fee_daily)
    resid_v = act_v - (lev * drv_v - (lev - 1.0) * rf_v - fee_daily)

    print(f"\n  Driver: {drv_label}")
    print(f"    TRAIN N={len(act_t):,}  "
          f"mean_resid={np.mean(resid_t)*252*100:+.3f}%/yr  "
          f"tm10={trimmed_mean(resid_t)*252*100:+.3f}%/yr  "
          f"median={np.median(resid_t)*252*100:+.3f}%/yr  "
          f"std={np.std(resid_t)*np.sqrt(252)*100:.2f}%/yr")
    print(f"    VALID  N={len(act_v):,}  "
          f"mean_resid={np.mean(resid_v)*252*100:+.3f}%/yr  "
          f"tm10={trimmed_mean(resid_v)*252*100:+.3f}%/yr  "
          f"median={np.median(resid_v)*252*100:+.3f}%/yr  "
          f"std={np.std(resid_v)*np.sqrt(252)*100:.2f}%/yr")

    # TRAIN year-by-year residual
    yrs_t = pd.DatetimeIndex(df[train_m].index).year
    print(f"    TRAIN year-by-year:")
    for yr in sorted(set(yrs_t.tolist())):
        m = (yrs_t == yr)
        r = resid_t[m]
        print(f"      {yr}: N={int(m.sum()):>4}  mean={np.mean(r)*252*100:>+8.3f}%/yr  "
              f"median={np.median(r)*252*100:>+8.3f}%/yr")

    # OOS pure M0 metrics
    m0_pred_v = lev * drv_v - (lev - 1.0) * rf_v - fee_daily
    m = oos_metrics(act_v, m0_pred_v)
    print(f"    OOS M0: CAGR_diff={m['cd']:+.3f}%/yr  TW_rel_err={m['tw']:.2f}%  "
          f"mean_resid={m['mr']:+.3f}%/yr  TE={m['te']:.3f}%  {gate(m['cd'], m['tw'])}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: Regime-intercept model with QQQ_Ret driver for TQQQ / QLD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 3: Regime-intercept model with QQQ_Ret as TQQQ/QLD driver")
print("=" * 90)

# For TQQQ and QLD, test with QQQ_Ret instead of QQQ_Ret_RealOnly.
# UPRO and SSO continue to use SPY_Ret_RealOnly (which equals SPY_Ret).

ASSETS_CORRECTED = [
    ('UPRO', 3.0, 'SPY_Ret_RealOnly'),
    ('SSO',  2.0, 'SPY_Ret_RealOnly'),
    ('TQQQ', 3.0, 'QQQ_Ret'),
    ('QLD',  2.0, 'QQQ_Ret'),
]

results = {}

for asset, lev, forced_driver in ASSETS_CORRECTED:
    real_col = f'{asset}_Real_Ret'
    syn_col  = f'{asset}_IsSynthetic'
    fee_d    = float(eng.ASSETS.get(asset, {}).get('expense_ratio', 0.01)) / 252.0

    if real_col not in df.columns or forced_driver not in df.columns:
        print(f"\n  {asset}: SKIP (missing columns)")
        continue

    is_syn  = df.get(syn_col, pd.Series(True, index=df.index)).fillna(True).astype(bool)
    act_s   = pd.to_numeric(df[real_col],       errors='coerce')
    drv_s   = pd.to_numeric(df[forced_driver],  errors='coerce')
    real_ok = act_s.notna() & drv_s.notna() & (~is_syn)

    train_m = real_ok & (dates <= pd.Timestamp(TRAIN_END))
    valid_m = real_ok & (dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))

    if train_m.sum() < 252 or valid_m.sum() < 252:
        print(f"\n  {asset}: SKIP (insufficient data)")
        continue

    # Rolling vol + regime from driver
    drv_full = pd.to_numeric(df[forced_driver].fillna(np.nan), errors='coerce')
    rvol = drv_full.rolling(window=VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
    rvol_tr = rvol[train_m].dropna()
    thr_mid = float(np.percentile(rvol_tr, P_MID))
    thr_hi  = float(np.percentile(rvol_tr, P_HI))
    vol_a   = rvol.to_numpy()
    reg_a   = np.zeros(len(df), dtype=int)
    reg_a[vol_a >= thr_mid] = 1
    reg_a[vol_a >= thr_hi]  = 2
    reg_a[~np.isfinite(vol_a)] = 0

    act_t  = act_s[train_m].to_numpy()
    drv_t  = drv_s[train_m].to_numpy()
    rf_t   = rf_arr[train_m]
    reg_t  = reg_a[train_m]
    m0r_t  = act_t - (lev * drv_t - (lev - 1.0) * rf_t - fee_d)

    act_v  = act_s[valid_m].to_numpy()
    drv_v  = drv_s[valid_m].to_numpy()
    rf_v   = rf_arr[valid_m]
    reg_v  = reg_a[valid_m]
    m0_v   = lev * drv_v - (lev - 1.0) * rf_v - fee_d

    # Fit intercept_alpha per regime on TRAIN
    ia = {}
    RNAME = {0: 'low', 1: 'high', 2: 'crash'}
    print(f"\n  {'─'*70}")
    print(f"  {asset}  L={lev:.0f}x  driver={forced_driver}  "
          f"TRAIN_N={train_m.sum():,}  VALID_N={valid_m.sum():,}")
    print(f"  Vol thresholds (TRAIN): P60={thr_mid*100:.2f}%/yr  P90={thr_hi*100:.2f}%/yr")
    print(f"  Overall TRAIN M0 residual tm10%: {trimmed_mean(m0r_t)*252*100:+.3f}%/yr")
    print(f"  Per-regime intercept_alpha (TRAIN):")
    for r in [0, 1, 2]:
        mask_r = (reg_t == r)
        n_r    = int(mask_r.sum())
        if n_r < 5:
            ia[r] = trimmed_mean(m0r_t)
            print(f"    {RNAME[r]:<6}: N={n_r:>4}  → overall fallback  {ia[r]*252*100:+.3f}%/yr")
        else:
            ia[r] = trimmed_mean(m0r_t[mask_r])
            print(f"    {RNAME[r]:<6}: N={n_r:>4} ({100.*n_r/len(reg_t):.1f}%)  "
                  f"tm10={ia[r]*252*100:+.3f}%/yr  "
                  f"median={np.median(m0r_t[mask_r])*252*100:+.3f}%/yr  "
                  f"std={np.std(m0r_t[mask_r])*np.sqrt(252)*100:.2f}%/yr")

    # Apply OOS
    alpha_v = np.array([ia[r] for r in reg_v])
    mreg_v  = m0_v + alpha_v

    m_m0   = oos_metrics(act_v, m0_v)
    m_reg  = oos_metrics(act_v, mreg_v)

    # VALID regime distribution
    print(f"  VALID regime distribution:")
    for r in [0, 1, 2]:
        nr = int((reg_v == r).sum())
        print(f"    {RNAME[r]:<6}: {nr:>5} ({100.*nr/len(reg_v):.1f}%)  "
              f"applied_α={ia[r]*252*100:+.3f}%/yr")

    wtd = float(np.mean(alpha_v)) * 252 * 100
    print(f"  Weighted OOS alpha: {wtd:+.3f}%/yr")

    print(f"\n  OOS M0:            CAGR_diff={m_m0['cd']:+.3f}%/yr  "
          f"TW_rel_err={m_m0['tw']:.2f}%  mean_resid={m_m0['mr']:+.3f}%/yr  {gate(m_m0['cd'],m_m0['tw'])}")
    print(f"  OOS regime_intercept: CAGR_diff={m_reg['cd']:+.3f}%/yr  "
          f"TW_rel_err={m_reg['tw']:.2f}%  mean_resid={m_reg['mr']:+.3f}%/yr  {gate(m_reg['cd'],m_reg['tw'])}")

    # Year-by-year OOS
    yrs_v = pd.DatetimeIndex(df[valid_m].index).year
    print(f"  Year-by-year OOS residual (regime_intercept model):")
    print(f"  {'Year':>5}  {'N':>4}  {'M0_resid%/yr':>14}  {'Mreg_resid%/yr':>16}  {'actual CAGR':>13}  {'dom regime':>10}")
    for yr in sorted(set(yrs_v.tolist())):
        m_yr = (yrs_v == yr)
        a_yr = act_v[m_yr]; p0_yr = m0_v[m_yr]; pr_yr = mreg_v[m_yr]; r_yr = reg_v[m_yr]
        dom  = RNAME[int(stats.mode(r_yr, keepdims=False).mode)]
        r0   = float(np.mean(a_yr - p0_yr)) * 252 * 100
        rr   = float(np.mean(a_yr - pr_yr)) * 252 * 100
        print(f"  {yr:>5}  {int(m_yr.sum()):>4}  {r0:>+14.3f}%  {rr:>+16.3f}%  "
              f"{_cagr(a_yr)*100:>+11.2f}%  {dom:>10s}")

    results[asset] = {'M0': m_m0, 'Mreg': m_reg, 'ia': ia, 'driver': forced_driver}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: Final summary table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("SECTION 4: OOS SUMMARY TABLE — corrected drivers (QQQ_Ret for TQQQ/QLD)")
print("=" * 90)
print(f"  {'Asset':<6}  {'Driver':<22}  "
      f"{'M0 CAGR_d':>11}  {'M0 TW_err':>11}  {'M0':>4}  "
      f"{'Mreg CAGR_d':>13}  {'Mreg TW_err':>12}  {'Mreg':>6}")
print(f"  " + "─"*100)
for asset, _, forced_driver in ASSETS_CORRECTED:
    if asset not in results:
        continue
    r = results[asset]
    m0, mr = r['M0'], r['Mreg']
    print(f"  {asset:<6}  {forced_driver:<22}  "
          f"{m0['cd']:>+11.3f}%  {m0['tw']:>9.2f}%  {gate(m0['cd'],m0['tw']):>4}  "
          f"{mr['cd']:>+13.3f}%  {mr['tw']:>10.2f}%  {gate(mr['cd'],mr['tw']):>6}")

print(f"\n  Gate: |CAGR_diff| < 0.5%/yr  AND  TW_rel_err < 10%")

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)
