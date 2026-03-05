"""
Diagnostic: debug_daily_letf_decomposition for UPRO
Goal: identify which reconstruction component drives systematic OOS CAGR drift.

Prints full per-day breakdown for:
  - 6 calm dates  (low VIX, small SPY moves)
  - 6 high-vol dates (large moves, elevated VIX)

All within the OOS window 2015-01-01 -- 2024-12-31.
"""

import sys
import os
import numpy as np
import pandas as pd

# ── Engine import ──────────────────────────────────────────────────────────────
os.chdir('/home/djmann')
sys.path.insert(0, '/home/djmann')
import LETF34_analysis as eng

# ── Load cached df ─────────────────────────────────────────────────────────────
print("Loading cached historical data...")
df = eng.load_cache(eng.DATA_CACHE)
if df is None:
    raise RuntimeError("historical_data.pkl not found — run the engine first to populate cache.")
df = eng._attach_real_only_driver_columns(df)
print(f"  df loaded: {len(df):,} rows  {df.index[0].date()} → {df.index[-1].date()}")

# ── Load/calibrate tracking residual model ─────────────────────────────────────
print("Loading tracking residual model...")
trk_raw = eng.load_cache(eng.TRACKING_RESIDUAL_CACHE)
if trk_raw is not None and isinstance(trk_raw, dict) and trk_raw.get('_version') == eng.CACHE_MODEL_VERSION:
    tracking_model = trk_raw
    print(f"  Tracking model loaded from cache (version {eng.CACHE_MODEL_VERSION})")
else:
    print("  Cache miss or stale — calibrating tracking_residual_model (full df)...")
    # Calibrate on the train window only (pre-OOS).
    train_df = df.loc[df.index < pd.Timestamp('2015-01-01')].copy()
    rf_train, _ = eng._load_daily_ff_rf_aligned(train_df)
    funding_model = {}  # minimal; spread calibration doesn't need full funding model here
    tracking_model = eng.calibrate_tracking_residual_model(
        train_df, funding_model=funding_model, bypass_cache=True
    )
    print("  Calibration complete.")

# ── RF series for full df ──────────────────────────────────────────────────────
rf_daily, rf_src = eng._load_daily_ff_rf_aligned(df)
print(f"  RF source: {rf_src}")

# ── Report key calibrated parameters for UPRO ─────────────────────────────────
upro_params = (tracking_model or {}).get('UPRO', {})
print(f"\n=== UPRO calibrated parameters ===")
for k in ['driver_alpha_daily', 'underlying_scale', 'spread_mean_daily',
          'negative_spread_flag', 'calibrated_spread']:
    v = upro_params.get(k, 'N/A')
    if isinstance(v, float):
        print(f"  {k:35s} = {v:.6f}  ({v*252*100:.3f} %/yr)" if 'daily' in k or 'spread' in k else f"  {k:35s} = {v:.6f}")
    else:
        print(f"  {k:35s} = {v}")

# ── Driver column for UPRO ─────────────────────────────────────────────────────
driver_col = eng._underlying_return_column_for_asset('UPRO', df=df, prefer_real_only=True)
print(f"  driver_col = {driver_col}")

# ── Target dates: calm and high-vol, all within OOS window ─────────────────────
#
# Calm (low VIX, small SPY moves):
#   2017-01-19, 2017-07-18, 2019-04-12, 2021-03-12, 2023-01-27, 2024-06-03
#
# High-vol (large |return|, elevated VIX):
#   2020-03-16 (COVID -12%), 2020-03-23 (COVID trough), 2022-01-24 (Fed fear),
#   2022-06-13 (CPI shock), 2018-12-24 (Xmas crash), 2020-11-09 (vaccine rally +2.9%)
#
calm_dates = [
    '2017-01-19', '2017-07-18', '2019-04-12',
    '2021-03-12', '2023-01-27', '2024-06-03',
]
hv_dates = [
    '2020-03-16', '2020-03-23', '2022-01-24',
    '2022-06-13', '2018-12-24', '2020-11-09',
]

# ── Convert dates to integer indices in df ─────────────────────────────────────
def nearest_idx(df_idx: pd.DatetimeIndex, date_str: str) -> int:
    ts = pd.Timestamp(date_str)
    loc = df_idx.searchsorted(ts, side='left')
    loc = int(np.clip(loc, 0, len(df_idx) - 1))
    # Walk forward to nearest valid date (up to 5 business days)
    for off in range(5):
        i = int(np.clip(loc + off, 0, len(df_idx) - 1))
        if df_idx[i] >= ts:
            return i
    return loc

df_idx = pd.DatetimeIndex(df.index)

calm_indices = [nearest_idx(df_idx, d) for d in calm_dates]
hv_indices   = [nearest_idx(df_idx, d) for d in hv_dates]

print(f"\n  Calm indices  : {[df.index[i].date() for i in calm_indices]}")
print(f"  High-vol indices: {[df.index[i].date() for i in hv_indices]}")

# ── Show actual VIX + UPRO return for context ──────────────────────────────────
print("\n  Context check (VIX, UPRO actual) for selected dates:")
for label, idxlist in [("CALM", calm_indices), ("HIGH-VOL", hv_indices)]:
    print(f"  --- {label} ---")
    for i in idxlist:
        date_str = str(df.index[i].date())
        vix_v = df['VIX'].iloc[i] if 'VIX' in df.columns else np.nan
        spy_r = df['SPY_Ret'].iloc[i] if 'SPY_Ret' in df.columns else np.nan
        upro_r = df['UPRO_Real_Ret'].iloc[i] if 'UPRO_Real_Ret' in df.columns else np.nan
        dr = df[driver_col].iloc[i] if driver_col in df.columns else np.nan
        print(f"    [{i:5d}] {date_str}  VIX={vix_v:5.1f}  SPY={spy_r*100:+.3f}%  "
              f"UPRO_actual={upro_r*100:+.4f}%  driver={dr*100:+.4f}%")

# ── Run decomposition: CALM dates ──────────────────────────────────────────────
print(f"\n{'='*100}")
print("DECOMPOSITION — CALM PERIODS")
print(f"{'='*100}")
eng.debug_daily_letf_decomposition(
    df=df,
    asset='UPRO',
    tracking_model=tracking_model,
    rf_daily=rf_daily,
    date_idx_list=calm_indices,
)

# ── Run decomposition: HIGH-VOL dates ─────────────────────────────────────────
print(f"\n{'='*100}")
print("DECOMPOSITION — HIGH-VOLATILITY PERIODS")
print(f"{'='*100}")
eng.debug_daily_letf_decomposition(
    df=df,
    asset='UPRO',
    tracking_model=tracking_model,
    rf_daily=rf_daily,
    date_idx_list=hv_indices,
)

# ── Aggregate error stats over the full OOS window ────────────────────────────
print(f"\n{'='*100}")
print("AGGREGATE ERROR STATS — OOS WINDOW (2015-01-01 to 2024-12-31)")
print(f"{'='*100}")

oos_mask = (df.index >= pd.Timestamp('2015-01-01')) & (df.index <= pd.Timestamp('2024-12-31'))
oos_df = df.loc[oos_mask].copy()

# Rebuild reconstructed series for OOS window only
recon = eng._build_reconstructed_letf_series(
    df=oos_df,
    asset='UPRO',
    tracking_model=tracking_model,
    rf_daily=rf_daily[oos_mask],
    seed=12345,
    use_replay_on_real_dates=False,
    prefer_real_only_driver=True,
    stochastic_shocks=False,
)

if recon is not None:
    real_mask = np.asarray(recon['real_mask'], dtype=bool)
    actual     = np.asarray(recon['actual'],       dtype=np.float64)[real_mask]
    modeled    = np.asarray(recon['reconstructed'], dtype=np.float64)[real_mask]
    spread_d   = np.asarray(recon['spread_daily'],  dtype=np.float64)[real_mask]
    resid_d    = np.asarray(recon['residual_daily'],dtype=np.float64)[real_mask]

    # Terminal wealth
    W_real = float(np.prod(1.0 + actual))
    W_syn  = float(np.prod(1.0 + modeled))
    n_days = len(actual)
    cagr_real = W_real ** (252.0 / n_days) - 1.0
    cagr_syn  = W_syn  ** (252.0 / n_days) - 1.0

    # Decompose per-day mean contributions
    upro_model_e = (upro_params or {})
    lev = float(eng.ASSETS['UPRO']['leverage'])
    exp_d = float(eng.ASSETS['UPRO']['expense_ratio']) / 252.0
    alpha_d = float(upro_model_e.get('driver_alpha_daily', 0.0))
    scale   = float(upro_model_e.get('underlying_scale', 1.0))

    rf_oos = rf_daily[oos_mask]
    if len(rf_oos) != len(oos_df):
        rf_oos = np.resize(rf_oos, len(oos_df))

    driver_col_vals = np.asarray(oos_df[driver_col].values, dtype=np.float64)
    actual_vals     = np.asarray(oos_df['UPRO_Real_Ret'].values, dtype=np.float64)

    finite_m = real_mask & np.isfinite(driver_col_vals) & np.isfinite(actual_vals)
    idx_sub  = driver_col_vals[finite_m] * scale
    rf_sub   = rf_oos[finite_m]
    act_sub  = actual_vals[finite_m]

    borrow_sub   = (lev - 1.0) * rf_sub
    implied_spread_raw = lev * idx_sub - act_sub - exp_d - borrow_sub
    implied_spread_adj = implied_spread_raw + alpha_d  # what model uses as spread

    print(f"\n  N real days in OOS window : {int(np.sum(real_mask)):,}")
    print(f"  UPRO actual   CAGR        : {cagr_real*100:+.3f}%/yr")
    print(f"  UPRO modeled  CAGR        : {cagr_syn*100:+.3f}%/yr")
    print(f"  CAGR error (model−actual) : {(cagr_syn-cagr_real)*100:+.3f}%/yr")
    print(f"  Terminal wealth rel err   : {abs(W_syn/W_real-1)*100:.2f}%")
    print(f"\n  --- Per-day mean component contributions (bps/day) ---")
    print(f"  L×idx_eff mean            : {np.nanmean(lev*idx_sub)*1e4:+.3f} bps/day")
    print(f"  borrow_cost mean          : {np.nanmean(borrow_sub)*1e4:+.3f} bps/day")
    print(f"  expense mean              : {exp_d*1e4:+.3f} bps/day")
    print(f"  driver_alpha_daily        : {alpha_d*1e4:+.3f} bps/day  ({alpha_d*252*100:+.3f}%/yr)")
    print(f"  implied_spread_raw mean   : {np.nanmean(implied_spread_raw)*1e4:+.3f} bps/day  ({np.nanmean(implied_spread_raw)*252*100:+.3f}%/yr)")
    print(f"  implied_spread_adj mean   : {np.nanmean(implied_spread_adj)*1e4:+.3f} bps/day  ({np.nanmean(implied_spread_adj)*252*100:+.3f}%/yr)")
    print(f"  residual mean             : {np.nanmean(resid_d)*1e4:+.3f} bps/day  ({np.nanmean(resid_d)*252*100:+.3f}%/yr)")
    print(f"  modeled return mean       : {np.nanmean(modeled)*1e4:+.3f} bps/day  ({np.nanmean(modeled)*252*100:+.3f}%/yr)")
    print(f"  actual  return mean       : {np.nanmean(actual)*1e4:+.3f} bps/day  ({np.nanmean(actual)*252*100:+.3f}%/yr)")
    print(f"  error mean (mod−act)      : {np.nanmean(modeled-actual)*1e4:+.3f} bps/day  ({np.nanmean(modeled-actual)*252*100:+.3f}%/yr)")

    # Show how driver differs from actual SPY_Ret (to check for price-vs-total return)
    if 'SPY_Ret' in oos_df.columns:
        spy_sub = np.asarray(oos_df['SPY_Ret'].values, dtype=np.float64)[finite_m]
        driver_raw_sub = driver_col_vals[finite_m]
        diff_mean = np.nanmean(driver_raw_sub - spy_sub) * 252 * 100
        print(f"\n  --- Driver vs SPY_Ret alignment check ---")
        print(f"  driver_col                : {driver_col}")
        print(f"  mean(driver - SPY_Ret)/yr : {diff_mean:+.4f}%/yr  (should ≈0 if driver IS SPY_Ret)")
        print(f"  corr(driver, SPY_Ret)     : {np.corrcoef(driver_raw_sub[np.isfinite(spy_sub)], spy_sub[np.isfinite(spy_sub)])[0,1]:.6f}")

    # Implied spread OOS vs training period
    if 'SPY_Ret' in df.columns:
        train_mask = df.index < pd.Timestamp('2015-01-01')
        train_df2 = df.loc[train_mask].copy()
        rf_train2, _ = eng._load_daily_ff_rf_aligned(train_df2)
        drv_t = np.asarray(train_df2[driver_col].values, dtype=np.float64) * scale
        act_t = np.asarray(train_df2['UPRO_Real_Ret'].values, dtype=np.float64) if 'UPRO_Real_Ret' in train_df2.columns else np.full(len(train_df2), np.nan)
        rft   = np.resize(rf_train2, len(train_df2))
        upro_syn_t = train_df2.get('UPRO_IsSynthetic', pd.Series(True, index=train_df2.index))
        real_t = np.isfinite(act_t) & np.isfinite(drv_t) & (~upro_syn_t.fillna(True).to_numpy(dtype=bool))
        if real_t.any():
            imp_t = lev * drv_t[real_t] - act_t[real_t] - exp_d - (lev-1)*rft[real_t]
            print(f"\n  --- Train vs OOS implied spread comparison ---")
            print(f"  Train implied_spread_raw mean : {np.nanmean(imp_t)*1e4:+.3f} bps/day  ({np.nanmean(imp_t)*252*100:+.3f}%/yr)")
            print(f"  OOS   implied_spread_raw mean : {np.nanmean(implied_spread_raw)*1e4:+.3f} bps/day  ({np.nanmean(implied_spread_raw)*252*100:+.3f}%/yr)")
            print(f"  Delta (OOS − train)           : {(np.nanmean(implied_spread_raw)-np.nanmean(imp_t))*252*100:+.3f}%/yr  ← CAGR drift source if nonzero")
else:
    print("  [ERROR] _build_reconstructed_letf_series returned None for OOS window")

print(f"\n{'='*100}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*100}")
