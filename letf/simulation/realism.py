"""
Forward Scenario Realism Validation Layer
==========================================

Assesses whether synthetic forward paths (from block bootstrap or parametric
engine) are statistically plausible relative to known historical market behavior.

Does NOT validate historical backtests — those have separate reconstruction
integrity checks.  Only validates FORWARD-LOOKING synthetic paths to guard
against distributional drift in the path generator.

Key metrics assessed (5 components × 20 pts each = 100 pts max):
  1. vol_clustering   – lag-1 autocorrelation of |daily returns|
  2. high_vol_frac    – fraction of days with VIX > 25 or |ret| > 2.5σ
  3. time_in_dd       – fraction of days the cumulative path is below its ATH
  4. tail_frequency   – fraction of days with |return| > 3 × rolling sigma
  5. vol_persistence  – AR(1) of 21-day rolling realized volatility

Scoring:
  Each component contributes proportionally to its weight (max 100 total).
  Component score = weight × 100 × clamp(1 − |error| / (3 × tolerance), 0, 1)
  (Linear penalty: within tolerance band → full score; 3× tolerance → 0 pts)

Aggregate pass thresholds for STRICT mode:
  median_score >= 70  AND  p10_score >= 50

ScenarioRealismMode:
  STRICT    -- Block unrealistic paths  (historical_bootstrap + neutral)
  WARN_ONLY -- Warn but do not block    (pessimistic / optimistic / custom)
  SKIP      -- No validation            (historical_backtest)
"""

from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
class ScenarioRealismMode(Enum):
    """Controls how aggressively the realism validator enforces plausibility."""
    STRICT    = auto()   # Block path if aggregate fails thresholds
    WARN_ONLY = auto()   # Warn but do not block
    SKIP      = auto()   # No validation applied


# ─────────────────────────────────────────────────────────────────────────────
# Historical baselines (SPY / broad-market, 1950-2025)
# These robust statistics should be matched by any realistic forward path.
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_BASELINES: Dict[str, float] = {
    'abs_ret_ac1':    0.19,   # lag-1 autocorr of |daily returns| (vol clustering)
    'high_vol_frac':  0.22,   # ~22% of days VIX > 25 or |ret| > 2.5σ
    'time_in_dd_pct': 0.62,   # ~62% of days below prior ATH (SPY, long-run)
    'tail_freq_3s':   0.040,  # ~4% of days with |return| > 3 × rolling sigma
    'vol_ac1':        0.87,   # AR(1) of 21-day rolling realized vol
}

# Tolerance bands: |observed − baseline| < tolerance → no penalty
REALISM_TOLERANCES: Dict[str, float] = {
    'abs_ret_ac1':    0.07,   # ±0.07 around 0.19
    'high_vol_frac':  0.08,   # ±8 pp around 22%
    'time_in_dd_pct': 0.12,   # ±12 pp around 62%
    'tail_freq_3s':   0.020,  # ±2 pp around 4%
    'vol_ac1':        0.10,   # ±0.10 around 0.87
}

# Component weights (must sum to 1.0)
COMPONENT_WEIGHTS: Dict[str, float] = {
    'vol_clustering':  0.20,   # uses abs_ret_ac1
    'high_vol_frac':   0.20,
    'time_in_dd':      0.20,
    'tail_frequency':  0.25,   # slightly higher: most problematic in practice
    'vol_persistence': 0.15,
}

# Aggregate score thresholds for STRICT mode
STRICT_MEDIAN_THRESHOLD: float = 70.0
STRICT_P10_THRESHOLD:    float = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_realized_vol(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Compute rolling realized volatility (annualized) with min_periods=5.

    Uses a plain rolling window to avoid pandas dependency here.
    NaN-fills the initial entries with the first valid value.

    Args:
        returns: Daily return series
        window:  Rolling window in days (default 21 ≈ 1 month)

    Returns:
        Array of annualised realized volatility, same shape as returns.
    """
    n = len(returns)
    rvol = np.full(n, np.nan)
    for i in range(window, n + 1):
        segment = returns[max(0, i - window):i]
        rvol[i - 1] = float(np.std(segment)) * np.sqrt(252.0)

    # Forward-fill NaN prefix with first valid value
    first_valid = np.where(np.isfinite(rvol))[0]
    if len(first_valid) > 0:
        rvol[:first_valid[0]] = rvol[first_valid[0]]
    return rvol


# ─────────────────────────────────────────────────────────────────────────────
# Core validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_forward_scenario_realism(
    spy_returns:    np.ndarray,
    vix_series:     np.ndarray,
    sigma_lookback: int = 21,
) -> Dict[str, float]:
    """
    Assess statistical realism of a single synthetic forward path.

    Computes five raw metrics and their signed deviations from historical
    baselines.  Does NOT compute the composite score — call
    compute_realism_score() for that.

    Args:
        spy_returns:    Daily SPY (or equivalent) return series, shape (n,).
        vix_series:     Daily VIX level series, shape (n,).
        sigma_lookback: Window for rolling sigma used in tail detection (days).

    Returns:
        Dict with raw metric values and signed error vs baseline:
          'high_vol_frac'        – observed fraction of high-vol days
          'time_in_dd_pct'       – observed time-in-drawdown fraction
          'abs_ret_ac1'          – lag-1 autocorr of |returns|
          'tail_freq_3s'         – fraction of days with |ret| > 3σ
          'vol_ac1'              – AR(1) of rolling 21-day realized vol
          '<metric>_err'         – signed deviation from baseline (positive = above)
    """
    r = np.asarray(spy_returns, dtype=float)
    v = np.asarray(vix_series,  dtype=float)
    n = len(r)

    result: Dict[str, float] = {}

    # ── 1. Rolling daily sigma for tail-detection ─────────────────────────
    rvol_daily = _rolling_realized_vol(r, sigma_lookback) / np.sqrt(252.0)
    global_sigma = float(np.nanstd(r)) if n > 1 else 0.01
    rvol_daily = np.where(np.isfinite(rvol_daily) & (rvol_daily > 0),
                          rvol_daily, global_sigma)

    # ── 2. High-vol fraction ─────────────────────────────────────────────
    # Day is "high vol" if VIX > 25 OR |return| > 2.5 × rolling daily sigma.
    hi_vix_flag = v > 25.0
    hi_ret_flag = np.abs(r) > (2.5 * rvol_daily)
    high_vol_frac = float(np.mean(hi_vix_flag | hi_ret_flag))
    result['high_vol_frac']     = high_vol_frac
    result['high_vol_frac_err'] = high_vol_frac - HISTORICAL_BASELINES['high_vol_frac']

    # ── 3. Time in drawdown ───────────────────────────────────────────────
    cum_ret = np.cumprod(1.0 + np.clip(r, -0.999, 10.0))
    ath     = np.maximum.accumulate(cum_ret)
    in_dd   = cum_ret < (ath * (1.0 - 1e-6))
    time_in_dd = float(np.mean(in_dd))
    result['time_in_dd_pct']  = time_in_dd
    result['time_in_dd_err']  = time_in_dd - HISTORICAL_BASELINES['time_in_dd_pct']

    # ── 4. Volatility clustering (lag-1 autocorr of |returns|) ───────────
    abs_r = np.abs(r)
    if n > 20:
        ac1 = float(np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1])
        if not np.isfinite(ac1):
            ac1 = 0.0
    else:
        ac1 = 0.0
    result['abs_ret_ac1']        = ac1
    result['vol_clustering_err'] = ac1 - HISTORICAL_BASELINES['abs_ret_ac1']

    # ── 5. Tail event frequency (|ret| > 3σ) ─────────────────────────────
    tail_threshold = 3.0 * rvol_daily
    tail_freq = float(np.sum(np.abs(r) > tail_threshold) / max(n, 1))
    result['tail_freq_3s']  = tail_freq
    result['tail_freq_err'] = tail_freq - HISTORICAL_BASELINES['tail_freq_3s']

    # ── 6. Volatility persistence (AR(1) of rolling 21-day realized vol) ─
    rvol_ann  = _rolling_realized_vol(r, sigma_lookback)
    rvol_ann  = rvol_ann[np.isfinite(rvol_ann)]
    if len(rvol_ann) > 20:
        vol_ac1 = float(np.corrcoef(rvol_ann[:-1], rvol_ann[1:])[0, 1])
        if not np.isfinite(vol_ac1):
            vol_ac1 = 0.0
    else:
        vol_ac1 = 0.0
    result['vol_ac1']            = vol_ac1
    result['vol_persistence_err'] = vol_ac1 - HISTORICAL_BASELINES['vol_ac1']

    return result


# ─────────────────────────────────────────────────────────────────────────────

def compute_realism_score(validation_result: Dict[str, float]) -> float:
    """
    Convert a per-path validation dict to a composite score in [0, 100].

    Scoring formula per component:
      component_score = weight × 100 × clamp(1 − |error| / (3 × tolerance), 0, 1)

    Linear penalty: error = 0 → full score; error = 3 × tolerance → 0 pts.

    Args:
        validation_result: Dict from validate_forward_scenario_realism().

    Returns:
        float in [0, 100].  Higher is more realistic.
    """
    # Map each component to its error key and tolerance
    component_specs = [
        ('vol_clustering',  'vol_clustering_err',  'abs_ret_ac1'),
        ('high_vol_frac',   'high_vol_frac_err',   'high_vol_frac'),
        ('time_in_dd',      'time_in_dd_err',      'time_in_dd_pct'),
        ('tail_frequency',  'tail_freq_err',        'tail_freq_3s'),
        ('vol_persistence', 'vol_persistence_err', 'vol_ac1'),
    ]

    score = 0.0
    for component, err_key, tol_key in component_specs:
        err = float(validation_result.get(err_key, 0.0))
        if not np.isfinite(err):
            err = 0.0
        tol   = REALISM_TOLERANCES[tol_key]
        frac  = max(0.0, 1.0 - abs(err) / (3.0 * tol))
        score += COMPONENT_WEIGHTS[component] * 100.0 * frac

    return float(np.clip(score, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────────────

def aggregate_forward_realism_across_paths(
    validation_results: List[Dict[str, float]],
    scores:             Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Aggregate per-path realism results across multiple simulation paths.

    Args:
        validation_results: List of dicts from validate_forward_scenario_realism().
        scores:             Pre-computed per-path scores (computed if not provided).

    Returns:
        Dict with median / p10 / p90 for the composite score and each raw metric.
        Keys include 'median_score', 'p10_score', 'p90_score', 'n_paths',
        and '<metric>_median', '<metric>_p10', '<metric>_p90' for all metrics.
    """
    if not validation_results:
        return {}

    if scores is None:
        scores = [compute_realism_score(r) for r in validation_results]

    scores_arr = np.array(scores, dtype=float)
    agg: Dict[str, float] = {
        'median_score': float(np.nanmedian(scores_arr)),
        'p10_score':    float(np.nanpercentile(scores_arr, 10)),
        'p90_score':    float(np.nanpercentile(scores_arr, 90)),
        'n_paths':      float(len(validation_results)),
    }

    # Per-metric aggregation
    metric_keys = [
        'high_vol_frac', 'time_in_dd_pct', 'abs_ret_ac1',
        'tail_freq_3s',  'vol_ac1',
        'high_vol_frac_err',  'time_in_dd_err',  'vol_clustering_err',
        'tail_freq_err',      'vol_persistence_err',
    ]
    for key in metric_keys:
        vals = np.array([r.get(key, np.nan) for r in validation_results], dtype=float)
        agg[f'{key}_median'] = float(np.nanmedian(vals))
        agg[f'{key}_p10']    = float(np.nanpercentile(vals, 10))
        agg[f'{key}_p90']    = float(np.nanpercentile(vals, 90))

    return agg


# ─────────────────────────────────────────────────────────────────────────────

def check_aggregate_realism(
    agg:  Dict[str, float],
    mode: ScenarioRealismMode,
) -> Dict[str, object]:
    """
    Apply scenario-aware gating logic to aggregate realism stats.

    Args:
        agg:  Output from aggregate_forward_realism_across_paths().
        mode: ScenarioRealismMode controlling enforcement level.

    Returns:
        Dict with 'should_block' (bool), 'mode', 'reason', and key scores.
    """
    median_score = float(agg.get('median_score', 100.0))
    p10_score    = float(agg.get('p10_score',    100.0))

    out: Dict[str, object] = {
        'mode':         mode.name,
        'median_score': median_score,
        'p10_score':    p10_score,
        'should_block': False,
        'reason':       '',
    }

    if mode == ScenarioRealismMode.SKIP:
        out['reason'] = 'SKIP mode: no realism check applied'
        return out

    fails_median = median_score < STRICT_MEDIAN_THRESHOLD
    fails_p10    = p10_score    < STRICT_P10_THRESHOLD
    failed       = fails_median or fails_p10

    if not failed:
        out['reason'] = 'PASS'
        return out

    reasons = []
    if fails_median:
        reasons.append(
            f'median_score={median_score:.1f} < threshold={STRICT_MEDIAN_THRESHOLD}'
        )
    if fails_p10:
        reasons.append(
            f'p10_score={p10_score:.1f} < threshold={STRICT_P10_THRESHOLD}'
        )
    out['reason'] = '; '.join(reasons)

    if mode == ScenarioRealismMode.STRICT:
        out['should_block'] = True
    # WARN_ONLY: should_block stays False

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic printing
# ─────────────────────────────────────────────────────────────────────────────

def print_realism_score_attribution(
    agg:       Dict[str, float],
    verbosity: int = 1,
    title:     str = 'FORWARD REALISM SCORE',
) -> None:
    """
    Print a diagnostic breakdown of realism score loss by component.

    This is the Part A diagnostic required by the work plan.  It reads
    directly from the aggregate result dict keys without touching
    compute_realism_score() itself.

    Args:
        agg:       Output from aggregate_forward_realism_across_paths().
        verbosity: 0 = silent, 1 = composite summary, 2 = per-component table.
        title:     Header text.
    """
    if verbosity <= 0:
        return

    sep = '─' * 64
    print(f'\n{sep}')
    print(f'  {title}')
    print(sep)
    print(
        f"  Composite score:  "
        f"median={agg.get('median_score', float('nan')):5.1f}  "
        f"p10={agg.get('p10_score', float('nan')):5.1f}  "
        f"p90={agg.get('p90_score', float('nan')):5.1f}  "
        f"n={int(agg.get('n_paths', 0))}"
    )

    if verbosity < 2:
        print(sep)
        return

    print(f'\n  Component scores (median observed vs historical baseline):')
    print(f"  {'Component':<20s}  {'Observed':>9s}  {'Baseline':>9s}  "
          f"{'Error':>9s}  {'Pts lost':>9s}")
    print(f"  {'-'*58}")

    component_specs = [
        ('Vol clustering',  'abs_ret_ac1_median',       'vol_clustering_err_median',  'abs_ret_ac1',    COMPONENT_WEIGHTS['vol_clustering']),
        ('High vol frac',   'high_vol_frac_median',      'high_vol_frac_err_median',   'high_vol_frac',  COMPONENT_WEIGHTS['high_vol_frac']),
        ('Time in DD',      'time_in_dd_pct_median',     'time_in_dd_err_median',      'time_in_dd_pct', COMPONENT_WEIGHTS['time_in_dd']),
        ('Tail freq (3σ)',  'tail_freq_3s_median',       'tail_freq_err_median',       'tail_freq_3s',   COMPONENT_WEIGHTS['tail_frequency']),
        ('Vol persistence', 'vol_ac1_median',             'vol_persistence_err_median', 'vol_ac1',        COMPONENT_WEIGHTS['vol_persistence']),
    ]

    total_max  = 0.0
    total_lost = 0.0
    for label, val_key, err_key, tol_key, w in component_specs:
        val      = float(agg.get(val_key, float('nan')))
        err      = float(agg.get(err_key, float('nan')))
        baseline = HISTORICAL_BASELINES[tol_key]
        tol      = REALISM_TOLERANCES[tol_key]
        max_pts  = w * 100.0
        frac     = max(0.0, 1.0 - abs(err) / (3.0 * tol)) if np.isfinite(err) else 0.0
        pts_got  = max_pts * frac
        pts_lost = max_pts - pts_got
        sign     = '+' if (np.isfinite(err) and err >= 0) else ''
        print(
            f"  {label:<20s}  {val:>9.4f}  {baseline:>9.4f}  "
            f"{sign}{err:>8.4f}  {pts_lost:>9.2f}"
        )
        total_max  += max_pts
        total_lost += pts_lost

    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<20s}  {'':>9s}  {'':>9s}  {'':>9s}  {total_lost:>9.2f}")
    print(sep)


def print_before_after_realism_summary(
    before_agg: Dict[str, float],
    after_agg:  Dict[str, float],
) -> None:
    """
    Print a BEFORE vs AFTER table for the key realism metrics.

    Deliverable D from the work plan.  Called from main() when
    cfg.SHOW_BOOTSTRAP_BEFORE_AFTER is True.

    Args:
        before_agg: Aggregate results with BEFORE (no crisis damping).
        after_agg:  Aggregate results with AFTER (crisis damping active).
    """
    sep = '=' * 72
    print(f'\n{sep}')
    print('  BOOTSTRAP REALISM: BEFORE vs AFTER CRISIS-CHAIN FIX')
    print(sep)
    print(
        f"  {'Metric':<32s}  {'Before':>9s}  {'After':>9s}  {'Delta':>10s}"
    )
    print(f"  {'-'*64}")

    metrics = [
        ('Realism score  (median)',     'median_score'),
        ('Realism score  (p10)',        'p10_score'),
        ('Time in DD     (median)',      'time_in_dd_pct_median'),
        ('Tail freq err  (median)',      'tail_freq_err_median'),
        ('High vol err   (median)',      'high_vol_frac_err_median'),
        ('Vol clustering (median)',      'abs_ret_ac1_median'),
        ('Vol persistence (median)',     'vol_ac1_median'),
    ]

    for label, key in metrics:
        b    = float(before_agg.get(key, float('nan')))
        a    = float(after_agg.get(key,  float('nan')))
        d    = a - b
        sign = '+' if d >= 0 else ''
        print(f"  {label:<32s}  {b:>9.3f}  {a:>9.3f}  {sign}{d:>9.3f}")

    print(sep)
