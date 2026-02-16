"""
Historical validation module for LETF simulation.

Compares Monte Carlo simulation results against actual historical LETF performance,
calculates rolling CAGRs, distribution overlap, percentile correlations, and
provides comprehensive quality assessments.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from letf import config as cfg
from letf.utils import infer_regime_from_vix
from letf.strategy import run_strategy_fixed


# ============================================================================
# VALIDATE MONTE CARLO VS HISTORICAL
# ============================================================================

def validate_monte_carlo_vs_historical(df, mc_results, time_horizon):
    """
    Validate Monte Carlo against historical LETF performance.

    WARNING: Only validates REAL data (post-inception).
    Pre-inception data is SYNTHETIC and cannot be validated.
    """
    from letf.calibration import get_default_correlations_time_varying

    print(f"\n{'='*80}")
    print(f"VALIDATING MONTE CARLO VS HISTORICAL DATA ({time_horizon}Y)")
    print(f"{'='*80}\n")

    validation_results = {}

    years_available = len(df) / 252

    if years_available < time_horizon:
        print(f"  Only {years_available:.2f} years available, need {time_horizon}")
        print(f"  Skipping validation for {time_horizon}Y horizon")
        return validation_results

    lookback_days = int(time_horizon * 252)

    for asset in ['TQQQ', 'SPY', 'SSO']:
        price_col = f'{asset}_Price'
        synthetic_col = f'{asset}_IsSynthetic'

        if price_col not in df.columns:
            continue

        # Only validate REAL data
        if synthetic_col in df.columns:
            real_data = df[~df[synthetic_col]]

            if len(real_data) < lookback_days:
                print(f"  {asset}: Insufficient REAL data ({len(real_data)/252:.2f} years)")
                continue

            df_validate = real_data
        else:
            df_validate = df

        if len(df_validate) >= lookback_days:
            historical_prices = df_validate[price_col].iloc[-lookback_days:]
            historical_return = historical_prices.iloc[-1] / historical_prices.iloc[0]

            strategy_map = {'TQQQ': 'S1', 'SPY': 'S2', 'SSO': 'S3'}
            sid = strategy_map.get(asset)

            if sid and sid in mc_results:
                # FIX: Don't pass regime_path from simulation - let it infer from historical VIX
                # We need correlation matrices for potential transaction cost calculations
                # though benchmark strategies ignore correlations.
                # Assuming empty or default correlation matrix if needed.
                dummy_correlations = get_default_correlations_time_varying()

                # We do NOT run strategy on historical data here because historical returns
                # are already baked into the price column. We just compare the final multiple.
                # BUT, if we were running a dynamic strategy (like SMA), we would need to run it.

                # Wait, this function compares *simulation distribution* vs *single historical scalar*.
                # The historical scalar is already computed above: `historical_return`.
                # We don't need to run_strategy_fixed here for benchmarks.

                # However, if we wanted to validate a complex strategy (like SMA), we WOULD need to run it
                # on historical data. Let's make sure that's possible.

                # Example: Validating SMA Strategy (S3) on history
                if asset == 'TQQQ':
                    sid_sma = 'S3'
                    if sid_sma in mc_results:
                         # Here we MUST run the strategy on historical data to get the historical return
                         # And this is where the BUG would manifest if we passed a short regime_path
                         equity_curve_hist, _ = run_strategy_fixed(
                             df_validate,
                             sid_sma,
                             regime_path=None,  # FIX: Let it infer from historical VIX
                             correlation_matrices=dummy_correlations,
                             apply_costs=True
                         )
                         historical_return_sma = equity_curve_hist.iloc[-1] / cfg.INITIAL_CAPITAL
                         # (Then compare this against MC distribution for S3)

                # Standard validation logic for Buy & Hold
                sim_results = mc_results[sid]
                sim_wealth = np.array([r['Final_Wealth'] for r in sim_results
                                      if r.get('Final_Wealth', 0) > 0])

                if len(sim_wealth) > 0:
                    sim_median = np.median(sim_wealth) / cfg.INITIAL_CAPITAL
                    sim_p10 = np.percentile(sim_wealth, 10) / cfg.INITIAL_CAPITAL
                    sim_p90 = np.percentile(sim_wealth, 90) / cfg.INITIAL_CAPITAL

                    in_range = sim_p10 <= historical_return <= sim_p90

                    deviation_pct = abs(historical_return - sim_median) / historical_return * 100

                    validation_results[asset] = {
                        'historical_multiple': historical_return,
                        'simulated_median': sim_median,
                        'simulated_p10': sim_p10,
                        'simulated_p90': sim_p90,
                        'in_range': in_range,
                        'deviation_pct': deviation_pct
                    }

                    print(f"  {asset:5s} (REAL DATA ONLY):")
                    print(f"    Historical:  {historical_return:.2f}x "
                          f"({((historical_return)**(1/time_horizon)-1)*100:+.2f}% CAGR)")
                    print(f"    Simulated:   {sim_median:.2f}x (median)")
                    print(f"    Range:       [{sim_p10:.2f}x, {sim_p90:.2f}x] (10th-90th %ile)")
                    print(f"    Deviation:   {deviation_pct:.2f}%")
                    print(f"    Status:      {'IN RANGE' if in_range else 'OUT OF RANGE'}")
                    print()

    if len(validation_results) > 0:
        in_range_count = sum(1 for v in validation_results.values() if v['in_range'])
        total_count = len(validation_results)

        print(f"  Validation Summary: {in_range_count}/{total_count} assets within simulated range")

        if in_range_count == total_count:
            print(f"  VALIDATION PASSED: Monte Carlo matches historical reality")
        elif in_range_count >= total_count * 0.7:
            print(f"  VALIDATION PARTIAL: Most assets match, review outliers")
        else:
            print(f"  VALIDATION FAILED: Monte Carlo diverges from reality")

    print(f"{'='*80}")

    return validation_results


# After the existing validate_monte_carlo_vs_historical function...

def calculate_historical_rolling_cagrs(df: pd.DataFrame, asset: str,
                                        years: int,
                                        step_days: int = 21) -> Dict:
    """
    Calculate all rolling N-year CAGRs from historical data.

    This gives us the ACTUAL distribution of historical returns, which we can
    compare against our simulated distribution.

    Args:
        df: Historical DataFrame with price columns
        asset: Asset name (e.g., 'SPY', 'TQQQ', 'SSO')
        years: Rolling window in years (e.g., 10 for 10-year CAGR)
        step_days: Days between each calculation (21 = monthly, 1 = daily)

    Returns:
        Dict with:
        - 'cagrs': List of all rolling CAGRs
        - 'start_dates': Corresponding start dates
        - 'end_dates': Corresponding end dates
        - 'percentiles': Dict of percentile values
        - 'stats': Basic statistics

    Example:
        For 10-year horizon with 20 years of data:
        - Start at day 0, end at day 2520 -> first 10-year CAGR
        - Start at day 21, end at day 2541 -> second 10-year CAGR
        - ... continue until we run out of data
    """
    price_col = f'{asset}_Price'
    synthetic_col = f'{asset}_IsSynthetic'

    if price_col not in df.columns:
        print(f"  Warning: {asset}: Price column not found")
        return None

    # Use only REAL data if synthetic column exists
    if synthetic_col in df.columns:
        df_real = df[~df[synthetic_col]].copy()
        data_type = "REAL"
    else:
        df_real = df.copy()
        data_type = "ALL"

    window_days = int(years * 252)

    if len(df_real) < window_days:
        print(f"  Warning: {asset}: Insufficient {data_type} data "
              f"({len(df_real)/252:.2f}Y available, {years}Y needed)")
        return None

    # Calculate rolling CAGRs
    cagrs = []
    start_dates = []
    end_dates = []

    prices = df_real[price_col].values
    dates = df_real.index

    for start_idx in range(0, len(df_real) - window_days + 1, step_days):
        end_idx = start_idx + window_days - 1

        start_price = prices[start_idx]
        end_price = prices[end_idx]

        if start_price > 0 and end_price > 0:
            # Calculate CAGR
            total_return = end_price / start_price
            cagr = total_return ** (1/years) - 1

            cagrs.append(cagr)
            start_dates.append(dates[start_idx])
            end_dates.append(dates[end_idx])

    if len(cagrs) == 0:
        print(f"  Warning: {asset}: No valid rolling periods found")
        return None

    cagrs = np.array(cagrs)

    # Calculate percentiles
    percentiles = {
        'p5': np.percentile(cagrs, 5),
        'p10': np.percentile(cagrs, 10),
        'p25': np.percentile(cagrs, 25),
        'p50': np.percentile(cagrs, 50),
        'p75': np.percentile(cagrs, 75),
        'p90': np.percentile(cagrs, 90),
        'p95': np.percentile(cagrs, 95)
    }

    # Basic stats
    stats_dict = {
        'mean': np.mean(cagrs),
        'median': np.median(cagrs),
        'std': np.std(cagrs),
        'min': np.min(cagrs),
        'max': np.max(cagrs),
        'count': len(cagrs),
        'data_type': data_type
    }

    return {
        'cagrs': cagrs,
        'start_dates': start_dates,
        'end_dates': end_dates,
        'percentiles': percentiles,
        'stats': stats_dict,
        'years': years,
        'asset': asset
    }


def find_percentile_rank(value: float, distribution: np.ndarray) -> float:
    """
    Find what percentile a value would be in a distribution.

    Args:
        value: The value to rank
        distribution: Array of values defining the distribution

    Returns:
        Percentile rank (0-100)

    Example:
        If value is greater than 75% of distribution, returns ~75
    """
    if len(distribution) == 0:
        return 50.0

    # Count how many values are less than or equal to the given value
    rank = np.sum(distribution <= value) / len(distribution) * 100

    return rank


def calculate_distribution_overlap(dist1: np.ndarray, dist2: np.ndarray,
                                   n_bins: int = 50) -> float:
    """
    Calculate the overlap between two distributions.

    Uses histogram intersection - a simple and intuitive measure.
    100% = identical distributions, 0% = no overlap.

    Args:
        dist1: First distribution (e.g., historical CAGRs)
        dist2: Second distribution (e.g., simulated CAGRs)
        n_bins: Number of bins for histogram

    Returns:
        Overlap percentage (0-100)
    """
    # Determine common range
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))

    # Create histograms with same bins
    bins = np.linspace(min_val, max_val, n_bins + 1)

    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    # Normalize to sum to 1
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

    # Calculate intersection (minimum at each bin)
    overlap = np.sum(np.minimum(hist1, hist2)) * 100

    return overlap


def calculate_percentile_correlation(hist_percentiles: Dict,
                                     sim_percentiles: Dict) -> Tuple[float, float]:
    """
    Calculate correlation between historical and simulated percentile curves.

    This measures whether the SHAPE of the distributions match, not just
    the absolute values.

    Args:
        hist_percentiles: Dict with p5, p10, p25, p50, p75, p90, p95 from historical
        sim_percentiles: Dict with same keys from simulation

    Returns:
        Tuple of (pearson_correlation, spearman_correlation)
    """
    from scipy.stats import pearsonr, spearmanr

    # Extract percentile values in order
    pct_keys = ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']

    hist_values = [hist_percentiles.get(k, 0) for k in pct_keys]
    sim_values = [sim_percentiles.get(k, 0) for k in pct_keys]

    # Calculate correlations
    if len(set(hist_values)) > 1 and len(set(sim_values)) > 1:
        pearson_r, _ = pearsonr(hist_values, sim_values)
        spearman_r, _ = spearmanr(hist_values, sim_values)
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    return pearson_r, spearman_r


def compare_simulated_vs_historical(df: pd.DataFrame, mc_results: Dict,
                                    time_horizon: int) -> Dict:
    """
    Comprehensive comparison of simulated vs historical returns.

    This is the main function that:
    1. Calculates rolling historical CAGRs
    2. Extracts simulated CAGRs from Monte Carlo results
    3. Compares the distributions
    4. Ranks historical performance in simulation (and vice versa)
    5. Calculates correlation metrics

    Args:
        df: Historical DataFrame
        mc_results: Monte Carlo results dict
        time_horizon: Simulation horizon in years

    Returns:
        Dict with comparison results for each asset
    """
    print(f"\n{'='*100}")
    print(f"HISTORICAL vs SIMULATED COMPARISON ({time_horizon}-YEAR HORIZON)")
    print(f"{'='*100}")
    print("\nThis compares your Monte Carlo simulations against actual historical rolling returns.")
    print("A good simulation should produce similar distributions to what actually happened.\n")

    comparison_results = {}

    # Map strategies to assets
    strategy_to_asset = {
        'S1': 'TQQQ',
        'S2': 'SPY',
        'S3': 'SSO'
    }

    for sid, asset in strategy_to_asset.items():
        print(f"\n{'-'*80}")
        print(f"{asset} BUY & HOLD (Strategy {sid})")
        print(f"{'-'*80}")

        # ====================================================================
        # STEP 1: Get Historical Rolling CAGRs
        # ====================================================================
        historical = calculate_historical_rolling_cagrs(
            df, asset, time_horizon, step_days=21  # Monthly rolling windows
        )

        if historical is None:
            print(f"  Skipping {asset} - insufficient historical data")
            continue

        hist_cagrs = historical['cagrs']
        hist_percentiles = historical['percentiles']
        hist_stats = historical['stats']

        print(f"\n  HISTORICAL DATA ({hist_stats['data_type']} only):")
        print(f"     Rolling {time_horizon}-year periods: {hist_stats['count']}")
        print(f"     CAGR Range: {hist_stats['min']*100:+.2f}% to {hist_stats['max']*100:+.2f}%")
        print(f"     CAGR Median: {hist_stats['median']*100:+.2f}%")
        print(f"     CAGR Std Dev: {hist_stats['std']*100:.2f}%")

        # ====================================================================
        # STEP 2: Get Simulated CAGRs
        # ====================================================================
        if sid not in mc_results or not mc_results[sid]:
            print(f"  No simulation results for {sid}")
            continue

        sim_results = mc_results[sid]
        sim_wealth = np.array([r['Final_Wealth'] for r in sim_results
                               if r.get('Final_Wealth', 0) > 0])

        if len(sim_wealth) == 0:
            print(f"  No valid simulation results for {sid}")
            continue

        # Convert wealth to CAGRs
        sim_cagrs = (sim_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1

        sim_percentiles = {
            'p5': np.percentile(sim_cagrs, 5),
            'p10': np.percentile(sim_cagrs, 10),
            'p25': np.percentile(sim_cagrs, 25),
            'p50': np.percentile(sim_cagrs, 50),
            'p75': np.percentile(sim_cagrs, 75),
            'p90': np.percentile(sim_cagrs, 90),
            'p95': np.percentile(sim_cagrs, 95)
        }

        sim_stats = {
            'mean': np.mean(sim_cagrs),
            'median': np.median(sim_cagrs),
            'std': np.std(sim_cagrs),
            'min': np.min(sim_cagrs),
            'max': np.max(sim_cagrs),
            'count': len(sim_cagrs)
        }

        print(f"\n  SIMULATED DATA:")
        print(f"     Monte Carlo simulations: {sim_stats['count']}")
        print(f"     CAGR Range: {sim_stats['min']*100:+.2f}% to {sim_stats['max']*100:+.2f}%")
        print(f"     CAGR Median: {sim_stats['median']*100:+.2f}%")
        print(f"     CAGR Std Dev: {sim_stats['std']*100:.2f}%")

        # ====================================================================
        # STEP 3: Calculate Percentile Rankings
        # ====================================================================

        # Where does historical median rank in simulation?
        hist_median_in_sim = find_percentile_rank(hist_stats['median'], sim_cagrs)

        # Where does simulated median rank in history?
        sim_median_in_hist = find_percentile_rank(sim_stats['median'], hist_cagrs)

        print(f"\n  PERCENTILE RANKINGS:")
        print(f"     Historical median ({hist_stats['median']*100:+.2f}%) would be P{hist_median_in_sim:.0f} in simulation")
        print(f"     Simulated median ({sim_stats['median']*100:+.2f}%) would be P{sim_median_in_hist:.0f} in history")

        # ====================================================================
        # STEP 4: Calculate Distribution Metrics
        # ====================================================================

        # Overlap between distributions
        overlap = calculate_distribution_overlap(hist_cagrs, sim_cagrs)

        # Correlation between percentile curves
        pearson_r, spearman_r = calculate_percentile_correlation(
            hist_percentiles, sim_percentiles
        )

        # Median difference
        median_diff = sim_stats['median'] - hist_stats['median']
        median_diff_pct = median_diff / abs(hist_stats['median']) * 100 if hist_stats['median'] != 0 else 0

        print(f"\n  DISTRIBUTION COMPARISON:")
        print(f"     Distribution Overlap: {overlap:.2f}%")
        print(f"     Percentile Correlation (Pearson): {pearson_r:.3f}")
        print(f"     Percentile Correlation (Spearman): {spearman_r:.3f}")
        print(f"     Median Difference: {median_diff*100:+.2f}% ({median_diff_pct:+.2f}% relative)")

        # ====================================================================
        # STEP 5: Print Percentile Comparison Table
        # ====================================================================

        print(f"\n  PERCENTILE COMPARISON TABLE:")
        print(f"     {'Percentile':<12} {'Historical':>12} {'Simulated':>12} {'Difference':>12}")
        print(f"     {'-'*48}")

        for pct_name in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            hist_val = hist_percentiles[pct_name] * 100
            sim_val = sim_percentiles[pct_name] * 100
            diff = sim_val - hist_val
            print(f"     {pct_name.upper():<12} {hist_val:>+11.2f}% {sim_val:>+11.2f}% {diff:>+11.2f}%")

        # ====================================================================
        # STEP 6: Quality Assessment
        # ====================================================================

        # Determine quality of simulation match
        quality_score = 0
        quality_notes = []

        # Check 1: Is historical median within P25-P75 of simulation?
        if 25 <= hist_median_in_sim <= 75:
            quality_score += 25
            quality_notes.append("OK: Historical median within simulation IQR")
        else:
            quality_notes.append("WARN: Historical median outside simulation IQR")

        # Check 2: Distribution overlap > 50%?
        if overlap > 50:
            quality_score += 25
            quality_notes.append(f"OK: Good distribution overlap ({overlap:.0f}%)")
        elif overlap > 25:
            quality_score += 10
            quality_notes.append(f"WARN: Moderate distribution overlap ({overlap:.0f}%)")
        else:
            quality_notes.append(f"BAD: Poor distribution overlap ({overlap:.0f}%)")

        # Check 3: Percentile correlation > 0.9?
        if spearman_r > 0.9:
            quality_score += 25
            quality_notes.append(f"OK: Strong percentile correlation ({spearman_r:.2f})")
        elif spearman_r > 0.7:
            quality_score += 15
            quality_notes.append(f"WARN: Moderate percentile correlation ({spearman_r:.2f})")
        else:
            quality_notes.append(f"BAD: Weak percentile correlation ({spearman_r:.2f})")

        # Check 4: Median difference < 3%?
        if abs(median_diff) < 0.03:
            quality_score += 25
            quality_notes.append(f"OK: Small median difference ({median_diff*100:+.2f}%)")
        elif abs(median_diff) < 0.05:
            quality_score += 15
            quality_notes.append(f"WARN: Moderate median difference ({median_diff*100:+.2f}%)")
        else:
            quality_notes.append(f"BAD: Large median difference ({median_diff*100:+.2f}%)")

        print(f"\n  SIMULATION QUALITY ASSESSMENT:")
        for note in quality_notes:
            print(f"     {note}")
        print(f"\n     Overall Score: {quality_score}/100", end="")

        if quality_score >= 80:
            print(" - EXCELLENT match to history")
        elif quality_score >= 60:
            print(" - GOOD match to history")
        elif quality_score >= 40:
            print(" - FAIR match to history (review assumptions)")
        else:
            print(" - POOR match to history (simulation may be unreliable)")

        # Store results
        comparison_results[asset] = {
            'historical': {
                'cagrs': hist_cagrs,
                'percentiles': hist_percentiles,
                'stats': hist_stats
            },
            'simulated': {
                'cagrs': sim_cagrs,
                'percentiles': sim_percentiles,
                'stats': sim_stats
            },
            'comparison': {
                'hist_median_percentile_in_sim': hist_median_in_sim,
                'sim_median_percentile_in_hist': sim_median_in_hist,
                'distribution_overlap': overlap,
                'pearson_correlation': pearson_r,
                'spearman_correlation': spearman_r,
                'median_difference': median_diff,
                'quality_score': quality_score
            }
        }

    # ========================================================================
    # OVERALL SUMMARY
    # ========================================================================

    print(f"\n{'='*100}")
    print("OVERALL SIMULATION VALIDATION SUMMARY")
    print(f"{'='*100}")

    if comparison_results:
        avg_quality = np.mean([r['comparison']['quality_score']
                              for r in comparison_results.values()])
        avg_overlap = np.mean([r['comparison']['distribution_overlap']
                              for r in comparison_results.values()])
        avg_corr = np.mean([r['comparison']['spearman_correlation']
                           for r in comparison_results.values()])

        print(f"\n  Average Quality Score: {avg_quality:.0f}/100")
        print(f"  Average Distribution Overlap: {avg_overlap:.2f}%")
        print(f"  Average Percentile Correlation: {avg_corr:.3f}")

        if avg_quality >= 70:
            print(f"\n  SIMULATION VALIDATED: Monte Carlo matches historical patterns well")
        elif avg_quality >= 50:
            print(f"\n  SIMULATION PARTIALLY VALIDATED: Some discrepancies with history")
        else:
            print(f"\n  SIMULATION CONCERNS: Significant differences from historical patterns")

    print(f"\n{'='*100}\n")

    return comparison_results


def calculate_rolling_cagrs_all_data(df: pd.DataFrame, asset: str,
                                      years: int,
                                      step_days: int = 21) -> Dict:
    """
    Calculate all rolling N-year CAGRs using ALL data (historical + synthetic).

    Unlike calculate_historical_rolling_cagrs() which only uses REAL post-inception
    data, this function uses EVERYTHING including synthetic reconstructed data
    back to 1926.

    This tests: "If this LETF had existed since 1926 with current expense ratios,
    borrowing costs, and tracking error, what would the return distribution look like?"

    Args:
        df: DataFrame with price columns (including synthetic data)
        asset: Asset name (e.g., 'SPY', 'TQQQ', 'SSO')
        years: Rolling window in years (e.g., 10 for 10-year CAGR)
        step_days: Days between each calculation (21 = monthly, 1 = daily)

    Returns:
        Dict with:
        - 'cagrs': Array of all rolling CAGRs
        - 'start_dates': Corresponding start dates
        - 'end_dates': Corresponding end dates
        - 'percentiles': Dict of percentile values
        - 'stats': Basic statistics
        - 'synthetic_count': Number of periods that include synthetic data
        - 'real_count': Number of periods that are purely real data
    """
    price_col = f'{asset}_Price'
    synthetic_col = f'{asset}_IsSynthetic'

    if price_col not in df.columns:
        print(f"  Warning: {asset}: Price column not found")
        return None

    # Use ALL data (don't filter out synthetic)
    df_all = df.copy()

    window_days = int(years * 252)

    if len(df_all) < window_days:
        print(f"  Warning: {asset}: Insufficient data "
              f"({len(df_all)/252:.2f}Y available, {years}Y needed)")
        return None

    # Calculate rolling CAGRs
    cagrs = []
    start_dates = []
    end_dates = []
    includes_synthetic = []  # Track which periods include synthetic data

    prices = df_all[price_col].values
    dates = df_all.index

    # Check if synthetic column exists for tracking
    has_synthetic_col = synthetic_col in df_all.columns
    if has_synthetic_col:
        is_synthetic = df_all[synthetic_col].values

    for start_idx in range(0, len(df_all) - window_days + 1, step_days):
        end_idx = start_idx + window_days - 1

        start_price = prices[start_idx]
        end_price = prices[end_idx]

        if start_price > 0 and end_price > 0 and not np.isnan(start_price) and not np.isnan(end_price):
            # Calculate CAGR
            total_return = end_price / start_price
            cagr = total_return ** (1/years) - 1

            cagrs.append(cagr)
            start_dates.append(dates[start_idx])
            end_dates.append(dates[end_idx])

            # Track if this period includes any synthetic data
            if has_synthetic_col:
                period_has_synthetic = np.any(is_synthetic[start_idx:end_idx+1])
                includes_synthetic.append(period_has_synthetic)
            else:
                includes_synthetic.append(False)

    if len(cagrs) == 0:
        print(f"  Warning: {asset}: No valid rolling periods found")
        return None

    cagrs = np.array(cagrs)
    includes_synthetic = np.array(includes_synthetic)

    # Calculate percentiles
    percentiles = {
        'p5': np.percentile(cagrs, 5),
        'p10': np.percentile(cagrs, 10),
        'p25': np.percentile(cagrs, 25),
        'p50': np.percentile(cagrs, 50),
        'p75': np.percentile(cagrs, 75),
        'p90': np.percentile(cagrs, 90),
        'p95': np.percentile(cagrs, 95)
    }

    # Count synthetic vs real periods
    synthetic_count = np.sum(includes_synthetic)
    real_count = len(cagrs) - synthetic_count

    # Basic stats
    stats_dict = {
        'mean': np.mean(cagrs),
        'median': np.median(cagrs),
        'std': np.std(cagrs),
        'min': np.min(cagrs),
        'max': np.max(cagrs),
        'count': len(cagrs),
        'data_type': 'ALL (Real + Synthetic)',
        'synthetic_count': synthetic_count,
        'real_count': real_count,
        'synthetic_pct': (synthetic_count / len(cagrs) * 100) if len(cagrs) > 0 else 0
    }

    # Find date range
    if len(start_dates) > 0:
        stats_dict['earliest_start'] = min(start_dates)
        stats_dict['latest_end'] = max(end_dates)

    return {
        'cagrs': cagrs,
        'start_dates': start_dates,
        'end_dates': end_dates,
        'percentiles': percentiles,
        'stats': stats_dict,
        'years': years,
        'asset': asset,
        'includes_synthetic': includes_synthetic
    }


def compare_simulated_vs_synthetic_historical(df: pd.DataFrame, mc_results: Dict,
                                               time_horizon: int) -> Dict:
    """
    Compare simulated returns vs ALL historical data (including synthetic).

    This is the SECOND comparison that uses:
    - Real post-inception data (e.g., TQQQ since 2010)
    - PLUS synthetic reconstructed data (e.g., "what if TQQQ existed since 1926")

    The synthetic data uses current expense ratios, borrowing costs, and tracking
    error applied to historical underlying returns.

    This gives us ~100 years of "what if" data to compare against simulations,
    including extreme events like the Great Depression.

    Args:
        df: Historical DataFrame (with both real and synthetic data)
        mc_results: Monte Carlo results dict
        time_horizon: Simulation horizon in years

    Returns:
        Dict with comparison results for each asset
    """
    print(f"\n{'='*100}")
    print(f"SYNTHETIC + HISTORICAL vs SIMULATED COMPARISON ({time_horizon}-YEAR HORIZON)")
    print(f"{'='*100}")
    print("\nThis compares Monte Carlo simulations against ALL data (real + synthetic reconstruction).")
    print("Synthetic data assumes CURRENT expense ratios, borrowing costs, and tracking error.")
    print("This includes extreme events like the Great Depression (1929-1932).\n")

    comparison_results = {}

    # Map strategies to assets
    strategy_to_asset = {
        'S1': 'TQQQ',
        'S2': 'SPY',
        'S3': 'SSO'
    }

    # Get asset configurations for display
    asset_configs = {
        'TQQQ': cfg.ASSETS.get('TQQQ', {}),
        'SSO': cfg.ASSETS.get('SSO', {}),
        'SPY': cfg.ASSETS.get('SPY', {})
    }

    for sid, asset in strategy_to_asset.items():
        print(f"\n{'-'*80}")
        print(f"{asset} BUY & HOLD (Strategy {sid}) - ALL DATA")
        print(f"{'-'*80}")

        # Show the parameters used for synthetic data
        config = asset_configs.get(asset, {})
        if config:
            print(f"\n  SYNTHETIC DATA PARAMETERS (applied to all pre-inception data):")
            print(f"     Leverage: {config.get('leverage', 'N/A')}x")
            print(f"     Expense Ratio: {config.get('expense_ratio', 0)*100:.2f}%")
            print(f"     Borrow Spread: {config.get('borrow_spread', 0)*100:.2f}%")
            print(f"     Tracking Error Base: {config.get('tracking_error_base', 0)*10000:.2f} bps")
            print(f"     Inception Date: {config.get('inception', 'N/A')}")

        # ====================================================================
        # STEP 1: Get Rolling CAGRs from ALL Data (Real + Synthetic)
        # ====================================================================
        all_data = calculate_rolling_cagrs_all_data(
            df, asset, time_horizon, step_days=21  # Monthly rolling windows
        )

        if all_data is None:
            print(f"  Skipping {asset} - insufficient data")
            continue

        all_cagrs = all_data['cagrs']
        all_percentiles = all_data['percentiles']
        all_stats = all_data['stats']

        print(f"\n  ALL DATA (Real + Synthetic):")
        print(f"     Rolling {time_horizon}-year periods: {all_stats['count']}")
        print(f"     - Periods with synthetic data: {all_stats['synthetic_count']} ({all_stats['synthetic_pct']:.2f}%)")
        print(f"     - Periods with real data only: {all_stats['real_count']}")
        if 'earliest_start' in all_stats:
            print(f"     Date range: {all_stats['earliest_start'].strftime('%Y-%m-%d') if hasattr(all_stats['earliest_start'], 'strftime') else all_stats['earliest_start']} to {all_stats['latest_end'].strftime('%Y-%m-%d') if hasattr(all_stats['latest_end'], 'strftime') else all_stats['latest_end']}")
        print(f"     CAGR Range: {all_stats['min']*100:+.2f}% to {all_stats['max']*100:+.2f}%")
        print(f"     CAGR Median: {all_stats['median']*100:+.2f}%")
        print(f"     CAGR Std Dev: {all_stats['std']*100:.2f}%")

        # ====================================================================
        # STEP 2: Get Simulated CAGRs
        # ====================================================================
        if sid not in mc_results or not mc_results[sid]:
            print(f"  No simulation results for {sid}")
            continue

        sim_results = mc_results[sid]
        sim_wealth = np.array([r['Final_Wealth'] for r in sim_results
                               if r.get('Final_Wealth', 0) > 0])

        if len(sim_wealth) == 0:
            print(f"  No valid simulation results for {sid}")
            continue

        # Convert wealth to CAGRs
        sim_cagrs = (sim_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1

        sim_percentiles = {
            'p5': np.percentile(sim_cagrs, 5),
            'p10': np.percentile(sim_cagrs, 10),
            'p25': np.percentile(sim_cagrs, 25),
            'p50': np.percentile(sim_cagrs, 50),
            'p75': np.percentile(sim_cagrs, 75),
            'p90': np.percentile(sim_cagrs, 90),
            'p95': np.percentile(sim_cagrs, 95)
        }

        sim_stats = {
            'mean': np.mean(sim_cagrs),
            'median': np.median(sim_cagrs),
            'std': np.std(sim_cagrs),
            'min': np.min(sim_cagrs),
            'max': np.max(sim_cagrs),
            'count': len(sim_cagrs)
        }

        print(f"\n  SIMULATED DATA:")
        print(f"     Monte Carlo simulations: {sim_stats['count']}")
        print(f"     CAGR Range: {sim_stats['min']*100:+.2f}% to {sim_stats['max']*100:+.2f}%")
        print(f"     CAGR Median: {sim_stats['median']*100:+.2f}%")
        print(f"     CAGR Std Dev: {sim_stats['std']*100:.2f}%")

        # ====================================================================
        # STEP 3: Calculate Percentile Rankings
        # ====================================================================

        # Where does historical+synthetic median rank in simulation?
        all_median_in_sim = find_percentile_rank(all_stats['median'], sim_cagrs)

        # Where does simulated median rank in historical+synthetic?
        sim_median_in_all = find_percentile_rank(sim_stats['median'], all_cagrs)

        print(f"\n  PERCENTILE RANKINGS:")
        print(f"     Historical+Synthetic median ({all_stats['median']*100:+.2f}%) would be P{all_median_in_sim:.0f} in simulation")
        print(f"     Simulated median ({sim_stats['median']*100:+.2f}%) would be P{sim_median_in_all:.0f} in historical+synthetic")

        # ====================================================================
        # STEP 4: Calculate Distribution Metrics
        # ====================================================================

        # Overlap between distributions
        overlap = calculate_distribution_overlap(all_cagrs, sim_cagrs)

        # Correlation between percentile curves
        pearson_r, spearman_r = calculate_percentile_correlation(
            all_percentiles, sim_percentiles
        )

        # Median difference
        median_diff = sim_stats['median'] - all_stats['median']
        median_diff_pct = median_diff / abs(all_stats['median']) * 100 if all_stats['median'] != 0 else 0

        print(f"\n  DISTRIBUTION COMPARISON:")
        print(f"     Distribution Overlap: {overlap:.2f}%")
        print(f"     Percentile Correlation (Pearson): {pearson_r:.3f}")
        print(f"     Percentile Correlation (Spearman): {spearman_r:.3f}")
        print(f"     Median Difference: {median_diff*100:+.2f}% ({median_diff_pct:+.2f}% relative)")

        # ====================================================================
        # STEP 5: Print Percentile Comparison Table
        # ====================================================================

        print(f"\n  PERCENTILE COMPARISON TABLE:")
        print(f"     {'Percentile':<12} {'Hist+Synth':>12} {'Simulated':>12} {'Difference':>12}")
        print(f"     {'-'*48}")

        for pct_name in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            all_val = all_percentiles[pct_name] * 100
            sim_val = sim_percentiles[pct_name] * 100
            diff = sim_val - all_val
            print(f"     {pct_name.upper():<12} {all_val:>+11.2f}% {sim_val:>+11.2f}% {diff:>+11.2f}%")

        # ====================================================================
        # STEP 6: Quality Assessment
        # ====================================================================

        quality_score = 0
        quality_notes = []

        # Check 1: Is historical+synthetic median within P25-P75 of simulation?
        if 25 <= all_median_in_sim <= 75:
            quality_score += 25
            quality_notes.append("OK: Hist+Synth median within simulation IQR")
        else:
            quality_notes.append("WARN: Hist+Synth median outside simulation IQR")

        # Check 2: Distribution overlap > 50%?
        if overlap > 50:
            quality_score += 25
            quality_notes.append(f"OK: Good distribution overlap ({overlap:.0f}%)")
        elif overlap > 25:
            quality_score += 10
            quality_notes.append(f"WARN: Moderate distribution overlap ({overlap:.0f}%)")
        else:
            quality_notes.append(f"BAD: Poor distribution overlap ({overlap:.0f}%)")

        # Check 3: Percentile correlation > 0.9?
        if spearman_r > 0.9:
            quality_score += 25
            quality_notes.append(f"OK: Strong percentile correlation ({spearman_r:.2f})")
        elif spearman_r > 0.7:
            quality_score += 15
            quality_notes.append(f"WARN: Moderate percentile correlation ({spearman_r:.2f})")
        else:
            quality_notes.append(f"BAD: Weak percentile correlation ({spearman_r:.2f})")

        # Check 4: Median difference < 5% (more lenient for synthetic data)?
        if abs(median_diff) < 0.05:
            quality_score += 25
            quality_notes.append(f"OK: Small median difference ({median_diff*100:+.2f}%)")
        elif abs(median_diff) < 0.08:
            quality_score += 15
            quality_notes.append(f"WARN: Moderate median difference ({median_diff*100:+.2f}%)")
        else:
            quality_notes.append(f"BAD: Large median difference ({median_diff*100:+.2f}%)")

        print(f"\n  SIMULATION QUALITY ASSESSMENT (vs Synthetic+Historical):")
        for note in quality_notes:
            print(f"     {note}")
        print(f"\n     Overall Score: {quality_score}/100", end="")

        if quality_score >= 80:
            print(" - EXCELLENT match to synthetic history")
        elif quality_score >= 60:
            print(" - GOOD match to synthetic history")
        elif quality_score >= 40:
            print(" - FAIR match to synthetic history (review assumptions)")
        else:
            print(" - POOR match to synthetic history (simulation may need calibration)")

        # ====================================================================
        # STEP 7: Extreme Event Analysis
        # ====================================================================

        # Find worst and best periods in the synthetic+historical data
        worst_idx = np.argmin(all_cagrs)
        best_idx = np.argmax(all_cagrs)

        print(f"\n  EXTREME EVENTS IN SYNTHETIC+HISTORICAL DATA:")
        print(f"     Worst {time_horizon}Y period: {all_cagrs[worst_idx]*100:+.2f}% CAGR", end="")
        if len(all_data['start_dates']) > worst_idx:
            start_dt = all_data['start_dates'][worst_idx]
            if hasattr(start_dt, 'strftime'):
                print(f" (starting {start_dt.strftime('%Y-%m-%d')})", end="")
        print()

        print(f"     Best {time_horizon}Y period:  {all_cagrs[best_idx]*100:+.2f}% CAGR", end="")
        if len(all_data['start_dates']) > best_idx:
            start_dt = all_data['start_dates'][best_idx]
            if hasattr(start_dt, 'strftime'):
                print(f" (starting {start_dt.strftime('%Y-%m-%d')})", end="")
        print()

        # What percentile would these extremes be in simulation?
        worst_in_sim = find_percentile_rank(all_cagrs[worst_idx], sim_cagrs)
        best_in_sim = find_percentile_rank(all_cagrs[best_idx], sim_cagrs)

        print(f"     Worst period would be P{worst_in_sim:.0f} in simulation")
        print(f"     Best period would be P{best_in_sim:.0f} in simulation")

        # Store results
        comparison_results[asset] = {
            'historical_synthetic': {
                'cagrs': all_cagrs,
                'percentiles': all_percentiles,
                'stats': all_stats
            },
            'simulated': {
                'cagrs': sim_cagrs,
                'percentiles': sim_percentiles,
                'stats': sim_stats
            },
            'comparison': {
                'all_median_percentile_in_sim': all_median_in_sim,
                'sim_median_percentile_in_all': sim_median_in_all,
                'distribution_overlap': overlap,
                'pearson_correlation': pearson_r,
                'spearman_correlation': spearman_r,
                'median_difference': median_diff,
                'quality_score': quality_score
            },
            'extremes': {
                'worst_cagr': all_cagrs[worst_idx],
                'best_cagr': all_cagrs[best_idx],
                'worst_percentile_in_sim': worst_in_sim,
                'best_percentile_in_sim': best_in_sim
            }
        }

    # ========================================================================
    # OVERALL SUMMARY
    # ========================================================================

    print(f"\n{'='*100}")
    print("OVERALL VALIDATION SUMMARY (Synthetic + Historical)")
    print(f"{'='*100}")

    if comparison_results:
        avg_quality = np.mean([r['comparison']['quality_score']
                              for r in comparison_results.values()])
        avg_overlap = np.mean([r['comparison']['distribution_overlap']
                              for r in comparison_results.values()])
        avg_corr = np.mean([r['comparison']['spearman_correlation']
                           for r in comparison_results.values()])

        print(f"\n  Average Quality Score: {avg_quality:.0f}/100")
        print(f"  Average Distribution Overlap: {avg_overlap:.2f}%")
        print(f"  Average Percentile Correlation: {avg_corr:.3f}")

        if avg_quality >= 70:
            print(f"\n  SIMULATION VALIDATED: Monte Carlo matches synthetic+historical patterns well")
        elif avg_quality >= 50:
            print(f"\n  SIMULATION PARTIALLY VALIDATED: Some discrepancies with synthetic history")
        else:
            print(f"\n  SIMULATION CONCERNS: Significant differences from synthetic historical patterns")

        # Note about synthetic data limitations
        print(f"\n  NOTE: Synthetic data uses current fund parameters (expense ratio, borrowing")
        print(f"        costs, tracking error). Historical actual costs would have varied.")

    print(f"\n{'='*100}\n")

    return comparison_results
