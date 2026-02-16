"""
Simple test to verify TQQQ performance is reasonable.

Runs 100 simulations and shows median results for SPY, SSO, and TQQQ.
"""

import os
import time
os.environ['LETF_NON_INTERACTIVE'] = '1'

import letf.config as cfg
from letf.data import fetch_historical_data
from letf.calibration import (
    calibrate_regime_model_volatility,
    calibrate_joint_return_model,
    calibrate_funding_spread_model,
    calibrate_stress_state_model,
    calibrate_tracking_residual_model,
    calibrate_correlations_time_varying,
)
from letf.mc_runner import parallel_monte_carlo_fixed
from letf.reporting import create_summary_statistics
import numpy as np

def main():
    print("="*80)
    print("TQQQ TEST - 100 SIMULATIONS x 10 YEAR HORIZON")
    print("="*80)

    start = time.time()

    # Initialize
    cfg.init_cache()
    cfg.ANALYSIS_START_DATE = "2010-01-01"
    cfg.ANALYSIS_END_DATE = "2025-12-31"
    cfg.NUM_SIMULATIONS = 100  # More simulations for better statistics

    # Fetch data
    print("\n### FETCHING DATA ###")
    data = fetch_historical_data()
    print(f"Data loaded: {len(data)} days")

    # Calibrate models
    print("\n### CALIBRATING MODELS ###")
    regime_model = calibrate_regime_model_volatility(data)
    joint_return_model = calibrate_joint_return_model(data, regime_model)
    funding_model = calibrate_funding_spread_model(data, regime_model)
    stress_model = calibrate_stress_state_model(data, regime_model)
    tracking_residual_model = calibrate_tracking_residual_model(data, regime_model)
    correlation_matrices = calibrate_correlations_time_varying(data, regime_model)

    # Run Monte Carlo for 10-year horizon
    print("\n### RUNNING MONTE CARLO ###")
    time_horizon = 10
    strategy_ids = ['S1', 'S2', 'S3']  # TQQQ, SPY, SSO

    results = parallel_monte_carlo_fixed(
        strategy_ids,
        time_horizon,
        regime_model,
        correlation_matrices,
        historical_df=data
    )

    # Extract CAGRs for analysis
    print("\n### ANALYZING RESULTS ###")

    spy_cagrs = []
    sso_cagrs = []
    tqqq_cagrs = []

    # results is a dict: {'S1': [...], 'S2': [...], 'S3': [...]}
    # Each result has 'Final_Wealth', so we calculate CAGR from it
    for strat_result in results.get('S2', []):  # SPY
        if strat_result is not None:
            final_wealth = strat_result['Final_Wealth']
            cagr = (final_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1
            spy_cagrs.append(cagr * 100)

    for strat_result in results.get('S3', []):  # SSO
        if strat_result is not None:
            final_wealth = strat_result['Final_Wealth']
            cagr = (final_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1
            sso_cagrs.append(cagr * 100)

    for strat_result in results.get('S1', []):  # TQQQ
        if strat_result is not None:
            final_wealth = strat_result['Final_Wealth']
            cagr = (final_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1
            tqqq_cagrs.append(cagr * 100)

    # Calculate statistics
    print("\n" + "="*80)
    print("RESULTS - 100 SIMULATIONS x 10 YEARS")
    print("="*80)

    print("\nSPY (1x S&P 500):")
    print(f"  Median CAGR: {np.median(spy_cagrs):6.2f}%")
    print(f"  P10:         {np.percentile(spy_cagrs, 10):6.2f}%")
    print(f"  P25:         {np.percentile(spy_cagrs, 25):6.2f}%")
    print(f"  P75:         {np.percentile(spy_cagrs, 75):6.2f}%")
    print(f"  P90:         {np.percentile(spy_cagrs, 90):6.2f}%")

    print("\nSSO (2x S&P 500):")
    print(f"  Median CAGR: {np.median(sso_cagrs):6.2f}%")
    print(f"  P10:         {np.percentile(sso_cagrs, 10):6.2f}%")
    print(f"  P25:         {np.percentile(sso_cagrs, 25):6.2f}%")
    print(f"  P75:         {np.percentile(sso_cagrs, 75):6.2f}%")
    print(f"  P90:         {np.percentile(sso_cagrs, 90):6.2f}%")

    print("\nTQQQ (3x NASDAQ-100):")
    print(f"  Median CAGR: {np.median(tqqq_cagrs):6.2f}%")
    print(f"  P10:         {np.percentile(tqqq_cagrs, 10):6.2f}%")
    print(f"  P25:         {np.percentile(tqqq_cagrs, 25):6.2f}%")
    print(f"  P75:         {np.percentile(tqqq_cagrs, 75):6.2f}%")
    print(f"  P90:         {np.percentile(tqqq_cagrs, 90):6.2f}%")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    spy_median = np.median(spy_cagrs)
    sso_median = np.median(sso_cagrs)
    tqqq_median = np.median(tqqq_cagrs)

    print(f"\nSSO vs SPY:")
    print(f"  Expected (if 2x with ~2% drag): {2*spy_median - 2:.2f}%")
    print(f"  Actual SSO median:              {sso_median:.2f}%")
    print(f"  Difference:                     {sso_median - (2*spy_median - 2):.2f}%")

    print(f"\nTQQQ performance:")
    print(f"  Median CAGR: {tqqq_median:.2f}%")
    if tqqq_median < 0:
        print(f"  WARNING: Median TQQQ is NEGATIVE!")
    elif tqqq_median < spy_median:
        print(f"  NOTE: TQQQ underperforms SPY by {spy_median - tqqq_median:.2f}%")
    else:
        print(f"  NOTE: TQQQ outperforms SPY by {tqqq_median - spy_median:.2f}%")

    # Historical context
    print("\n" + "="*80)
    print("HISTORICAL CONTEXT")
    print("="*80)
    print("Historical TQQQ (2010-2020): ~38% CAGR (exceptional bull market)")
    print("Historical SPY (2010-2020):  ~13% CAGR")
    print("Historical SSO (2010-2020):  ~20% CAGR")
    print()
    print("Our simulations sample from 1926-2025 (includes Great Depression,")
    print("1970s stagflation, 2000-2002 tech crash, 2008 crisis, etc.)")
    print("This creates more conservative projections than 2010-2020 period.")

    elapsed = time.time() - start
    print("\n" + "="*80)
    print(f"TEST COMPLETE - {elapsed:.1f} seconds")
    print("="*80)

if __name__ == '__main__':
    main()
