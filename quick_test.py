"""Quick simulation test - runs a minimal Monte Carlo to validate the system."""
import sys
import time
import os

# Set non-interactive mode for automated testing
os.environ['LETF_NON_INTERACTIVE'] = '1'

sys.path.insert(0, '.')

# Temporarily reduce simulation count for quick test
import letf.config as cfg
original_sims = cfg.NUM_SIMULATIONS
cfg.NUM_SIMULATIONS = 10  # Just 10 simulations for quick test

from letf import config as cfg
from letf.tax.engine import run_golden_tests
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


def main():
    """Main test function - wrapped for multiprocessing compatibility."""
    print("="*80)
    print("QUICK TEST - 10 SIMULATIONS x 10 YEAR HORIZON")
    print("="*80)

    start = time.time()

    # Initialize
    cfg.init_cache()
    cfg.ANALYSIS_START_DATE = "2010-01-01"  # Recent data only for speed
    cfg.ANALYSIS_END_DATE = "2025-12-31"

    # Test tax engine
    print("\n### VALIDATING TAX ENGINE ###")
    try:
        run_golden_tests(trace_failures=False)
        print("Tax engine: PASSED")
    except Exception as e:
        print(f"Tax engine: FAILED - {e}")
        sys.exit(1)

    # Fetch data
    print("\n### FETCHING DATA ###")
    df = fetch_historical_data()
    print(f"Data loaded: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

    # Calibrate models
    print("\n### CALIBRATING MODELS ###")
    print("Calibrating regime model...")
    regime_model = calibrate_regime_model_volatility(df)

    print("Calibrating joint return model...")
    regime_model['joint_return_model'] = calibrate_joint_return_model(
        df, regime_model['regimes_historical']
    )

    print("Calibrating funding spread model...")
    regime_model['funding_model'] = calibrate_funding_spread_model(df)

    print("Calibrating stress-state model...")
    regime_model['stress_state_model'] = calibrate_stress_state_model(
        df, regime_model['regimes_historical']
    )

    print("Calibrating tracking residual model...")
    regime_model['tracking_residual_model'] = calibrate_tracking_residual_model(
        df, funding_model=regime_model['funding_model']
    )

    print("Calibrating correlations...")
    correlation_matrices = calibrate_correlations_time_varying(df, regime_model)

    # Run quick simulation
    print("\n### RUNNING MONTE CARLO ###")
    time_horizon = 10
    strategy_ids = ['S1', 'S2', 'S3']  # Just test a few strategies

    mc_results = parallel_monte_carlo_fixed(
        strategy_ids=strategy_ids,
        time_horizon=time_horizon,
        regime_model=regime_model,
        correlation_matrices=correlation_matrices,
        historical_df=df,
    )

    # Show results
    print("\n### RESULTS ###")
    create_summary_statistics(mc_results, time_horizon)

    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"QUICK TEST COMPLETE - {elapsed:.1f} seconds")
    print(f"{'='*80}")

    # Restore original config
    cfg.NUM_SIMULATIONS = original_sims


if __name__ == '__main__':
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
