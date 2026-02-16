"""
LETF Analysis Package - Corrected Leveraged ETF Monte Carlo Simulator

Entry point: letf.run()
"""

import time
from letf import config as cfg


def _fmt_elapsed(seconds):
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def run():
    """Main execution - runs complete LETF analysis with percentile reporting."""

    run_start = time.time()
    step_times = []

    def _step(label):
        """Print step timing and record it."""
        now = time.time()
        if step_times:
            prev_label, prev_start = step_times[-1]
            elapsed = now - prev_start
            print(f"  [{_fmt_elapsed(elapsed)}] {prev_label}")
        step_times.append((label, now))

    # Initialize (no side effects at import time)
    cfg.init_cache()
    cfg.init_plotting()
    cfg.print_banner()

    # Lazy imports to avoid circular deps and heavy import-time cost
    from letf.tax.engine import run_golden_tests
    from letf.ui import get_start_date_interactive, validate_time_horizons_for_start_date
    from letf.data import fetch_historical_data
    from letf.calibration import (
        calibrate_regime_model_volatility,
        calibrate_joint_return_model,
        calibrate_funding_spread_model,
        calibrate_stress_state_model,
        calibrate_tracking_residual_model,
        calibrate_correlations_time_varying,
    )
    from letf.validation import run_validation_tests
    from letf.mc_runner import parallel_monte_carlo_fixed
    from letf.reporting import create_summary_statistics, get_tax_config_interactive
    from letf.historical import (
        compare_simulated_vs_historical,
        compare_simulated_vs_synthetic_historical,
    )

    # ========================================================================
    # STEP 0: Validate Tax Engine (mandatory)
    # ========================================================================
    print("\n" + "=" * 80)
    print("LETF ULTIMATE v6.0 - FULLY INTEGRATED")
    print("=" * 80)

    _step("Tax engine validation")
    print("\n### VALIDATING TAX ENGINE ###\n")
    try:
        run_golden_tests(trace_failures=True)
        print("\nTax engine validated - proceeding with simulation\n")
    except Exception as e:
        print(f"\nGOLDEN TESTS FAILED: {e}")
        print("STOPPING - System is broken")
        return

    # ========================================================================
    # STEP 1: Select Analysis Date Range
    # ========================================================================
    print("\n" + "=" * 80)
    print("LETF ANALYSIS WITH PERCENTILE REPORTING")
    print("=" * 80)

    _step("Date selection")
    selected_start, selected_end = get_start_date_interactive()
    print(f"\n  Using date range: {selected_start} to {selected_end}")

    # ========================================================================
    # STEP 2: Fetch and Calibrate
    # ========================================================================
    _step("Fetch historical data")
    print("\nFetching historical data...")
    df = fetch_historical_data()
    print(f"\n  Data loaded: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total days: {len(df):,} ({len(df)/252:.2f} years)")

    # ========================================================================
    # STEP 2b: Tax & Income Configuration
    # ========================================================================
    _step("Tax configuration")
    tax_config = get_tax_config_interactive()

    _step("Calibrate regime model")
    print("\nCalibrating regime model...")
    regime_model = calibrate_regime_model_volatility(df)

    _step("Calibrate joint return model")
    print("Calibrating joint return model...")
    regime_model['joint_return_model'] = calibrate_joint_return_model(
        df, regime_model['regimes_historical']
    )

    _step("Calibrate funding spread model")
    print("Calibrating funding spread model...")
    regime_model['funding_model'] = calibrate_funding_spread_model(df)

    _step("Calibrate stress-state model")
    print("Calibrating stress-state model...")
    regime_model['stress_state_model'] = calibrate_stress_state_model(
        df, regime_model['regimes_historical']
    )

    _step("Calibrate tracking residual model")
    print("Calibrating tracking residual model...")
    regime_model['tracking_residual_model'] = calibrate_tracking_residual_model(
        df, funding_model=regime_model['funding_model']
    )

    _step("Calibrate correlations")
    print("Calibrating correlations...")
    correlation_matrices = calibrate_correlations_time_varying(df, regime_model)

    _step("Validation tests")
    print("Running validation tests...")
    run_validation_tests(df=df, regime_model=regime_model)

    # ========================================================================
    # STEP 3: Monte Carlo Simulation per Horizon
    # ========================================================================
    requested_horizons = [10, 20, 30]
    time_horizons = validate_time_horizons_for_start_date(
        cfg.ANALYSIS_START_DATE, requested_horizons
    )

    if not time_horizons:
        print("\nERROR: Not enough data for any requested time horizon!")
        print(f"  Requested horizons: {requested_horizons}")
        print(f"  Start date: {cfg.ANALYSIS_START_DATE}")
        return

    for horizon in time_horizons:
        print(f"\n{'=' * 80}")
        print(f"MONTE CARLO SIMULATION: {horizon}-YEAR HORIZON")
        print(f"{'=' * 80}")

        _step(f"MC simulation {horizon}Y")
        mc_results = parallel_monte_carlo_fixed(
            strategy_ids=list(cfg.STRATEGIES.keys()),
            time_horizon=horizon,
            regime_model=regime_model,
            correlation_matrices=correlation_matrices,
            historical_df=df,
        )

        _step(f"Summary stats {horizon}Y")
        create_summary_statistics(mc_results, horizon, tax_config=tax_config)

        _step(f"Historical comparison {horizon}Y")
        compare_simulated_vs_historical(df, mc_results, horizon)
        compare_simulated_vs_synthetic_historical(df, mc_results, horizon)

    # Print final step timing
    _step("done")

    # ========================================================================
    # Timing Summary
    # ========================================================================
    total_elapsed = time.time() - run_start
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    for i in range(len(step_times) - 1):
        label, start = step_times[i]
        _, end = step_times[i + 1]
        elapsed = end - start
        pct = (elapsed / total_elapsed) * 100 if total_elapsed > 0 else 0
        print(f"  {label:<40s} {_fmt_elapsed(elapsed):>8s}  ({pct:5.1f}%)")
    print(f"  {'':->56s}")
    print(f"  {'TOTAL':<40s} {_fmt_elapsed(total_elapsed):>8s}")

    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n  Analysis Start Date: {cfg.ANALYSIS_START_DATE}")
    print(f"  Tax Engine: v6.0 with proper marginal rates")
    print(f"  Golden Tests: 6/6 passing")
    print(f"  LETF Strategies: 19 (S1-S19)")
    print(f"  Regime Model: Volatility-based switching")
    print(f"  Total runtime: {_fmt_elapsed(total_elapsed)}")
    print("=" * 80)
