"""
Detailed diagnostic for TQQQ performance in simulations.

Analyzes a single simulation path to understand:
1. Underlying (QQQ) vs SPY returns
2. Leveraged returns before costs
3. Impact of costs (expense ratio + borrow costs)
4. Impact of tracking error
5. Final TQQQ returns
"""

import numpy as np
import os
os.environ['LETF_NON_INTERACTIVE'] = '1'

import letf.config as cfg
from letf.calibration import (
    calibrate_regime_model_volatility,
    calibrate_joint_return_model,
)
from letf.data import fetch_historical_data
from letf.simulation.engine import (
    simulate_regime_path_semi_markov,
    generate_fat_tailed_returns,
    compute_letf_return_correct,
    generate_tracking_error_ar1,
    _stable_asset_seed,
)

def main():
    print("\n" + "="*80)
    print("TQQQ DETAILED DIAGNOSTIC - SINGLE 10-YEAR PATH")
    print("="*80)

    # Initialize
    cfg.init_cache()
    cfg.ANALYSIS_START_DATE = "2010-01-01"
    cfg.ANALYSIS_END_DATE = "2025-12-31"

    # Fetch data
    print("\nLoading historical data...")
    data = fetch_historical_data()

    # Calibrate models
    print("Calibrating models...")
    regime_model = calibrate_regime_model_volatility(data)
    joint_model = calibrate_joint_return_model(data, regime_model)

    # Run single simulation
    np.random.seed(42)
    sim_id = 0
    n_years = 10
    n_days = n_years * 252

    # Generate regime path
    start_regime = 0
    regime_path = simulate_regime_path_semi_markov(
        n_days,
        start_regime,
        regime_model['transition_matrix'],
        regime_model['duration_samples'],
        np.random.default_rng(42)
    )

    # Generate returns
    returns_dict = generate_fat_tailed_returns(
        n_days=n_days,
        regime_path=regime_path,
        joint_model=joint_model,
        bootstrap_sampler=None,  # Will use Student-t only
        rng=np.random.default_rng(42),
        antithetic=False
    )

    spy_returns = returns_dict['SPY_Ret']
    qqq_returns = returns_dict['QQQ_Ret']
    vix = returns_dict['VIX']

    # TQQQ parameters
    leverage = 3.0
    expense_ratio = 0.0086  # 0.86% annual
    borrow_spread = 0.0075  # 0.75% spread
    risk_free_rate = 0.02   # 2% annual (simplified)

    # Calculate borrowing costs
    daily_borrow = (risk_free_rate + borrow_spread) / 252 * np.ones(n_days)

    # Step 1: Perfect 3x leverage (no costs)
    perfect_3x = 3.0 * qqq_returns

    # Step 2: With costs (no tracking error)
    with_costs = np.array([
        compute_letf_return_correct(qqq_returns[t], leverage, 0, expense_ratio, daily_borrow[t])
        for t in range(n_days)
    ])

    # Step 3: Generate tracking error
    te_scale = 0.0002  # 2 bps base
    te_df = 5
    te_rho = 0.3
    te_downside = 1.30

    tracking_errors = generate_tracking_error_ar1(
        n_days,
        regime_path,
        vix,
        qqq_returns,
        te_scale,
        te_df,
        rng=np.random.default_rng(_stable_asset_seed(sim_id, 'TQQQ')),
        rho=te_rho,
        downside_asymmetry=te_downside,
        liquidity_series=None,
        clip_limit=None
    )

    # Step 4: Final TQQQ returns (with tracking error)
    tqqq_returns = with_costs + tracking_errors
    tqqq_returns = np.clip(tqqq_returns, -0.999, 10.0)

    # Calculate terminal wealth
    spy_final = np.exp(np.sum(np.log(1 + spy_returns)))
    qqq_final = np.exp(np.sum(np.log(1 + qqq_returns)))
    perfect_3x_final = np.exp(np.sum(np.log(1 + np.clip(perfect_3x, -0.999, 10))))
    with_costs_final = np.exp(np.sum(np.log(1 + np.clip(with_costs, -0.999, 10))))
    tqqq_final = np.exp(np.sum(np.log(1 + tqqq_returns)))

    # Calculate CAGRs
    spy_cagr = (spy_final ** (1/n_years) - 1) * 100
    qqq_cagr = (qqq_final ** (1/n_years) - 1) * 100
    perfect_cagr = (perfect_3x_final ** (1/n_years) - 1) * 100
    costs_cagr = (with_costs_final ** (1/n_years) - 1) * 100
    tqqq_cagr = (tqqq_final ** (1/n_years) - 1) * 100

    # Calculate realized volatilities
    spy_vol = np.std(spy_returns) * np.sqrt(252) * 100
    qqq_vol = np.std(qqq_returns) * np.sqrt(252) * 100
    perfect_vol = np.std(perfect_3x) * np.sqrt(252) * 100
    tqqq_vol = np.std(tqqq_returns) * np.sqrt(252) * 100

    # Theoretical volatility drag
    theoretical_drag = -0.5 * 3.0 * (3.0 - 1) * (qqq_vol/100)**2 * 100

    # Results
    print("\n" + "="*80)
    print("UNDERLYING RETURNS")
    print("="*80)
    print(f"SPY CAGR:  {spy_cagr:6.2f}% | Volatility: {spy_vol:5.2f}%")
    print(f"QQQ CAGR:  {qqq_cagr:6.2f}% | Volatility: {qqq_vol:5.2f}%")
    print(f"QQQ vs SPY: {qqq_cagr - spy_cagr:+6.2f}% outperformance")

    print("\n" + "="*80)
    print("LEVERAGED RETURNS BREAKDOWN")
    print("="*80)
    print(f"Step 1 - Perfect 3x QQQ (no costs):        {perfect_cagr:6.2f}%")
    print(f"Step 2 - With costs (expense + borrow):   {costs_cagr:6.2f}%")
    print(f"Step 3 - Final TQQQ (with tracking error): {tqqq_cagr:6.2f}%")

    print("\n" + "="*80)
    print("DRAG ANALYSIS")
    print("="*80)
    print(f"Theoretical vol drag: {theoretical_drag:6.2f}% (from -0.5*L*(L-1)*sigma^2)")
    print(f"Actual drag:")
    print(f"  Perfect 3x -> QQQ:    {perfect_cagr - 3*qqq_cagr:6.2f}% (volatility drag)")
    print(f"  Costs impact:         {costs_cagr - perfect_cagr:6.2f}% (expense + borrow)")
    print(f"  Tracking error impact: {tqqq_cagr - costs_cagr:6.2f}%")
    print(f"  Total drag:           {tqqq_cagr - 3*qqq_cagr:6.2f}%")

    print("\n" + "="*80)
    print("VOLATILITY COMPARISON")
    print("="*80)
    print(f"QQQ volatility:         {qqq_vol:5.2f}%")
    print(f"Perfect 3x volatility:  {perfect_vol:5.2f}% (expect ~3x = {3*qqq_vol:5.2f}%)")
    print(f"TQQQ final volatility:  {tqqq_vol:5.2f}%")

    print("\n" + "="*80)
    print("REGIME DISTRIBUTION")
    print("="*80)
    regime_counts = {int(r): int((regime_path == r).sum()) for r in np.unique(regime_path)}
    for regime, count in regime_counts.items():
        pct = 100 * count / n_days
        regime_name = "Low Vol" if regime == 0 else "High Vol"
        print(f"Regime {regime} ({regime_name}): {count:4d} days ({pct:5.2f}%)")

    print("\n" + "="*80)
    print("TRACKING ERROR STATISTICS")
    print("="*80)
    print(f"Mean tracking error:  {np.mean(tracking_errors)*10000:6.2f} bps")
    print(f"Std tracking error:   {np.std(tracking_errors)*10000:6.2f} bps")
    print(f"Min tracking error:   {np.min(tracking_errors)*10000:6.2f} bps")
    print(f"Max tracking error:   {np.max(tracking_errors)*10000:6.2f} bps")
    print(f"P5 tracking error:    {np.percentile(tracking_errors, 5)*10000:6.2f} bps")
    print(f"P95 tracking error:   {np.percentile(tracking_errors, 95)*10000:6.2f} bps")

    # Check if tracking error is too negative on average
    if np.mean(tracking_errors) < -0.00001:  # Less than -0.1 bps average
        print(f"\nWARNING: Tracking error has negative mean ({np.mean(tracking_errors)*10000:.2f} bps)")
        print("This suggests systematic drag - tracking error should be mean-zero!")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if tqqq_cagr > 0:
        print(f"TQQQ achieved {tqqq_cagr:.2f}% CAGR")
        if tqqq_cagr < qqq_cagr:
            print(f"  -> UNDERPERFORMED QQQ by {qqq_cagr - tqqq_cagr:.2f}%")
        else:
            print(f"  -> OUTPERFORMED QQQ by {tqqq_cagr - qqq_cagr:.2f}%")
    else:
        print(f"TQQQ had NEGATIVE returns: {tqqq_cagr:.2f}% CAGR")
        print(f"  -> This is {abs(tqqq_cagr):.2f}% worse than buy-and-hold QQQ")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
