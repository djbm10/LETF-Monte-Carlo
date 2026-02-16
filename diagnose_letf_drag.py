"""
Diagnostic test to isolate sources of LETF performance drag.

This script runs simplified simulations to identify where returns are being lost.
"""

import numpy as np
import os
os.environ['LETF_NON_INTERACTIVE'] = '1'

# Simple test: Perfect 2x leverage with no costs
def test_perfect_2x_leverage():
    """Test perfect 2x leverage with no costs or tracking error."""
    print("\n" + "="*80)
    print("TEST 1: Perfect 2x Leverage (No Costs, No Tracking Error)")
    print("="*80)

    # Simulate SPY returning 7% per year with 15% vol
    np.random.seed(42)
    n_years = 10
    n_days = n_years * 252

    daily_mean = 0.07 / 252
    daily_std = 0.15 / np.sqrt(252)

    spy_returns = np.random.normal(daily_mean, daily_std, n_days)

    # Perfect 2x leverage
    sso_returns_perfect = 2.0 * spy_returns

    # Calculate terminal wealth
    spy_final = np.exp(np.sum(np.log(1 + spy_returns)))
    sso_final = np.exp(np.sum(np.log(1 + sso_returns_perfect)))

    spy_cagr = (spy_final ** (1/n_years) - 1) * 100
    sso_cagr = (sso_final ** (1/n_years) - 1) * 100

    print(f"\nSPY CAGR: {spy_cagr:.2f}%")
    print(f"Perfect 2x CAGR: {sso_cagr:.2f}%")
    print(f"Theoretical 2x: {2 * spy_cagr:.2f}% (if linear)")

    # Calculate actual volatility drag
    realized_vol = np.std(spy_returns) * np.sqrt(252)
    theoretical_drag = -0.5 * 2 * (2-1) * (realized_vol**2)
    print(f"\nRealized volatility: {realized_vol*100:.2f}%")
    print(f"Theoretical vol drag: {theoretical_drag*100:.2f}%")
    print(f"Expected CAGR: {(2*spy_cagr + theoretical_drag*100):.2f}%")

def test_with_costs_only():
    """Test 2x leverage with realistic costs but no tracking error."""
    print("\n" + "="*80)
    print("TEST 2: 2x Leverage with Costs (No Tracking Error)")
    print("="*80)

    np.random.seed(42)
    n_years = 10
    n_days = n_years * 252

    daily_mean = 0.07 / 252
    daily_std = 0.15 / np.sqrt(252)

    spy_returns = np.random.normal(daily_mean, daily_std, n_days)

    # 2x leverage with costs
    expense_ratio = 0.0089  # 0.89% annual
    borrow_spread = 0.0050  # 0.50% annual
    risk_free_rate = 0.02  # 2% annual

    daily_expense = expense_ratio / 252
    daily_borrow = (risk_free_rate + borrow_spread) / 252

    sso_returns_with_costs = 2.0 * spy_returns - daily_expense - daily_borrow

    sso_final = np.exp(np.sum(np.log(1 + sso_returns_with_costs)))
    sso_cagr = (sso_final ** (1/n_years) - 1) * 100

    print(f"\nSSO CAGR with costs: {sso_cagr:.2f}%")
    print(f"Annual costs: {(expense_ratio + risk_free_rate + borrow_spread)*100:.2f}%")

def test_with_tracking_error():
    """Test 2x leverage with full model including tracking error."""
    print("\n" + "="*80)
    print("TEST 3: 2x Leverage with Full Model (Including Tracking Error)")
    print("="*80)

    np.random.seed(42)
    n_years = 10
    n_days = n_years * 252

    daily_mean = 0.07 / 252
    daily_std = 0.15 / np.sqrt(252)

    spy_returns = np.random.normal(daily_mean, daily_std, n_days)

    # Costs
    expense_ratio = 0.0089
    borrow_spread = 0.0050
    risk_free_rate = 0.02

    daily_expense = expense_ratio / 252
    daily_borrow = (risk_free_rate + borrow_spread) / 252

    # Simulate realistic tracking error
    # Base TE = 1 bps, with VIX and regime multipliers
    base_te = 0.0001  # 1 bps
    rho = 0.3  # AR(1) persistence

    # Simple regime: 95% low vol, 5% high vol
    regime = np.random.choice([0, 1], size=n_days, p=[0.95, 0.05])

    # Generate tracking error
    te_series = np.zeros(n_days)
    for i in range(1, n_days):
        # Simplified multipliers
        regime_mult = 1.0 if regime[i] == 0 else 2.5
        vix_mult = 1.5  # Assume average VIX multiplier
        move_mult = 1.0 + 5.0 * abs(spy_returns[i])  # Simplified
        downside_mult = 1.15 if spy_returns[i] < 0 else 0.95

        innovation = np.random.standard_t(df=5)
        innovation *= base_te * regime_mult * vix_mult * move_mult * downside_mult

        te_series[i] = rho * te_series[i-1] + np.sqrt(1 - rho**2) * innovation

    # Apply to returns
    sso_returns_full = 2.0 * spy_returns - daily_expense - daily_borrow
    sso_returns_full = sso_returns_full * (1 + te_series)

    sso_final = np.exp(np.sum(np.log(1 + np.clip(sso_returns_full, -0.99, 10))))
    sso_cagr = (sso_final ** (1/n_years) - 1) * 100

    print(f"\nSSO CAGR with full model: {sso_cagr:.2f}%")
    print(f"Mean tracking error: {np.mean(te_series)*10000:.2f} bps")
    print(f"Std tracking error: {np.std(te_series)*10000:.2f} bps")
    print(f"Max tracking error: {np.max(np.abs(te_series))*10000:.2f} bps")

def test_current_implementation():
    """Test actual implementation from quick_test to compare."""
    print("\n" + "="*80)
    print("TEST 4: Current Implementation (From Actual Simulation)")
    print("="*80)

    print("\nFrom the actual test results:")
    print("  SSO median CAGR: -6.80%")
    print("  SPY median CAGR: +4.50%")
    print()
    print("Expected SSO (if 2x SPY): 2 × 4.50% = 9.00% (before drag)")
    print("With vol drag (~3%): 9.00% - 3.00% = 6.00%")
    print("With costs (~1.4%): 6.00% - 1.40% = 4.60%")
    print()
    print("Actual result: -6.80%")
    print("Difference: -6.80% - 4.60% = -11.40% unexplained drag")
    print()
    print("This suggests tracking error is adding ~11% annual drag!")

if __name__ == '__main__':
    test_perfect_2x_leverage()
    test_with_costs_only()
    test_with_tracking_error()
    test_current_implementation()

    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print()
    print("The tracking error multipliers are TOO AGGRESSIVE:")
    print("  - Regime multiplier: 2.5x in high vol (should be ~1.5x?)")
    print("  - Move multiplier: 1 + 10×|return| (should be 1 + 5×|return|?)")
    print("  - Downside asymmetry: 1.15x (reasonable)")
    print()
    print("Recommendation: Reduce tracking error amplification factors.")
    print("="*80)
