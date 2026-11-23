"""
Reporting module for LETF simulation.

Contains interactive tax configuration, simplified tax estimation,
percentile scenario explanations, and summary statistics generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from letf import config as cfg
from letf.tax.marginal import calculate_comprehensive_tax_v6


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

# ============================================================================
# PERCENTILE-BASED REPORTING (v7.0 REPLACEMENT)
# ============================================================================

STATE_TAX_INFO = {
    'CA': {'name': 'California', 'rate': 0.093},
    'NY': {'name': 'New York', 'rate': 0.065},
    'TX': {'name': 'Texas', 'rate': 0.0},
    'FL': {'name': 'Florida', 'rate': 0.0},
    'WA': {'name': 'Washington', 'rate': 0.07},
    'NV': {'name': 'Nevada', 'rate': 0.0},
    'IL': {'name': 'Illinois', 'rate': 0.0495},
    'MA': {'name': 'Massachusetts', 'rate': 0.05},
    'NJ': {'name': 'New Jersey', 'rate': 0.0637}
}


def get_tax_config_interactive():
    """Interactive tax configuration menu. Uses defaults if stdin is not a terminal."""
    import sys, os

    # Check if running in non-interactive mode
    if not sys.stdin.isatty() or os.getenv('LETF_NON_INTERACTIVE') or os.getenv('LETF_NONINTERACTIVE'):
        print("\n  [Non-interactive mode] Federal-only scenario; explicit account and income required for tax replay")
        return {
            'state': None, 'state_name': 'FEDERAL ONLY', 'federal_only': True,
            'ordinary_income': None, 'filing_status': 'single',
            'career_stage': 'mid', 'years_until_retirement': None,
            'retirement_income': None
        }

    print("\n" + "="*120)
    print("TAX CONFIGURATION - Customize for YOUR Situation")
    print("="*120)

    print("\nSelect Your State:")
    print("  1. California (progressive 1-13.3%)")
    print("  2. New York (progressive 4-10.9%)")
    print("  3. Texas (NO state tax)")
    print("  4. Florida (NO state tax)")
    print("  5. Washington (7% on cap gains >$250k)")
    print("  6. Nevada (NO state tax)")
    print("  7. Illinois (flat 4.95%)")
    print("  8. Massachusetts (flat 5%)")
    print("  9. New Jersey (progressive 1.4-10.75%)")

    state_map = {'1': 'CA', '2': 'NY', '3': 'TX', '4': 'FL', '5': 'WA', '6': 'NV', '7': 'IL', '8': 'MA', '9': 'NJ'}
    state_choice = input("\nEnter (1-9) [default 1]: ").strip() or '1'
    state = state_map.get(state_choice, 'CA')

    print("\nYour CURRENT Ordinary Income (W-2, salary, etc.):")
    print("  (Note: Income will grow over time based on Monte Carlo simulation)")
    income_str = input("  Enter amount [default 150000]: ").strip() or "150000"
    try:
        ordinary_income = int(income_str.replace(',', '').replace('$', ''))
    except:
        ordinary_income = 150000

    print("\nFiling Status:")
    print("  1. Single")
    print("  2. Married")
    filing_choice = input("\nEnter (1-2) [default 1]: ").strip() or '1'
    filing_status = 'married' if filing_choice == '2' else 'single'

    print("\nCareer Stage (for income growth simulation):")
    print("  1. Early Career (20s-early 30s) - Rapid growth, promotions, job hopping")
    print("  2. Mid Career (30s-40s) - Steady growth, senior roles [DEFAULT]")
    print("  3. Late Career (40s-50s) - Slower growth, near peak earnings")
    career_choice = input("\nEnter (1-3) [default 2]: ").strip() or '2'
    career_map = {'1': 'early', '2': 'mid', '3': 'late'}
    career_stage = career_map.get(career_choice, 'mid')

    print("\n" + "="*120)
    print("RETIREMENT PLANNING (Optional)")
    print("="*120)
    print("\nWill you retire during the simulation horizon?")
    print("  (This drops income to retirement level, lowering taxes)")
    retire_str = input("\nHow many years until retirement? [press Enter to skip]: ").strip()

    years_until_retirement = None
    retirement_income = None

    if retire_str:
        try:
            years_until_retirement = int(retire_str)

            print(f"\nRetirement income (after {years_until_retirement} years):")
            print(f"  Typical: 40-60% of peak salary from Social Security + pension/401k")
            print(f"  Examples:")
            print(f"    - If peak salary is $300k: Retirement income ~$150k (50%)")
            print(f"    - If peak salary is $150k: Retirement income ~$75k (50%)")

            ret_income_str = input(f"\nRetirement income [default 50% of peak]: ").strip()

            if ret_income_str:
                try:
                    retirement_income = int(ret_income_str.replace(',', '').replace('$', ''))
                except:
                    retirement_income = None  # Will auto-calculate as 50% of peak
            # else: retirement_income stays None, will auto-calculate

        except:
            years_until_retirement = None

    config = {
        'state': state,
        'state_name': STATE_TAX_INFO[state]['name'],
        'ordinary_income': ordinary_income,
        'filing_status': filing_status,
        'career_stage': career_stage,
        'years_until_retirement': years_until_retirement,
        'retirement_income': retirement_income
    }

    print("\n" + "="*120)
    print("YOUR TAX CONFIG")
    print("="*120)
    print(f"  State: {config['state_name']}")
    print(f"  Starting Income: ${ordinary_income:,}")
    print(f"  Status: {filing_status.title()}")
    print(f"  Career Stage: {career_stage.title()}")

    if years_until_retirement:
        print(f"\n  Retirement Planning:")
        print(f"    Years until retirement: {years_until_retirement}")
        if retirement_income:
            print(f"    Retirement income: ${retirement_income:,}/year")
        else:
            print(f"    Retirement income: 50% of peak salary (auto-calculated)")

    print(f"\n  Note: Income will grow via Monte Carlo simulation accounting for:")
    print(f"        - Promotions, job changes, layoffs")
    print(f"        - Career stage progression")
    print(f"        - Random market volatility")
    if years_until_retirement:
        print(f"        - Retirement income drop after year {years_until_retirement}")
    print("="*120 + "\n")

    return config


def estimate_tax_simple(gains, ordinary_income, state, filing_status):
    """Simplified tax estimation"""

    if gains <= 0:
        return {'total_tax': 0, 'effective_rate': 0}

    st_gains = gains * 0.50
    lt_gains = gains * 0.50
    total_income = ordinary_income + gains

    # Federal rates
    if filing_status == 'single':
        if total_income < 100525:
            fed_st_rate, fed_lt_rate = 0.22, 0.0
        elif total_income < 191950:
            fed_st_rate, fed_lt_rate = 0.24, 0.15
        elif total_income < 518900:
            fed_st_rate, fed_lt_rate = 0.32, 0.15
        else:
            fed_st_rate, fed_lt_rate = 0.37, 0.20
        niit_thresh = 200000
    else:
        if total_income < 201050:
            fed_st_rate, fed_lt_rate = 0.22, 0.0
        elif total_income < 383900:
            fed_st_rate, fed_lt_rate = 0.24, 0.15
        elif total_income < 583750:
            fed_st_rate, fed_lt_rate = 0.32, 0.15
        else:
            fed_st_rate, fed_lt_rate = 0.37, 0.20
        niit_thresh = 250000

    federal = st_gains * fed_st_rate + lt_gains * fed_lt_rate
    state_tax = gains * STATE_TAX_INFO[state]['rate']
    niit = min(gains, max(0, total_income - niit_thresh)) * 0.038 if total_income > niit_thresh else 0

    total = federal + state_tax + niit
    return {'total_tax': total, 'effective_rate': (total/gains*100) if gains > 0 else 0}


def explain_percentile(p, pre_cagr, horizon, spy_cagr=0):
    """Market scenario explanation with SPY comparison"""

    scenarios = {
        10: f"""
+===========================================================================================+
| P10 - WORST 10% (You beat this in 90% of cases)                                          |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P10: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT GOES WRONG:
* 2-3 major crashes (2008-level events)
* VIX stays >30 for months
* Strategy whipsaws badly
* Worst {horizon}-year period since Depression

Historical: 2000-2010 (tech+housing crashes)
Probability: 1 in 10
""",
        25: f"""
+===========================================================================================+
| P25 - BELOW AVERAGE (You beat this in 75% of cases)                                      |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P25: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT HAPPENS:
* 1 major crash (COVID/2008-style)
* VIX averages 22-28
* Slow 3-5yr recovery
* Below-average decade

Historical: 2007-2013 (crisis+recovery)
Probability: 1 in 4
""",
        40: f"""
+===========================================================================================+
| P40 - SLIGHTLY BELOW MEDIAN                                                               |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P40: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT HAPPENS:
* 2-3 moderate 15-20% corrections
* Normal volatility (VIX 18-22)
* Mixed years
* Average decade

Historical: 1980-1990, 2010-2020
Probability: Common
""",
        60: f"""
+===========================================================================================+
| P60 - SLIGHTLY ABOVE MEDIAN                                                               |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P60: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT HAPPENS:
* Minor 10-15% corrections only
* Low volatility (VIX 15-18)
* More good years than bad
* Good decade

Historical: 2010-2018, 1982-1987
Probability: Common
""",
        75: f"""
+===========================================================================================+
| P75 - ABOVE AVERAGE (Need luck)                                                           |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P75: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT GOES RIGHT:
* Max 10% pullbacks
* Low volatility (VIX 12-15)
* 70-80% time in bull
* Great decade

Historical: 2012-2017, 1995-1999
Probability: 1 in 4
""",
        90: f"""
+===========================================================================================+
| P90 - BEST 10% (DON'T PLAN ON THIS!)                                                      |
| Strategy Pre-Tax CAGR: {pre_cagr:>5.2f}% | SPY B&H at P90: {spy_cagr:>5.2f}%                                    |
+===========================================================================================+

WHAT GOES PERFECTLY:
* No corrections (max 5-8% dips)
* VIX 10-12 throughout
* 85-90% time in bull
* Best {horizon}-year period ever

Historical: 2013-2017
Probability: 1 in 10 - RARE
WARNING: DO NOT PLAN RETIREMENT ON THIS
"""
    }
    return scenarios.get(p, "")


def create_summary_statistics(mc_results, time_horizon, tax_config=None):
    """NEW: Percentile-based analysis with tax customization - Option A Format"""
    from letf.integration import process_trades_with_wired_engine
    from letf.tax.engine import TaxpayerElections

    from letf.accuracy import UnsupportedModelError
    raise UnsupportedModelError(
        'Legacy rankings are gated: journals do not reconcile execution, tax cash payments, '
        'terminal liquidation and broker permissions. Use the audited account replay API.')
