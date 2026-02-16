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
        print("\n  [Non-interactive mode] Using default tax config: CA, $150k income, Single, Mid-career, no retirement")
        return {
            'state': 'CA', 'state_name': 'California',
            'ordinary_income': 150000, 'filing_status': 'single',
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

    # Use provided tax_config, or fall back to interactive prompt
    global TAX_CONFIG
    if tax_config is not None:
        TAX_CONFIG = tax_config
    elif 'TAX_CONFIG' not in globals():
        TAX_CONFIG = get_tax_config_interactive()

    # ========================================================================
    # MARKET SCENARIOS - SHOWN ONCE AT START
    # ========================================================================

    # Get SPY percentiles for comparison (only once, not per strategy)
    global MARKET_SCENARIOS_SHOWN
    if 'MARKET_SCENARIOS_SHOWN' not in globals():
        MARKET_SCENARIOS_SHOWN = True

        spy_pcts = {}
        if 'S2' in mc_results and mc_results['S2']:
            spy_wealth = np.array([r['Final_Wealth'] for r in mc_results['S2']])
            for pname, pval in [('p10',10), ('p25',25), ('p40',40), ('p60',60), ('p75',75), ('p90',90)]:
                spy_w = np.percentile(spy_wealth, pval)
                spy_pcts[pname] = (spy_w / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1

        print(f"\n{'='*140}")
        print(f"MARKET SCENARIOS FOR {time_horizon}-YEAR HORIZON")
        print(f"{'='*140}")
        print("\nThese scenarios apply to ALL strategies - they describe the market conditions.")
        print("SPY Buy & Hold CAGRs shown for reference.\n")

        for pval in [10, 25, 40, 60, 75, 90]:
            pname = f"p{pval}"
            spy_cagr = spy_pcts.get(pname, 0) * 100
            print(explain_percentile(pval, spy_cagr, time_horizon, spy_cagr))

        print("="*140 + "\n")

    # ========================================================================
    # ROTH IRA SECTION (NO TAX)
    # ========================================================================

    print("\n" + "="*100)
    print(f"ROTH IRA COMPATIBLE - {time_horizon}-YEAR HORIZON")
    print("="*100)
    print(f"{'Rank':<5} {'ID':<5} {'Strategy':<18} {'Win%':>8} {'p10':>7} {'p25':>7} {'p40':>8} {'Median$':>9} {'CAGR':>8} {'p60':>7} {'p75':>7} {'p90':>7}| {'MaxDD':>9} {'Trd/Y':>7}")
    print("-"*100)

    roth_ids = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    roth_data = []

    for sid in roth_ids:
        if sid not in mc_results or not mc_results[sid]:
            continue

        results = mc_results[sid]
        wealth = np.array([r['Final_Wealth'] for r in results])
        median = np.median(wealth)
        cagr = (median / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1
        p10 = np.percentile(wealth, 10)
        p25 = np.percentile(wealth, 25)
        p40 = np.percentile(wealth, 40)
        p60 = np.percentile(wealth, 60)
        p75 = np.percentile(wealth, 75)
        p90 = np.percentile(wealth, 90)

        # Win rate vs SPY
        spy_wealth = np.array([r['Final_Wealth'] for r in mc_results.get('S2', [])])
        if len(spy_wealth) == len(wealth):
            win_rate = sum(w > s for w, s in zip(wealth, spy_wealth)) / len(wealth) * 100
        else:
            win_rate = 0

        max_dd = np.median([r.get('Max_DD', 0) for r in results])
        trades = np.mean([r.get('Trades_Per_Year', 0) for r in results])

        roth_data.append({
            'id': sid,
            'p10': p10, 'p25': p25, 'p40': p40, 'median': median,
            'p60': p60, 'p75': p75, 'p90': p90,
            'cagr': cagr, 'win': win_rate,
            'dd': max_dd, 'trades': trades, 'name': cfg.STRATEGIES[sid]['name']
        })

    roth_data.sort(key=lambda x: x['median'], reverse=True)
    for i, d in enumerate(roth_data, 1):
        print(f"{i:<5} {d['id']:<5} {d['name']:<18} {d['win']:>8.2f}% {d['p10']:>7,.0f} {d['p25']:>7,.0f} {d['p40']:>8,.0f} {d['median']:>9,.0f} {d['cagr']*100:>8.2f}% {d['p60']:>7,.0f} {d['p75']:>7,.0f} {d['p90']:>7,.0f} | {d['dd']*100:>8.1f}% {d['trades']:>7.1f}")

    print("="*100 + "\n")

    # ========================================================================
    # TAXABLE BROKERAGE SECTION (WITH TAX)
    # ========================================================================

    print(f"\n{'='*140}")
    print(f"TAXABLE BROKERAGE (High Frequency / Advanced Risk Management)")
    print(f"  Requires margin and generates significant short-term capital gains:")
    print(f"  Tax Config: {TAX_CONFIG['state_name']} | ${TAX_CONFIG['ordinary_income']:,} | {TAX_CONFIG['filing_status'].title()}")
    print("-"*140)
    print(f"{'Rank':<5} {'ID':<5} {'Strategy':<30} {'Pre Tax':>15} {'Post Tax':>15} {'Post Tax':>12} {'Win%':>8} | {'MaxDD':>} {'Trd/Y':>7}")
    print(f"{'':>5} {'':>5} {'':>30} {'Median$':>15} {'Median$':>15} {'CAGR':>12} {'':>8} | {'':>9} {'':>7}")
    print("-"*140)

    # Collect taxable data with percentiles
    taxable_ids = ['S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19']
    data = []

    # Get SPY percentiles for comparison
    spy_pcts = {}
    if 'S2' in mc_results and mc_results['S2']:
        spy_wealth = np.array([r['Final_Wealth'] for r in mc_results['S2']])
        for pname, pval in [('p10',10), ('p25',25), ('p40',40), ('p60',60), ('p75',75), ('p90',90)]:
            spy_w = np.percentile(spy_wealth, pval)
            spy_pcts[pname] = (spy_w / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1

    # Pre-compute SPY post-tax wealth ONCE (reused for all taxable strategy win-rate comparisons)
    spy_post_tax_cache = None
    if 'S2' in mc_results and mc_results['S2']:
        spy_results_for_tax = mc_results['S2']
        spy_post_tax_cache = []
        for spy_sim in spy_results_for_tax:
            spy_pre = spy_sim['Final_Wealth']
            spy_trades = spy_sim.get('Trade_List', [])
            if spy_trades and len(spy_trades) > 0:
                spy_tax_result = process_trades_with_wired_engine(
                    trades=spy_trades,
                    time_horizon_years=time_horizon,
                    elections=TaxpayerElections(),
                    initial_capital=cfg.INITIAL_CAPITAL,
                    debug=False,
                    strategy_id="SPY_tax",
                    tax_config=TAX_CONFIG
                )
                spy_post = spy_pre - spy_tax_result['total_tax']
            else:
                spy_post = spy_pre
            spy_post_tax_cache.append(spy_post)
        spy_post_tax_cache = np.array(spy_post_tax_cache)

    for sid in taxable_ids:
        if sid not in mc_results or not mc_results[sid]:
            continue

        results = mc_results[sid]
        wealth = np.array([r['Final_Wealth'] for r in results])
    # Calculate post-tax wealth for ALL simulations first
        # Then take percentiles of the post-tax distribution

        if len(wealth) == 0:
            continue

        # Step 1: Sample-based tax computation for scalability
        # Computing tax for ALL sims is O(N) with expensive per-trade tax engine.
        # Instead, sample MAX_TAX_SIMS evenly across the wealth distribution,
        # compute exact tax for those, and interpolate the rest.
        MAX_TAX_SIMS = 50

        n_sims = len(results)
        sorted_indices = np.argsort(wealth)

        if n_sims <= MAX_TAX_SIMS:
            sample_indices = list(range(n_sims))
        else:
            # Pick evenly spaced positions from sorted wealth distribution
            positions = np.linspace(0, n_sims - 1, MAX_TAX_SIMS, dtype=int)
            sample_indices = [int(sorted_indices[p]) for p in positions]

        # Compute exact tax for sampled sims
        sample_wealth_vals = []
        sample_tax_vals = []
        for idx in sample_indices:
            sim_result = results[idx]
            pre_w = sim_result['Final_Wealth']
            trade_list = sim_result.get('Trade_List', [])

            if trade_list and len(trade_list) > 0:
                tax_result = process_trades_with_wired_engine(
                    trades=trade_list,
                    time_horizon_years=time_horizon,
                    elections=TaxpayerElections(),
                    initial_capital=cfg.INITIAL_CAPITAL,
                    debug=False,
                    strategy_id=f"{sid}_bulk",
                    tax_config=TAX_CONFIG
                )
                total_tax = tax_result['total_tax']
            else:
                total_tax = 0

            sample_wealth_vals.append(pre_w)
            sample_tax_vals.append(total_tax)

        sample_wealth_arr = np.array(sample_wealth_vals)
        sample_tax_arr = np.array(sample_tax_vals)

        # Sort samples by wealth for monotonic interpolation
        sort_order = np.argsort(sample_wealth_arr)
        sw_sorted = sample_wealth_arr[sort_order]
        st_sorted = sample_tax_arr[sort_order]

        # Interpolate tax for all sims based on their pre-tax wealth
        if n_sims <= MAX_TAX_SIMS:
            # Used exact values for all
            all_tax = np.zeros(n_sims)
            for i, idx in enumerate(sample_indices):
                all_tax[idx] = sample_tax_vals[i]
        else:
            all_tax = np.interp(wealth, sw_sorted, st_sorted)

        post_tax_wealths = wealth - all_tax

        # Step 2: Now take percentiles from BOTH distributions
        pcts = {}

        for pname, pval in [('p10',10), ('p25',25), ('p40',40), ('p60',60), ('p75',75), ('p90',90)]:
            pre_wealth = np.percentile(wealth, pval)
            post_wealth = np.percentile(post_tax_wealths, pval)
            tax_paid = pre_wealth - post_wealth

            pre_cagr = (pre_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1
            post_cagr = (post_wealth / cfg.INITIAL_CAPITAL) ** (1/time_horizon) - 1 if post_wealth > 0 else float('nan')

            # Tax drag: percentage of pre-tax CAGR consumed by taxes.
            # Cap at 100% - drag >100% means taxes exceed gains (tax asymmetry:
            # gains taxed immediately at full rate, losses only deductible $3k/yr).
            if pre_cagr > 0.001 and not np.isnan(post_cagr):
                drag = min(((pre_cagr - post_cagr) / pre_cagr * 100), 100.0)
            elif pre_cagr > 0.001:
                drag = 100.0  # post_cagr is nan (negative post-tax wealth)
            else:
                drag = 0  # No meaningful drag on zero/negative pre-tax returns

            pcts[pname] = {
                'pre_cagr': pre_cagr, 'post_cagr': post_cagr, 'drag': drag,
                'pre_wealth': pre_wealth, 'post_wealth': post_wealth,
                'tax_paid': tax_paid
            }

        # Get other metrics
        max_dd = np.median([r.get('Max_DD', 0) for r in results])
        trades = np.mean([r.get('Trades_Per_Year', 0) for r in results])

        # Win rate vs SPY (post-tax for taxable strategies)
        spy_results = mc_results.get('S2', [])
        spy_wealth = np.array([r['Final_Wealth'] for r in spy_results])

        if len(spy_wealth) == len(wealth) and sid in taxable_ids and spy_post_tax_cache is not None:
            # Use pre-computed SPY post-tax wealth
            win_rate = sum(w > s for w, s in zip(post_tax_wealths, spy_post_tax_cache)) / len(post_tax_wealths) * 100
        elif len(spy_wealth) == len(wealth):
            # For ROTH strategies, compare pre-tax (no tax in Roth)
            win_rate = sum(w > s for w, s in zip(wealth, spy_wealth)) / len(wealth) * 100
        else:
            win_rate = 0

        data.append({
            'id': sid, 'name': cfg.STRATEGIES[sid]['name'], 'pcts': pcts,
            'max_dd': max_dd, 'trades': trades, 'win': win_rate
        })

    # Sort by post-tax median
    data.sort(key=lambda x: x['pcts']['p60']['post_wealth'], reverse=True)

    # Print compact main table
    for i, item in enumerate(data, 1):
        pre_wealth = item['pcts']['p60']['pre_wealth']
        post_wealth = item['pcts']['p60']['post_wealth']
        post_cagr = item['pcts']['p60']['post_cagr'] * 100

        print(f"{i:<5} {item['id']:<5} {item['name']:<30} "
              f"${pre_wealth:>13,.0f} ${post_wealth:>13,.0f} {post_cagr:>11.2f}% {item['win']:>8.2f}% | "
              f"{item['max_dd']*100:>9.2f}% {item['trades']:>7.2f}")

    print("="*140)
    print("\nNote: Ranked by P60 (60th percentile) post-tax CAGR")
    print("      Median = Pre-tax CAGR -> Post-tax CAGR | Drag = Tax drag as % of pre-tax CAGR (capped at 100%)")
    print("      Negative post-tax = tax asymmetry: gains taxed at full rate, losses only deductible $3k/yr (IRC 1211)")
    print("="*140)

    # ========================================================================
    # PERCENTILE DISTRIBUTION - TOP 5 DETAILED
    # ========================================================================

    print(f"\n{'='*140}")
    print("PERCENTILE DISTRIBUTION - Top 5 Strategies (Post-Tax)")
    print(f"{'='*140}\n")

    for rank, item in enumerate(data[:5], 1):
        print(f"\nStrategy: {item['name']} ({item['id']})")
        print("-"*140)
        print(f"{'':>10} {'P10':>13} {'P25':>13} {'P40':>13} {'P60':>13} {'P75':>13} {'P90':>13}")
        print("-"*140)

        # Pre-tax row
        pre_line = f"{'Pre:':>10}"
        for pn in ['p10', 'p25', 'p40', 'p60', 'p75', 'p90']:
            d = item['pcts'][pn]
            pre_line += f" ${d['pre_wealth']:>10,.0f}"
        print(pre_line)

        pre_cagr = f"{'':>10}"
        for pn in ['p10', 'p25', 'p40', 'p60', 'p75', 'p90']:
            d = item['pcts'][pn]
            pre_cagr += f"    {d['pre_cagr']*100:>10.2f}%"
        print(pre_cagr)

        print()

        # Post-tax row
        post_line = f"{'Post:':>10}"
        for pn in ['p10', 'p25', 'p40', 'p60', 'p75', 'p90']:
            d = item['pcts'][pn]
            post_line += f" ${d['post_wealth']:>10,.0f}"
        print(post_line)

        post_cagr = f"{'':>10}"
        for pn in ['p10', 'p25', 'p40', 'p60', 'p75', 'p90']:
            d = item['pcts'][pn]
            post_cagr += f"    {d['post_cagr']*100:>10.2f}%"
        print(post_cagr)

        print()

        # Tax drag row
        drag_line = f"{'Drag:':>10}"
        for pn in ['p10', 'p25', 'p40', 'p60', 'p75', 'p90']:
            d = item['pcts'][pn]
            drag_line += f"      {d['drag']:>9.2f}%"
        print(drag_line)

        print("-"*140)

    print("\n" + "="*140)
