from typing import List, Dict, Tuple
from letf.tax.brackets import (
    FEDERAL_TAX_BRACKETS_2024, LTCG_BRACKETS_2024,
    STANDARD_DEDUCTION_2024, NIIT_THRESHOLD_2024,
    STATE_TAX_BRACKETS, NIIT_RATE
)


def calculate_marginal_tax(income: float, brackets: List[Tuple[float, float]],
                           standard_deduction: float = 0) -> float:
    """
    Calculate tax using proper marginal brackets.

    THIS IS THE CORRECT WAY - NOT FLAT RATES!

    Args:
        income: Gross income
        brackets: List of (upper_limit, rate) tuples
        standard_deduction: Amount to deduct before calculating tax

    Returns:
        Tax liability
    """
    if income <= 0:
        return 0

    # Apply standard deduction
    taxable_income = max(0, income - standard_deduction)

    tax = 0
    prev_bracket = 0

    for bracket_limit, rate in brackets:
        if taxable_income <= prev_bracket:
            break

        amount_in_bracket = min(taxable_income, bracket_limit) - prev_bracket
        tax += amount_in_bracket * rate
        prev_bracket = bracket_limit

        if taxable_income <= bracket_limit:
            break

    return tax


def calculate_ltcg_tax_stacked(
    ltcg_amount: float,
    ordinary_income_after_deduction: float,
    ltcg_brackets: List[Tuple[float, float]]
) -> Tuple[float, Dict]:
    """
    Calculate LTCG tax with proper bracket stacking.

    LTCG is "stacked" on top of ordinary income and taxed through each bracket
    progressively, just like ordinary income tax works.

    This is the CORRECT method per IRS rules. The old method of applying a single
    flat rate based on total income is WRONG.

    Args:
        ltcg_amount: Total long-term capital gains to be taxed
        ordinary_income_after_deduction: Taxable ordinary income (after std deduction)
                                         This determines where LTCG "starts" in brackets
        ltcg_brackets: List of (upper_limit, rate) tuples for LTCG
                       e.g., [(47025, 0.00), (518900, 0.15), (inf, 0.20)]

    Returns:
        Tuple of (total_ltcg_tax, breakdown_dict)
        breakdown_dict contains details for each bracket

    Example:
        Single filer, $40,000 ordinary income, $50,000 LTCG
        Brackets: [(47025, 0.00), (518900, 0.15), (inf, 0.20)]

        LTCG stacks starting at $40,000:
        - $7,025 fills 0% bracket ($40k -> $47,025) = $0
        - $42,975 fills 15% bracket ($47,025 -> $90k) = $6,446.25
        Total LTCG tax = $6,446.25
    """
    if ltcg_amount <= 0:
        return 0.0, {'breakdown': [], 'total': 0.0}

    # Where LTCG starts stacking (ordinary income fills up to here)
    stack_start = max(0, ordinary_income_after_deduction)

    # Track tax and breakdown
    total_tax = 0.0
    breakdown = []
    remaining_ltcg = ltcg_amount
    current_position = stack_start

    for i, (bracket_ceiling, rate) in enumerate(ltcg_brackets):
        if remaining_ltcg <= 0:
            break

        # Skip brackets that ordinary income has already filled past
        if current_position >= bracket_ceiling:
            continue

        # How much room is left in this bracket?
        room_in_bracket = bracket_ceiling - current_position

        # How much LTCG goes into this bracket?
        ltcg_in_bracket = min(remaining_ltcg, room_in_bracket)

        # Calculate tax for this portion
        tax_in_bracket = ltcg_in_bracket * rate

        # Record breakdown
        breakdown.append({
            'bracket_num': i + 1,
            'bracket_ceiling': bracket_ceiling,
            'rate': rate,
            'rate_pct': f"{rate*100:.0f}%",
            'ltcg_in_bracket': ltcg_in_bracket,
            'tax_in_bracket': tax_in_bracket,
            'income_range': (current_position, current_position + ltcg_in_bracket)
        })

        # Update totals
        total_tax += tax_in_bracket
        remaining_ltcg -= ltcg_in_bracket
        current_position += ltcg_in_bracket

    return total_tax, {
        'breakdown': breakdown,
        'total': total_tax,
        'ordinary_income_base': stack_start,
        'ltcg_amount': ltcg_amount,
        'effective_ltcg_rate': (total_tax / ltcg_amount) if ltcg_amount > 0 else 0
    }


def calculate_comprehensive_tax_v6(
    taxable_st: float,
    taxable_lt: float,
    capital_loss_deduction: float,
    ordinary_income: float = 0,
    include_state: bool = True,
    include_niit: bool = True,
    filing_status: str = 'single',
    bracket_multiplier: float = 1.0,  # Inflate brackets for future years
    state_code: str = 'CA'  # State for tax calculation (was hardcoded to CA)
) -> Dict:
    """
    Calculate ACTUAL tax liability with proper marginal rates.

    THIS REPLACES THE BROKEN FLAT-RATE CALCULATION.

    Args:
        taxable_st: Short-term capital gains
        taxable_lt: Long-term capital gains
        capital_loss_deduction: Capital loss deduction (reduces ordinary income)
        ordinary_income: W-2, interest, etc.
        include_state: Include state tax
        include_niit: Include Net Investment Income Tax (3.8%)
        bracket_multiplier: Multiplier for bracket thresholds (for inflation adjustment)

    Returns:
        Dict with federal, state, NIIT, and total tax
    """
    ordinary_brackets = FEDERAL_TAX_BRACKETS_2024.get(filing_status, FEDERAL_TAX_BRACKETS_2024['single'])
    std_deduction = STANDARD_DEDUCTION_2024.get(filing_status, STANDARD_DEDUCTION_2024['single'])
    ltcg_brackets = LTCG_BRACKETS_2024.get(filing_status, LTCG_BRACKETS_2024['single'])
    state_data = STATE_TAX_BRACKETS.get(state_code, STATE_TAX_BRACKETS['CA'])
    state_brackets = state_data.get(filing_status, state_data['single'])
    state_std_ded = state_data['std_deduction'].get(filing_status, state_data['std_deduction']['single'])
    niit_threshold = NIIT_THRESHOLD_2024.get(filing_status, NIIT_THRESHOLD_2024['single'])

    # Apply bracket inflation if needed (for future years)
    if bracket_multiplier != 1.0:
        # Inflate ordinary income brackets
        ordinary_brackets = [(threshold * bracket_multiplier, rate)
                             for threshold, rate in ordinary_brackets]

        # Inflate LTCG brackets
        ltcg_brackets = [(threshold * bracket_multiplier, rate)
                         for threshold, rate in ltcg_brackets]

        # Inflate standard deduction
        std_deduction = std_deduction * bracket_multiplier

        # Inflate NIIT threshold
        niit_threshold = niit_threshold * bracket_multiplier

        # Inflate state brackets
        state_brackets = [(threshold * bracket_multiplier, rate)
                          for threshold, rate in state_brackets]
        state_std_ded = state_std_ded * bracket_multiplier

    # Ordinary income (includes W-2, interest, capital loss deduction offset)
    # Capital loss deduction REDUCES ordinary income
    adjusted_ordinary = max(0, ordinary_income - capital_loss_deduction)

    # Federal tax on ordinary income (progressive brackets)
    federal_ordinary = calculate_marginal_tax(
        adjusted_ordinary,
        ordinary_brackets,
        std_deduction
    )

    # Short-term capital gains taxed as ordinary income
    # Stack on top of ordinary income for proper marginal rate
    total_ordinary_income = adjusted_ordinary + taxable_st
    federal_with_st = calculate_marginal_tax(
        total_ordinary_income,
        ordinary_brackets,
        std_deduction
    )
    federal_st_tax = federal_with_st - federal_ordinary

    # Long-term capital gains (preferential rates)
    # Based on total income (ordinary + LTCG)
    total_income = total_ordinary_income + taxable_lt

    # ========================================================================
    # LTCG TAX WITH PROPER BRACKET STACKING
    # ========================================================================
    # LTCG is "stacked" on top of ordinary income (including ST gains) and
    # taxed progressively through each LTCG bracket.
    #
    # This is the CORRECT method. The old flat-rate method was WRONG.
    #
    # Example: Single, $40k ordinary income, $50k LTCG
    #   - First $7,025 of LTCG fills 0% bracket ($40k -> $47,025) = $0
    #   - Remaining $42,975 at 15% ($47,025 -> $90k) = $6,446
    #   - Total: $6,446 (NOT $7,500 from flat 15% rate!)

    # The "base" for LTCG stacking is ordinary income AFTER standard deduction
    # (This includes both W-2 income and short-term capital gains)
    ordinary_income_for_ltcg_stacking = max(0, total_ordinary_income - std_deduction)

    federal_ltcg_tax = 0
    ltcg_breakdown = None

    if taxable_lt > 0:
        federal_ltcg_tax, ltcg_breakdown = calculate_ltcg_tax_stacked(
            ltcg_amount=taxable_lt,
            ordinary_income_after_deduction=ordinary_income_for_ltcg_stacking,
            ltcg_brackets=ltcg_brackets
        )

    federal_total = federal_ordinary + federal_st_tax + federal_ltcg_tax

    # State tax
    state_tax = 0
    if include_state:
        if state_data.get('cap_gains_only', False):
            # States like WA: tax applies ONLY to capital gains, not ordinary income.
            # The $250K threshold is a separate deduction against capital gains alone.
            cap_gains = taxable_st + taxable_lt
            state_tax = calculate_marginal_tax(cap_gains, state_brackets, state_std_ded)
        else:
            # Most states (CA, NY, NJ, IL, MA): all income taxed through state brackets
            state_income = total_income
            state_tax = calculate_marginal_tax(state_income, state_brackets, state_std_ded)

    # NIIT (3.8% on investment income over threshold)
    niit_tax = 0
    if include_niit and total_income > niit_threshold:
        investment_income = taxable_st + taxable_lt
        niit_base = min(investment_income, total_income - niit_threshold)
        niit_tax = niit_base * NIIT_RATE

    total_tax = federal_total + state_tax + niit_tax

    return {
        'federal_ordinary': federal_ordinary,
        'federal_st': federal_st_tax,
        'federal_ltcg': federal_ltcg_tax,
        'federal_total': federal_total,
        'state_tax': state_tax,
        'niit_tax': niit_tax,
        'total_tax': total_tax,
        'effective_rate': (total_tax / total_income) if total_income > 0 else 0,
        'marginal_rate_used': True,
        # NEW: LTCG bracket breakdown for analysis
        'ltcg_breakdown': ltcg_breakdown,
        'ltcg_stacking_used': True
    }


def test_ltcg_stacking():
    """
    Test that LTCG stacking is working correctly.

    This verifies that LTCG is taxed progressively through brackets,
    not at a flat rate based on total income.
    """
    print(f"\n{'='*80}")
    print("TEST: LTCG BRACKET STACKING")
    print(f"{'='*80}\n")

    # Test case: Single filer, $40k ordinary income, $50k LTCG
    # LTCG brackets for single: [(47025, 0.00), (518900, 0.15), (inf, 0.20)]

    ordinary_income = 40000
    ltcg = 50000
    std_deduction = 14600  # 2024 single

    # Calculate using stacked method
    ordinary_after_deduction = max(0, ordinary_income - std_deduction)  # $25,400

    ltcg_brackets = [(47025, 0.00), (518900, 0.15), (float('inf'), 0.20)]

    tax, breakdown = calculate_ltcg_tax_stacked(
        ltcg_amount=ltcg,
        ordinary_income_after_deduction=ordinary_after_deduction,
        ltcg_brackets=ltcg_brackets
    )

    print(f"  Test Case: Single filer")
    print(f"    Ordinary income: ${ordinary_income:,}")
    print(f"    Standard deduction: ${std_deduction:,}")
    print(f"    Ordinary after deduction: ${ordinary_after_deduction:,}")
    print(f"    LTCG: ${ltcg:,}")
    print()

    # Expected calculation:
    # Ordinary fills up to $25,400
    # LTCG stacks from $25,400:
    #   - $21,625 fills 0% bracket ($25,400 -> $47,025) = $0
    #   - $28,375 fills 15% bracket ($47,025 -> $75,400) = $4,256.25
    # Total = $4,256.25

    expected_0pct_portion = 47025 - ordinary_after_deduction  # $21,625
    expected_15pct_portion = ltcg - expected_0pct_portion  # $28,375
    expected_tax = expected_0pct_portion * 0.00 + expected_15pct_portion * 0.15

    print(f"  EXPECTED (manual calculation):")
    print(f"    LTCG in 0% bracket: ${expected_0pct_portion:,.0f} × 0% = $0")
    print(f"    LTCG in 15% bracket: ${expected_15pct_portion:,.0f} × 15% = ${expected_15pct_portion * 0.15:,.2f}")
    print(f"    Total expected: ${expected_tax:,.2f}")
    print()

    print(f"  ACTUAL (from stacking function):")
    print(f"    Total LTCG tax: ${tax:,.2f}")
    if breakdown and 'breakdown' in breakdown:
        for b in breakdown['breakdown']:
            print(f"    Bracket {b['bracket_num']} ({b['rate_pct']}): "
                  f"${b['ltcg_in_bracket']:,.0f} -> ${b['tax_in_bracket']:,.2f}")
    print(f"    Effective LTCG rate: {breakdown['effective_ltcg_rate']*100:.2f}%")
    print()

    # OLD (WRONG) method for comparison
    total_income = ordinary_income + ltcg - std_deduction  # $75,400
    if total_income > 47025:
        old_rate = 0.15
    else:
        old_rate = 0.00
    old_tax = ltcg * old_rate

    print(f"  OLD METHOD (flat rate - WRONG):")
    print(f"    Total income: ${total_income:,}")
    print(f"    Rate used: {old_rate*100:.0f}% (because total > $47,025)")
    print(f"    Tax: ${ltcg:,} × {old_rate*100:.0f}% = ${old_tax:,.2f}")
    print()

    # Compare
    difference = old_tax - tax
    print(f"  COMPARISON:")
    print(f"    Old method (wrong): ${old_tax:,.2f}")
    print(f"    New method (correct): ${tax:,.2f}")
    print(f"    Difference (tax savings): ${difference:,.2f}")
    print()

    # Verify
    tolerance = 0.01
    if abs(tax - expected_tax) < tolerance:
        print(f"  [OK] TEST PASSED: Stacking calculation is correct!")
    else:
        print(f"  [FAIL] TEST FAILED: Expected ${expected_tax:,.2f}, got ${tax:,.2f}")

    return abs(tax - expected_tax) < tolerance
