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
    taxable_st: float, taxable_lt: float, capital_loss_deduction: float,
    ordinary_income: float = 0, include_state: bool = True,
    include_niit: bool = True, filing_status: str = 'single',
    bracket_multiplier: float = 1.0, state_code: str = None, *,
    tax_year: int = 2026, qualified_dividends: float = 0,
    ordinary_dividends: float = 0, interest_income: float = 0,
    itemized_deduction: float = None, state_inputs: Dict = None,
    section199a_dividends: float = 0,
) -> Dict:
    """Regular federal income tax + NIIT for the explicitly supported scope.

    No AMT, credits, foreign tax, collectibles/1250 rates, SALT optimization,
    or investment-interest election is inferred. State models fail closed
    unless independently implemented. See TAX_RULES.md for restrictions.
    ``ordinary_dividends`` means NONQUALIFIED dividends only (1099-DIV box
    1a minus box 1b), so qualified income is not counted a second time.
    """
    import math
    from letf.tax.years import federal_parameters
    from letf.accuracy import UnsupportedModelError
    values = (taxable_st, taxable_lt, capital_loss_deduction, ordinary_income,
              qualified_dividends, ordinary_dividends, interest_income, section199a_dividends)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Income inputs must be finite and nonnegative')
    if bracket_multiplier != 1.0:
        raise UnsupportedModelError('Use an enacted tax-year table; bracket extrapolation is hypothetical')
    ordinary_brackets, ltcg_brackets, standard, threshold = federal_parameters(tax_year, filing_status)
    loss_limit = 1500 if filing_status == 'married_filing_separately' else 3000
    if capital_loss_deduction > loss_limit:
        raise ValueError('Capital loss deduction exceeds statutory limit')
    if itemized_deduction is not None and (not math.isfinite(itemized_deduction) or itemized_deduction < 0):
        raise ValueError('Itemized deduction must be a verified nonnegative amount')
    deduction = standard if itemized_deduction is None else max(standard, itemized_deduction)
    if section199a_dividends and tax_year not in (2024,2025):
        raise UnsupportedModelError('Qualified RIC/REIT dividend deduction verified only for 2024-25')
    ordinary_gross = ordinary_income + ordinary_dividends + interest_income + section199a_dividends
    # A capital-loss deduction can also reduce AGI consisting of qualified
    # dividends. Preserve a negative ordinary component for that allocation.
    ordinary_base = ordinary_gross - capital_loss_deduction
    with_st = ordinary_base + taxable_st
    preferred = taxable_lt + qualified_dividends
    # The part of the deduction unused by ordinary income shelters preferred
    # income too (Qualified Dividends and Capital Gain Tax Worksheet).
    ordinary_taxable = max(0., with_st - deduction)
    preferred_taxable = max(0., preferred - max(0., deduction - with_st))
    # Form 8995 REIT/RIC-only subset: no business/PTP losses or other QBI.
    # Below-line deduction does not reduce AGI/NIIT or MA income.
    qbi_deduction=min(.20*section199a_dividends,
        .20*max(0.,ordinary_taxable+preferred_taxable-preferred))
    ordinary_taxable-=qbi_deduction
    federal_ordinary = calculate_marginal_tax(ordinary_base, ordinary_brackets, deduction+qbi_deduction)
    federal_st = calculate_marginal_tax(with_st, ordinary_brackets, deduction+qbi_deduction) - federal_ordinary
    federal_ltcg, breakdown = calculate_ltcg_tax_stacked(
        preferred_taxable, ordinary_taxable, ltcg_brackets)
    federal_total = federal_ordinary + federal_st + federal_ltcg
    magi = ordinary_gross + taxable_st + preferred - capital_loss_deduction
    investment_income = max(0., taxable_st + preferred + ordinary_dividends + interest_income + section199a_dividends
                             - capital_loss_deduction)
    niit_tax = .038 * min(investment_income, max(0., magi - threshold)) if include_niit else 0.
    state_tax = 0.
    state_calculation = None
    if include_state:
        if state_code is None:
            raise UnsupportedModelError('Select a state explicitly, or request federal-only tax')
        # No general individual income tax in these three states. Other
        # states require dated capital-gain, conformity and surcharge rules.
        if state_code == 'MA':
            from letf.tax.massachusetts import massachusetts_tax
            if state_inputs is None and capital_loss_deduction:
                raise UnsupportedModelError('MA losses require raw activity and independent state carryovers')
            activity = state_inputs if state_inputs is not None else dict(st_gains=taxable_st, lt_gains=taxable_lt)
            state_calculation = massachusetts_tax(tax_year, ordinary_income, interest=interest_income,
                dividends=ordinary_dividends + qualified_dividends + section199a_dividends, filing_status=filing_status, **activity)
            state_tax = state_calculation['total_tax']
        elif state_code not in ('TX', 'FL', 'NV'):
            raise UnsupportedModelError(f'No validated state model for {state_code}/{tax_year}')
    total_tax = federal_total + niit_tax + state_tax
    return {
        'federal_ordinary': federal_ordinary, 'federal_st': federal_st,
        'federal_ltcg': federal_ltcg, 'federal_total': federal_total,
        'state_tax': state_tax, 'state_calculation': state_calculation, 'niit_tax': niit_tax, 'total_tax': total_tax,
        'effective_rate': total_tax / magi if magi > 0 else 0.,
        'marginal_rate_used': True, 'ltcg_breakdown': breakdown,
        'ltcg_stacking_used': True, 'tax_year': tax_year,
        'scope': 'regular federal tax and NIIT; excludes AMT/credits/special-rate gains',
        'state_scope': state_code if include_state else 'FEDERAL ONLY',
        'section199a_dividend_deduction':qbi_deduction,
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
