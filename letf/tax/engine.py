import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class RuleBasis(Enum):
    STATUTORY = "IRC/Treasury Regulation"
    HEURISTIC = "Modeling assumption"
    AMBIGUOUS = "Unclear/litigated"
    TAXPAYER_ELECTION = "Elective"

@dataclass
class TaxRule:
    name: str
    basis: RuleBasis
    citation: Optional[str] = None
    confidence: float = 1.0
    notes: Optional[str] = None

class CapitalLossUsageStrategy(Enum):
    """Legacy configuration values. Statutory ordering is mandatory."""
    MAXIMIZE_CURRENT_YEAR = "use_all_asap"
    MINIMIZE_ST_FIRST = "offset_st_first"  # Statutory safe
    MINIMIZE_LT_FIRST = "offset_lt_first"
    DEFER_TO_FUTURE = "defer_maximum"


class LotSelectionMethod(Enum):
    """
    Method for selecting which lot to sell when you own multiple lots.

    This is a taxpayer election - you can choose any method, but must be consistent
    and properly identify the lots at time of sale (per IRS rules).

    FIFO: First In, First Out (default, simplest)
    LIFO: Last In, First Out (sell newest first)
    HIFO: Highest In, First Out (sell highest cost basis first - minimizes gains)
    LOFO: Lowest In, First Out (sell lowest cost basis first - realizes gains early)
    LTFO: Long-Term First Out (sell long-term lots first - preferential tax rates)
    STFO: Short-Term First Out (sell short-term lots first)
    MINTAX: Minimum Tax (smart selection to minimize current year tax)
    SPEC_ID: Specific Identification (requires explicit lot identifiers)
    """
    FIFO = "fifo"           # First In, First Out (current default)
    LIFO = "lifo"           # Last In, First Out
    HIFO = "hifo"           # Highest cost basis first (minimize gains)
    LOFO = "lofo"           # Lowest cost basis first (realize gains)
    LTFO = "ltfo"           # Long-term lots first (preferential rates)
    STFO = "stfo"           # Short-term lots first
    MINTAX = "mintax"       # Algorithmic minimum tax selection
    SPEC_ID = "specific"    # Requires explicit lot identifiers

class AMTCreditTiming(Enum):
    USE_IMMEDIATELY = "immediate"
    DEFER_TO_LOW_INCOME = "defer_low"
    DEFER_TO_HIGH_GAINS = "defer_gains"

@dataclass
class TaxpayerElections:
    capital_loss_strategy: CapitalLossUsageStrategy = CapitalLossUsageStrategy.MINIMIZE_ST_FIRST
    amt_credit_timing: AMTCreditTiming = AMTCreditTiming.USE_IMMEDIATELY
    lot_selection_method: LotSelectionMethod = LotSelectionMethod.FIFO  # NEW: Lot selection for sales


@dataclass
class CapitalGainsResult:
    """Output from capital gains netting"""
    taxable_st: float
    taxable_lt: float
    st_loss_cf_out: float
    lt_loss_cf_out: float
    capital_loss_deduction: float

    # Audit trail
    steps: List[str] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)


def compute_capital_gains(
    st_gains: float, st_losses: float, lt_gains: float, lt_losses: float,
    st_loss_cf_in: float, lt_loss_cf_in: float, elections: TaxpayerElections,
    trace: bool = False, *, filing_status: str = 'single',
    taxable_income_before_loss: Optional[float] = None,
) -> CapitalGainsResult:
    """Schedule D netting, including carryovers in their original character.

    Sources: Schedule D lines 6/14 and Capital Loss Carryover Worksheet;
    IRC 1211(b), 1212(b), 1222. See docs/accuracy-audit/TAX_RULES.md.
    ``taxable_income_before_loss`` is the worksheet's nonnegative income
    available to absorb the deduction, before deducting the capital loss.
    If omitted, sufficient income is assumed (legacy low-level API only).
    """
    values = (st_gains, st_losses, lt_gains, lt_losses,
              st_loss_cf_in, lt_loss_cf_in)
    if any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError('Gains, losses and carryovers must be finite nonnegative amounts')
    if elections.capital_loss_strategy not in (
        CapitalLossUsageStrategy.MINIMIZE_ST_FIRST,
        CapitalLossUsageStrategy.MAXIMIZE_CURRENT_YEAR,
    ):
        raise ValueError('Capital-loss netting is statutory; deferral/character ordering is not an election')
    if filing_status not in ('single', 'married', 'married_filing_jointly',
                              'head_of_household', 'married_filing_separately'):
        raise ValueError('Unsupported filing status')
    if taxable_income_before_loss is not None and (
        not np.isfinite(taxable_income_before_loss) or taxable_income_before_loss < 0
    ):
        raise ValueError('Income available for the carryover worksheet must be nonnegative')

    net_st = st_gains - st_losses - st_loss_cf_in
    net_lt = lt_gains - lt_losses - lt_loss_cf_in
    steps = [f'Net by character including carryovers: ST {net_st}, LT {net_lt}']
    if net_st > 0 > net_lt:
        offset = min(net_st, -net_lt)
        net_st -= offset
        net_lt += offset
    elif net_lt > 0 > net_st:
        offset = min(net_lt, -net_st)
        net_lt -= offset
        net_st += offset
    limit = 1500.0 if filing_status == 'married_filing_separately' else 3000.0
    deduction = min(limit, max(0.0, -(net_st + net_lt)))
    # Low taxable income can preserve otherwise deductible losses in the
    # carryover worksheet. This is not a discretionary deferral election.
    used = deduction if taxable_income_before_loss is None else min(
        deduction, taxable_income_before_loss)
    st_loss = max(0.0, -net_st)
    lt_loss = max(0.0, -net_lt)
    st_used = min(used, st_loss)
    result = CapitalGainsResult(
        taxable_st=max(0.0, net_st), taxable_lt=max(0.0, net_lt),
        st_loss_cf_out=st_loss - st_used,
        lt_loss_cf_out=max(0.0, lt_loss - (used - st_used)),
        capital_loss_deduction=deduction, steps=steps,
        rules_applied=['IRC 1222', 'IRC 1211(b)', 'IRC 1212(b)',
                       'Schedule D lines 6/14; Capital Loss Carryover Worksheet'],
    )
    if trace:
        print('\n'.join(steps))
    return result


@dataclass
class GoldenTestCase:
    """Hand-crafted scenario with known correct outcome"""
    name: str
    description: str

    # Inputs
    st_gains: float
    st_losses: float
    lt_gains: float
    lt_losses: float
    st_carryforward_in: float
    lt_carryforward_in: float

    # Expected outputs (hand-calculated regression cases, not IRS certification)
    expected_taxable_st: float
    expected_taxable_lt: float
    expected_st_cf_out: float
    expected_lt_cf_out: float
    expected_capital_loss_deduction: float

    # Election for this test
    election_strategy: CapitalLossUsageStrategy = CapitalLossUsageStrategy.MINIMIZE_ST_FIRST

    # Statutory basis
    statutory_basis: List[str] = field(default_factory=list)

    tolerance: float = 0.01  # $0.01 tolerance

    def run(self, trace: bool = False) -> Tuple[bool, str]:
        """
        Run test against REAL engine.
        NO MOCKING.

        If this fails, the engine is broken.
        """

        elections = TaxpayerElections(capital_loss_strategy=self.election_strategy)

        actual = compute_capital_gains(
            st_gains=self.st_gains,
            st_losses=self.st_losses,
            lt_gains=self.lt_gains,
            lt_losses=self.lt_losses,
            st_loss_cf_in=self.st_carryforward_in,
            lt_loss_cf_in=self.lt_carryforward_in,
            elections=elections,
            trace=trace
        )

        # Validate against expected
        checks = [
            ('taxable_st', self.expected_taxable_st, actual.taxable_st),
            ('taxable_lt', self.expected_taxable_lt, actual.taxable_lt),
            ('st_cf_out', self.expected_st_cf_out, actual.st_loss_cf_out),
            ('lt_cf_out', self.expected_lt_cf_out, actual.lt_loss_cf_out),
            ('capital_loss_deduction', self.expected_capital_loss_deduction,
             actual.capital_loss_deduction)
        ]

        failures = []
        for name, expected, actual_val in checks:
            if abs(expected - actual_val) > self.tolerance:
                failures.append(
                    f"  {name}: expected ${expected:,.2f}, got ${actual_val:,.2f} "
                    f"(diff ${abs(expected - actual_val):,.2f})"
                )

        if failures:
            msg = f"FAILED: {self.name}\n" + "\n".join(failures)
            if trace:
                msg += "\n\nTrace:\n" + "\n".join(actual.steps)
            return False, msg
        else:
            return True, f"PASSED: {self.name}"


# Regression cases; original baseline preserved in docs/accuracy-audit/baseline.json
GOLDEN_TESTS = [
    GoldenTestCase(
        name="Basic Netting",
        description="Simple gains and losses, no carryforwards",
        st_gains=50000, st_losses=10000,
        lt_gains=20000, lt_losses=5000,
        st_carryforward_in=0, lt_carryforward_in=0,
        expected_taxable_st=40000,
        expected_taxable_lt=15000,
        expected_st_cf_out=0,
        expected_lt_cf_out=0,
        expected_capital_loss_deduction=0,
        statutory_basis=["IRC §1222"]
    ),

    GoldenTestCase(
        name="$3k Loss Deduction",
        description="Net loss allows $3k deduction, rest carries",
        st_gains=5000, st_losses=20000,
        lt_gains=0, lt_losses=0,
        st_carryforward_in=0, lt_carryforward_in=0,
        # Current: ST -15k
        # Use $3k deduction
        # Carry forward: $12k ST
        expected_taxable_st=0,
        expected_taxable_lt=0,
        expected_st_cf_out=12000,
        expected_lt_cf_out=0,
        expected_capital_loss_deduction=3000,
        statutory_basis=["IRC §1211(b)"]
    ),

    GoldenTestCase(
        name="Cross-Offset Current Year",
        description="ST gains offset by LT losses (current year)",
        st_gains=50000, st_losses=0,
        lt_gains=0, lt_losses=30000,
        st_carryforward_in=0, lt_carryforward_in=0,
        # Current: ST +50k, LT -30k
        # Cross-net: ST becomes +20k, LT becomes 0
        expected_taxable_st=20000,
        expected_taxable_lt=0,
        expected_st_cf_out=0,
        expected_lt_cf_out=0,
        expected_capital_loss_deduction=0,
        statutory_basis=["IRC §1222", "Treas. Reg. §1.1222-1"]
    ),

    GoldenTestCase(
        name="Carryforward Application Order",
        description="Carryovers enter Schedule D by character before cross-netting",
        st_gains=100000, st_losses=0,
        lt_gains=0, lt_losses=60000,
        st_carryforward_in=25000, lt_carryforward_in=15000,
        # Current year: ST +100k, LT -60k
        # Cross-net current: ST +40k, LT 0
        # Apply ST CF: ST +40k - 25k = +15k
        # Apply LT CF (cross): ST +15k - 15k = 0
        expected_taxable_st=0,
        expected_taxable_lt=0,
        expected_st_cf_out=0,
        expected_lt_cf_out=0,
        expected_capital_loss_deduction=0,
        statutory_basis=["IRC §1212", "Rev. Rul. 84-8"]
    ),

    GoldenTestCase(
        name="Large Loss Year",
        description="Massive loss generates large carryforward",
        st_gains=10000, st_losses=500000,
        lt_gains=5000, lt_losses=200000,
        st_carryforward_in=0, lt_carryforward_in=0,
        # Current: ST -490k, LT -195k
        # No cross-netting needed (both losses)
        # Total loss: -685k
        # Use $3k deduction (from ST first)
        # Carry forward: ST -487k, LT -195k
        expected_taxable_st=0,
        expected_taxable_lt=0,
        expected_st_cf_out=487000,
        expected_lt_cf_out=195000,
        expected_capital_loss_deduction=3000,
        statutory_basis=["IRC §1211(b)", "IRC §1212"]
    ),

    GoldenTestCase(
        name="Same-character carryovers",
        description="Both characters remain gains after their own carryovers",
        st_gains=50000, st_losses=0,
        lt_gains=30000, lt_losses=0,
        st_carryforward_in=40000, lt_carryforward_in=25000,
        election_strategy=CapitalLossUsageStrategy.MINIMIZE_ST_FIRST,
        # With DEFER: only offset to zero, don't cross-apply
        # ST: +50k - 40k CF = +10k taxable
        # LT: +30k - 25k CF = +5k taxable
        # No cross-application
        expected_taxable_st=10000,
        expected_taxable_lt=5000,
        expected_st_cf_out=0,
        expected_lt_cf_out=0,
        expected_capital_loss_deduction=0,
        statutory_basis=["IRC §1212(b); Schedule D lines 6 and 14"]
    )
]


def run_golden_tests(trace_failures: bool = False) -> Dict:
    """
    Run all golden tests against REAL engine.

    If ANY test fails, system is broken.
    """

    results = {
        'total': len(GOLDEN_TESTS),
        'passed': 0,
        'failed': 0,
        'details': []
    }

    print("\n" + "="*80)
    print("GOLDEN-CASE REGRESSION TESTS (WIRED TO REAL ENGINE)")
    print("="*80)
    print(f"Running {len(GOLDEN_TESTS)} hand-crafted test cases...")
    print("NO MOCKING - Tests can actually fail\n")

    for test in GOLDEN_TESTS:
        passed, message = test.run(trace=trace_failures and results['failed'] == 0)

        results['details'].append({
            'test': test.name,
            'passed': passed,
            'message': message
        })

        if passed:
            results['passed'] += 1
            print(f"  PASS: {test.name}")
        else:
            results['failed'] += 1
            print(f"  FAIL: {test.name}")
            print(message)

    print("\n" + "="*80)
    print(f"RESULTS: {results['passed']}/{results['total']} passed")
    if results['failed'] > 0:
        print(f"CRITICAL: {results['failed']} TESTS FAILED - SYSTEM BROKEN")
        print(f"DO NOT USE UNTIL ALL TESTS PASS")
    else:
        print("Covered regression cases passed; this is not comprehensive tax certification")
        print("See docs/accuracy-audit/TAX_RULES.md for scope and limitations")
    print("="*80)

    if results['failed']:
        raise AssertionError(f"{results['failed']} capital-gains regression cases failed")
    return results
