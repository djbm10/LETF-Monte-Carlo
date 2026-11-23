from datetime import date

import pytest

from letf.accounts import PortfolioLedger, Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.tax.engine import (compute_capital_gains, TaxpayerElections,
                             CapitalLossUsageStrategy, run_golden_tests)
from letf.tax.marginal import calculate_comprehensive_tax_v6 as tax
from letf.tax.roth import contribution_limit, RothHistory, Conversion
from letf.accuracy import UnsupportedModelError


def ledger(cash=100000):
    result = PortfolioLedger()
    result.add_account('t', TAXABLE_CASH, cash)
    result.add_account('r', CONSERVATIVE_ROTH, cash)
    result.add_security(Security('X', kind='stock'))
    return result


def test_existing_regression_outcomes():
    assert run_golden_tests()['passed'] == 6


def test_carryovers_keep_character_before_cross_netting():
    # Schedule D lines 6/14: ST is zero, LT is -10k, then $3k used.
    result = compute_capital_gains(10000, 0, 0, 10000, 10000, 0, TaxpayerElections())
    assert result.lt_loss_cf_out == 7000
    assert result.st_loss_cf_out == 0
    assert result.capital_loss_deduction == 3000


def test_carry_only_year_uses_loss_deduction():
    result = compute_capital_gains(0, 0, 0, 0, 5000, 0, TaxpayerElections())
    assert result.capital_loss_deduction == 3000
    assert result.st_loss_cf_out == 2000


def test_low_income_carryover_worksheet_preserves_unabsorbed_losses():
    result = compute_capital_gains(0, 0, 0, 0, 5000, 0, TaxpayerElections(),
                                   taxable_income_before_loss=1000)
    assert result.st_loss_cf_out == 4000


def test_separate_filer_loss_limit():
    result = compute_capital_gains(0, 5000, 0, 0, 0, 0, TaxpayerElections(),
                                   filing_status='married_filing_separately')
    assert result.capital_loss_deduction == 1500
    assert result.st_loss_cf_out == 3500


@pytest.mark.parametrize('election', [CapitalLossUsageStrategy.DEFER_TO_FUTURE,
                                    CapitalLossUsageStrategy.MINIMIZE_LT_FIRST])
def test_nonstatutory_elections_rejected(election):
    with pytest.raises(ValueError, match='not an election'):
        compute_capital_gains(0, 0, 0, 0, 0, 0, TaxpayerElections(capital_loss_strategy=election))


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
def test_negative_or_nonfinite_loss_inputs_rejected(bad):
    with pytest.raises(ValueError):
        compute_capital_gains(0, bad, 0, 0, 0, 0, TaxpayerElections())


def test_standard_deduction_shelters_preferred_income():
    assert tax(0, 60000, 0, include_state=False, tax_year=2024)['total_tax'] == 0
    assert tax(0, 70000, 0, include_state=False, tax_year=2026)['total_tax'] == pytest.approx(667.5)
    assert tax(0, 70000, 3000, include_state=False, tax_year=2026)['total_tax'] == pytest.approx(217.5)


def test_wash_facade_is_idempotent_and_requires_calendar_dates():
    from letf.tax.wash_sale import WashSaleTracker
    tracker = WashSaleTracker()
    with pytest.raises(UnsupportedModelError, match='calendar'):
        tracker.record_trade('X', 0, 'BUY', 100, 10)
    tracker.record_trade('X', '2024-01-01', 'BUY', 100, 10)
    tracker.record_trade('X', '2024-12-20', 'SELL', 100, 6)
    tracker.record_trade('X', '2025-01-01', 'BUY', 50, 6)
    tracker.process_all_wash_sales()
    tracker.process_all_wash_sales()
    assert tracker.get_total_disallowed() == 200
    assert tracker.get_total_allowed() == 200
    assert tracker.get_disallowed_for_year('X', 2024) == 200
    assert tracker.get_basis_adjustment('X', '2025-01-01') == 200


@pytest.mark.parametrize('year, deduction, bottom', [(2024, 14600, 11600),
                                                   (2025, 15750, 11925), (2026, 16100, 12400)])
def test_dated_bottom_brackets(year, deduction, bottom):
    assert tax(0, 0, 0, ordinary_income=deduction + bottom, include_state=False,
               tax_year=year)['federal_total'] == pytest.approx(.1 * bottom)


def test_niit_uses_interest_dividends_and_fixed_threshold():
    result = tax(0, 0, 0, ordinary_income=199000, interest_income=2000,
                 include_state=False, tax_year=2026)
    assert result['niit_tax'] == 38
    with pytest.raises(UnsupportedModelError):
        tax(0, 0, 0, include_state=False, bracket_multiplier=2)


@pytest.mark.parametrize('kwargs', [{'state_code': None}, {'state_code': 'XX'},
                                   {'state_code': 'CA'}, {'include_state': False, 'tax_year': 1973},
                                   {'include_state': False, 'filing_status': 'nonsense'}])
def test_unsupported_tax_configuration_fails(kwargs):
    with pytest.raises(UnsupportedModelError):
        tax(0, 0, 0, **kwargs)


def test_irs_550_partial_replacement_example_1():
    # Pub 550, Wash Sales, More or less stock bought than sold, Example 1.
    book = ledger()
    book.execute('2024-09-20', 't', 'X', 'BUY', 100, 50)
    book.execute('2024-12-13', 't', 'X', 'BUY', 50, 55)
    book.execute('2024-12-20', 't', 'X', 'BUY', 25, 45)
    book.execute('2025-01-03', 't', 'X', 'SELL', 100, 40)
    sale = book.realizations[0]
    assert sale.disallowed == 750
    assert sale.gain == -250
    remaining = [l for l in book.lots if l.shares > 0]
    assert [l.shares * l.basis_per_share for l in remaining] == [3250, 1375]


def test_irs_550_partial_replacement_example_2():
    book = ledger()
    book.execute('2024-09-16', 't', 'X', 'BUY', 100, 50)
    book.execute('2025-01-29', 't', 'X', 'SELL', 100, 40)
    for day in range(3, 7):
        book.execute(f'2025-02-0{day}', 't', 'X', 'BUY', 50, 40)
    assert book.realizations[0].disallowed == 1000
    assert [l.basis_per_share for l in book.lots[1:]] == [50, 50, 40, 40]


def test_irs_2008_5_roth_replacement_permanently_disallowed():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 100, 10)
    book.execute('2024-12-20', 't', 'X', 'SELL', 100, 6)
    book.execute('2024-12-21', 'r', 'X', 'BUY', 100, 6)
    assert book.realizations[0].gain == 0
    assert book.realizations[0].permanent_disallowed == 400
    assert book.lots[-1].basis_per_share == 6  # No IRA basis increase.


def test_roth_internal_trades_have_no_annual_personal_tax():
    book = ledger()
    book.execute('2024-01-01', 'r', 'X', 'BUY', 100, 10)
    book.execute('2024-12-20', 'r', 'X', 'SELL', 100, 20)
    assert not book.realizations
    assert book.annual_tax('r', 2024, 100000)['tax'] == 0


@pytest.mark.parametrize('offset, disallowed', [(0, 100), (30, 100), (31, 0)])
def test_calendar_window_includes_same_day_and_day_30(offset, disallowed):
    from datetime import timedelta
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-12-20', 't', 'X', 'SELL', 10, 90)
    book.execute(date(2024, 12, 20) + timedelta(days=offset), 't', 'X', 'BUY', 10, 90)
    assert book.realizations[0].disallowed == disallowed


def test_sold_original_lot_is_not_its_own_replacement():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-01-02', 't', 'X', 'SELL', 10, 90)
    assert book.realizations[0].disallowed == 0


def test_tax_loss_harvesting_preserves_wash_rules_and_cash():
    book = ledger()
    book.add_security(Security('Y', kind='stock'))
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    sales = book.harvest_losses('2024-12-01', 't', {'X': 90, 'Y': 45},
                                minimum_loss=50, replacements={'X': 'Y'})
    assert sales[0].gain == -100
    assert book.quantity('t', 'X') == 0 and book.quantity('t', 'Y') == 20
    book.execute('2024-12-02', 'r', 'X', 'BUY', 5, 90)
    assert sales[0].gain == -50 and sales[0].permanent_disallowed == 50


def test_gain_lots_are_consumed_and_loss_matching_is_per_lot():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 50)
    book.execute('2024-01-02', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-06-01', 't', 'X', 'SELL', 10, 80)
    book.execute('2024-12-20', 't', 'X', 'SELL', 10, 80)
    book.execute('2024-12-21', 't', 'X', 'BUY', 5, 80)
    assert [r.gain for r in book.realizations] == [300, -100]
    assert book.quantity('t', 'X') == 5


def test_split_replacement_lots_have_distinct_basis_and_tacking():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-12-20', 't', 'X', 'SELL', 10, 90)
    book.execute('2024-12-21', 't', 'X', 'BUY', 20, 90)
    live = [l for l in book.lots if l.shares]
    assert [(l.shares, l.basis_per_share) for l in live] == [(10, 100), (10, 90)]
    assert live[0].tacked_days == 354
    assert live[1].tacked_days == 0


def test_pending_year_end_tax_rejected_then_cross_year_match():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-12-20', 't', 'X', 'SELL', 10, 90)
    with pytest.raises(ValueError, match='provisional'):
        book.annual_tax('t', 2024, 100000, federal_only=True, as_of='2024-12-31')
    book.execute('2025-01-10', 't', 'X', 'BUY', 5, 90)
    result = book.annual_tax('t', 2024, 100000, federal_only=True, as_of='2025-01-31')
    assert result['tax'] == pytest.approx(-11.)  # $50 loss saves 22%.


def test_holding_period_calendar_anniversary_in_leap_year():
    book = ledger()
    book.execute('2023-03-01', 't', 'X', 'BUY', 2, 100)
    book.execute('2024-03-01', 't', 'X', 'SELL', 1, 200)
    book.execute('2024-03-02', 't', 'X', 'SELL', 1, 200)
    assert [r.long_term for r in book.realizations] == [False, True]


def test_loss_tax_benefit_not_cancelled_in_baseline():
    book = ledger()
    book.execute('2024-01-01', 't', 'X', 'BUY', 10, 100)
    book.execute('2024-12-20', 't', 'X', 'SELL', 10, 90)
    assert book.annual_tax('t', 2024, 100000, federal_only=True,
                            as_of='2025-02-01')['tax'] == pytest.approx(-22.)


@pytest.mark.parametrize('year, age, expected', [(2024, 49, 7000), (2025, 50, 8000),
                                               (2026, 49, 7500), (2026, 50, 8600)])
def test_roth_dated_limits(year, age, expected):
    assert contribution_limit(year, age, 50000, 100000) == expected


def test_roth_phaseout_and_shared_ira_cap():
    assert contribution_limit(2026, 30, 160500, 100000) == 3750
    assert contribution_limit(2026, 30, 168000, 100000) == 0
    assert contribution_limit(2026, 30, 160500, 100000, other_ira_contributions=5000) == 2500
    assert contribution_limit(2026, 30, 167999, 100000) == 200
    assert contribution_limit(2026, 30, 10000, 500) == 500


def test_roth_contribution_excess_and_unqualified_ordering():
    book = ledger()
    history = RothHistory(book, 'owner', '1990-01-01')
    history.contribute('2024-01-01', 'r', 7000, tax_year=2024, magi=100000, compensation=100000)
    with pytest.raises(ValueError):
        history.contribute('2024-01-02', 'r', 1, tax_year=2024, magi=100000, compensation=100000)
    history.convert('2024-02-01', 'r', 10000, 6000)
    result = history.withdraw('2025-01-01', 'r', 18000)
    assert result['regular_basis'] == 7000
    assert result['conversion_principal'] == 10000
    assert result['personal_taxable_income'] == 1000
    assert result['additional_tax'] == 700


def test_roth_separate_conversion_clock_and_qualified_clock():
    book = ledger()
    history = RothHistory(book, 'owner', '1950-01-01', first_tax_year=2024, regular_basis=7000)
    history.convert('2024-01-01', 'r', 10000, 10000)
    result = history.withdraw('2028-12-31', 'r', 18000)
    assert not result['qualified']
    assert result['personal_taxable_income'] == 1000  # age alone does not satisfy five years
    assert result['additional_tax'] == 0
    result = history.withdraw('2029-01-01', 'r', 1000)
    assert result['qualified'] and result['personal_taxable_income'] == 0


def test_irs_590b_amelia_distribution_example():
    book = ledger()
    history = RothHistory(book, 'owner', '1965-01-01', first_tax_year=2021,
                           regular_basis=5000, conversions=[Conversion(2021, 60000, 20000)])
    result = history.withdraw('2025-11-08', 'r', 7000)
    assert result['regular_basis'] == 5000
    assert result['conversion_principal'] == 2000
    assert result['additional_tax'] == 0 and result['personal_taxable_income'] == 0
