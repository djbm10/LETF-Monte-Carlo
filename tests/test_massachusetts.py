import pytest
from letf.tax.massachusetts import massachusetts_tax
from letf.accounts import PortfolioLedger, Security, TAXABLE_CASH
from letf.accuracy import UnsupportedModelError


@pytest.mark.parametrize('year, threshold', [(2024, 1053750), (2025, 1083150), (2026, 1107750)])
def test_dor_surtax_table_and_no_federal_standard_deduction(year, threshold):
    x = massachusetts_tax(year, threshold + 4400, st_gains=10000)
    assert x['surtax'] == 400
    assert x['five_percent_tax'] == threshold * .05
    assert x['short_term_tax'] == pytest.approx(850.)


def test_ma_schedule_b_losses_offset_interest_before_long_term_gains():
    # Schedule B lines 20/22 and Schedule D: ST loss -5000, interest 3000,
    # LT gain 4000 -> interest 1000, LT gain 1000. No federal $3k wage loss.
    x = massachusetts_tax(2025, 100000, st_losses=5000, lt_gains=4000, interest=3000)
    assert x['loss_offset_interest_dividends'] == 2000
    assert x['st_cf'] == x['lt_cf'] == 0
    assert x['total_tax'] == pytest.approx((100000 - 4400 + 2000) * .05)


def test_ma_loss_carryforward_independent_of_federal_wage_deduction():
    book = PortfolioLedger();book.add_account('t', TAXABLE_CASH, 10000)
    book.add_security(Security('X', kind='stock'))
    book.execute('2024-01-02', 't', 'X', 'BUY', 100, 100)
    book.execute('2024-12-02', 't', 'X', 'SELL', 100, 50)
    x = book.annual_tax('t', 2024, 100000, state_code='MA', as_of='2025-01-31')
    assert x['st_cf'] == 2000
    assert x['state_st_cf'] == 5000
    assert x['calculation']['state_tax'] == pytest.approx(4780.)
    y = book.annual_tax('t', 2025, 100000, state_code='MA', as_of='2026-01-31',
                        st_cf=x['st_cf'], state_st_cf=x['state_st_cf'])
    assert y['state_st_cf'] == 5000


def test_ma_unmodeled_low_income_or_old_year_is_explicit():
    with pytest.raises(UnsupportedModelError):
        massachusetts_tax(2025, 0, lt_gains=10000)
    with pytest.raises(UnsupportedModelError):
        massachusetts_tax(2022, 100000)
