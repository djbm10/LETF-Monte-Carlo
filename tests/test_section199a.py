import pytest
from letf.tax.marginal import calculate_comprehensive_tax_v6
from letf.accounts import PortfolioLedger,TAXABLE_CASH
from letf.accuracy import UnsupportedModelError


def tax(**kw):return calculate_comprehensive_tax_v6(0.,0.,0.,include_state=False,tax_year=2024,**kw)


def test_form_8995_reit_only_deduction_is_twenty_percent_and_below_agi():
    base=tax(ordinary_income=100000.)
    r=tax(ordinary_income=100000.,section199a_dividends=1000.)
    assert r['section199a_dividend_deduction']==200.
    assert r['total_tax']-base['total_tax']==pytest.approx(176.)
    # NIIT is not reduced by a below-line QBI deduction.
    r=tax(ordinary_income=200000.,section199a_dividends=1000.)
    assert r['niit_tax']==38.


def test_form_8995_taxable_income_cap_excludes_qualified_dividends_and_gains():
    r=tax(ordinary_income=0.,section199a_dividends=20000.)
    assert r['section199a_dividend_deduction']==1080. # .2*(20000-14600)
    assert r['federal_total']==pytest.approx(432.)
    r=calculate_comprehensive_tax_v6(0.,50000.,0.,include_state=False,tax_year=2024,
        section199a_dividends=20000.)
    assert r['section199a_dividend_deduction']==1080.
    with pytest.raises(UnsupportedModelError):
        calculate_comprehensive_tax_v6(0.,0.,0.,include_state=False,tax_year=2026,section199a_dividends=100.)


def test_ma_does_not_take_federal_qbi_deduction_and_ledger_does_not_double_count_box5():
    b=PortfolioLedger();b.add_account('a',TAXABLE_CASH,0.)
    with pytest.raises(UnsupportedModelError):b.income('2024-06-01','a',1000.,'section199a_dividend')
    b.income('2024-06-01','a',1000.,'section199a_dividend',section199a_verified=True)
    r=b.annual_tax('a',2024,100000.,state_code='MA',as_of='2025-01-31')
    assert b.accounts['a'].cash==1000.
    assert r['tax']==pytest.approx(176.+50.)
