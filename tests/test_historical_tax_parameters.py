"""Independent published schedule constants, not tests recomputing our tables."""
import pytest
from letf.tax.marginal import calculate_comprehensive_tax_v6 as tax
from letf.tax.roth import contribution_limit
from letf.accuracy import UnsupportedModelError
from letf.tax.massachusetts import massachusetts_tax


@pytest.mark.parametrize('year,status,gross,published',[
    (2021,'single',12550+40525,4664),
    (2021,'head_of_household',18800+86350,13293),
    (2021,'married_filing_separately',12550+314150,84496.75),
    (2022,'single',12950+89075,15213.50),
    (2022,'head_of_household',19400+170050,33148),
    (2023,'single',13850+182100,37104),
    (2023,'head_of_household',20800+95350,14678),
    (2023,'married',27700+190750,32580),
])
def test_official_schedule_tax_at_bracket_boundary(year,status,gross,published):
    assert tax(0,0,0,ordinary_income=gross,include_state=False,include_niit=False,
               tax_year=year,filing_status=status)['federal_total']==pytest.approx(published)


@pytest.mark.parametrize('year,ordinary,lt,expected',[
    (2021,12550+39400,2000,150),
    (2022,12950+40675,2000,150),
    (2023,13850+43625,2000,150),
])
def test_historical_qualified_capital_income_stacks_over_ordinary(year,ordinary,lt,expected):
    result=tax(0,lt,0,ordinary_income=ordinary,include_state=False,tax_year=year)
    assert result['federal_ltcg']==pytest.approx(expected)


@pytest.mark.parametrize('year,limit,low,high',[(2021,6000,125000,140000),(2022,6000,129000,144000),(2023,6500,138000,153000)])
def test_dated_roth_limits_phaseout_and_compensation(year,limit,low,high):
    assert contribution_limit(year,40,low,100000)==limit
    assert contribution_limit(year,40,high,100000)==0
    assert contribution_limit(year,50,0,100000)==limit+1000
    assert contribution_limit(year,40,0,1000)==1000


def test_no_unverified_old_law_extrapolation():
    with pytest.raises(UnsupportedModelError):tax(0,0,0,include_state=False,tax_year=1930)


def test_ma_2023_enacted_short_rate_and_first_surtax_threshold():
    # 2023 Form 1: 5% wage base after $4,400 personal exemption, 8.5% ST,
    # 4% surtax on aggregate taxable income above exactly $1,000,000.
    result=massachusetts_tax(2023,1004400,st_gains=10000)
    assert result['five_percent_tax']==50000
    assert result['short_term_tax']==pytest.approx(850)
    assert result['surtax']==pytest.approx(400)
    assert result['total_tax']==pytest.approx(51250)
    assert massachusetts_tax(2023,100000,st_losses=3000,interest=1000)['st_cf']==2000


def test_export_handles_top_legal_bracket_without_accepting_nonfinite_wealth():
    import json
    from letf.account_cli import _report_safe
    result=tax(0,1000000,0,ordinary_income=100000,include_state=False,tax_year=2023)
    safe=_report_safe(result)
    assert safe['ltcg_breakdown']['breakdown'][-1]['bracket_ceiling']=='UNBOUNDED'
    json.dumps(safe,allow_nan=False)
    with pytest.raises(ValueError):_report_safe({'ending_wealth':float('inf')})
    with pytest.raises(ValueError):_report_safe({'tax':float('nan')})
