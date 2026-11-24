from hashlib import sha256
import json
import pandas as pd
import pytest

from letf.account_scenarios import compare_scenarios
from letf.accuracy import UnsupportedModelError


def bundle(tmp_path):
    f=tmp_path/'prices.csv'
    pd.DataFrame({'Open':100.,'Close':100.,'Dividends':0.,'Stock Splits':0.},
        index=pd.bdate_range('2024-01-02',periods=30)).to_csv(f)
    (tmp_path/'manifest.json').write_text(json.dumps({'data_status':'OBSERVED','symbol':'X',
        'price_convention':'as_traded','distributions_complete':True,'instrument_kind':'stock',
        'frequency':'daily','currency':'USD','files':{'prices':{'path':f.name,
        'sha256':sha256(f.read_bytes()).hexdigest(),'provider':'Deterministic test fixture','url':'https://example.invalid'}}}))
    return tmp_path


def test_paired_paths_use_canonical_fills_and_taxes_with_frozen_future_law(tmp_path):
    source=bundle(tmp_path)
    a=compare_scenarios(source,paths=2,horizons=[5],seed=7,freeze_tax_year=2026)
    b=compare_scenarios(source,paths=2,horizons=[5],seed=7,freeze_tax_year=2026)
    assert a==b
    assert a['evidence']=='HYPOTHETICAL_STRESS'
    assert a['investment_performance_certification']=='BLOCKED'
    # Constant stock, independently priced buy/sell with 2bp full spread + 1bp slippage.
    gross=100000/1.0002*.9998
    for row in a['rows']:
        assert row['accounts']['roth']['ending_wealth']==pytest.approx(gross)
        assert row['accounts']['taxable']['ending_wealth']==pytest.approx(gross+(100000-gross)*.22)
    assert a['paired_terminal_difference_ci']['5']['independent_path_count']==2


def test_future_unknown_tax_law_and_dividend_paths_are_rejected(tmp_path):
    source=bundle(tmp_path)
    with pytest.raises(UnsupportedModelError,match='frozen'):
        compare_scenarios(source,paths=1,horizons=[5],seed=7,freeze_tax_year=2030)
    p=source/'manifest.json';m=json.loads(p.read_text());m['instrument_kind']='etf';p.write_text(json.dumps(m))
    with pytest.raises(UnsupportedModelError,match='Dividend/ETF'):
        compare_scenarios(source,paths=1,horizons=[5],seed=7,freeze_tax_year=2026)
