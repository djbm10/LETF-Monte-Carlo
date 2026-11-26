from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from letf.lab.portfolio import run_observed,validate_request,load_portfolio
from letf.lab.strategies import make_allocator,registry
from letf.accuracy import UnsupportedModelError


def bundle(root,symbol):
    root.mkdir()
    index=xcals.get_calendar('XNYS').sessions_in_range('2024-01-02','2025-12-31').tz_localize(None)
    price=np.linspace(100.,120. if symbol=='X' else 95.,len(index))
    frame=pd.DataFrame({'Open':price,'Close':price,'Dividends':0.,'Stock Splits':0.},index=index)
    frame.loc['2024-06-03','Dividends']=1.
    frame.to_csv(root/'prices.csv')
    dist=pd.DataFrame([dict(ex_date='2024-06-03',pay_date='2024-06-05',amount_per_share=1.,category='ordinary_dividend',source='Independent software fixture')]).set_index('ex_date')
    dist.to_csv(root/'distributions.csv')
    files={name:dict(path=name+'.csv',sha256=sha256((root/(name+'.csv')).read_bytes()).hexdigest(),provider='SOFTWARE_FIXTURE',url='https://example.invalid/fixture') for name in ('prices','distributions')}
    manifest=dict(symbol=symbol,data_status='OBSERVED',classification='SOFTWARE_FIXTURE',currency='USD',frequency='daily',
        price_convention='as_traded',distributions_complete=True,files=files,supported_tax_states=['FEDERAL_ONLY'],
        account_coverage_start='2024-01-02',account_coverage_end='2025-12-31')
    (root/'manifest.json').write_text(json.dumps(manifest))
    return str(root)


def request(tmp_path):
    return dict(schema=1,mode='observed',bundles={s:bundle(tmp_path/s,s) for s in ('X','Y')},
        start='2024-01-02',end='2025-12-31',strategy=dict(id='fixed_weights',version='cash_v1',weights={'X':.6,'Y':.3}),
        initial_capital=10000.,rebalance_sessions=63,cash_sweep_annual_rate=.03,
        tax=dict(state='FEDERAL_ONLY',ordinary_income=100000.,filing_status='single',law_mode='ACTUAL_DATED',external_wash_policy='NO_OTHER_OWNER_REPLACEMENTS'),
        cash_flows=[dict(date='2025-01-02',kind='contribution',amount=1000.,magi=100000.,compensation=100000.)],
        roth_owner=dict(birth_date='1980-01-01',first_tax_year=2010,regular_basis=10000.))


def test_full_multiasset_flow_interest_distribution_tax_cash_pair_is_independently_audited(tmp_path):
    r=request(tmp_path);progress=[]
    result=run_observed(r,output=tmp_path/'output',progress=progress.append)
    assert result['classification']=='SOFTWARE_FIXTURE' and result['same_pre_tax_policy_and_fills']
    assert result['accounts']['taxable']['tax_paid']>0
    assert result['accounts']['roth']['tax_paid']==result['accounts']['roth']['terminal_tax_payable']==0
    assert result['accounts']['taxable']['ending_wealth_net_payable']<result['accounts']['roth']['ending_wealth_net_payable']
    for name,row in result['accounts'].items():
        assert row['independent_audit']['passed'] and row['max_cash_residual']<1e-7
        assert row['metrics']['cagr'] is None and row['metrics']['money_weighted_return'] is not None
    assert len(progress)==4*502  # pinned XNYS calendar, including 2025 Carter closure
    exported=json.loads((tmp_path/'output/export-manifest.json').read_text())
    for name,digest in exported['sha256'].items():assert sha256((tmp_path/'output'/name).read_bytes()).hexdigest()==digest
    with pytest.raises(FileExistsError):run_observed(r,output=tmp_path/'output')


def test_joint_calendar_gap_or_state_distribution_gap_never_silently_filled(tmp_path):
    r=request(tmp_path);r['tax']['state']='MA'
    with pytest.raises(UnsupportedModelError,match='adjustments'):load_portfolio(r)
    r['tax']['state']='FEDERAL_ONLY'
    p=Path(r['bundles']['Y']);data=pd.read_csv(p/'prices.csv');data=data.drop(index=10);data.to_csv(p/'prices.csv',index=False)
    m=json.loads((p/'manifest.json').read_text());m['files']['prices']['sha256']=sha256((p/'prices.csv').read_bytes()).hexdigest();(p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(UnsupportedModelError,match='calendars'):load_portfolio(r)


def test_unknown_tax_law_external_accounts_and_frozen_mislabel_are_blocked(tmp_path):
    r=request(tmp_path)
    with pytest.raises(ValueError):validate_request(dict(r,fictional_option_returns=1))
    r['tax']['freeze_year']=2026
    with pytest.raises(ValueError,match='Frozen'):validate_request(r)
    r['tax'].pop('freeze_year');r['tax']['external_wash_policy']='UNKNOWN'
    with pytest.raises(UnsupportedModelError,match='Linked'):validate_request(r)


def test_string_false_cannot_grant_broker_permission_or_distribution_completeness(tmp_path):
    from dataclasses import replace
    from letf.accounts import CONSERVATIVE_ROTH
    from letf.account_cli import load_bundle
    for value in ('false','False',0,1):
        with pytest.raises(ValueError,match='boolean'):replace(CONSERVATIVE_ROTH,fractional_shares=value)
    folder=Path(bundle(tmp_path/'X','X'));m=json.loads((folder/'manifest.json').read_text())
    for value in ('false','False',1):
        m['distributions_complete']=value;(folder/'manifest.json').write_text(json.dumps(m))
        with pytest.raises(UnsupportedModelError,match='incomplete'):load_bundle(folder)


def test_multiyear_comparison_exports_actual_payment_dates_and_fixture_evidence(tmp_path):
    from letf.lab.workflows import tournament
    r=request(tmp_path);strategy=r.pop('strategy')
    result=tournament({'schema':1,'base':r,'strategies':[strategy]},output=tmp_path/'comparison')
    assert result['classification']=='SOFTWARE_FIXTURE' and result['all_requested_cells_supported']
    saved=json.loads((tmp_path/'comparison/report.json').read_text())
    taxes=saved['cells'][0]['accounts']['taxable']['tax_rows']
    assert [row['paid_on'] for row in taxes if 'paid_on'in row]==['2025-04-15',None] # next tax year remains terminal payable
    assert not saved['rankings_released']


def test_original_versions_and_corrected_cash_signals_are_explicit():
    records=registry();assert {f'S{i}' for i in range(1,20)}<={r['id'] for r in records}
    with pytest.raises(UnsupportedModelError):make_allocator({'id':'S16','version':'legacy_rules_v1'})
    with pytest.raises(UnsupportedModelError):make_allocator({'id':'S1','version':'legacy_vector_v0'})
    with pytest.raises(ValueError):make_allocator({'id':'fixed_weights','version':'cash_v1','weights':{'X':1.1}})
    with pytest.raises(ValueError):make_allocator({'id':'HFEA','version':'cash_v1','weights':{'UPRO':.6}})


def test_control_fill_audit_allows_only_sub_reconciliation_roundoff():
    from letf.lab.portfolio import compare_control_trades
    row=dict(date='2025-01-02',action='BUY',security='X',price=100.,shares=1.,commission=0.,reason='STRATEGY')
    assert compare_control_trades([[row],[dict(row,shares=1.+1e-13)]])['passed']
    with pytest.raises(AssertionError):compare_control_trades([[row],[dict(row,shares=1.+1e-8)]])
    with pytest.raises(AssertionError):compare_control_trades([[row],[dict(row,price=100.+1e-12)]])


def test_frozen_35_selector_matches_preserved_original_without_import_side_effects():
    import ast,math
    from letf.lab.strategies import __file__ as strategies_file
    path=Path(strategies_file).parents[2]/'research/frozen_35_0_actual_futures_validation_20261004.py'
    tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen_exposure')
    scope=dict(np=np,pd=pd,math=math,TD=252,BULL_VOL_TARGET=.35,BEAR_VOL_TARGET=0.,SMA_DAYS=200,VOL_DAYS=20,EXPOSURE_CAP=3.)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    rng=np.random.default_rng(72);idx=pd.bdate_range('2020-01-01',periods=600)
    q=pd.Series(100*np.exp(np.cumsum(rng.normal(.0003,.012,600))),index=idx)
    rf=pd.Series(np.linspace(.00001,.0003,600),index=idx)
    expected=scope['frozen_exposure'](q,rf)
    allocate,_,_,_=make_allocator(dict(id='35_0',version='frozen_ETF_wrapper_v1'))
    frame=pd.DataFrame({'QQQ_TR_Close':q,'RF_DAILY':rf})
    for i in range(1,len(idx)):
        actual=allocate(frame.iloc[:i],{}).get('TQQQ',0.)*3
        assert actual==pytest.approx(expected.iloc[i],abs=1e-12)
