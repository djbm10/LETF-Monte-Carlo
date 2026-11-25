from datetime import date
import numpy as np
import pandas as pd
import pytest
from letf.accounts import Security,CONSERVATIVE_ROTH,TAXABLE_CASH
from letf.account_backtest import run_account_backtest
from letf.accuracy import UnsupportedModelError


def replay(market,**kwargs):
    return run_account_backtest(market,[Security('X')],CONSERVATIVE_ROTH,initial_capital=1000.,
        allocator=lambda h,w:{'X':1.},price_convention='as_traded',distributions_complete=True,
        terminal_liquidation=True,**kwargs)


def test_intraday_extinction_does_not_avoid_the_last_open_fill_and_cannot_recover():
    idx=pd.bdate_range('2024-01-02',periods=4)
    m=pd.DataFrame({'X_Open':[100.,100.,0.,0.],'X_Close':[100.,0.,0.,0.]},index=idx)
    r=replay(m,liquidation_events=[{'date':'2024-01-03','security':'X','cash_per_share':0.,'timing':'close','source':'Explicit crash scenario'}])
    assert r['ending_wealth_net_payable']==0.
    assert r['after_tax_metrics']['cagr']==-1.
    assert r['ledger'].quantity('main','X')==0.
    assert len([e for e in r['ledger'].events if e.get('reason')=='FUND_LIQUIDATION'])==1
    assert r['reconciliation'].quantity_residual.abs().max()==0.


def test_new_cash_after_extinction_remains_cash_and_is_not_a_recovery_return():
    idx=pd.bdate_range('2027-01-04',periods=6)
    m=pd.DataFrame({'X_Open':[100.,100.,0.,0.,0.,0.],'X_Close':[100.,0.,0.,0.,0.,0.]},index=idx)
    r=replay(m,liquidation_events=[{'date':'2027-01-05','security':'X','cash_per_share':0.,'timing':'close','source':'Explicit crash scenario'}],
        cash_flows=[{'date':'2027-01-07','kind':'contribution','amount':100.,'magi':100000.,'compensation':100000.}],
        roth_history_config={'birth_date':'1980-01-01','first_tax_year':2010,'regular_basis':1000.},
        freeze_roth_contribution_year=2026)
    assert r['ending_wealth_net_payable']==100.
    assert r['metrics']['time_weighted_cagr']==-1.
    assert r['ledger'].accounts['main'].cash==100.
    assert r['ledger'].quantity('main','X')==0.


def test_split_normalized_and_total_return_signal_references_do_not_create_crash():
    idx=pd.bdate_range('2024-01-02',periods=5)
    m=pd.DataFrame({'X_Open':[100.,100.,50.,50.,50.],'X_Close':[100.,100.,50.,50.,50.]},index=idx)
    observed=[]
    def allocator(h,w):
        if len(h)>=3:observed.append((h['X_Close'].pct_change().iloc[-1],h['X_TR_Close'].pct_change().iloc[-1]))
        return {'X':1.}
    run_account_backtest(m,[Security('X')],CONSERVATIVE_ROTH,initial_capital=1000.,allocator=allocator,
        price_convention='as_traded',distributions_complete=True,
        split_events=[{'date':'2024-01-04','security':'X','ratio':2.}])
    assert observed[0]==(0.,0.)


def test_roth_does_not_exist_in_1930_and_future_contributions_need_explicit_table():
    idx=pd.bdate_range('1930-01-02',periods=5)
    m=pd.DataFrame({'X_Open':100.,'X_Close':100.},index=idx)
    with pytest.raises(UnsupportedModelError,match='1998'):replay(m)
