from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from letf.accounts import TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.accuracy import UnsupportedModelError
from letf.derivatives import FuturesContract
from letf.research_adapters import replay_futures_account, optionmetrics_contract
from research.free_validation.futures_reconcile import FuturesCosts, replay_contract_settlements
from research.optionmetrics_leaps_backtest import load, backtest_one
from research.bottleneck_winner_tournament import prepare_crsp, prepare_fund


def test_canonical_futures_matches_independent_research_for_lagged_signal_and_roll():
    dates=pd.bdate_range('2024-11-01',periods=35)
    q=pd.Series(100+np.arange(35)*.5+.01*np.sin(np.arange(35)),index=dates)
    warm=pd.bdate_range(end=dates[0]-pd.Timedelta(days=1),periods=10)
    q=pd.concat([pd.Series(np.linspace(95,99.5,10),index=warm),q])
    exp_a=dates[19]+pd.Timedelta(days=1);exp_b=dates[-1]+pd.Timedelta(days=45)
    data=pd.concat([pd.DataFrame({'date':dates[:20],'contract':'MNQZ24','expiry':exp_a,'settlement':100+np.arange(20)*.75}),
        pd.DataFrame({'date':dates[6:],'contract':'MNQH25','expiry':exp_b,'settlement':140+np.arange(29)*.75})])
    cfg=FuturesCosts(sma_days=4,volatility_days=3,roll_business_days=3,commission_per_side=0.,slippage_ticks_per_side=0.)
    contracts={s:FuturesContract(s,exp.date(),2.,.25,25.,20.,date(2024,1,1),'Explicit test margin scenario',True,'cash')
        for s,exp in [('MNQZ24',exp_a),('MNQH25',exp_b)]}
    profile=replace(TAXABLE_CASH,allowed_kinds=frozenset({'future'}))
    result=replay_futures_account(data,q,contracts,profile,costs=cfg,terminal_liquidation=False)
    # Shift the independent research signal one session for causal comparison.
    reference=replay_contract_settlements(data,q.shift(1).fillna(q.iloc[0]),cfg)
    np.testing.assert_allclose(result['daily'].cash,reference.equity,atol=1e-7)
    assert result['ledger'].accounts['main'].income[2024]['section1256']==pytest.approx(reference.equity.iloc[-1]-cfg.initial_cash)
    with pytest.raises(UnsupportedModelError,match='permit futures'):
        replay_futures_account(data,q,contracts,CONSERVATIVE_ROTH,costs=cfg)


def test_public_optionmetrics_contract_preserves_strike_expiry_and_no_quote_invention():
    rows=load(Path(__file__).resolve().parents[1]/'research/fixtures/optionmetrics_public_sample_20050915.csv')
    row=rows.iloc[0]
    c=optionmetrics_contract(row,underlying='SPX',settlement='cash',multiplier=100,source='Published OptionMetrics sample')
    assert c.strike==1600 and c.expiry==date(2006,6,17) and c.style=='european'
    assert not hasattr(c,'timestamp')
    row=row.copy();row['cfadj']=2
    with pytest.raises(UnsupportedModelError):
        optionmetrics_contract(row,underlying='SPX',settlement='cash',multiplier=100,source='sample')


def test_missing_held_option_quote_blocks_legacy_scenario():
    dates=pd.date_range('2020-01-31',periods=25,freq='ME')
    rows=pd.DataFrame({'date':dates,'exdate':pd.Timestamp('2023-01-20'),'cp_flag':'C','strike':100.,
        'best_bid':9.,'best_offer':10.,'mid':9.5,'spread':1.,'dte':500,'delta':.8,
        'underlying_price':120.,'ticker':'X','optionid':np.where(np.arange(25)==1,2,1)})
    with pytest.raises(ValueError,match='Missing held option'):
        backtest_one(rows,'X',.8,548,1.)


def test_momentum_never_shifts_a_previous_securitys_value_into_the_next():
    dates=pd.date_range('2020-01-31',periods=15,freq='ME')
    crsp=pd.DataFrame({'permno':np.repeat([1,2],15),'date':list(dates)*2,'ret':np.repeat([.1,.01],15),'me':100.})
    out=prepare_crsp(crsp)
    second=out[out.permno.eq(2)]
    assert np.isnan(second.mom12_1.iloc[0])
    assert second.mom12_1.iloc[10]==pytest.approx(1.01**10-1)


def test_fundamental_columns_at_and_lt_are_not_pandas_attributes():
    dates=pd.date_range('2020-01-01',periods=5,freq='QE')
    x=pd.DataFrame({'gvkey':'1','available_date':dates,'sale':100.,'cogs':50.,'oibdp':20.,
        'oancf':10.,'capx':2.,'at':[100.,110.,120.,130.,140.],'lt':25.,'csho':10.,'prcc':20.})
    out=prepare_fund(x)
    assert out.leverage.iloc[0]==.25
    assert out.asset_growth_yoy.iloc[4]==pytest.approx(.4)
    assert out.ev_sales_proxy.iloc[0]==2.25
