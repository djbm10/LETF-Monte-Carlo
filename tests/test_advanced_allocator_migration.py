"""Compare migrated decisions with preserved legacy return formulas."""
import numpy as np
import pandas as pd
import pytest

from letf import config as cfg
from letf.strategy import run_strategy_fixed
from letf.strategy_allocators import LegacyAllocator,SUPPORTED
from letf.accounts import CONSERVATIVE_ROTH,Security
from letf.account_backtest import run_account_backtest
from letf.accuracy import UnsupportedModelError


def fixture(n=330):
    rng=np.random.default_rng(45);idx=pd.bdate_range('2023-01-02',periods=n)
    spy=rng.normal(.0004,.012,n);letf=3*spy-.0002+rng.normal(0,.001,n)
    data=pd.DataFrame({'SPY_Ret':spy,'TQQQ_Ret':letf,'SPY_Price':100*np.cumprod(1+spy),
        'TQQQ_Price':100*np.cumprod(1+letf),'Cash_Ret':.0001,'VIX':20+5*np.sin(np.arange(n)/13)},index=idx)
    history=pd.DataFrame({'SPY_TR_Close':data.SPY_Price,'TQQQ_TR_Close':data.TQQQ_Price,
        'SPY_Signal_Ret':data.SPY_Ret,'TQQQ_Signal_Ret':data.TQQQ_Ret,'VIX':data.VIX},index=idx)
    return data,history


@pytest.mark.parametrize('sid',[s for s in SUPPORTED if s!='S16'])
def test_legacy_causal_signal_formulas_match_independent_preserved_returns(sid):
    data,history=fixture()
    equity,_=run_strategy_fixed(data,sid,np.zeros(len(data),dtype=int),None,apply_costs=False)
    legacy=equity.pct_change()
    adapter=LegacyAllocator(sid)
    for i in range(1,len(data)):
        weights=adapter(history.iloc[:i],{})
        predicted=(weights.get('TQQQ',0)*data.TQQQ_Ret.iloc[i]+weights.get('SPY',0)*data.SPY_Ret.iloc[i]+
                   (1-sum(weights.values()))*data.Cash_Ret.iloc[i])
        assert predicted==pytest.approx(legacy.iloc[i],abs=2e-12),(sid,i)


def test_s16_current_bar_vix_rule_has_no_silently_causal_alias():
    with pytest.raises(UnsupportedModelError,match='current-close VIX'):LegacyAllocator('S16')
    _,history=fixture();a=LegacyAllocator('S16',version='causal_cash_v2')
    assert 0<=a(history.iloc[:200],{})['TQQQ']<=1
    with pytest.raises(UnsupportedModelError,match='VIX'):a(history.drop(columns='VIX'),{})


@pytest.mark.parametrize('sid',SUPPORTED)
def test_every_named_cash_v2_allocator_executes_through_canonical_book(sid):
    _,h=fixture(30)
    market=pd.DataFrame({'SPY_Open':h.SPY_TR_Close,'SPY_Close':h.SPY_TR_Close,
        'TQQQ_Open':h.TQQQ_TR_Close,'TQQQ_Close':h.TQQQ_TR_Close,'VIX':h.VIX},index=h.index)
    adapter=LegacyAllocator(sid,version='causal_cash_v2')
    result=run_account_backtest(market,[Security('SPY'),Security('TQQQ')],CONSERVATIVE_ROTH,
        initial_capital=10000.,allocator=adapter,price_convention='as_traded',distributions_complete=True,
        terminal_liquidation=True)
    assert result['reconciliation'].cash.min()>=-1e-8
    assert result['reconciliation'].quantity_residual.abs().max()<1e-8
    assert adapter.metadata['version']=='causal_cash_v2'
