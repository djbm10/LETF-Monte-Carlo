"""Offline tests: no proprietary data, internet access or broker account required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'))
import numpy as np
import pandas as pd
import pytest
from free_validation.futures_reconcile import FuturesCosts, target_exposure, validate_contracts, replay_contract_settlements
from free_validation.sec_pit import facts_pit, get_companyfacts


def fixture():
    dates = pd.bdate_range('2026-01-02', periods=35)
    q = pd.Series(np.linspace(100, 130, len(dates)), index=dates, name='QQQ')
    expiry_a = dates[19] + pd.Timedelta(days=1)
    expiry_b = dates[-1] + pd.Timedelta(days=45)
    a = pd.DataFrame(dict(date=dates[:20],contract='MNQH26',expiry=expiry_a,settlement=100+np.arange(20)*.75))
    # Next contract starts 40 points higher; that DIFFERENCE must not count as P&L
    b = pd.DataFrame(dict(date=dates[6:],contract='MNQM26',expiry=expiry_b,settlement=140+np.arange(len(dates)-6)*.75))
    return pd.concat([a,b],ignore_index=True),q


def test_futures_roll_gap_not_realized_as_pnl():
    market,q = fixture()
    cfg=FuturesCosts(initial_cash=100_000,sma_days=4,volatility_days=3,roll_business_days=3,
                     commission_per_side=0,slippage_ticks_per_side=0)
    x=replay_contract_settlements(market,q,cfg)
    assert len(x)==35 and (x.equity>0).all()
    assert x.rolled.any()
    # When old and new contracts are both rising only 0.75 points/day, no 40-point jump.
    active = x.loc[x.rolled].index
    assert len(active) == 1
    day = active[0]; prior = x.index[x.index.get_loc(day)-1]
    assert x.loc[day,'pnl'] == pytest.approx(x.loc[prior,'contracts'] * 2 * .75)
    assert x.pnl.min() >= -1e-9


def test_high_cost_and_margin_impact():
    market,q=fixture()
    low=replay_contract_settlements(market,q,FuturesCosts(sma_days=4,volatility_days=3,roll_business_days=3,
                                                          commission_per_side=0,slippage_ticks_per_side=0))
    high=replay_contract_settlements(market,q,FuturesCosts(sma_days=4,volatility_days=3,roll_business_days=3,
                                                          commission_per_side=5,slippage_ticks_per_side=2,
                                                          initial_margin_fraction=.5,maintenance_margin_fraction=.4))
    assert high.equity.iloc[-1] < low.equity.iloc[-1]
    assert (high.actual_exposure.dropna() <= 2.05).all()


def test_invalid_contracts_rejected():
    market,q=fixture()
    with pytest.raises(ValueError,match='Duplicate'):
        validate_contracts(pd.concat([market,market.iloc[:1]]))
    with pytest.raises(ValueError,match='Missing QQQ'):
        replay_contract_settlements(market,q.iloc[1:],FuturesCosts(sma_days=4,volatility_days=3))


def test_missing_held_contract_day_fails_closed():
    market,q=fixture()
    market=market.loc[~((market.contract=='MNQH26') & (market.date==q.index[12]))]
    with pytest.raises(ValueError,match='missing settlement'):
        replay_contract_settlements(market,q,FuturesCosts(sma_days=4,volatility_days=3,roll_business_days=3))


def test_sec_pit_restated_future_value_not_leaked():
    data={'facts':{'us-gaap':{'Revenues':{'units':{'USD':[
        dict(end='2023-12-31',filed='2024-02-15',accn='first',form='10-K',val=100),
        dict(end='2023-12-31',filed='2024-09-10',accn='amend',form='10-K/A',val=200),
        dict(end='2024-03-31',filed='2024-05-01',accn='future',form='10-Q',val=42),
    ]}}}}}
    before=facts_pit(data,'Revenues','2024-03-01')
    assert len(before)==1 and before.value.iloc[0]==100
    after=facts_pit(data,'Revenues','2024-10-01')
    assert after.loc[after.end==pd.Timestamp('2023-12-31'),'value'].iloc[0]==200
    assert len(facts_pit(data,'NetIncomeLoss','2024-10-01'))==0


def test_sec_requires_real_contact_identity():
    with pytest.raises(ValueError,match='SEC_USER_AGENT'):
        get_companyfacts(320193,user_agent='Anonymous Bot')

from free_validation.risk_scenarios import shock_joint_series, compare_shocks
from free_validation.shadow_signals import compute_frozen_signals, append_immutable


def test_joint_crash_shocks_cannot_look_ahead():
    raw, q = fixture()
    day = q.index[15]
    r, z = shock_joint_series(raw,q,[(str(day.date()),-.20)])
    assert z.loc[q.index[14]] == q.loc[q.index[14]]
    assert z.loc[day] == pytest.approx(q.loc[day]*.8)
    untouched = raw.loc[raw.date < day,'settlement'].to_numpy()
    np.testing.assert_allclose(untouched, r.loc[r.date < day,'settlement'].to_numpy())


def test_stress_suite_explicitly_labels_hypothetical_paths():
    raw,q = fixture()
    d=q.index[14].date().isoformat()
    z=compare_shocks(raw,q,FuturesCosts(sma_days=4,volatility_days=3,roll_business_days=3),
                     {'crash20':{'shocks':[(d,-.20)]},'stressed_margin':{'initial_margin_fraction':.5,'maintenance_margin_fraction':.4}})
    assert len(z)==3 and not z.loc[z.scenario=='unchanged_input','synthetic_scenario'].iloc[0]
    assert z.loc[z.scenario=='crash20','synthetic_scenario'].iloc[0]


def test_forward_signal_ledger_immutable(tmp_path):
    dates=pd.bdate_range('2024-01-01',periods=290)
    names=['QQQ','SPY']+list({'XLK':'ROM','XLE':'DIG'})+['ROM','DIG']
    q=pd.DataFrame({name:pd.Series(100+np.arange(290)*(1.5 if name!='SPY' else .5),index=dates)
                    for name in names})
    # Restore all full 10-sector columns as required by real frozen rule
    from free_validation.shadow_signals import SECTOR_PAIRS
    for k,v in SECTOR_PAIRS.items():
        for sym in (k,v):
            if sym not in q:q[sym]=q.QQQ*.9
    day=dates[-1].date().isoformat()
    r=compute_frozen_signals(q,day)
    p=tmp_path/'forward_signals.jsonl'
    assert append_immutable(r,p)
    assert not append_immutable(r,p)
    with pytest.raises(ValueError,match='frozen signal changed'):
        append_immutable({**r,'candidate_etf':'FAKE'},p)
