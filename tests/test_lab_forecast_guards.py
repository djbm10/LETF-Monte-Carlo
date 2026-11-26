from datetime import datetime,timezone
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from letf.macro.data import CORE
from letf.macro.prospective_daily import assess_daily
from letf.macro.retraining import issue_candidates,evaluate_issued,evaluate_pending
from letf.macro.learning import ModelRegistry
from test_macro_explorer import FixtureSources
from test_prospective_retraining import protocol
from letf.lab.workflows import account_sampling_diagnostics


def test_future_injected_months_cannot_be_scored_by_candidate_pipeline(tmp_path):
    source=FixtureSources();registry=ModelRegistry(tmp_path/'registry');p=protocol()
    issued=issue_candidates(source,registry,p,output=tmp_path/'issued')
    frozen,_=registry.store.load('forecasts',issued['forecast_ids'][0])
    original_panel=source.panel
    def fake_future(*args,**kwargs):
        frame,meta=original_panel(*args,**kwargs)
        future=pd.DataFrame(0.,index=pd.to_datetime(frozen['dates']),columns=frame.columns)
        return pd.concat([frame,future]),meta
    source.panel=fake_future
    scored=evaluate_issued(source,registry,p,issued['forecast_ids'],output=tmp_path/'scored',allow_promotion=False)
    assert scored['status']=='PENDING_COMPLETE_OUTCOMES' and not scored['promoted']
    assert all(not r['complete_evaluation'] for r in scored['evaluations'])
    assert not (registry.root/'champion.json').exists()


def test_reissues_are_preserved_but_earliest_forecast_selected_before_outcomes(tmp_path):
    source=FixtureSources();registry=ModelRegistry(tmp_path/'registry');p=protocol()
    first=issue_candidates(source,registry,p,output=tmp_path/'first')
    original=source.panel
    def revised_training(*args,**kwargs):
        frame,meta=original(*args,**kwargs);frame=frame.copy()
        frame.iloc[-1,0]+=.0001 # explicitly revised SOFTWARE_FIXTURE, not a future outcome
        return frame,meta
    source.panel=revised_training
    second=issue_candidates(source,registry,p,output=tmp_path/'second')
    assert set(first['forecast_ids']).isdisjoint(second['forecast_ids'])
    with pytest.raises(ValueError,match='same-kind'):
        evaluate_issued(source,registry,p,first['forecast_ids']+second['forecast_ids'],output=tmp_path/'bad',allow_promotion=False)
    r=evaluate_pending(source,registry,p,output=tmp_path/'scored',allow_promotion=False)
    ids=[e['forecast_id'] for c in r['cohorts'] for e in c['evaluations']]
    assert set(ids)==set(first['forecast_ids'])
    assert len(r['ineligible_old_issue_cohorts'])==len(second['forecast_ids'])
    for fid in second['forecast_ids']:registry.store.load('forecasts',fid)
    assert not r['promoted']


def test_complete_daily_month_with_missing_cpi_never_becomes_complete_joint_outcome(tmp_path,monkeypatch):
    import letf.macro.prospective_daily as module
    monkeypatch.setattr(module,'utc_now',lambda:datetime(2027,2,1,tzinfo=timezone.utc))
    dates=xcals.get_calendar('XNYS',start='2026-12-01',end='2027-01-02').sessions_in_range('2026-12-01','2026-12-31').tz_localize(None)
    factors=pd.DataFrame({'Mkt-RF':0.,'RF':0.},index=dates)
    class Outcomes:
        catalog_hash='SOFTWARE_FIXTURE'
        def daily_factors(self):return factors
        def panel(self):
            frame=pd.DataFrame(0.,index=pd.to_datetime(['2026-12-31']),columns=CORE)
            frame.loc[:,'inflation_log']=np.nan
            return frame,{}
    prediction={'kind':'joint_block','monthly_quantiles':[[[0.]*7 for _ in CORE]],'daily_marginal_quantiles':[[0.]*7],'prefix_tails':[]}
    f={'issued_at':'2026-11-25T12:00:00+00:00','outcome_dates':['2026-12-31'],'columns':list(CORE),'common_training_scale':[1.]*5,'models':[prediction]}
    path=tmp_path/'frozen.json';path.write_text(json.dumps(f));before=path.read_bytes()
    result=assess_daily(Outcomes(),path,output=tmp_path/'scored.json')
    assert result['status']=='PENDING_VERIFIED_OUTCOME_DATA' and result['evaluated_months']==0
    assert result['outcome_coverage'][0]['daily_complete']
    assert result['outcome_coverage'][0]['missing_macro_variables']==['inflation_log']
    assert path.read_bytes()==before


def test_account_sampling_intervals_never_promote_one_precise_seed():
    rows=[dict(horizon=40,strategy='X',account='roth',path=i,net_terminal_wealth=0.) for i in range(1000)]
    r=account_sampling_diagnostics(rows)
    assert r['results'][0]['interval_error_goal_passed']
    assert not r['account_convergence_certified']
    tiny=account_sampling_diagnostics(rows[:1])
    assert not tiny['results'][0]['interval_error_goal_passed']
    with pytest.raises(ValueError):account_sampling_diagnostics(rows+rows)
