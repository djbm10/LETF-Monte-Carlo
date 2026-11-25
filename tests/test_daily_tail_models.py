"""Independent block-duration, daily compounding and lineage checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from letf.accuracy import UnsupportedModelError
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage,EXTENDED_BOOTSTRAPS
from letf.macro.daily_tails import (realized_monthly_variance,training_templates,daily_bridge,
    evaluate_daily,longest_underwater,_crps_many)
from letf.macro.scoring import ensemble_crps
from letf.macro.tail_study import daily_gates
from letf.macro.learning import ModelRegistry,REQUIRED_GATES


def fixture():
    p=json.loads((Path(__file__).resolve().parents[1]/'docs/tail-validation-2026-11-24/protocol.json').read_text())
    rng=np.random.default_rng(90);idx=pd.bdate_range('1990-01-01','2009-12-31')
    daily=pd.Series(rng.standard_t(6,len(idx))*.01,index=idx)
    frame=pd.DataFrame({'equity_log':daily.resample('ME').sum()})
    for j,k in enumerate(('inflation_log','short_rate','credit_spread','production_log')):
        frame[k]=rng.normal(.001,.01,len(frame))+frame.equity_log*.1*(j+1)
    frame['equity_log_variance']=realized_monthly_variance(daily)
    return p,frame,daily


@pytest.mark.parametrize('kind',['joint_block',*EXTENDED_BOOTSTRAPS])
def test_lineage_recovers_joint_observations_and_rng_is_local(kind):
    p,x,d=fixture();m=MacroModel.fit(kind,x,p)
    np.random.seed(88);expected=np.random.random(2);np.random.seed(88)
    assert np.random.random()==expected[0]
    out,ix=sample_with_lineage(m,120,20,31)
    assert np.random.random()==expected[1]
    np.testing.assert_array_equal(out,x.to_numpy()[ix])
    np.testing.assert_array_equal(out,m.simulate(120,20,31))
    np.testing.assert_array_equal(ix,sample_with_lineage(m,120,20,31)[1])


def test_geometric_duration_matches_registered_mean_and_missing_rows_force_restart():
    p,x,_=fixture();p=dict(p,stationary_mean_months=12)
    m=MacroModel.fit('stationary_block',x,p)
    _,ix=sample_with_lineage(m,20000,5,213)
    breaks=ix[:,1:]!=(ix[:,:-1]+1)%len(x)
    assert breaks.mean()==pytest.approx((1/12)*(1-1/len(x)),abs=.003)
    x.iloc[33:36]=np.nan;m=MacroModel.fit('stationary_block',x,p)
    out,ix=sample_with_lineage(m,1000,30,213)
    assert np.isfinite(out).all() and not np.isin(ix,[33,34,35]).any()
    # 32 cannot be followed by a fabricated/filled 33.
    assert not np.any((ix[:,:-1]==32)&(ix[:,1:]==33))


def test_daily_bridge_preserves_monthly_return_variance_and_chunk_identity():
    p,x,d=fixture();m=MacroModel.fit('volatility_stationary_block',x,p)
    monthly,ix=sample_with_lineage(m,12,41,12);templates,lengths=training_templates(d,x)
    daily=daily_bridge(monthly,ix,m.columns,templates,lengths,900)
    grouped=daily.reshape(41,12,21)
    np.testing.assert_allclose(grouped.sum(2),monthly[:,:,0],atol=1e-15)
    np.testing.assert_allclose(((grouped-grouped.mean(2,keepdims=True))**2).sum(2),np.exp(monthly[:,:,-1]),rtol=1e-12)
    # Equivalent multiplicative total returns, not just additive simple returns.
    np.testing.assert_allclose(np.prod(np.exp(grouped),axis=2),np.exp(monthly[:,:,0]),rtol=1e-12)
    chunks=[]
    for ids in (np.arange(0,13),np.arange(13,41)):
        chunks.append(daily_bridge(monthly[ids],ix[ids],m.columns,templates,lengths,900,path_ids=ids))
    np.testing.assert_array_equal(np.concatenate(chunks),daily)
    with pytest.raises(ValueError,match='Future daily'):
        training_templates(pd.concat([d,pd.Series([.1],index=[pd.Timestamp('2010-01-04')])]),x)
    damaged=x.copy();damaged.iloc[0,-1]+=.1
    with pytest.raises(ValueError,match='variance mismatch'):training_templates(d,damaged)


def test_daily_scores_match_pairwise_crps_and_account_for_actual_session_count():
    samples=np.array([-2.,0.,1.,5.]);truth=np.array([-3.,.3,6.])
    np.testing.assert_allclose(_crps_many(samples,truth),[ensemble_crps(samples,t) for t in truth])
    p,x,d=fixture();m=MacroModel.fit('joint_block',x,p)
    monthly,ix=sample_with_lineage(m,2,100,12);templates,lengths=training_templates(d,x)
    predicted=daily_bridge(monthly,ix,m.columns,templates,lengths,900)
    observed=d.loc['1995-01-01':'1995-02-28']
    score=evaluate_daily(predicted,observed,pd.to_datetime(['1995-01-31','1995-02-28']),d.std())
    assert score['observed_sessions']==len(observed) and score['synthetic_sessions']==42
    assert score['normalized_daily_marginal_crps']>0
    assert longest_underwater(np.log([.8,1.,1.1,1.2]))[0]==3
    with pytest.raises(ValueError,match='Missing/extra'):evaluate_daily(predicted,observed.iloc[:15],pd.to_datetime(['1995-01-31','1995-02-28']),d.std())


def test_new_tail_gates_reject_inaccurate_tails_despite_good_average_coverage():
    p,_,_=fixture()
    scores=[{'drawdown_95_breach':i<2,'coverage95':.95,'crashes_per_252_sessions':{
        str(c):{'upper95_breach':False} for c in (-.05,-.10)}} for i in range(5)]
    gates,_=daily_gates(scores,p['daily_tail_gate'])
    assert not gates['daily_drawdown_tail95'] and gates['daily_coverage95']


def test_registry_cannot_approve_a_tail_batch_with_only_original_monthly_gates(tmp_path):
    registry=ModelRegistry(tmp_path)
    batch=registry.store.put('batches',{'windows':[['1955-01-31','1959-12-31']],
        'additional_required_gates':['daily_drawdown_tail95']})
    decision={'candidate':{'passed':True,'gates':dict.fromkeys(REQUIRED_GATES,True),'relative_crps_improvement':.1}}
    with pytest.raises(ValueError,match='every registered'):registry.consider(batch,decision,candidate_release_ids={})
    decision['candidate']['gates']['daily_drawdown_tail95']=False
    with pytest.raises(ValueError,match='every registered'):registry.consider(batch,decision,candidate_release_ids={})


def test_extended_bootstraps_fail_explicitly_for_unregistered_independent_shocks():
    p,x,_=fixture();m=MacroModel.fit('long_joint_block',x,p)
    with pytest.raises(UnsupportedModelError):m.simulate(12,5,3,constraints={'short_rate':.06})
    with pytest.raises(ValueError):sample_with_lineage(m,0,5,1)
