"""Independent proper-score, chronology, covariance and immutable-history tests."""
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from letf.accuracy import UnsupportedModelError
from letf.macro.artifacts import ArtifactStore,fingerprint
from letf.macro.data import french_daily,train_slice
from letf.macro.models import MacroModel
from letf.macro.scoring import ensemble_crps,evaluate_paths,drawdowns
from letf.macro.learning import forecast_training,evaluate_forecast,compare_models,ModelRegistry


def protocol():return json.loads((Path(__file__).resolve().parents[1]/'docs/macro-learning-2026-11-24/protocol.json').read_text())


def training(n=240):
    rng=np.random.default_rng(17)
    shock=rng.normal(size=(n,5));shock[:,1]+=.4*shock[:,0]
    scale=np.array([.04,.002,.01,.004,.01]);mean=np.array([.006,.002,.03,.01,.001])
    return pd.DataFrame(shock*scale+mean,index=pd.date_range('1980-01-31',periods=n,freq='ME'),
        columns=['equity_log','inflation_log','short_rate','credit_spread','production_log'])


def test_crps_matches_independent_pairwise_definition_and_perfect_prediction():
    x=np.array([-2.,0.,1.,5.]);y=.3
    direct=np.mean(abs(x-y))-.5*np.mean(abs(x[:,None]-x[None,:]))
    assert ensemble_crps(x,y)==pytest.approx(direct)
    assert ensemble_crps(np.full(10,3.),3.)==0.
    assert ensemble_crps(np.zeros(10),3.)==3.


def test_french_parser_never_merges_second_equal_weight_block_or_fills_sentinel():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:z.writestr('data.csv','Header\n,Mkt-RF,SMB,HML,RF\n19260701,1,2,3,.01\n19260702,-1,0,0,.01\n\nEqual weighted\n19260701,99,0,0,0\n')
    x=french_daily(buf.getvalue())
    assert len(x)==2 and x.iloc[0,0]==.01
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:z.writestr('data.csv',',Mkt-RF,SMB,HML,RF\n19260701,-99.99,2,3,.01\n')
    with pytest.raises(UnsupportedModelError):french_daily(buf.getvalue())


@pytest.mark.parametrize('kind',['joint_block','regime_block','conditional_var_t'])
def test_joint_models_repeat_without_global_rng_or_future_information(kind):
    x=training();m=MacroModel.fit(kind,x,protocol())
    np.random.seed(123);a=np.random.random();left=m.simulate(24,32,44);b=np.random.random()
    np.random.seed(123);assert a==np.random.random();assert b==np.random.random()
    np.testing.assert_array_equal(left,m.simulate(24,32,44))
    assert left.shape==(32,24,5)
    if kind=='joint_block':
        for p in left:
            assert any(np.array_equal(p[:3],x.iloc[i:i+3].to_numpy()) for i in range(len(x)-2))


def test_multivariate_t_innovations_have_the_declared_covariance_not_raw_t_variance():
    m=MacroModel.fit('conditional_var_t',training(),protocol())
    out=m.simulate(1,100000,81)[:,0]
    scale=np.asarray(m.parameters['scale'])
    expected=np.asarray(m.parameters['covariance'])*scale[:,None]*scale[None,:]
    np.testing.assert_allclose(np.diag(np.cov(out,rowvar=False)),np.diag(expected),rtol=.06)


def test_joint_student_conditioning_sets_rates_and_preserves_free_uncertainty():
    m=MacroModel.fit('conditional_var_t',training(),protocol())
    out=m.simulate(6,2000,81,constraints={'short_rate':.06,'inflation_log':np.log1p(.04)/12})
    np.testing.assert_allclose(out[:,:,2],.06,atol=1e-15)
    np.testing.assert_allclose(out[:,:,1],np.log1p(.04)/12,atol=1e-15)
    assert out[:,:,0].std()>.001
    with pytest.raises(UnsupportedModelError):MacroModel.fit('joint_block',training(),protocol()).simulate(5,2,1,constraints={'short_rate':.06})


def test_forecast_is_content_addressed_and_frozen_before_scoring(tmp_path):
    store=ArtifactStore(tmp_path);x=training();origin=x.index[-1]
    meta={'point_in_time_verified':False,'source_hashes':{'x':'fixture'}}
    fid=forecast_training(store,x,meta,origin,'joint_block',protocol(),seed=3,months=12,paths=64)
    manifest=next((tmp_path/'forecasts').rglob('manifest.json'));before=manifest.read_bytes()
    actual=training(252).iloc[-12:]
    _,evaluation=evaluate_forecast(store,fid,actual,meta,protocol())
    assert evaluation['forecast_was_frozen_before_scoring']
    assert manifest.read_bytes()==before
    changed=actual*20
    evaluate_forecast(store,fid,changed,meta,protocol())
    assert manifest.read_bytes()==before
    assert len(list((tmp_path/'evaluations').iterdir()))==2
    arrays=manifest.parent/'arrays.npz';arrays.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='tampered'):store.load('forecasts',fid)


def test_future_rows_cannot_change_fitted_training_prefix_and_missing_month_rejects():
    x=training(300);origin=x.index[239]
    a=train_slice(x,origin,240,120)
    altered=x.copy();altered.loc[altered.index>origin]=100000
    b=train_slice(altered,origin,240,120)
    assert fingerprint(a)==fingerprint(b)
    damaged=x.copy();damaged.iloc[20,0]=np.nan
    with pytest.raises(UnsupportedModelError,match='Missing interior'):train_slice(damaged,origin,240,120)


def test_drawdown_counts_initial_loss_and_scores_report_missing_macro_outcomes():
    assert drawdowns(np.log([.8,1.1]))==pytest.approx(.2)
    x=training(12);paths=np.broadcast_to(x.to_numpy(),(64,12,5)).copy()
    actual=x.to_numpy().copy();actual[3,1]=np.nan
    s=evaluate_paths(paths,actual,np.ones(5),list(x.columns),score_paths=32)
    assert s['primary_crps']==pytest.approx(0)
    assert s['marginals']['inflation_log']['missing_months']==1
    assert s['complete_joint_months']==11


def test_registry_rejection_consumes_evidence_and_relabeling_cannot_reuse_eras(tmp_path):
    registry=ModelRegistry(tmp_path)
    batch=registry.store.put('batches',{'windows':[['2000-01-31','2004-12-31']]})
    r=registry.consider(batch,{'candidate':{'passed':False,'gates':{'tail':False}}},candidate_release_ids={})
    assert r['winner'] is None and not (tmp_path/'champion.json').exists()
    with pytest.raises(ValueError,match='already consumed'):
        registry.consider(batch,{},candidate_release_ids={})
    relabeled=registry.store.put('batches',{'windows':[['2002-01-31','2005-12-31']],'label':'new'})
    with pytest.raises(ValueError,match='overlaps'):
        registry.consider(relabeled,{},candidate_release_ids={})


def test_comparison_rejects_good_average_fit_with_bad_tail_or_unverified_vintages():
    from copy import deepcopy
    spec=protocol();evaluations=[]
    for year in [1950,1960,1970,1980,1990]:
        for kind,loss in [('joint_block',1.),('regime_block',.7),('conditional_var_t',.6)]:
            evaluations.append({'origin':f'{year-1}-12-31','evaluation_start':f'{year}-01-31',
                'evaluation_end':f'{year+4}-12-31','complete_evaluation':True,'point_in_time_verified':True,'model':kind,
                'score':{'primary_crps':loss,'paths':2000,'drawdown_95_breach':False,
                    'marginals':{'equity_log':{'coverage80':.8,'lower_5_breach_rate':.05}}}})
    result=compare_models(evaluations,spec)
    assert result['regime_block']['passed'] and result['conditional_var_t']['passed']
    bad=deepcopy(evaluations)
    for e in bad:
        if e['model']=='conditional_var_t':e['score']['marginals']['equity_log']['lower_5_breach_rate']=.5
        if e['model']=='regime_block':e['point_in_time_verified']=False
    result=compare_models(bad,spec)
    assert not result['regime_block']['passed'] and not result['conditional_var_t']['passed']


def test_registry_approval_requires_all_gates_and_preserves_prior_decisions(tmp_path):
    from letf.macro.learning import REQUIRED_GATES
    registry=ModelRegistry(tmp_path)
    model=registry.store.put('models',{'kind':'candidate','source_metadata':{'point_in_time_verified':True}})
    batch=registry.store.put('batches',{'windows':[['2000-01-31','2004-12-31']]})
    with pytest.raises(ValueError,match='every registered'):
        registry.consider(batch,{'candidate':{'passed':True,'gates':{'fake':True},'relative_crps_improvement':.1}},candidate_release_ids={'candidate':model})
    approved={'candidate':{'passed':True,'gates':dict.fromkeys(REQUIRED_GATES,True),'relative_crps_improvement':.1}}
    event=registry.consider(batch,approved,candidate_release_ids={'candidate':model})
    assert event['winner']=='candidate'
    before=(tmp_path/'decisions.jsonl').read_bytes()
    new=registry.store.put('batches',{'windows':[['2010-01-31','2014-12-31']]})
    with pytest.raises(UnsupportedModelError,match='incumbent'):
        registry.consider(new,approved,candidate_release_ids={'candidate':model})
    registry.consider(new,{'candidate':{'passed':False,'gates':{'tail':False}}},
        candidate_release_ids={'candidate':model},comparison_incumbent=model)
    assert (tmp_path/'decisions.jsonl').read_bytes().startswith(before)
    assert json.loads((tmp_path/'champion.json').read_text())['release_id']==model


def test_extended_market_closure_requires_evidence_and_accrues_calendar_costs():
    from letf.reconstruction import SourceRecord,daily_reset_returns
    src=SourceRecord('fixture','https://example.invalid','2026-11-24','fixture','a'*64,
        '1933-03-02','USD','daily total return','daily',False)
    idx=pd.to_datetime(['1933-03-02','1933-03-03','1933-03-15','1933-03-16'])
    returns=pd.Series([0.,0.,.1,0.],index=idx);days=pd.Series([1.,1.,12.,1.],index=idx)
    with pytest.raises(UnsupportedModelError,match='closure evidence'):
        daily_reset_returns(returns,3,source=src,elapsed_calendar_days=days)
    notice={'1933-03-15':{'source':'Independent dated closure fixture','elapsed_days':12,'previous_close':'1933-03-03'}}
    r=daily_reset_returns(returns,3,source=src,annual_fee=.01,funding_rate_at_previous_close=.06,
        elapsed_calendar_days=days,verified_closures=notice)
    assert r['return'].iloc[2]==pytest.approx(.3-.01*12/365-2*.06*12/360)
