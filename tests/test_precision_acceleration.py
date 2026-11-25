from math import comb
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from letf.macro.precision import MonthlyLookup,period_summary,compose_periods,quantile_interval,exact_probability
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage
from letf.macro.daily_tails import training_templates,daily_bridge,realized_monthly_variance
from letf.reconstruction import SourceRecord,daily_reset_returns
from letf.macro.precision_run import generate_block,stream_seeds,assess


def fixture():
    idx=pd.bdate_range('2000-01-03','2019-12-31');rng=np.random.default_rng(989)
    logs=pd.Series(rng.normal(.0003,.015,len(idx)),index=idx)
    frame=pd.DataFrame({'equity_log':logs.resample('ME').sum(),
        'equity_log_variance':realized_monthly_variance(logs)})
    frame['short_rate']=np.resize([.0,.02,.06],len(frame))
    p={'block_months':3,'stationary_mean_months':12,'minimum_bootstrap_months':120}
    model=MacroModel.fit('stationary_block',frame,p)
    return model,training_templates(logs,frame)


def test_exact_quantile_ranks_match_enumerated_binomial_coverage_and_zero_atom():
    r=quantile_interval(np.arange(10.),.5,.05)
    assert (r['lower_rank'],r['upper_rank'],r['low'],r['high'])==(2,9,1.,8.)
    assert sum(comb(10,k)*.5**10 for k in range(2,9))>=.95
    z=quantile_interval(np.full(100,-np.inf),.1,.05)
    assert z['status']=='IDENTIFIED_ZERO_ATOM' and z['maximum_error']==0.
    p=exact_probability(0,100,.05)
    assert p['high']==pytest.approx(1-.025**.01)
    with pytest.raises(ValueError):quantile_interval([np.nan],.1,.05)


def test_month_composition_matches_daily_cash_growth_drawdown_and_absorbing_loss():
    r=np.array([[.1,-.2,.15,-.1,.03,.05],[.1,-1.1,10.,.5,.5,.5],[.1,.1,.1,-.5,.1,.1]])
    summaries=period_summary(r.reshape(3,2,3));lognav,dd,dead,_=compose_periods(summaries)
    for i,row in enumerate(r):
        wealth=1.;peak=1.;worst=0.;closed=False
        for j,v in enumerate(row):
            if not closed:
                if v<=-1:wealth=0.;closed=True
                else:wealth*=1+v
            peak=max(peak,wealth);worst=max(worst,1-wealth/peak)
            if j%3==2:
                assert dd[i,j//3]==pytest.approx(worst,abs=2e-15)
                assert bool(dead[i,j//3])==closed
                assert (np.isneginf(lognav[i,j//3]) if closed else np.exp(lognav[i,j//3])==pytest.approx(wealth))
    # Floating underflow must not become a fictitious product liquidation.
    tiny=np.full((1,10,100),-.9)
    nav,_,liquidated,_=compose_periods(period_summary(tiny))
    assert np.isfinite(nav).all() and nav[0,-1]<-2000 and not liquidated.any()


def test_lookup_matches_frozen_bridge_and_canonical_daily_reset_for_every_path():
    model,(templates,lengths)=fixture()
    monthly,lineage=sample_with_lineage(model,60,33,42)
    lookup=MonthlyLookup.build(model,templates,lengths)
    initial=np.full(33,lookup.rate_indices[-1])
    results,shape_ids=lookup.evaluate(lineage,initial,100,horizons=(1,5))
    logs=daily_bridge(monthly,lineage,model.columns,templates,lengths,100)
    reconstructed=lookup.shapes[shape_ids].reshape(logs.shape)
    np.testing.assert_array_equal(logs,reconstructed)
    source=SourceRecord('fixture','https://example.invalid','2026-11-25','fixture','a'*64,'2000-01-01','USD','daily TR','daily',False)
    dates=pd.date_range('2000-01-01',periods=logs.shape[1],freq='D')
    j=model.columns.index('short_rate')
    for p in range(33):
        rate=np.repeat(np.r_[model.train_values[-1,j],monthly[p,:-1,j]],21)
        for leverage in (1,2,3):
            r=daily_reset_returns(pd.Series(np.expm1(logs[p]),index=dates),leverage,source=source,
                annual_fee=.01 if leverage>1 else 0.,borrowing_spread=.015,
                funding_rate_at_previous_close=pd.Series(rate,index=dates))
            growth=np.cumprod(1+r['return'].to_numpy());peak=np.maximum.accumulate(np.r_[1.,growth])[1:]
            for h in (1,5):
                row=results[h,leverage];last=h*252
                assert row['log_nav'][p]==pytest.approx(np.log(growth[last-1]),abs=2e-13)
                assert row['drawdown'][p]==pytest.approx(np.max(1-growth[:last]/peak[:last]),abs=2e-14)
                assert bool(row['liquidated'][p])==bool(r.liquidated.iloc[last-1])
    empty,ix=sample_with_lineage(model,60,33,42,return_values=False)
    assert empty is None;np.testing.assert_array_equal(ix,lineage)


def test_fixed_stream_blocks_are_repeatable_independent_and_exclude_generation_order():
    model,(templates,lengths)=fixture();lookup=MonthlyLookup.build(model,templates,lengths)
    p={'fixed_stream_block_paths':32,'horizons':[1,5],'leverages':[1,2,3],'origin':'2020-01-31'}
    first=generate_block(model,lookup,p,123,0);second=generate_block(model,lookup,p,123,1)
    np.testing.assert_array_equal(first,generate_block(model,lookup,p,123,0))
    np.testing.assert_array_equal(second,generate_block(model,lookup,p,123,1))
    assert not np.array_equal(first,second)
    assert stream_seeds(123,0)!=stream_seeds(124,0)
