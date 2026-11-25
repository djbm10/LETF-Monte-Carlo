import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from letf.macro.explorer import conditional_daily_bridge,explore
from letf.macro.models import MacroModel
from letf.macro.data import train_slice
from letf.accuracy import UnsupportedModelError


def test_future_bridge_preserves_monthly_log_totals_without_claiming_observed_daily_data():
    idx=xcals.get_calendar('XNYS').sessions_in_range('2027-01-04','2027-02-26').tz_localize(None)
    columns=['equity_log','inflation_log','short_rate','credit_spread','production_log']
    values=np.array([[.02,.002,.06,.01,0.],[-.03,.003,.07,.02,-.01]])
    a=conditional_daily_bridge(values,columns,idx,seed=1,daily_equity_vol=.01,initial_short_rate=.04)
    b=conditional_daily_bridge(values,columns,idx,seed=2,daily_equity_vol=.01,initial_short_rate=.04)
    sums=np.log1p(a.equity_return).groupby(a.index.to_period('M')).sum()
    np.testing.assert_allclose(sums,[.02,-.03],atol=1e-12)
    assert not a.equity_return.equals(b.equity_return)
    assert a.short_rate.iloc[0]==.04 and a.short_rate.iloc[-1]==.06
    crash=conditional_daily_bridge(values,columns,idx,seed=1,daily_equity_vol=.01,
        crash_month=1,crash_return=-.4)
    assert crash.equity_return.min()==pytest.approx(-.4)
    np.testing.assert_allclose(np.log1p(crash.equity_return).groupby(crash.index.to_period('M')).sum(),[.02,-.03],atol=1e-12)


def training():
    rng=np.random.default_rng(44);idx=pd.date_range('2000-01-31','2024-12-31',freq='ME')
    x=rng.normal(size=(len(idx),5))*[.04,.002,.005,.002,.01]+[.006,.002,.04,.01,.002]
    return pd.DataFrame(x,index=idx,columns=['equity_log','inflation_log','short_rate','credit_spread','production_log'])


def test_missing_aware_version_uses_adjacent_pairs_and_does_not_stitch_across_gap():
    spec=json.loads((Path(__file__).resolve().parents[1]/'docs/macro-learning-2026-11-24/protocol.json').read_text())
    x=training();x.iloc[100,1]=np.nan
    with pytest.raises(UnsupportedModelError):train_slice(x,x.index[-1],300,120)
    model=MacroModel.fit('conditional_var_t',train_slice(x,x.index[-1],300,120,allow_missing_transitions=True),dict(spec,allow_incomplete_training=True))
    assert model.parameters['incomplete_training_months']==1
    assert model.parameters['observed_adjacent_transitions']==len(x)-3
    assert np.isnan(model.train_values[100,1])
    assert np.isfinite(model.simulate(12,10,7)).all()


class FixtureSources:
    catalog_hash='fixture-catalog-hash'
    catalog={'sources':{'ff_daily':{'provider':'Independent deterministic fixture','url':'https://example.invalid/fixture',
        'retrieved_at':'2026-01-01T00:00:00+00:00','source_header':['Fixture only'],'sha256':'a'*64}}}
    def panel(self,*args,**kwargs):
        return training(),{'point_in_time_verified':False,'source_hashes':{'ff_daily':'a'*64}}
    def daily_factors(self):
        idx=pd.bdate_range('2000-01-03','2024-12-31');rng=np.random.default_rng(21)
        return pd.DataFrame({'Mkt-RF':rng.normal(.0002,.01,len(idx)),'RF':.0001},index=idx)


def test_macro_retirement_scenario_uses_canonical_tax_and_roth_books(tmp_path):
    r=explore(FixtureSources(),calibration_end='2024-12-31',output=tmp_path/'result',paths=2,horizons=[1],
        annual_contribution=1200.,rate=.06,inflation=.04,strategies=['US1X_CF','US2X_CF'],
        roth_birth_date='1980-01-01',roth_first_tax_year=2010,roth_regular_basis=10000.)
    assert len(r['path_results'])==8
    assert r['evidence']=='HYPOTHETICAL_STRESS' and r['investment_performance_certification']=='BLOCKED'
    assert r['frozen_tax_and_roth_eligibility_year']==2026
    assert all(x['cash_residual_max']<1e-8 and x['nav_residual_max']<1e-7 for x in r['path_results'])
    assert all(x['tax_paid']==0. and x['terminal_tax_payable']==0. for x in r['path_results'] if x['account']=='roth')
    assert all(x['metrics']['cagr'] is None for x in r['path_results'])


def test_retirement_cash_flows_require_owner_history(tmp_path):
    with pytest.raises(UnsupportedModelError,match='explicit owner'):
        explore(FixtureSources(),calibration_end='2024-12-31',output=tmp_path/'missing',annual_contribution=1200.)
