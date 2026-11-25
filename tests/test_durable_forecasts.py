from hashlib import sha256
import json
import numpy as np
import pytest
from letf.macro.artifacts import ArtifactStore
from letf.macro.data import HistoricalSources
from letf.macro.prospective_daily import issue_daily,assess_daily,weighted_interval_score,seal_snapshot,verify_snapshot
from test_macro_explorer import FixtureSources


def test_proper_interval_score_and_actual_future_forecasts_are_not_assessed_early(tmp_path):
    assert weighted_interval_score([0.]*7,0.)==0.
    assert weighted_interval_score([0.]*7,2.)==pytest.approx(2.)
    source=FixtureSources();path=tmp_path/'public.json'
    f=issue_daily(source,registry=tmp_path/'registry',output=path,paths=40,months=3)
    assert len(f['models'])==2 and not f['model_approved']
    before=path.read_bytes()
    class NoOutcomeLoad:
        def panel(self):raise AssertionError('Future outcomes must not be loaded')
    r=assess_daily(NoOutcomeLoad(),path,output=tmp_path/'pending.json')
    assert r['status']=='PENDING_GENUINE_FUTURE_OUTCOMES' and not r['promoted']
    assert path.read_bytes()==before
    with pytest.raises(ValueError):issue_daily(source,registry=tmp_path/'registry',output=path,paths=40)


def test_artifact_write_failure_never_publishes_a_partial_identity(tmp_path,monkeypatch):
    store=ArtifactStore(tmp_path/'registry');original=np.savez_compressed
    def fail(*args,**kwargs):raise OSError('Simulated disk interruption')
    with monkeypatch.context() as m:
        m.setattr(np,'savez_compressed',fail)
        with pytest.raises(OSError):store.put('forecasts',{'case':1},{'x':np.arange(3.)})
    assert not list((tmp_path/'registry/forecasts').iterdir())
    fid=store.put('forecasts',{'case':1},{'x':np.arange(3.)})
    np.testing.assert_array_equal(store.load('forecasts',fid)[1]['x'],np.arange(3.))


def test_portable_source_and_forecast_archive_verifies_every_byte_and_rejects_extra_files(tmp_path):
    source=tmp_path/'sources';source.mkdir();(source/'input.csv').write_text('date,X\n2024-01-01,1\n')
    catalog={'sources':{'X':{'path':'input.csv','sha256':sha256((source/'input.csv').read_bytes()).hexdigest()}}}
    (source/'catalog.json').write_text(json.dumps(catalog));sources=HistoricalSources(source)
    registry=tmp_path/'registry';store=ArtifactStore(registry)
    store.put('forecasts',{'fixture':True},{'x':np.arange(4.)})
    commitment=tmp_path/'public.json';commitment.write_text('{"fixture":true}')
    archive=tmp_path/'sealed.tar.gz';r=seal_snapshot(sources,registry,commitment,archive)
    assert r['status']=='VERIFIED_LOCAL_ARCHIVE' and not r['off_host_backup_verified']
    assert verify_snapshot(archive)==r
    with pytest.raises(ValueError):seal_snapshot(sources,registry,commitment,archive)
    (registry/'.env').write_text('FAKE_TEST_ONLY')
    with pytest.raises(ValueError,match='Unreviewed'):seal_snapshot(sources,registry,commitment,tmp_path/'other.tar.gz')


def test_elapsed_assessment_scores_frozen_tails_and_rejects_partial_daily_month(tmp_path,monkeypatch):
    # Artificial clock/data are test-only. Production CLI has no backdating flag.
    from datetime import datetime,timezone
    import pandas as pd
    import exchange_calendars as xcals
    import letf.macro.prospective_daily as module
    from letf.macro.data import CORE
    monkeypatch.setattr(module,'utc_now',lambda:datetime(2027,2,1,tzinfo=timezone.utc))
    dates=xcals.get_calendar('XNYS',start='2026-12-01',end='2027-01-01').sessions_in_range('2026-12-01','2026-12-31').tz_localize(None)
    factors=pd.DataFrame({'Mkt-RF':0.,'RF':0.},index=dates);factors.iloc[0,0]=-.06
    class Outcomes:
        catalog_hash='test-only'
        def panel(self):
            p=pd.DataFrame(0.,index=pd.to_datetime(['2026-12-31']),columns=CORE)
            p.loc[:,'equity_log']=np.log(.94)
            return p,{}
        def daily_factors(self):return factors
    prediction={'kind':'joint_block','monthly_quantiles':[[[0.]*7 for _ in CORE]],'daily_marginal_quantiles':[[0.]*7],
        'prefix_tails':[{'months':1,'max_drawdown_p50_p95_p99':[.02,.05,.07],
            'underwater_sessions_p50_p95':[3.,5.],'crash_rate_per252_p50_p95':{'-0.05':[0.,2.]}}]}
    commitment={'issued_at':'2026-11-25T12:00:00+00:00','outcome_dates':['2026-12-31'],'columns':list(CORE),
        'common_training_scale':[1.]*len(CORE),'models':[prediction]}
    path=tmp_path/'frozen.json';path.write_text(json.dumps(commitment));before=path.read_bytes()
    result=assess_daily(Outcomes(),path,output=tmp_path/'scored.json')
    row=result['evaluated_prefix_tails'][0]
    assert row['observed_max_drawdown']==pytest.approx(.06)
    assert row['drawdown_exceeds_p95'] and not row['drawdown_exceeds_p99']
    assert row['observed_underwater_sessions']==len(dates) and row['underwater_exceeds_p95']
    assert row['observed_crash_rate_per252']['-0.05']==pytest.approx(252/len(dates))
    assert row['crash_rate_exceeds_p95']['-0.05']
    assert result['evaluated_rows'][0]['daily_mean_wis']==pytest.approx(abs(np.log(.94))/len(dates))
    assert not result['promoted'] and path.read_bytes()==before
    factors=factors.iloc[:-1]
    pending=assess_daily(Outcomes(),path,output=tmp_path/'incomplete.json')
    assert pending['status']=='PENDING_VERIFIED_OUTCOME_DATA'
    assert pending['evaluated_rows']==pending['evaluated_prefix_tails']==[]
