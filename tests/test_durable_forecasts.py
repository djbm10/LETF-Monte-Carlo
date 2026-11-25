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
