from datetime import datetime,timezone
import json
from pathlib import Path

import pytest

from letf.accuracy import UnsupportedModelError
from letf.macro.learning import ModelRegistry
from letf.macro.retraining import issue_candidates,evaluate_issued
from test_macro_explorer import FixtureSources


def protocol():
    p=json.loads((Path(__file__).resolve().parents[1]/'docs/macro-learning-2026-11-24/protocol.json').read_text())
    return dict(p,paths_per_origin=40,evaluation_months=12,ensemble_score_sample_paths=40)


def test_prospective_freeze_deduplicates_data_and_cannot_promote_before_outcomes(tmp_path):
    registry=ModelRegistry(tmp_path/'registry');sources=FixtureSources();p=protocol()
    report=issue_candidates(sources,registry,p,output=tmp_path/'issue')
    assert report['models_created']==3 and not report['promoted']
    assert report['first_future_month']>str(datetime.now(timezone.utc).date())
    for fid in report['forecast_ids']:
        f,a=registry.store.load('forecasts',fid)
        assert f['source_metadata']['point_in_time_verified']
        assert f['research_status']=='PROSPECTIVE_FROZEN_CANDIDATE'
        assert a['paths'].shape==(40,12,5)
    duplicate=issue_candidates(sources,registry,p,output=tmp_path/'duplicate')
    assert duplicate['status']=='NO_NEW_VERIFIED_TRAINING_DATA' and duplicate['models_created']==0
    evaluation=evaluate_issued(sources,registry,p,report['forecast_ids'],output=tmp_path/'evaluation')
    assert evaluation['status']=='PENDING_COMPLETE_OUTCOMES' and not evaluation['promoted']
    assert not (registry.root/'champion.json').exists()
    assert not (registry.root/'decisions.jsonl').exists()
    changed=json.loads(json.dumps(p));changed['promotion']['maximum_drawdown_95_breach_fraction']=.99
    with pytest.raises(UnsupportedModelError,match='immutable forecast registration'):
        evaluate_issued(sources,registry,changed,report['forecast_ids'],output=tmp_path/'mutated')


def test_lost_registry_and_backdating_fail_closed(tmp_path):
    registry=ModelRegistry(tmp_path/'registry')
    r=issue_candidates(FixtureSources(),registry,protocol(),output=tmp_path/'missing',require_existing=True)
    assert r['status']=='BLOCKED'
    with pytest.raises(UnsupportedModelError,match='backdated'):
        issue_candidates(FixtureSources(),registry,protocol(),output=tmp_path/'backdated',asof='2007-12-31')


def test_current_capture_after_issue_time_is_rejected(tmp_path):
    class FutureSources(FixtureSources):
        catalog={'sources':{'ff_daily':{'retrieved_at':'2099-01-01T00:00:00Z'}}}
    with pytest.raises(ValueError,match='after issue'):
        issue_candidates(FutureSources(),ModelRegistry(tmp_path/'registry'),protocol(),output=tmp_path/'future')


def test_tax_verification_strings_are_not_python_truthiness():
    from letf.account_cli import _verified_flag
    assert not _verified_flag('False') and _verified_flag('True')
    assert not _verified_flag(float('nan'))
    with pytest.raises(ValueError):_verified_flag('unreviewed')
