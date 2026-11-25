"""Prospective candidate issuance and later evaluation, never live trading."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.macro.artifacts import canonical,fingerprint,lock
from letf.macro.data import train_slice
from letf.macro.models import MacroModel
from letf.macro.learning import implementation_hash,evaluate_forecast,compare_models


def issue_candidates(sources,registry,protocol,*,output,require_existing=False,asof=None):
    """Retrain on genuinely changed observed training data; freeze forecasts.

    A past current-vintage history is permissible at today's issue timestamp.
    That same snapshot would NOT be permissible for an earlier hindcast origin.
    No candidate is promoted without later, complete, unconsumed outcomes.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    now=datetime.now(timezone.utc)
    issue=pd.Timestamp(asof or now.date())
    if issue.date()>now.date():raise ValueError('Cannot issue a prospective forecast from a future information timestamp')
    if asof and issue.date()!=now.date():
        raise UnsupportedModelError('A past issue timestamp requires genuine archived market vintages, not a backdated current capture')
    ledger=registry.root/'issues.jsonl'
    if require_existing and not ledger.exists():
        result={'status':'BLOCKED','reason':'Persistent registry history absent; cache loss is not authority to reinitialize evidence'}
        (output/'report.json').write_bytes(canonical(result));return result
    panel,meta=sources.panel()
    training=train_slice(panel,issue,protocol['training_window_months'],protocol['minimum_conditional_months'],allow_missing_transitions=True)
    # Verify actual source snapshots were available at this real issue time.
    for name in meta['source_hashes']:
        item=sources.catalog['sources'][name]
        if pd.Timestamp(item['retrieved_at'])>pd.Timestamp(now):raise ValueError('Source capture is after issue time')
    meta=dict(meta,point_in_time_verified=True,verification_scope='Current archived data available at this actual prospective issue timestamp; not historical-origin certification')
    data_id=fingerprint(training)
    spec=dict(protocol,allow_incomplete_training=True,implementation_variant='adjacent-observation-v2')
    code=implementation_hash()
    with lock(registry.root):
        records=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        if any(r['training_sha256']==data_id and r['implementation_sha256']==code and r['protocol_sha256']==sha256(canonical(spec)).hexdigest() for r in records):
            result={'status':'NO_NEW_VERIFIED_TRAINING_DATA','training_sha256':data_id,'models_created':0}
            (output/'report.json').write_bytes(canonical(result));return result
    origin=issue+pd.offsets.MonthEnd(0)
    gap=(origin.to_period('M')-training.index[-1].to_period('M')).n
    forecast_ids=[];models=[]
    for j,kind in enumerate(spec['models']):
        model=MacroModel.fit(kind,training,spec)
        model_meta={'kind':kind,'columns':model.columns,'parameters':model.parameters,
            'training_sha256':data_id,'source_metadata':meta,'protocol_sha256':sha256(canonical(spec)).hexdigest(),
            'implementation_sha256':code,'issuance_code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'issued_at':now.isoformat(),'status':'CANDIDATE_NOT_APPROVED'}
        mid=registry.store.put('models',model_meta,{'training':model.train_values});models.append(mid)
        seed=spec['seed']+j
        paths=model.simulate(spec['evaluation_months']+gap,spec['paths_per_origin'],seed)[:,gap:]
        dates=pd.date_range(origin+pd.offsets.MonthEnd(1),periods=spec['evaluation_months'],freq='ME')
        metadata={'kind':kind,'model_id':mid,'origin':str(origin.date()),'information_asof':str(issue.date()),
            'issued_at':now.isoformat(),'months':spec['evaluation_months'],'dates':[str(d.date()) for d in dates],
            'columns':model.columns,'paths':spec['paths_per_origin'],'seed':seed,'source_metadata':meta,
            'training_end':str(training.index[-1].date()),'training_scale':training.std(ddof=1).tolist(),
            'unobserved_state_bridge_months':gap,'research_status':'PROSPECTIVE_FROZEN_CANDIDATE',
            'evidence':'HYPOTHETICAL_STRESS','strategy_change_authority':False}
        forecast_ids.append(registry.store.put('forecasts',metadata,{'paths':paths}))
    champion_file=registry.root/'champion.json'
    if champion_file.exists():
        champion=json.loads(champion_file.read_text());release=champion['release_id']
        approved,arrays=registry.store.load('models',release)
        incumbent=MacroModel(approved['kind'],approved['columns'],arrays['training'],approved['parameters'])
        if incumbent.columns!=list(training.columns):raise UnsupportedModelError('Approved incumbent schema differs from new data')
        initial=training.iloc[-1].to_numpy()
        # Parameters remain approved/frozen; only the observable initial state
        # is updated for a fair forecast at the same new information date.
        predicted=incumbent.simulate(spec['evaluation_months']+gap,spec['paths_per_origin'],spec['seed']+99,
            initial_state=initial)[:,gap:]
        info=dict(metadata,kind='incumbent_approved',model_id=release,seed=spec['seed']+99,
                  training_end=approved['parameters']['training_end'])
        forecast_ids.append(registry.store.put('forecasts',info,{'paths':predicted}))
    result={'status':'CANDIDATES_ISSUED_PENDING_FUTURE_VALIDATION','issued_at':now.isoformat(),
        'training_sha256':data_id,'implementation_sha256':code,'protocol_sha256':sha256(canonical(spec)).hexdigest(),
        'issuance_code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'forecast_ids':forecast_ids,'model_ids':models,'training_end':str(training.index[-1].date()),
        'first_future_month':str(dates[0].date()),'models_created':len(models),'promoted':False,
        'reason':'Actual future outcomes and disjoint evidence gates are not yet available'}
    with lock(registry.root):
        with ledger.open('a') as f:f.write(json.dumps(result,sort_keys=True)+'\n')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def evaluate_issued(sources,registry,protocol,forecast_ids,*,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    actual,meta=sources.panel();evaluations=[]
    champion_file=registry.root/'champion.json'
    incumbent=json.loads(champion_file.read_text())['release_id'] if champion_file.exists() else None
    implementations=set();specifications=set()
    expected_spec=sha256(canonical(dict(protocol,allow_incomplete_training=True,
        implementation_variant='adjacent-observation-v2'))).hexdigest()
    for fid in forecast_ids:
        f,_=registry.store.load('forecasts',fid)
        if f.get('research_status')!='PROSPECTIVE_FROZEN_CANDIDATE':
            raise UnsupportedModelError('Retrospective forecasts cannot be relabeled as prospective validation')
        outcome_start=pd.Timestamp(f['dates'][0]).to_period('M').start_time.tz_localize('UTC')
        if pd.Timestamp(f['issued_at']).tz_convert('UTC')>=outcome_start:
            raise ValueError('Forecast was not frozen before its outcome period')
        model_meta,_=registry.store.load('models',f['model_id'])
        if f['kind']=='incumbent_approved' and f['model_id']!=incumbent:
            raise UnsupportedModelError('Scored incumbent differs from the exact current approved release')
        if f['kind']!='incumbent_approved':
            implementations.add(model_meta['implementation_sha256']);specifications.add(model_meta['protocol_sha256'])
            if model_meta['protocol_sha256']!=expected_spec:
                raise UnsupportedModelError('Evaluation protocol differs from the immutable forecast registration')
        if len(implementations)>1 or len(specifications)>1:
            raise UnsupportedModelError('Different candidate code/protocol versions require separate registered evidence batches')
        first=pd.Timestamp(f['dates'][0])
        # A newly frozen forecast has no realized equity outcomes yet. This
        # is pending evidence, not permission to initialize a champion.
        if not actual.reindex(pd.to_datetime(f['dates'])).equity_log.notna().any():
            evaluations.append({'forecast_id':fid,'complete_evaluation':False,
                'status':'PENDING_UNOBSERVED_OUTCOMES','evaluation_start':str(first.date())})
            continue
        _,e=evaluate_forecast(registry.store,fid,actual,meta,protocol);evaluations.append(e)
    complete=[e for e in evaluations if e['complete_evaluation']]
    if not complete:
        result={'status':'PENDING_COMPLETE_OUTCOMES','promoted':False,'evaluations':evaluations}
    else:
        decisions=compare_models(complete,protocol)
        if not all(d['gates']['minimum_disjoint_eras'] for d in decisions.values()):
            result={'status':'PENDING_MINIMUM_DISJOINT_ERAS','promoted':False,
                'evaluations':evaluations,'comparisons':decisions,'evidence_consumed':False}
            (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
        completed_ids=[e['forecast_id'] for e in complete]
        windows=sorted(set((e['evaluation_start'],e['evaluation_end']) for e in complete))
        batch=registry.store.put('batches',{'windows':windows,'forecast_ids':forecast_ids,
            'protocol_sha256':sha256(canonical(protocol)).hexdigest()})
        releases={registry.store.load('forecasts',f)[0]['kind']:registry.store.load('forecasts',f)[0]['model_id'] for f in completed_ids}
        # An incumbent release must itself be represented in scored forecasts.
        if incumbent and incumbent not in releases.values():
            raise UnsupportedModelError('Future batch lacks the exact approved incumbent forecast')
        decision=registry.consider(batch,decisions,candidate_release_ids=releases,comparison_incumbent=incumbent)
        result={'status':'EVALUATED','promoted':decision['winner'] is not None,'decision':decision,'evaluations':evaluations}
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def evaluate_pending(sources,registry,protocol,*,output):
    """Evaluate each fixed implementation cohort, excluding consumed intervals.

    Accumulate the required disjoint eras before making a selection decision.
    Monthly overlapping forecast starts are never independent evidence units.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    history=registry.root/'decisions.jsonl';consumed=[]
    if history.exists():
        for line in history.read_text().splitlines():
            batch,_=registry.store.load('batches',json.loads(line)['batch_id']);consumed+=batch['windows']
    ledger=registry.root/'issues.jsonl';groups={}
    champion=registry.root/'champion.json'
    incumbent=json.loads(champion.read_text())['release_id'] if champion.exists() else None
    skipped=[]
    for line in ledger.read_text().splitlines() if ledger.exists() else []:
        issue=json.loads(line);eligible=[]
        records=[registry.store.load('forecasts',fid)[0] for fid in issue['forecast_ids']]
        if incumbent and not any(f['kind']=='incumbent_approved' and f['model_id']==incumbent for f in records):
            skipped.append({'forecast_ids':issue['forecast_ids'],'reason':'Exact current incumbent was not prospectively frozen at this old issue date'})
            continue
        for fid in issue['forecast_ids']:
            f,_=registry.store.load('forecasts',fid);a,b=f['dates'][0],f['dates'][-1]
            if not any(a<=d and c<=b for c,d in consumed):eligible.append(fid)
        if eligible:groups.setdefault((issue['implementation_sha256'],issue['protocol_sha256']),[]).extend(eligible)
    results=[]
    for i,ids in enumerate(groups.values()):
        results.append(evaluate_issued(sources,registry,protocol,ids,output=output/f'cohort-{i}'))
    report={'status':'RESEARCH_EVALUATION_ONLY','cohorts':results,
        'ineligible_old_issue_cohorts':skipped,
        'promoted':any(r['promoted'] for r in results),'strategy_change_authority':False}
    (output/'report.json').write_bytes(canonical(report));return report
