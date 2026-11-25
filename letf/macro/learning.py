"""Forecast-first walk-forward research and controlled model registry.

A scientific rejection is persisted as evidence. No function modifies trading
rules, data snapshots, prior model artifacts, historical holdouts or main.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.macro.artifacts import ArtifactStore, canonical, fingerprint, lock
from letf.macro.data import train_slice
from letf.macro.models import MacroModel
from letf.macro.scoring import evaluate_paths
from letf.simulation.paired import paired_mean_difference_ci

REQUIRED_GATES={'minimum_disjoint_eras','registered_path_count','relative_crps_improvement',
    'paired_ci','coverage80','equity_tail5','drawdown_tail95','decision_time_vintages','incumbent_comparison'}


def implementation_hash():
    return sha256(b''.join(Path(__file__).with_name(n).read_bytes() for n in
        ('learning.py','models.py','data.py','scoring.py','artifacts.py'))).hexdigest()


def forecast_training(store,training,source_meta,origin,kind,protocol,*,seed,months,paths):
    origin=pd.Timestamp(origin)
    if training.index[-1]>origin:raise ValueError('Training includes future observations')
    model=MacroModel.fit(kind,training,protocol)
    model_meta={'kind':kind,'columns':model.columns,'parameters':model.parameters,
        'training_sha256':fingerprint(training),'source_metadata':source_meta,
        'protocol_sha256':sha256(canonical(protocol)).hexdigest(),'implementation_sha256':implementation_hash()}
    mid=store.put('models',model_meta,{'training':model.train_values})
    gap=(origin.to_period('M')-training.index[-1].to_period('M')).n
    generated=model.simulate(months+gap,paths,seed)[:,gap:]
    dates=pd.date_range(origin+pd.offsets.MonthEnd(1),periods=months,freq='ME')
    forecast_meta={'model_id':mid,'kind':kind,'origin':str(origin.date()),'months':months,
        'dates':[str(d.date()) for d in dates],'paths':paths,'seed':seed,'columns':model.columns,
        'unobserved_state_bridge_months':gap,'training_end':str(training.index[-1].date()),
        'source_metadata':source_meta,'training_scale':training.std(ddof=1).tolist(),
        'research_status':'RETROSPECTIVE_CHRONOLOGICAL_HINDCAST',
        'never_claimed':'A forecast made in real time or an untouched human/universe holdout',
        'evidence':'HYPOTHETICAL_STRESS'}
    fid=store.put('forecasts',forecast_meta,{'paths':generated})
    return fid


def evaluate_forecast(store,forecast_id,actual,actual_metadata,protocol):
    forecast,arrays=store.load('forecasts',forecast_id)
    dates=pd.to_datetime(forecast['dates'])
    available=actual.reindex(dates)[forecast['columns']]
    # Score only the elapsed contiguous prefix, never a fabricated future.
    present=np.flatnonzero(available.equity_log.notna().to_numpy())
    if not len(present) or not np.array_equal(present,np.arange(present[-1]+1)):
        raise UnsupportedModelError('Missing/unevaluated equity outcome interval')
    n=present[-1]+1
    score=evaluate_paths(arrays['paths'][:,:n],available.iloc[:n].to_numpy(),
        np.asarray(forecast['training_scale']),forecast['columns'],
        score_paths=protocol['ensemble_score_sample_paths'],seed=protocol['seed'])
    result={'forecast_id':forecast_id,'model':forecast['kind'],'origin':forecast['origin'],
        'evaluation_start':str(dates[0].date()),'evaluation_end':str(dates[n-1].date()),
        'complete_evaluation':bool(n==len(dates)),'score':score,
        'actual_sha256':fingerprint(available.iloc[:n]),'actual_metadata':actual_metadata,
        'forecast_was_frozen_before_scoring':True,
        'point_in_time_verified':bool(forecast['source_metadata']['point_in_time_verified'])}
    eid=store.put('evaluations',result)
    return eid,result


def compare_models(evaluations,protocol):
    """Disjoint-era paired tests, fixed gates, familywise CI correction."""
    by={}
    for e in evaluations:
        if e['complete_evaluation']:
            group=by.setdefault(e['origin'],{})
            if e['model'] in group:raise ValueError('Duplicate model/origin evaluations cannot be selected by overwrite')
            group[e['model']]=e
    incumbent_present=any(e['model']=='incumbent_approved' for e in evaluations)
    common=[];last_end=None
    for origin,models in sorted(by.items()):
        if set(protocol['models'])-set(models):continue
        if incumbent_present and 'incumbent_approved' not in models:continue
        first=models['joint_block']
        if last_end is not None and first['evaluation_start']<=last_end:continue
        common.append(models);last_end=first['evaluation_end']
    policy=protocol['promotion'];decisions={}
    confidence=1-.05/max(1,len(protocol['models'])-1)
    for model in protocol['models']:
        if model=='joint_block':continue
        candidate=[g[model] for g in common];baseline=[g['joint_block'] for g in common]
        losses=[e['score']['primary_crps'] for e in candidate]
        ref=[e['score']['primary_crps'] for e in baseline]
        improvement=1-np.mean(losses)/np.mean(ref) if losses else None
        interval=paired_mean_difference_ci(losses,ref,seed=protocol['seed'],confidence=confidence) if len(losses)>=2 else None
        incumbent_ci=None;incumbent_gain=None
        if incumbent_present and len(losses)>=2:
            inc=[g['incumbent_approved']['score']['primary_crps'] for g in common]
            incumbent_ci=paired_mean_difference_ci(losses,inc,seed=protocol['seed'],confidence=confidence)
            incumbent_gain=1-np.mean(losses)/np.mean(inc)
        coverage=[v['coverage80'] for e in candidate for v in e['score']['marginals'].values() if v['coverage80'] is not None]
        breaches=[e['score']['marginals']['equity_log']['lower_5_breach_rate'] for e in candidate]
        dd=[e['score']['drawdown_95_breach'] for e in candidate]
        gates={'minimum_disjoint_eras':len(candidate)>=policy['minimum_disjoint_eras'],
            'registered_path_count':bool(candidate and all(e['score']['paths']>=protocol['paths_per_origin'] for e in candidate)),
            'relative_crps_improvement':improvement is not None and improvement>=policy['minimum_relative_crps_improvement'],
            'paired_ci':interval is not None and interval['high']<=policy['paired_era_mean_loss_difference_ci_upper'],
            'coverage80':bool(coverage and policy['minimum_80_interval_coverage']<=np.mean(coverage)<=policy['maximum_80_interval_coverage']),
            'equity_tail5':bool(breaches and np.mean(breaches)<=policy['maximum_5_percent_breach_rate']),
            'drawdown_tail95':bool(dd and np.mean(dd)<=policy['maximum_drawdown_95_breach_fraction']),
            'decision_time_vintages':bool(candidate and all(e['point_in_time_verified'] for e in candidate))}
        gates['incumbent_comparison']=not incumbent_present or (incumbent_ci is not None and
            incumbent_ci['high']<=0 and incumbent_gain>=policy['minimum_relative_crps_improvement'])
        gates={k:bool(v) for k,v in gates.items()}
        decisions[model]={'passed':all(gates.values()),'gates':gates,'relative_crps_improvement':float(improvement) if improvement is not None else None,
            'paired_loss_difference_ci':interval,'disjoint_origins':[e['origin'] for e in candidate],
            'incumbent_loss_difference_ci':incumbent_ci,
            'coverage80_mean':float(np.mean(coverage)) if coverage else None,
            'tail5_breach_mean':float(np.mean(breaches)) if breaches else None,
            'drawdown95_breach_fraction':float(np.mean(dd)) if dd else None}
    return decisions


class ModelRegistry:
    def __init__(self,root):
        self.root=Path(root);self.store=ArtifactStore(self.root/'artifacts')

    def consider(self,batch_id,decisions,*,candidate_release_ids,comparison_incumbent=None):
        """Consume a selection batch once; a current incumbent must be tested.

        Rejections consume the batch too. Data/code revisions do not turn an
        already inspected evaluation interval into new independent evidence.
        """
        with lock(self.root):
            history=self.root/'decisions.jsonl'
            rows=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
            if any(r['batch_id']==batch_id for r in rows):raise ValueError('Selection batch already consumed')
            pointer=self.root/'champion.json'
            incumbent=json.loads(pointer.read_text()) if pointer.exists() else None
            if incumbent and comparison_incumbent!=incumbent['release_id']:
                raise UnsupportedModelError('Candidate was not compared with the current approved incumbent')
            # Selection windows are checked against every previously consumed
            # batch, even when someone changes its human-readable label.
            batch,_=self.store.load('batches',batch_id)
            for prior in rows:
                old,_=self.store.load('batches',prior['batch_id'])
                for a,b in batch['windows']:
                    if any(a<=d and c<=b for c,d in old['windows']):
                        raise ValueError('Evaluation window overlaps previously consumed evidence')
            for d in decisions.values():
                required=REQUIRED_GATES|set(batch.get('additional_required_gates',[]))
                if d['passed'] and (set(d['gates'])!=required or not all(d['gates'].values())):
                    raise ValueError('Approval requires every registered evidence gate')
            approved=[(name,d) for name,d in decisions.items() if d['passed'] and all(d['gates'].values())]
            winner=min(approved,key=lambda x:-x[1]['relative_crps_improvement'])[0] if approved else None
            event={'batch_id':batch_id,'decisions':decisions,'winner':winner,'previous_champion':incumbent,
                'created_at':datetime.now(timezone.utc).isoformat(),'authority':'Research model only; never trading strategy/live deployment'}
            if winner:
                release=candidate_release_ids[winner]
                release_metadata,_=self.store.load('models',release)
                if not release_metadata['source_metadata']['point_in_time_verified']:
                    raise UnsupportedModelError('Immutable model source provenance does not permit promotion')
                new={'family':winner,'release_id':release,'batch_id':batch_id,'approved_at':event['created_at']}
                tmp=self.root/'champion.pending.json';tmp.write_bytes(canonical(new));os.replace(tmp,pointer)
                event['champion']=new
            with history.open('a') as f:f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n')
            return event


def walkforward_study(sources,registry,protocol,*,output,origins=None,paths=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    store=registry.store;forecasts=[];blocked=[];releases={}
    # All forecasts are written BEFORE any evaluation frame is loaded here.
    for i,origin in enumerate(origins or protocol['origins']):
        panel,metadata=sources.panel(origin)
        for j,kind in enumerate(protocol['models']):
            try:
                minimum=protocol['minimum_conditional_months'] if kind=='conditional_var_t' else protocol['minimum_bootstrap_months']
                training=train_slice(panel,origin,protocol['training_window_months'],minimum)
                fid=forecast_training(store,training,metadata,origin,kind,protocol,seed=protocol['seed']+100*i+j,
                    months=protocol['evaluation_months'],paths=paths or protocol['paths_per_origin'])
                forecasts.append(fid);releases[kind]=store.load('forecasts',fid)[0]['model_id']
                print('forecast frozen',origin,kind,fid[:12],flush=True)
            except UnsupportedModelError as exc:blocked.append({'origin':origin,'model':kind,'reason':str(exc)})
    actual,actual_meta=sources.panel()
    evaluations=[]
    for fid in forecasts:
        _,result=evaluate_forecast(store,fid,actual,actual_meta,protocol);evaluations.append(result)
    comparisons=compare_models(evaluations,protocol)
    windows=sorted(set((e['evaluation_start'],e['evaluation_end']) for e in evaluations if e['complete_evaluation']))
    batch_id=store.put('batches',{'protocol_sha256':sha256(canonical(protocol)).hexdigest(),
        'forecast_ids':forecasts,'windows':windows,'source_catalog_sha256':sources.catalog_hash})
    decision=registry.consider(batch_id,comparisons,candidate_release_ids=releases)
    report={'protocol':protocol,'protocol_sha256':sha256(canonical(protocol)).hexdigest(),'forecast_ids':forecasts,
        'source_catalog_sha256':sources.catalog_hash,'blocked_models':blocked,'evaluations':evaluations,
        'model_comparison':comparisons,'registry_decision':decision,'selection_batch_id':batch_id,
        'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'scope':'Retrospective chronological out-of-sample research; equity data revisions prevent real-time certification',
        'limitations':['Forecasts are immutable before scoring, but historical outcomes are known to human researchers',
            'All declared eras are diagnostics; only complete disjoint eras enter selection',
            'Early missing macro vintages and current revised equity history block verified promotion',
            'Synthetic future Student-t log returns have no finite expected arithmetic wealth; use quantiles, not a mean wealth forecast']}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
