"""Frozen chronological monthly AND daily-tail bootstrap comparisons.

Existing forecasts stay immutable. The new retrospective windows are checked
against every prior complete forecast interval; revised data still prevents
promotion. Human-known history is never represented as a pristine holdout.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.macro.artifacts import canonical,fingerprint
from letf.macro.daily_tails import (VARIANCE_COLUMN,realized_monthly_variance,
    training_templates,daily_bridge,evaluate_daily,longest_underwater,serial_correlation)
from letf.macro.data import CORE,train_slice
from letf.macro.learning import compare_models,implementation_hash
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage
from letf.macro.scoring import evaluate_paths,drawdowns


def tail_implementation_hash():
    return sha256(implementation_hash().encode()+b''.join(Path(__file__).with_name(n).read_bytes()
        for n in ('resampling.py','daily_tails.py','tail_study.py'))).hexdigest()


def _logs(sources):
    d=sources.daily_factors()
    return np.log1p(d['Mkt-RF']+d.RF).rename('equity_log')


def daily_gates(scores,policy):
    """Era-level failures, not thousands of pseudo-independent daily trials."""
    dd=np.mean([s['drawdown_95_breach'] for s in scores])
    crash=max(np.mean([s['crashes_per_252_sessions'][str(c)]['upper95_breach'] for s in scores]) for c in (-.05,-.10))
    coverage=np.mean([s['coverage95'] for s in scores])
    return {'daily_drawdown_tail95':bool(dd<=policy['maximum_drawdown_95_breach_fraction']),
        'daily_crash_tail95':bool(crash<=policy['maximum_crash_rate_95_breach_fraction']),
        'daily_coverage95':bool(policy['minimum_daily_95_interval_coverage']<=coverage<=policy['maximum_daily_95_interval_coverage'])}, {
        'drawdown95_breach_fraction':float(dd),'maximum_crash95_breach_fraction':float(crash),'coverage95_mean':float(coverage)}


def run_tail_study(sources,registry,protocol,*,prior_report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prior_bytes=Path(prior_report).read_bytes();prior=json.loads(prior_bytes)
    previous=sorted(set((e['evaluation_start'],e['evaluation_end']) for e in prior['evaluations'] if e['complete_evaluation']))
    for origin in protocol['origins']:
        start=pd.Timestamp(origin)+pd.offsets.MonthEnd(1)
        end=pd.Timestamp(origin)+pd.offsets.MonthEnd(protocol['evaluation_months'])
        if any(str(start.date())<=b and a<=str(end.date()) for a,b in previous):
            raise ValueError('Candidate evaluation overlaps previously inspected forecast outcomes')
    store=registry.store
    # Preserve the consumption constraint in this NEW registry without editing
    # the original registry or copying/replacing any original artifacts.
    inherited=store.put('batches',{'windows':previous,'purpose':'Inherited consumed evidence reference only',
        'original_report_sha256':sha256(prior_bytes).hexdigest()})
    registry.consider(inherited,{},candidate_release_ids={})
    forecasts=[];releases={}
    for oi,origin in enumerate(protocol['origins']):
        panel,meta=sources.panel(origin)
        daily=_logs(sources).loc[:pd.Timestamp(origin)]
        panel[VARIANCE_COLUMN]=realized_monthly_variance(daily).reindex(panel.index)
        panel=panel[[*CORE,VARIANCE_COLUMN]]
        for kind in protocol['models']:
            frame=train_slice(panel,origin,protocol['training_window_by_model'][kind],
                protocol['minimum_bootstrap_months'],allow_missing_transitions=protocol['allow_incomplete_training'])
            model=MacroModel.fit(kind,frame,protocol)
            model_meta={'kind':kind,'columns':model.columns,'parameters':model.parameters,
                'source_metadata':meta,'training_sha256':fingerprint(frame),
                'protocol_sha256':sha256(canonical(protocol)).hexdigest(),'implementation_sha256':tail_implementation_hash()}
            mid=store.put('models',model_meta,{'training':model.train_values})
            releases[kind]=mid
            # Same seed by origin gives a paired initial RNG source; differing
            # block rules do not imply identical subsequent monthly outcomes.
            seed=protocol['seed']+oi*100
            gap=(pd.Timestamp(origin).to_period('M')-frame.index[-1].to_period('M')).n
            monthly,lineage=sample_with_lineage(model,protocol['evaluation_months']+gap,protocol['paths_per_origin'],seed)
            monthly=monthly[:,gap:];lineage=lineage[:,gap:]
            observed=daily.loc[frame.index[0].to_period('M').start_time:frame.index[-1]]
            templates,lengths=training_templates(observed,frame)
            dates=pd.date_range(pd.Timestamp(origin)+pd.offsets.MonthEnd(1),periods=protocol['evaluation_months'],freq='ME')
            metadata={'kind':kind,'origin':origin,'dates':[str(d.date()) for d in dates],
                'model_id':mid,'columns':model.columns,'paths':protocol['paths_per_origin'],
                'training_end':str(frame.index[-1].date()),'training_start':str(frame.index[0].date()),
                'training_scale':frame.std(ddof=1).tolist(),'source_metadata':meta,
                'fixed_training_daily_sd':float(observed.std(ddof=1)),'seed':seed,
                'daily_seed':seed+10000,'daily_bridge_code_sha256':tail_implementation_hash(),
                'daily_bridges':protocol['daily_bridges'],'sessions_per_month':21,
                'unobserved_state_bridge_months':gap,'evidence':'HYPOTHETICAL_STRESS',
                'scope':protocol['confirmation_scope'],'status':'FROZEN_BEFORE_OUTCOME_LOADING'}
            fid=store.put('forecasts',metadata,{'monthly':monthly,'lineage':lineage,'templates':templates,'lengths':lengths})
            forecasts.append(fid);print('forecast frozen',origin,kind,fid[:12],flush=True)
    # A receipt identifies ALL forecasts before the outcome frame is requested.
    receipt={'forecast_ids':forecasts,'implementation_sha256':tail_implementation_hash(),
        'protocol_sha256':sha256(canonical(protocol)).hexdigest(),'source_catalog_sha256':sources.catalog_hash}
    (output/'frozen-receipt.json').write_bytes(canonical(receipt))
    evaluations=score_frozen_tail(sources,store,forecasts,protocol)
    decisions=compare_models(evaluations,protocol);daily_summary={}
    for kind in protocol['models']:
        daily_summary[kind]={}
        for bridge in protocol['daily_bridges']:
            scores=[e['daily'][bridge] for e in evaluations if e['model']==kind]
            gates,metrics=daily_gates(scores,protocol['daily_tail_gate'])
            daily_summary[kind][bridge]={'gates':gates,**metrics}
            if kind!='joint_block' and bridge=='empirical_variance_template':
                decisions[kind]['gates'].update(gates)
                decisions[kind]['passed']=bool(all(decisions[kind]['gates'].values()))
    windows=sorted(set((e['evaluation_start'],e['evaluation_end']) for e in evaluations))
    extra=['daily_drawdown_tail95','daily_crash_tail95','daily_coverage95']
    batch=store.put('batches',{'windows':windows,'forecast_ids':forecasts,'additional_required_gates':extra,
        'protocol_sha256':receipt['protocol_sha256'],'inherited_consumption_batch':inherited})
    decision=registry.consider(batch,decisions,candidate_release_ids=releases)
    report={'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'protocol':protocol,'receipt':receipt,'prior_report_sha256':sha256(prior_bytes).hexdigest(),
        'evaluations':evaluations,'model_comparison':decisions,'daily_summary':daily_summary,
        'registry_decision':decision,'selection_batch_id':batch,
        'limitations':['Historical source revisions and human-known history prevent real-time/untouched certification',
            'Five retrospective eras give wide statistical uncertainty; calibration is not a future guarantee',
            'Twenty-one synthetic sessions per month do not establish actual future exchange closures',
            'Daily variance templates are a conditional scenario assumption, not reconstructed observations',
            'No bond total-return series in this panel: stock/bond correlation forecasts remain BLOCKED',
            'No executable strategy or account wealth is calculated in this forecast-validation study']}
    (output/'report.json').write_bytes(canonical(report));return report


def common_reference_scales(store,forecast_ids):
    """Proper losses require common units, even with different training windows."""
    references={}
    for fid in forecast_ids:
        f,_=store.load('forecasts',fid)
        if f['kind']=='joint_block':
            if f['origin'] in references:raise ValueError('Duplicate baseline reference')
            references[f['origin']]={'forecast_id':fid,'monthly':f['training_scale'],
                'daily':f['fixed_training_daily_sd']}
    return references


def score_frozen_tail(sources,store,forecast_ids,protocol,*,allow_scoring_revision=False):
    references=common_reference_scales(store,forecast_ids)
    actual,actual_meta=sources.panel();actual_daily=_logs(sources);evaluations=[]
    for fid in forecast_ids:
        f,a=store.load('forecasts',fid);dates=pd.to_datetime(f['dates'])
        if not allow_scoring_revision and f['daily_bridge_code_sha256']!=tail_implementation_hash():
            raise ValueError('Frozen bridge implementation changed; explicit inspected rescoring required')
        reference=references[f['origin']]
        truth=actual.reindex(dates)[list(CORE)]
        score=evaluate_paths(a['monthly'][:,:,:len(CORE)],truth.to_numpy(),np.array(reference['monthly'][:len(CORE)]),list(CORE),
            score_paths=protocol['ensemble_score_sample_paths'],seed=protocol['seed'])
        result={'forecast_id':fid,'model':f['kind'],'origin':f['origin'],
            'evaluation_start':f['dates'][0],'evaluation_end':f['dates'][-1],'complete_evaluation':True,
            'point_in_time_verified':bool(f['source_metadata']['point_in_time_verified']),
            'score':score,'actual_sha256':fingerprint(truth),'actual_metadata':actual_meta,
            'forecast_was_frozen_before_scoring':True,'daily':{},
            'common_loss_scale_reference_forecast_id':reference['forecast_id'],
            'scoring_implementation_sha256':tail_implementation_hash()}
        observed=actual_daily.loc[dates[0].to_period('M').start_time:dates[-1]]
        for bridge in protocol['daily_bridges']:
            daily=daily_bridge(a['monthly'],a['lineage'],f['columns'],a['templates'],a['lengths'],f['daily_seed'],
                bridge=bridge,fixed_daily_sd=f['fixed_training_daily_sd'])
            result['daily'][bridge]=evaluate_daily(daily,observed,dates,reference['daily'])
        eid=store.put('evaluations',result);result['evaluation_id']=eid;evaluations.append(result)
        print('evaluated',f['origin'],f['kind'],flush=True)
    return evaluations


def rescore_inspected(sources,store,original_report,*,output):
    """Correct units on frozen paths; NEVER consume outcomes again for promotion."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=json.loads(Path(original_report).read_text());protocol=before['protocol']
    evaluations=score_frozen_tail(sources,store,before['receipt']['forecast_ids'],protocol,allow_scoring_revision=True)
    for after,old in zip(evaluations,before['evaluations'],strict=True):
        if after['forecast_id']!=old['forecast_id']:raise AssertionError('Frozen forecast replaced')
        for bridge in protocol['daily_bridges']:
            # The bridge recipe is unchanged for 21 sessions. Only score units
            # change; tail distributions/coverage MUST remain identical.
            for key,value in old['daily'][bridge].items():
                if key!='normalized_daily_marginal_crps' and after['daily'][bridge][key]!=value:
                    raise AssertionError('Frozen daily diagnostic changed during units correction: '+key)
    decisions=compare_models(evaluations,protocol)
    for d in decisions.values():d['passed']=False  # inspected, consumed evidence cannot promote
    report={'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'status':'SCORING_CORRECTION_ON_CONSUMED_EVIDENCE_NOT_NEW_VALIDATION',
        'original_report_sha256':sha256(Path(original_report).read_bytes()).hexdigest(),
        'receipt':before['receipt'],'protocol_unchanged':protocol,'evaluations':evaluations,
        'model_comparison':decisions,'daily_summary_unchanged':before['daily_summary'],
        'normalization_correction':'All candidates use the SAME baseline training scale by origin, including daily marginal CRPS',
        'registry_decision':'NO_PROMOTION; no second consumption or registry overwrite',
        'source_vintage_gate':False,'original_forecasts_unchanged':True}
    (output/'report.json').write_bytes(canonical(report));return report


def diagnose_frozen(sources,store,prior_report,*,output):
    """Inspect old artifacts read-only; no refit, revised forecast or promotion."""
    prior=json.loads(Path(prior_report).read_text());actual,_=sources.panel();rows=[]
    for old in prior['evaluations']:
        f,a=store.load('forecasts',old['forecast_id']);dates=pd.to_datetime(f['dates'])
        n=old['score']['months'];truth=actual.reindex(dates[:n])[f['columns']]
        paths=a['paths'][:,:n];j=f['columns'].index('equity_log');eq=paths[:,:,j];real=truth.equity_log.to_numpy()
        def diagnostic(sample,value):
            return {'actual':float(value),'p05_p50_p95':np.quantile(sample,[.05,.5,.95]).tolist(),
                'rank':float(np.mean(sample<=value))}
        row={'forecast_id':old['forecast_id'],'origin':f['origin'],'model':f['kind'],
            'complete':old['complete_evaluation'],'original_score':old['score'],
            'monthly_volatility':diagnostic(eq.std(1,ddof=1)*np.sqrt(12),real.std(ddof=1)*np.sqrt(12)),
            'max_drawdown':diagnostic(drawdowns(eq),drawdowns(real)),
            'longest_underwater_months':diagnostic(longest_underwater(eq),longest_underwater(real)[0]),
            'squared_return_lag1':diagnostic(serial_correlation(eq**2,1),serial_correlation(real**2,1)[0]),
            'crash_months_below_minus10_percent':diagnostic((eq<np.log(.9)).sum(1),(real<np.log(.9)).sum()),
            'crash_months_below_minus20_percent':diagnostic((eq<np.log(.8)).sum(1),(real<np.log(.8)).sum()),
            'stock_bond_correlation':'BLOCKED: no bond total-return observation in original forecast',
            'daily_tail_certification':'BLOCKED: monthly forecasts do not identify daily paths'}
        rows.append(row)
    report={'original_report_sha256':sha256(Path(prior_report).read_bytes()).hexdigest(),
        'forecasts_verified_and_diagnosed':len(rows),'evaluations':rows,'original_forecasts_modified':False}
    Path(output).write_bytes(canonical(report));return report
