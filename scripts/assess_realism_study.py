"""Preserve raw stress diagnostics; apply legal/evidence gates and separate uncertainties.

This is an evidence assessment, not a change to market paths, tax rates, scenario
weights or forecasts. Earlier synthetic zero-price tax-sale values are excluded
from supported after-tax outputs, with their raw file hashes retained.
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.account_backtest import run_account_backtest
from letf.accounts import Security,TAXABLE_CASH
from letf.accuracy import UnsupportedModelError
from letf.account_stress import write_evidence_json
from letf.macro.precision_run import file_hash


def assess(study,precision,sensitivity,output,raw_engine_commit):
    if output.exists():raise ValueError('Evidence assessments are immutable')
    raw=json.loads((study/'report.json').read_text());inputs={r['id']:r for r in raw['inputs']}
    actions={}
    for key,entry in inputs.items():
        for name,digest in entry['files'].items():
            if file_hash(study/key/name)!=digest:raise ValueError('Raw scenario source hash changed')
        actions[key]=json.loads((study/key/'actions.json').read_text())
    rows=[];blocked_checks=0
    for old in raw['results']:
        ai=raw['execution_scope']['selected_anchors'].index(old['anchor'])
        key=f"anchor{ai}-{old['case']}-{old['leverage']}x";end=pd.Timestamp(year=2026+old['horizon'],month=12,day=31)
        directory=study/key/f"{old['horizon']}y-{old['account']}"
        for name,digest in old['exports'].items():
            if file_hash(directory/name)!=digest:raise ValueError('Raw daily account export changed')
        events=[e for e in actions[key]['liquidations'] if pd.Timestamp(e['date'])<=end]
        identity={k:old[k] for k in ('anchor','case','leverage','horizon','account')}
        identity['raw_result_sha256']=file_hash(directory/'result.json')
        if old['account']=='taxable' and any(e['cash_per_share']==0 for e in events):
            market=pd.read_csv(study/key/'prices.csv',index_col=0,parse_dates=True).loc[:end]
            try:
                run_account_backtest(market,[Security(inputs[key]['manifest']['symbol'])],TAXABLE_CASH,
                    initial_capital=10000.,allocator=lambda h,w:{},price_convention='scenario_nominal',
                    distributions_complete=True,source_manifest=inputs[key]['manifest'],liquidation_events=events)
            except UnsupportedModelError as error:
                if 'IRC165' not in str(error):raise
                reason=str(error)
            else:raise AssertionError('Current canonical API failed to block unsupported zero-price taxable loss')
            blocked_checks+=1;identity.update(status='BLOCKED_TAX_DISPOSITION',reason=reason,summary=None,
                superseded_raw_tax_values='Not valid after-tax evidence; no same-day zero-NAV disposition assumed')
        elif old['status']=='EXECUTED_UNAPPROVED':
            summary={k:v for k,v in old['summary'].items() if k!='annual_tax_rows'}
            summary['annual_tax_years']=len(old['summary']['annual_tax_rows'])
            summary['ending_carryforwards']={k:old['summary']['annual_tax_rows'][-1].get(k,0.)
                for k in ('st_cf','lt_cf','state_st_cf','state_lt_cf')} if old['summary']['annual_tax_rows'] else {}
            identity.update(status='SUPPORTED_HYPOTHETICAL_MECHANICS',summary=summary)
        else:identity.update(status='BLOCKED',reason=old.get('error'),summary=None)
        rows.append(identity)
    old_precision=json.loads(precision.read_text());old_spec=json.loads(sensitivity.read_text());uncertainty=[]
    for b in old_precision['results']:
        h,l=b['horizon'],b['leverage'];point=b['quantiles'][1]
        specs={'old_reference':{'median_unit_NAV':float(np.exp(point['estimate'])),
            'within_model_log_quantile_CI':[point['low'],point['high']], 'precision_passed':b['passed']}}
        for name,s in old_spec['specification_sensitivity'].items():
            q=next(r for r in s['results'] if (r['horizon'],r['leverage'])==(h,l))['quantiles'][1]
            specs[name]={'median_unit_NAV':float(np.exp(q['estimate'])),'within_model_log_quantile_CI':[q['low'],q['high']],
                'complete_specification_precision_passed':s['passed_precision']}
        uncertainty.append({'horizon':h,'leverage':l,'specifications':specs,
            'scope':'Existing fixed-21-session, no-distribution UNIT-NAV studies; not directly comparable account profiles',
            'scenario_range_is_probability_interval':False})
    report={'evidence':'HYPOTHETICAL_STRESS','model_approved':False,'investment_certification':'BLOCKED',
        'raw_study_sha256':file_hash(study/'report.json'),'raw_engine_commit_declared':raw_engine_commit,
        'assessment_implementation_sha256':file_hash(Path(__file__)),
        'complete_registered_matrix':raw['execution_scope']['complete_registered_matrix'],
        'counts':dict(Counter(r['status'] for r in rows)),'current_canonical_zero_tax_rejections_verified':blocked_checks,
        'supported_arithmetic_checks_passed':all(r['summary']['independent_audit']['passed'] for r in rows if r['summary']),
        'scenario_probabilities':'UNSPECIFIED_NOT_ESTIMATED','results':rows,
        'prior_within_model_sampling_and_specification_uncertainty':uncertainty,
        'prior_precision_report_sha256':file_hash(precision),'prior_specification_report_sha256':file_hash(sensitivity),
        'distinctions':{'sampling':'Prior independent-seed numerical intervals conditional on fixed assumptions',
            'regime_and_financing':'Unweighted specification/cost sensitivity; no calibrated coverage or model weights',
            'structural':'Explicit shocks outside historical support; no event probabilities or assured execution through halts',
            'historical_sources':'Current revisions, incomplete daily financing and index publication discrepancies are not a posterior probability distribution',
            'future_tax_policy':'Frozen 2026 federal/eligibility assumptions, unknown future law',
            'forecast_skill':'Rejected models stay rejected; genuine December2026 outcomes pending'},
        'limitations':raw['limitations']+['All taxable zero-residual cases blocked; raw initial ordinary-sale tax values superseded',
            'Only two registered old anchor paths reused, not an empirical probability or convergence study']}
    output.parent.mkdir(parents=True,exist_ok=True);write_evidence_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study','precision','sensitivity','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--raw-engine-commit',required=True);a=p.parse_args()
    r=assess(a.study,a.precision,a.sensitivity,a.output,a.raw_engine_commit)
    print(json.dumps({'counts':r['counts'],'supported_checks_passed':r['supported_arithmetic_checks_passed'],'model_approved':False},indent=2))
