"""Common-path scenarios and compatible, unranked historical comparisons."""
import copy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from letf.accuracy import UnsupportedModelError
from letf.account_cli import _serialize,_report_safe
from letf.lab.catalog import file_hash, private_json
from letf.lab.evidence import evidence
from letf.lab.portfolio import run_observed
from letf.macro.artifacts import canonical
from letf.macro.precision import quantile_interval
from letf.macro.precision_run import json_safe


def account_sampling_diagnostics(rows, *, quantiles=(.1,.5,.9),family_alpha=.025):
    """Fixed-look Bonferroni order statistics; not an account convergence pass.

    The original .02 log-error goal is retained. Independent seeds and latest
    doubling stability still require separate evidence, so this diagnostic
    never certifies convergence from one conveniently precise sample.
    """
    keys=sorted(set((r['horizon'],r['strategy'],r['account']) for r in rows))
    if not keys:raise ValueError('No completed full-ledger paths')
    alpha=family_alpha/(len(keys)*len(quantiles));results=[]
    for horizon,strategy,account in keys:
        selected=[r for r in rows if (r['horizon'],r['strategy'],r['account'])==(horizon,strategy,account)]
        ids=[r['path'] for r in selected]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate path identities are not independent samples')
        values=np.array([r['net_terminal_wealth'] for r in selected])
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid account sample; never drop failed paths')
        logs=np.full(len(values),-np.inf);np.log(values,out=logs,where=values>0)
        intervals={str(q):quantile_interval(logs,q,alpha) for q in quantiles}
        results.append({'horizon':horizon,'strategy':strategy,'account':account,'paths':len(values),
            'log_wealth_order_statistic_intervals':intervals,
            'interval_error_goal_passed':all(x['maximum_error']<=.02 for x in intervals.values())})
    return json_safe({'classification':'CONDITIONAL_STRESS','fixed_look_family_alpha':family_alpha,
        'quantile_log_error_goal':.02,'results':results,'account_convergence_certified':False,
        'missing_required_evidence':['Independent root-seed replication','Registered latest-doubling stability','Account-specific probability precision'],
        'scope':'Sampling error conditional on exactly this model, cash/tax/flow policy; not model or future-law uncertainty'})


def simulate(config, *, output):
    from letf.macro.data import HistoricalSources
    from letf.macro.explorer import explore
    if config.get('schema')!=1 or config.get('mode')!='future':raise ValueError('Expected schema1/future configuration')
    kwargs={k:v for k,v in config.items() if k not in ('schema','mode','sources')}
    allowed=set(inspect.signature(explore).parameters)-{'sources','output'}
    if set(kwargs)-allowed:raise ValueError('Unsupported simulation inputs: '+','.join(sorted(set(kwargs)-allowed)))
    if not {'calibration_end','state','frozen_tax_year'}<=set(kwargs):raise ValueError('Explicit calibration, state and future-law scenario required')
    kwargs.setdefault('model_type','stationary_block');kwargs.setdefault('paths',4)
    if type(kwargs['paths'])is not int or kwargs['paths']<1:raise ValueError('Positive integer paths required')
    start=time.perf_counter()
    report=explore(HistoricalSources(config['sources']),output=output,**kwargs)
    sampling=account_sampling_diagnostics(report['path_results'])
    output=Path(output)
    envelope={**evidence('CONDITIONAL_STRESS',limitations=report['limitations']),
        'schema':1,'mode':'future','source_catalog_sha256':report['source_catalog_sha256'],
        'model_type':report['model_type'],'model_id':report['model_id'],'forecast_id':report['forecast_id'],
        'report_sha256':file_hash(output/'report.json'),'report':'report.json','request':config,
        'paired_market_paths':True,'conditional_distributions':report['results'],
        'sampling_uncertainty':sampling,'structural_uncertainty':{
            'weights_or_probabilities_assigned':False,
            'not_in_sampling_CI':['Historical regime support/duration','Unprecedented catastrophe frequencies',
                'Funding/collateral/fee contracts','Future tax law and income/eligibility',
                'Daily bridge, monetary regime, source revisions and unobserved daily tail behavior'],
            'sensitivity_method':'Separate explicit configurations/seeds; no mixture weights or Bayesian credible intervals implied'},
        'elapsed_seconds':time.perf_counter()-start,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform!='darwin' else 1),
        'runtime_scope':'Wall time for this call; peak RSS is process lifetime including source/model setup',
        'rankings_released':False}
    private_json(output/'laboratory-report.json',envelope)
    return envelope


def comparison_fingerprint(report):
    common=copy.deepcopy(report['request']);common.pop('strategy',None)
    # All instruments and costs remain in this fingerprint, including signal
    # bundles with zero portfolio weights. No post-hoc shorter common window.
    payload={'common_inputs':common,'source_manifests':report['source_manifests']}
    return sha256(canonical(payload)).hexdigest()


def tournament(config, *, output):
    if set(config)-{'schema','base','strategies'} or config.get('schema')!=1:raise ValueError('Unknown tournament schema/fields')
    base=config['base'];strategies=config['strategies']
    if 'rebalance_sessions'not in base:raise ValueError('Fair cohort requires an explicit common execution/reinvestment interval')
    if not strategies or len({(s['id'],s['version']) for s in strategies})!=len(strategies):raise ValueError('Unique strategy/version cells required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700);cells=[]
    for i,strategy in enumerate(strategies):
        request=dict(copy.deepcopy(base),strategy=strategy)
        try:
            r=run_observed(request,output=output/f'cell-{i:03}')
            cells.append({'strategy':strategy,'status':'SUPPORTED_RESEARCH_MECHANICS','classification':r['classification'],
                'comparison_fingerprint':comparison_fingerprint(r),'accounts':r['accounts'],
                'report':f'cell-{i:03}/report.json','report_sha256':file_hash(output/f'cell-{i:03}/report.json')})
        except (ValueError,FileNotFoundError,UnsupportedModelError) as error:
            cells.append({'strategy':strategy,'status':'BLOCKED','classification':'BLOCKED','reason':str(error)})
    completed=[r for r in cells if r['status']!='BLOCKED']
    compatible=bool(completed) and len({r['comparison_fingerprint'] for r in completed})==1
    labels={r['classification'] for r in completed}
    report={'schema':1,'classification':next(iter(labels)) if len(labels)==1 else 'BLOCKED','cells':cells,
            'all_requested_cells_supported':len(completed)==len(cells),'common_inputs_verified':compatible,
            'rankings_released':False,'model_approved':False,
            'scope':'Unsorted retrospective comparison under identical reviewed sources/flows/costs; blocked cells never ranked',
            'rebalance_override_sessions':base['rebalance_sessions'],
            'strategy_version_scope':'Explicit common rebalance override; not a claim of unchanged original research execution intervals',
            'reason':'Model approval/execution/source limits remain separate from compatible numerical research results'}
    safe=json.loads(json.dumps(_report_safe(report),default=_serialize,allow_nan=False))
    private_json(output/'report.json',safe)
    return safe
