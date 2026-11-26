"""Resumable, preregistered precision studies of full canonical account paths.

No unit-fund accelerator is used. Thresholds match the existing unit precision
protocol; new simultaneous intervals cover account quantities and planned looks.
Numerical acceptance never promotes a model or calibrates real event frequencies.
"""
from pathlib import Path
import json

import numpy as np

from letf.lab.catalog import file_hash,private_json
from letf.macro.artifacts import canonical
from letf.macro.precision import quantile_interval,exact_probability,log_distance
from letf.macro.precision_run import json_safe
from letf.accuracy import UnsupportedModelError


def assess_samples(looks,*,planned_looks,seeds):
    """Pooled IID paths, separately checked roots and latest-doubling stability."""
    if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('At least two distinct independent root seeds required')
    groups={};rows=[];previous={}
    for count,by_seed in sorted(looks.items()):
        if set(by_seed)!=set(seeds):continue # an incomplete look cannot be accepted
        cells=None
        for seed,samples in by_seed.items():
            keys={(r['horizon'],r['strategy'],r['account']) for r in samples}
            if cells is not None and cells!=keys:raise ValueError('Missing account cells; never discard failed scenarios')
            cells=keys
            for key in keys:
                selected=[r for r in samples if (r['horizon'],r['strategy'],r['account'])==key]
                if len(selected)!=count or {r['path'] for r in selected}!=set(range(count)):
                    raise ValueError('Missing/duplicate canonical account paths')
                groups[(count,seed,key)]=selected
        if not cells:raise ValueError('Empty account look')
        aq=.025/(planned_looks*len(cells)*3);ap=.025/(planned_looks*len(cells)*4)
        for key in sorted(cells):
            samples=[r for seed in seeds for r in groups[count,seed,key]]
            wealth=np.array([r['net_terminal_wealth'] for r in samples])
            if not np.isfinite(wealth).all() or (wealth<0).any():raise ValueError('Invalid account wealth; failed paths cannot be dropped')
            logs=np.full(len(wealth),-np.inf);np.log(wealth,out=logs,where=wealth>0)
            quantiles=[]
            for q in (.1,.5,.9):
                ci=quantile_interval(logs,q,aq);by_root=[]
                for seed in seeds:
                    x=np.array([r['net_terminal_wealth'] for r in groups[count,seed,key]])
                    y=np.full(len(x),-np.inf);np.log(x,out=y,where=x>0)
                    by_root.append(quantile_interval(y,q,aq)['estimate'])
                difference=max(log_distance(a,b) for a in by_root for b in by_root)
                old=previous.get((key,q));stability=log_distance(ci['estimate'],old) if old is not None else np.inf
                quantiles.append(dict(ci,quantile=q,independent_seed_difference=difference,
                    latest_doubling_change=stability,passed=ci['maximum_error']<=.02 and difference<=.02 and stability<=.02))
                previous[key,q]=ci['estimate']
            reference={(seed,r['path']):r['net_terminal_wealth'] for seed in seeds for r in by_seed[seed]
                       if (r['horizon'],r['account'],r['strategy'])==(key[0],key[2],'US1X_CF')}
            if len(reference)!=len(samples):raise ValueError('Paired1x benchmark absent')
            flags={'fund_liquidation':[bool(r['fund_liquidation']) for r in samples],
                'drawdown_over50':[r['metrics']['max_drawdown']>.5 for r in samples],
                'drawdown_over90':[r['metrics']['max_drawdown']>.9 for r in samples],
                'outperform_paired1x':[r['net_terminal_wealth']>reference[seed,r['path']]
                    for seed in seeds for r in groups[count,seed,key]]}
            probabilities={k:exact_probability(sum(v),len(v),ap) for k,v in flags.items()}
            rows.append({'paths_per_seed':count,'root_seeds':seeds,'horizon':key[0],'strategy':key[1],'account':key[2],
                'quantiles':quantiles,'probabilities':probabilities,
                'passed':all(q['passed'] for q in quantiles) and all(p['half_width']<=.025 for p in probabilities.values())})
    final_count=max(looks) if looks else None
    final=[r for r in rows if r['paths_per_seed']==final_count]
    return json_safe({'classification':'CONDITIONAL_STRESS','results':rows,'model_approved':False,
        'maximum_log_quantile_error':.02,'maximum_probability_interval_half_width':.025,
        'family_alpha_quantiles':.025,'family_alpha_probabilities':.025,
        'account_sampling_gates_passed':bool(final) and all(r['passed'] for r in final),
        'scope':'Numerical sampling of this exact account/model/flow/law scenario only; no structural confidence or prediction certification'})


def study(spec,output,*,max_new_runs=1,runner=None):
    from letf.lab.workflows import simulate
    from letf.macro.data import HistoricalSources
    from hashlib import sha256
    if set(spec)!={'schema','experiment','counts','root_seeds'} or spec['schema']!=1:raise ValueError('Registered account-precision schema1 required')
    counts=spec['counts'];seeds=spec['root_seeds']
    if (len(counts)<2 or any(type(n)is not int or n<1 for n in counts) or any(b!=2*a for a,b in zip(counts,counts[1:]))):
        raise ValueError('At least two predeclared doubling counts required')
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or any(type(s)is not int or s<0 for s in seeds):raise ValueError('Distinct nonnegative integer root seeds required')
    if type(max_new_runs)is not int or max_new_runs<0:raise ValueError('Nonnegative run budget required')
    if {'paths','seed'}&set(spec['experiment']):raise ValueError('Paths/seeds belong to frozen study protocol, not experiment overrides')
    sources=HistoricalSources(spec['experiment']['sources'])
    for name in sources.catalog['sources']:sources.bytes(name)
    root=Path(output);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    code={str(p.relative_to(Path(__file__).parents[2])):file_hash(p) for p in Path(__file__).parents[1].rglob('*.py')}
    plan={'spec':spec,'source_catalog_sha256':sources.catalog_hash,'implementation_hashes':code,
        'thresholds':{'log_quantile_error':.02,'probability_interval_half_width':.025},
        'sampling':'Full canonical ledger; fixed planned looks; independent root seeds; no optional stopping promotion'}
    registration=root/'registration.json'
    if registration.exists():
        if json.loads(registration.read_text())!=plan:raise ValueError('Registered source/code/protocol changed; preserve this study and use a new namespace')
    else:private_json(registration,plan)
    runner=runner or simulate;new=0;looks={};blocked=[];pending=[]
    for count in counts:
        for seed in seeds:
            target=root/f'paths-{count}-seed-{seed}';receipt=target/'completion.json'
            if not target.exists() and new<max_new_runs:
                request=dict(spec['experiment'],paths=count,seed=seed);new+=1
                try:
                    runner(request,output=target)
                    private_json(receipt,{'report_sha256':file_hash(target/'report.json')})
                except (ValueError,FileNotFoundError,UnsupportedModelError) as error:
                    target.mkdir(parents=True,exist_ok=True,mode=0o700)
                    private_json(target/'blocked.json',{'classification':'BLOCKED','reason':str(error)})
            if receipt.exists():
                if file_hash(target/'report.json')!=json.loads(receipt.read_text())['report_sha256']:raise ValueError('Completed account checkpoint corrupted')
                looks.setdefault(count,{})[seed]=json.loads((target/'report.json').read_text())['path_results']
            elif target.exists():blocked.append({'paths':count,'seed':seed,'reason':'Incomplete or blocked checkpoint retained; no path is discarded or overwritten'})
            else:pending.append({'paths':count,'seed':seed})
    report=assess_samples(looks,planned_looks=len(counts),seeds=seeds)
    complete=not blocked and not pending
    report.update(registration_sha256=file_hash(registration),study_complete=complete,pending=pending,blocked=blocked,
        account_convergence_certified=complete and report['account_sampling_gates_passed'],new_runs_completed_or_attempted=new,
        future_policy_uncertainty='Not included in sampling intervals; frozen-law scenario only')
    blob=canonical(report);path=root/('assessment-'+sha256(blob).hexdigest()+'.json')
    if not path.exists():private_json(path,report)
    return dict(report,assessment_path=str(path))
