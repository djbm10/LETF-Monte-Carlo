"""Bounded-memory ordinary Monte Carlo with registered simultaneous gates.

Immutable checkpoints, independent root seeds, fixed path-stream blocks. This
does NOT turn an unapproved market model into an investment forecast.
"""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from letf.macro.artifacts import canonical,fingerprint,ArtifactStore
from letf.macro.data import train_slice
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage
from letf.macro.daily_tails import realized_monthly_variance,training_templates
from letf.macro.precision import MonthlyLookup,quantile_interval,exact_probability,log_distance


def json_safe(value):
    """Represent absorbing zero / unbounded CI explicitly, never as NaN JSON."""
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(np.floating,float)):
        if np.isnan(value):raise ValueError('NaN scientific result')
        return 'NEGATIVE_INFINITY_ZERO_NAV' if np.isneginf(value) else 'POSITIVE_INFINITY_UNBOUNDED' if np.isposinf(value) else float(value)
    if isinstance(value,np.generic):return value.item()
    return value


def stream_seeds(master,block):
    if type(master)is not int or master<0 or type(block)is not int or block<0:raise ValueError('Nonnegative stream identity required')
    return tuple(int(np.random.SeedSequence([master,namespace,block]).generate_state(1,dtype=np.uint64)[0]) for namespace in (1,2))


def file_hash(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def generate_block(model,lookup,protocol,master,block):
    size=protocol['fixed_stream_block_paths'];months=max(protocol['horizons'])*12
    gap=(pd.Period(protocol['origin'],'M')-pd.Period(model.parameters['training_end'],'M')).n
    if gap<0:raise ValueError('Training after origin')
    monthly_seed,bridge_seed=stream_seeds(master,block)
    _,full=sample_with_lineage(model,months+gap,size,monthly_seed,return_values=False)
    initial=lookup.rate_indices[full[:,gap-1]] if gap else np.full(size,lookup.rate_indices[-1])
    lineage=full[:,gap:]
    results,_=lookup.evaluate(lineage,initial,bridge_seed,horizons=protocol['horizons'])
    inflation=np.cumsum(model.train_values[lineage,model.columns.index('inflation_log')],axis=1)
    keys=[(h,l) for h in protocol['horizons'] for l in protocol['leverages']]
    out=np.empty((size,len(keys),5))
    for j,(h,l) in enumerate(keys):
        r=results[h,l]
        out[:,j]=np.column_stack([r['log_nav'],r['drawdown'],r['liquidated'],r['minimum_daily_gross'],r['log_nav']-inflation[:,h*12-1]])
    return out


def assess(samples,protocol,previous=None):
    n=len(samples[0]);looks=len(protocol['checkpoints_per_seed']);keys=[(h,l) for h in protocol['horizons'] for l in protocol['leverages']]
    alpha=protocol['family_alpha_quantiles']/(len(keys)*len(protocol['quantiles'])*looks)
    probability_alpha=protocol['family_alpha_probabilities']/(len(keys)*3*looks)
    rows=[]
    for j,(h,l) in enumerate(keys):
        pooled=np.concatenate([a[:,j,0] for a in samples]);quantiles=[]
        for q in protocol['quantiles']:
            interval=quantile_interval(pooled,q,alpha)
            by_seed=[float(np.quantile(a[:,j,0],q,method='inverted_cdf')) for a in samples]
            seed_difference=log_distance(*by_seed)
            prefix_difference=log_distance(interval['estimate'],previous[j]['quantiles'][len(quantiles)]['estimate']) if previous else np.inf
            quantiles.append(dict(interval,quantile=q,by_seed=by_seed,independent_seed_difference=seed_difference,
                previous_doubling_difference=prefix_difference,
                passed=bool(max(interval['maximum_error'],seed_difference,prefix_difference)<=protocol['maximum_log_quantile_error'])))
        probabilities={}
        masks={'loss':lambda a:a[:,j,0]<0,'drawdown_over_90':lambda a:a[:,j,1]>.9,'liquidation':lambda a:a[:,j,2].astype(bool)}
        for name,mask in masks.items():
            count=sum(int(mask(a).sum()) for a in samples)
            probabilities[name]=exact_probability(count,2*n,probability_alpha)
        rows.append({'horizon':h,'leverage':l,'paths_per_seed':n,'quantiles':quantiles,
            'real_log_nav_p10_p50_p90':[float(np.quantile(np.concatenate([a[:,j,4] for a in samples]),q,method='inverted_cdf')) for q in protocol['quantiles']],
            'drawdown_p10_p50_p90':np.quantile(np.concatenate([a[:,j,1] for a in samples]),protocol['quantiles']).tolist(),
            'probabilities':probabilities,'minimum_daily_gross':min(float(a[:,j,3].min()) for a in samples),
            'numeric_underflows_without_liquidation':sum(int(((a[:,j,0]<np.log(np.nextafter(0.,1.)))&~a[:,j,2].astype(bool)).sum()) for a in samples),
            'passed':bool(all(q['passed'] for q in quantiles) and all(p['half_width']<=protocol['maximum_probability_interval_half_width'] for p in probabilities.values()))})
    return rows


def run_precision(sources,protocol,*,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    panel,meta=sources.panel(protocol['calibration_end']);ff=sources.daily_factors().loc[:pd.Timestamp(protocol['calibration_end'])]
    logs=np.log1p(ff['Mkt-RF']+ff.RF);panel['equity_log_variance']=realized_monthly_variance(logs).reindex(panel.index)
    training=train_slice(panel,protocol['calibration_end'],protocol['training_months'],120,allow_missing_transitions=True)
    spec={'block_months':3,'minimum_bootstrap_months':120,'allow_incomplete_training':True,'stationary_mean_months':protocol['stationary_mean_months']}
    model=MacroModel.fit(protocol['model'],training,spec)
    templates,lengths=training_templates(logs.loc[training.index[0].to_period('M').start_time:training.index[-1]],training)
    lookup=MonthlyLookup.build(model,templates,lengths,fee=protocol['annual_fee_leveraged'],spread=protocol['borrowing_spread'])
    store=ArtifactStore(output/'artifacts')
    implementation=sha256(b''.join(Path(__file__).with_name(f).read_bytes() for f in ('precision_run.py','precision.py','resampling.py','daily_tails.py','models.py'))).hexdigest()
    mid=store.put('models',{'kind':model.kind,'columns':model.columns,'parameters':model.parameters,
        'training_sha256':fingerprint(training),'source_metadata':meta,'protocol_sha256':sha256(canonical(protocol)).hexdigest(),
        'implementation_sha256':implementation},{'training':model.train_values,'templates':templates,'lengths':lengths})
    receipt={'model_id':mid,'protocol':protocol,'implementation_sha256':implementation,'source_catalog_sha256':sources.catalog_hash,
        'scope':'Conditional fund UNIT NAV, not after-tax account balances or validated future probabilities',
        'created_at':datetime.now(timezone.utc).isoformat(),'no_live_authority':True}
    (output/'registration.json').write_bytes(canonical(receipt))
    files=[output/f'seed-{s}.pending.bin' for s in protocol['seeds']]
    for path in files:path.touch(exist_ok=False)
    completed=0;checkpoints=[];previous=None;keys=len(protocol['horizons'])*len(protocol['leverages']);size=protocol['fixed_stream_block_paths']
    for count in protocol['checkpoints_per_seed']:
        if count<=completed or count%size:raise ValueError('Increasing checkpoints must align to frozen stream blocks')
        for seed,path in zip(protocol['seeds'],files,strict=True):
            with path.open('ab') as f:
                for block in range(completed//size,count//size):
                    generated=generate_block(model,lookup,protocol,seed,block)
                    f.write(generated.astype('<f8').tobytes())
            print('seed checkpoint generated',seed,count,flush=True)
        samples=[np.memmap(p,dtype='<f8',mode='r',shape=(count,keys,5)) for p in files]
        rows=assess(samples,protocol,previous);passed=all(r['passed'] for r in rows)
        checkpoint={'paths_per_seed':count,'results':rows,'passed':passed,'elapsed_seconds':time.perf_counter()-started,
            'prefix_sha256':[file_hash(p) for p in files]}
        (output/f'checkpoint-{count}.json').write_bytes(canonical(json_safe(checkpoint)))
        checkpoints.append({'paths_per_seed':count,'passed':passed,'passed_combinations':sum(r['passed'] for r in rows),
            'maximum_quantile_ci_log_error':max(q['maximum_error'] for r in rows for q in r['quantiles']),
            'maximum_independent_seed_difference':max(q['independent_seed_difference'] for r in rows for q in r['quantiles']),
            'elapsed_seconds':checkpoint['elapsed_seconds']})
        print('precision assessed',count,sum(r['passed'] for r in rows),'/',len(rows),flush=True)
        previous=rows;completed=count
        del samples
        if passed:break
    identities=[]
    for path in files:
        final=path.with_name(path.name.replace('.pending',''));path.rename(final)
        identities.append({'path':final.name,'sha256':file_hash(final),'shape':[completed,keys,5],
            'dtype':'<f8','fields':['log_nav','drawdown','liquidated','minimum_daily_gross','real_log_nav']})
    report={'registration':receipt,'paths_per_seed':completed,'passed':passed,'checkpoints':checkpoints,'results':rows,
        'samples':identities,'elapsed_seconds':time.perf_counter()-started,
        'evidence':'HYPOTHETICAL_STRESS','model_approved':False,'investment_certification':'BLOCKED',
        'legacy_15_failed_tests':'Preserved unchanged; this is larger-sample numerical evidence with the same error tolerance',
        'uncertainty':'Intervals quantify sampling error conditional on the model/data/assumptions, excluding model/source/parameter/future-tax uncertainty'}
    (output/'report.json').write_bytes(canonical(json_safe(report)));return report


def main():
    import argparse
    from letf.macro.data import HistoricalSources
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run_precision(HistoricalSources(a.sources),json.loads(a.protocol.read_text()),output=a.output)
    print(json.dumps(json_safe({'passed':r['passed'],'paths_per_seed':r['paths_per_seed'],'checkpoints':r['checkpoints']}),indent=2))


if __name__=='__main__':main()
