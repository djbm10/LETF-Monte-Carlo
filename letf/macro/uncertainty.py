"""Training-only parameter sensitivity and conditional Monte Carlo precision.

Market-variable diagnostics are not portfolio/account balances or forecasts of
investable fund returns. No outcome observations enter these refits.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.macro.artifacts import ArtifactStore,canonical,fingerprint
from letf.macro.models import MacroModel
from letf.macro.data import train_slice
from letf.macro.learning import implementation_hash
from letf.accuracy import UnsupportedModelError


def wilson_interval(successes,n,z=1.959963984540054):
    """95% conditional binomial sampling interval, not model validity evidence."""
    if type(n)is not int or n<=0 or not 0<=successes<=n:raise ValueError('Valid binomial count required')
    p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(max(0.,center-half)),float(min(1.,center+half))]


def parameter_sensitivity(sources,*,calibration_end,output,replicates=12,paths=256,
                          horizons=(5,10,20,30,40),seed=20261124,block_months=12):
    """Joint block refits, excluding transitions across resampled block joins.

    Report across-refit dispersion of forecast medians separately from within-
    refit market uncertainty. It is a finite-history sensitivity estimate, not
    a calibrated Bayesian posterior or a future wealth confidence interval.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if replicates<2 or paths<2 or block_months<2:raise ValueError('Multiple refits/paths and contiguous blocks required')
    if not horizons or any(not 1<=h<=40 for h in horizons):raise ValueError('Horizons must be 1-40 years')
    protocol=json.loads((Path(__file__).resolve().parents[2]/'docs/macro-learning-2026-11-24/protocol.json').read_text())
    panel,meta=sources.panel(calibration_end)
    training=train_slice(panel,calibration_end,240,120,allow_missing_transitions=True)
    spec=dict(protocol,allow_incomplete_training=True,implementation_variant='training-block-parameter-sensitivity-v1')
    x=training.to_numpy();length=len(x)
    starts=[i for i in range(length-block_months+1) if np.isfinite(x[i:i+block_months]).all()]
    if not starts:raise UnsupportedModelError('No observed contiguous training blocks')
    rng=np.random.default_rng(seed);store=ArtifactStore(output/'artifacts');rows=[]
    for b in range(replicates):
        selections=rng.choice(starts,size=int(np.ceil(length/block_months)))
        sampled=np.concatenate([x[i:i+block_months] for i in selections])[:length].copy()
        # Never pretend the last observation of a random block was followed
        # historically by the first observation of another block.
        sampled[np.arange(block_months,length,block_months)]=np.nan
        # A final, partial resampled block must contain an observable end state.
        if not np.isfinite(sampled[-1]).all():raise UnsupportedModelError('Resampled end state is missing')
        frame=pd.DataFrame(sampled,index=training.index,columns=training.columns)
        model=MacroModel.fit('conditional_var_t',frame,spec)
        mid=store.put('models',{'parameters':model.parameters,'columns':model.columns,
            'original_training_sha256':fingerprint(training),'resampled_training_sha256':fingerprint(frame),
            'selected_block_starts':selections.tolist(),'seed':seed,'replicate':b,
            'implementation_sha256':implementation_hash(),'source_metadata':meta}, {'training':sampled})
        generated=model.simulate(12*max(horizons),paths,seed+1000+b,initial_state=x[-1])
        fid=store.put('forecasts',{'model_id':mid,'paths':paths,'seed':seed+1000+b,
            'purpose':'Macro-variable parameter sensitivity; not account or fund investment performance'}, {'paths':generated})
        for h in horizons:
            sample=generated[:,:12*h]
            rows.append({'replicate':b,'horizon':h,'model_id':mid,'forecast_id':fid,
                'equity_log_cumulative_p10_p50_p90':np.quantile(sample[:,:,model.columns.index('equity_log')].sum(axis=1),[.1,.5,.9]).tolist(),
                'inflation_log_cumulative_p10_p50_p90':np.quantile(sample[:,:,model.columns.index('inflation_log')].sum(axis=1),[.1,.5,.9]).tolist(),
                'terminal_rate_p10_p50_p90':np.quantile(sample[:,-1,model.columns.index('short_rate')],[.1,.5,.9]).tolist()})
    summary=[]
    for h in horizons:
        selected=[r for r in rows if r['horizon']==h]
        summary.append({'horizon':h,'across_refit_p10_p50_p90_of_conditional_medians':{
            key:np.quantile([r[key][1] for r in selected],[.1,.5,.9]).tolist()
            for key in ('equity_log_cumulative_p10_p50_p90','inflation_log_cumulative_p10_p50_p90','terminal_rate_p10_p50_p90')}})
    report={'evidence':'HYPOTHETICAL_STRESS','certification':'BLOCKED','replicates':replicates,'paths_per_refit':paths,
        'seed':seed,'block_months':block_months,'training_start':str(training.index[0].date()),
        'training_end':str(training.index[-1].date()),'training_sha256':fingerprint(training),
        'source_catalog_sha256':sources.catalog_hash,'results':summary,'refits':rows,
        'limitations':['Parameter sensitivity of an unapproved model, not verified future investment performance',
            'No portfolio/account wealth is calculated here; use the canonical account explorer',
            'Block length and finite historical coverage limit this sensitivity estimate',
            'Across-refit and within-refit percentiles have different meanings and are not added',
            'Regime changes, source revisions and model misspecification exceed parameter-resampling uncertainty']}
    (output/'report.json').write_bytes(canonical(report));return report
