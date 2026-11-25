"""Registered bootstrap extensions with auditable observed-row lineage.

Stationary blocks have geometric duration. Their circular boundary is an
explicit stationarity assumption, never a claim of an observed transition.
Missing rows always force a restart; observations are never interpolated.
"""
import numpy as np

from letf.accuracy import UnsupportedModelError

EXTENDED_BOOTSTRAPS=('joint_block_long_history','long_joint_block',
                     'stationary_block','volatility_stationary_block')


def bootstrap_parameters(kind,training,protocol):
    p={'lineage_version':'observed-month-indices-v1'}
    if kind=='long_joint_block':
        p['block_months']=protocol['long_block_months']
    if kind in ('stationary_block','volatility_stationary_block'):
        mean=protocol['stationary_mean_months']
        if not np.isfinite(mean) or mean<1:raise ValueError('Stationary mean duration must be at least one month')
        p.update(stationary_mean_months=mean,circular_boundary=True)
    if kind=='volatility_stationary_block':
        if 'equity_log_variance' not in training:raise UnsupportedModelError('Observed daily realized variance required')
        width=protocol['volatility_restart_bandwidth'];mix=protocol['unconditional_restart_weight'];n=protocol['volatility_state_months']
        if not np.isfinite(width) or width<=0 or not 0<mix<=1 or type(n)is not int or n<1:
            raise ValueError('Invalid registered volatility restart parameters')
        v=training.equity_log_variance.to_numpy()
        states=np.array([np.mean(v[max(0,i-n):i]) if i else np.nan for i in range(len(v))])
        valid=np.isfinite(states)&np.isfinite(training.to_numpy()).all(1)
        scale=np.std(states[valid],ddof=1)
        if not np.isfinite(scale) or scale<=0:raise UnsupportedModelError('Unidentified volatility state')
        p.update(volatility_state_months=n,volatility_restart_bandwidth=width,
                 unconditional_restart_weight=mix,state_scale=float(scale))
    block=p.get('block_months',protocol['block_months'])
    if type(block)is not int or block<1:raise ValueError('Positive integral block duration required')
    return p


def sample_with_lineage(model,months,paths,seed,*,initial_state=None):
    """Return joint monthly rows and their original training row indices.

    The original joint-block loop order is retained byte-for-byte. A restart
    distribution depends only on simulated/past variance, never future outcomes.
    The positive uniform mixture preserves support for every complete month.
    """
    if type(months)is not int or months<1 or type(paths)is not int or paths<1:raise ValueError('Positive integer sizes required')
    if model.kind not in ('joint_block',*EXTENDED_BOOTSTRAPS):raise UnsupportedModelError('No observed-row lineage for this model')
    x=model.train_values;par=model.parameters;rng=np.random.default_rng(seed)
    lineage=np.empty((paths,months),dtype=np.int32);valid=np.isfinite(x).all(1)
    if model.kind in ('joint_block','joint_block_long_history','long_joint_block'):
        block=par['block_months']
        starts=np.array([i for i in range(len(x)-block+1) if valid[i:i+block].all()],dtype=int)
        if not len(starts):raise UnsupportedModelError('No complete historical blocks remain')
        for t in range(0,months,block):
            n=min(block,months-t)
            for p in range(paths):
                start=int(rng.choice(starts));lineage[p,t:t+n]=np.arange(start,start+n)
        return x[lineage],lineage
    eligible=np.flatnonzero(valid)
    if not len(eligible):raise UnsupportedModelError('No complete historical months remain')
    conditional=model.kind=='volatility_stationary_block'
    if conditional:
        j=model.columns.index('equity_log_variance');n=par['volatility_state_months']
        v=x[:,j]
        states=np.array([np.mean(v[max(0,i-n):i]) if i else np.nan for i in range(len(v))])
        eligible=eligible[np.isfinite(states[eligible])]
        if not len(eligible):raise UnsupportedModelError('No observed restart states remain')
        if initial_state is None:
            history=np.broadcast_to(v[-n:],(paths,n)).copy()
            if not np.isfinite(history).all():raise UnsupportedModelError('Missing initial volatility history')
        else:
            initial=np.broadcast_to(np.asarray(initial_state,float),(paths,len(model.columns)))
            if not np.isfinite(initial).all():raise ValueError('Nonfinite initial state')
            history=np.repeat(initial[:,j,None],n,axis=1)
    current=np.zeros(paths,dtype=int)
    for t in range(months):
        following=(current+1)%len(x)
        restart=np.ones(paths,dtype=bool) if t==0 else (rng.random(paths)<1/par['stationary_mean_months'])|~valid[following]
        current=following
        selected=np.flatnonzero(restart)
        if conditional and len(selected):
            # Bounded-memory matrix; 2000 x <=480 probabilities in studies.
            distance=(states[eligible][None,:]-history[selected].mean(1)[:,None])/par['state_scale']
            weights=np.exp(-.5*(distance/par['volatility_restart_bandwidth'])**2)
            total=weights.sum(1,keepdims=True)
            weights=np.divide(weights,total,out=np.full_like(weights,1/len(eligible)),where=total>0)
            mix=par['unconditional_restart_weight'];weights=(1-mix)*weights+mix/len(eligible)
            u=rng.random(len(selected));cdf=np.cumsum(weights,axis=1);cdf[:,-1]=1.
            choices=np.sum(cdf<u[:,None],axis=1);current[selected]=eligible[choices]
        else:current[selected]=rng.choice(eligible,size=len(selected))
        lineage[:,t]=current
        if conditional:history=np.column_stack([history[:,1:],v[current]])
    return x[lineage],lineage
