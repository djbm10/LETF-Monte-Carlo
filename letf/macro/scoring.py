"""Proper ensemble scores and explicit missing-outcome/tail diagnostics."""
import numpy as np


def ensemble_crps(samples,observation):
    """E|X-y| - E|X-X'|/2, exact finite-ensemble version, O(P log P)."""
    samples=np.asarray(samples,float)
    if samples.ndim!=1 or not np.isfinite(samples).all() or not np.isfinite(observation):raise ValueError('Finite univariate ensemble required')
    ordered=np.sort(samples);n=len(ordered)
    if n<2:raise ValueError('At least two predictive samples required')
    weights=2*np.arange(1,n+1)-n-1
    return float(np.mean(abs(samples-observation))-np.dot(weights,ordered)/n**2)


def drawdowns(log_returns):
    logs=np.cumsum(log_returns,axis=-1)
    peak=np.maximum.accumulate(np.concatenate([np.zeros((*logs.shape[:-1],1)),logs],axis=-1),axis=-1)[...,1:]
    return 1-np.exp(np.min(logs-peak,axis=-1))


def evaluate_paths(paths,actual,scale,columns,*,score_paths=256,seed=0):
    paths=np.asarray(paths,float);actual=np.asarray(actual,float);scale=np.asarray(scale,float)
    if paths.ndim!=3 or actual.shape!=paths.shape[1:] or scale.shape!=(paths.shape[2],):raise ValueError('Forecast/outcome schema mismatch')
    if not np.isfinite(paths).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid predictive ensemble/scale')
    p,m,d=paths.shape;rng=np.random.default_rng(seed)
    selected=paths[rng.choice(p,min(p,score_paths),replace=False)]/scale
    normalized=actual/scale
    scores=[];energy=[];variogram=[];marginals={}
    for j,name in enumerate(columns):
        valid=np.isfinite(actual[:,j]);loss=[];coverage80=[];coverage95=[];breaches=[]
        for t in np.flatnonzero(valid):
            sample=paths[:,t,j]/scale[j];truth=normalized[t,j]
            loss.append(ensemble_crps(sample,truth))
            q=np.quantile(sample,[.025,.05,.1,.9,.975])
            coverage80.append(q[2]<=truth<=q[3]);coverage95.append(q[0]<=truth<=q[4]);breaches.append(truth<q[1])
        marginals[name]={'observed_months':int(valid.sum()),'missing_months':int((~valid).sum()),
            'crps':float(np.mean(loss)) if loss else None,'coverage80':float(np.mean(coverage80)) if loss else None,
            'coverage95':float(np.mean(coverage95)) if loss else None,'lower_5_breach_rate':float(np.mean(breaches)) if loss else None}
        scores.extend(loss)
    for t in np.flatnonzero(np.isfinite(actual).all(1)):
        sample=selected[:,t];truth=normalized[t]
        # Independent forecast draws for an unbiased Monte Carlo energy term.
        pairs=rng.integers(0,len(sample),size=(len(sample),2))
        energy.append(np.linalg.norm(sample-truth,axis=1).mean()-.5*np.linalg.norm(sample[pairs[:,0]]-sample[pairs[:,1]],axis=1).mean())
        error=0.
        for i in range(d):
            for j in range(i):
                error+=(abs(truth[i]-truth[j])**.5-np.mean(abs(sample[:,i]-sample[:,j])**.5))**2
        variogram.append(error/max(1,d*(d-1)/2))
    equity=columns.index('equity_log');valid_eq=np.isfinite(actual[:,equity])
    if not valid_eq.all():raise ValueError('Missing equity outcome: terminal wealth/tail comparison cannot skip a month')
    predicted_dd=drawdowns(paths[:,:,equity]);actual_dd=float(drawdowns(actual[:,equity]))
    terminal_log=paths[:,:,equity].sum(1);truth=float(actual[:,equity].sum())
    actual_cov=np.corrcoef(actual[np.isfinite(actual).all(1)],rowvar=False)
    if not np.isfinite(actual_cov).all():correlation_error=None
    else:
        predicted_cov=np.mean([np.corrcoef(x,rowvar=False) for x in paths[:min(p,score_paths)]],axis=0)
        correlation_error=float(np.linalg.norm(predicted_cov-actual_cov)/d)
    return {'primary_crps':float(np.mean(scores)),'energy_score':float(np.mean(energy)),
        'variogram_score':float(np.mean(variogram)),'marginals':marginals,
        'equity_drawdown_actual':actual_dd,'predicted_drawdown_p95':float(np.quantile(predicted_dd,.95)),
        'drawdown_95_breach':bool(actual_dd>np.quantile(predicted_dd,.95)),
        'terminal_log_return_actual':truth,'terminal_predictive_rank':float(np.mean(terminal_log<=truth)),
        'terminal_log_p10':float(np.quantile(terminal_log,.1)),'terminal_log_p50':float(np.median(terminal_log)),
        'terminal_log_p90':float(np.quantile(terminal_log,.9)),
        'correlation_error_frobenius_per_dimension':correlation_error,
        'months':m,'paths':p,'complete_joint_months':int(np.isfinite(actual).all(1).sum())}
