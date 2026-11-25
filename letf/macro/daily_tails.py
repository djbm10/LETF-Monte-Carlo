"""Empirical daily forecast bridge and tail diagnostics, never historical quotes.

Monthly returns do not determine a daily path. These are explicit conditional
scenarios using ONLY training daily shapes and joint forecast variance. They
cannot establish actual future closures, executable fills or model validity.
"""
import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.macro.scoring import drawdowns

VARIANCE_COLUMN='equity_log_variance'


def realized_monthly_variance(daily_logs):
    if not isinstance(daily_logs,pd.Series) or not daily_logs.index.is_unique or not daily_logs.index.is_monotonic_increasing:
        raise ValueError('Unique chronological observed daily log returns required')
    if not np.isfinite(daily_logs).all():raise ValueError('Nonfinite daily returns')
    groups=daily_logs.groupby(daily_logs.index.to_period('M'))
    v=groups.apply(lambda x:float(np.sum((x-x.mean())**2)))
    if (v<=0).any():raise UnsupportedModelError('Positive observed within-month variance required; no variance floor')
    v.index=v.index.to_timestamp('M')
    return np.log(v).rename(VARIANCE_COLUMN)


def training_templates(daily_logs,training):
    """No daily observation later than the final complete training month."""
    if len(daily_logs) and daily_logs.index[-1]>training.index[-1]:
        raise ValueError('Future daily observation in training templates')
    groups={str(k):v.to_numpy(float) for k,v in daily_logs.groupby(daily_logs.index.to_period('M'))}
    lengths=np.zeros(len(training),dtype=int);templates=[]
    for i,day in enumerate(training.index):
        values=groups.get(str(day.to_period('M')))
        if values is None or len(values)<2:
            if np.isfinite(training.iloc[i].to_numpy()).all():raise UnsupportedModelError('Missing complete training month daily evidence')
            templates.append(np.zeros(1));continue
        centered=values-values.mean();variance=np.dot(centered,centered)
        if variance<=0:raise UnsupportedModelError('Unidentified within-month daily shape')
        if VARIANCE_COLUMN in training and np.isfinite(training.iloc[i][VARIANCE_COLUMN]):
            if not np.isclose(np.log(variance),training.iloc[i][VARIANCE_COLUMN],atol=1e-12):
                raise ValueError('Daily/monthly training variance mismatch')
        lengths[i]=len(values);templates.append(centered)
    padded=np.zeros((len(training),max(map(len,templates))))
    for i,v in enumerate(templates):padded[i,:len(v)]=v
    return padded,lengths


def daily_bridge(monthly,lineage,columns,templates,lengths,seed,*,bridge='empirical_variance_template',
                 sessions=21,fixed_daily_sd=None,path_ids=None):
    """Keep every monthly log total; empirical bridge also keeps sampled variance.

    A circular training template window is re-centered/rescaled to 21 synthetic
    sessions. This transformation is hypothetical, not a fabricated observation.
    Month-indexed RNG plus explicit path IDs makes chunked and full runs equal.
    """
    monthly=np.asarray(monthly,float);lineage=np.asarray(lineage)
    if monthly.ndim!=3 or lineage.shape!=monthly.shape[:2] or not np.isfinite(monthly).all():raise ValueError('Invalid monthly ensemble/lineage')
    if type(sessions)is not int or sessions<2:raise ValueError('At least two sessions required')
    if lineage.dtype.kind not in 'iu' or (lineage<0).any() or (lineage>=len(lengths)).any():raise ValueError('Invalid observed-row lineage')
    p,m,_=monthly.shape
    ids=np.arange(p) if path_ids is None else np.asarray(path_ids)
    if ids.shape!=(p,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=p or (ids<0).any():raise ValueError('Unique nonnegative path IDs required')
    if bridge not in ('empirical_variance_template','gaussian_fixed_reference'):raise UnsupportedModelError('Unknown daily bridge')
    if bridge=='gaussian_fixed_reference' and (fixed_daily_sd is None or not np.isfinite(fixed_daily_sd) or fixed_daily_sd<=0):
        raise ValueError('Positive training daily standard deviation required')
    eq=columns.index('equity_log');out=np.empty((p,m*sessions))
    if bridge=='empirical_variance_template':vi=columns.index(VARIANCE_COLUMN)
    for t in range(m):
        rng=np.random.default_rng(np.random.SeedSequence([seed,t]))
        if bridge=='empirical_variance_template':
            selected=lineage[:,t];n=lengths[selected]
            if (n<2).any():raise UnsupportedModelError('Forecast selected missing daily template')
            # Uniform draws, rather than bounded integer draws, preserve path-ID
            # invariance when different paths have different template lengths.
            offset=(rng.random(int(ids.max())+1)[ids]*n).astype(int)
            index=(offset[:,None]+np.arange(sessions))%n[:,None]
            residual=templates[selected[:,None],index]
            residual=residual-residual.mean(1,keepdims=True)
            ss=np.sum(residual**2,axis=1)
            if (ss<=0).any():raise UnsupportedModelError('Degenerate sampled daily shape; no silent redraw')
            residual*=np.sqrt(np.exp(monthly[:,t,vi])/ss)[:,None]
        else:
            residual=rng.standard_normal((int(ids.max())+1,sessions))[ids]*fixed_daily_sd
            residual-=residual.mean(1,keepdims=True)
        out[:,t*sessions:(t+1)*sessions]=residual+monthly[:,t,eq,None]/sessions
    if not np.isfinite(out).all():raise ValueError('Nonfinite daily paths; no dropped paths')
    return out


def longest_underwater(logs):
    x=np.atleast_2d(logs);wealth=np.cumsum(x,axis=1)
    peak=np.maximum.accumulate(np.column_stack([np.zeros(len(x)),wealth]),axis=1)[:,1:]
    underwater=wealth<peak-1e-14;clock=np.arange(1,x.shape[1]+1)
    last=np.maximum.accumulate(np.where(underwater,0,clock),axis=1)
    return (clock-last).max(1)


def serial_correlation(x,lag):
    x=np.atleast_2d(x)
    if x.shape[1]<=lag:return np.full(len(x),np.nan)
    a=x[:,:-lag];b=x[:,lag:];a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    den=np.sqrt(np.sum(a*a,1)*np.sum(b*b,1))
    return np.divide(np.sum(a*b,1),den,out=np.full(len(x),np.nan),where=den>0)


def _crps_many(samples,observations):
    ordered=np.sort(samples);n=len(ordered);prefix=np.r_[0.,np.cumsum(ordered)]
    k=np.searchsorted(ordered,observations)
    absolute=(observations*(2*k-n)+prefix[-1]-2*prefix[k])/n
    pair=np.dot(2*np.arange(1,n+1)-n-1,ordered)/n**2
    return absolute-pair


def evaluate_daily(paths,actual_logs,dates,training_daily_sd,*,sessions=21):
    """Daily marginal scores plus whole-path tails; no pseudo-independent era CI."""
    paths=np.asarray(paths,float);dates=pd.DatetimeIndex(dates)
    if paths.ndim!=2 or paths.shape[1]!=sessions*len(dates) or not np.isfinite(paths).all():raise ValueError('Daily forecast clock/schema mismatch')
    if not isinstance(actual_logs,pd.Series) or not np.isfinite(actual_logs).all() or not actual_logs.index.is_unique or not actual_logs.index.is_monotonic_increasing:
        raise ValueError('Observed chronological daily outcomes required')
    if not np.isfinite(training_daily_sd) or training_daily_sd<=0:raise ValueError('Training scale required')
    if set(actual_logs.index.to_period('M'))!=set(dates.to_period('M')):raise ValueError('Missing/extra daily outcome months')
    coverage=[];loss=[]
    for t,day in enumerate(dates):
        values=actual_logs.loc[actual_logs.index.to_period('M')==day.to_period('M')].to_numpy()
        sample=paths[:,t*sessions:(t+1)*sessions].ravel()
        lo,hi=np.quantile(sample,[.025,.975])
        coverage.extend((values>=lo)&(values<=hi));loss.extend(_crps_many(sample,values)/training_daily_sd)
    actual=actual_logs.to_numpy();pred_dd=drawdowns(paths);truth_dd=float(drawdowns(actual))
    def summary(sample,truth):
        return {'actual':float(truth),'p05_p50_p95':np.quantile(sample,[.05,.5,.95]).tolist(),
                'predictive_rank':float(np.mean(sample<=truth))}
    crashes={}
    for cut in (-.05,-.10):
        predicted=(paths<np.log1p(cut)).sum(1)*252/paths.shape[1]
        truth=(actual<np.log1p(cut)).sum()*252/len(actual)
        crashes[str(cut)]=dict(summary(predicted,truth),upper95_breach=bool(truth>np.quantile(predicted,.95)))
    return {'paths':len(paths),'observed_sessions':len(actual),'synthetic_sessions':paths.shape[1],
        'normalized_daily_marginal_crps':float(np.mean(loss)),'coverage95':float(np.mean(coverage)),
        'drawdown':summary(pred_dd,truth_dd),'drawdown_95_breach':bool(truth_dd>np.quantile(pred_dd,.95)),
        'crashes_per_252_sessions':crashes,'annualized_volatility':summary(paths.std(1,ddof=1)*np.sqrt(252),actual.std(ddof=1)*np.sqrt(252)),
        'longest_underwater_sessions':summary(longest_underwater(paths),longest_underwater(actual)[0]),
        'squared_return_lag1_correlation':summary(serial_correlation(paths**2,1),serial_correlation(actual**2,1)[0]),
        'squared_return_lag21_correlation':summary(serial_correlation(paths**2,21),serial_correlation(actual**2,21)[0]),
        'scope':'Conditional research paths; fixed forecast clock differs from realized closures; whole eras are the inference units'}
