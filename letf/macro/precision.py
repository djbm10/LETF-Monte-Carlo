"""Exact finite-support acceleration and simultaneous Monte Carlo uncertainty.

No forecast model is fitted here. The existing bootstrap and empirical daily
bridge define the distribution. Finite monthly shapes and previous rates are
memoized; log NAV/peak/drawdown summaries compose exactly across months.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom,beta


def quantile_interval(values,q,alpha):
    """Distribution-free order-statistic interval, conservative with atoms.

    For K~Bin(n,q), P(K<k)<=alpha/2 and P(K>=l)<=alpha/2. Thus
    [X_(k),X_(l)] covers the population lower quantile conservatively.
    Negative infinity is a genuine zero-NAV atom, not a clipped positive NAV.
    """
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or np.isnan(x).any() or np.isposinf(x).any() or not 0<q<1 or not 0<alpha<1:
        raise ValueError('Nonempty finite-or-liquidated sample and valid quantile/confidence required')
    n=len(x);k=int(binom.ppf(alpha/2,n,q));l=int(binom.ppf(1-alpha/2,n,q))+1
    ordered=np.sort(x)
    estimate=float(ordered[int(np.ceil(n*q))-1])
    lo=float(ordered[k-1]) if k else -np.inf
    hi=float(ordered[l-1]) if l<=n else np.inf
    if np.isneginf(lo) and np.isneginf(hi):error=0.;kind='IDENTIFIED_ZERO_ATOM'
    elif np.isfinite([lo,hi,estimate]).all():error=max(estimate-lo,hi-estimate);kind='FINITE'
    else:error=np.inf;kind='UNBOUNDED_OR_ZERO_BOUNDARY'
    return {'estimate':estimate,'low':lo,'high':hi,'maximum_error':float(error),
        'lower_rank':k,'upper_rank':l,'alpha':alpha,'status':kind}


def exact_probability(count,n,alpha):
    if type(n)is not int or n<1 or type(count)is not int or not 0<=count<=n or not 0<alpha<1:raise ValueError('Valid binomial count/confidence required')
    lo=0. if not count else float(beta.ppf(alpha/2,count,n-count+1))
    hi=1. if count==n else float(beta.ppf(1-alpha/2,count+1,n-count))
    return {'estimate':count/n,'low':lo,'high':hi,'half_width':(hi-lo)/2,'count':count,'paths':n,'alpha':alpha}


def log_distance(a,b):
    if np.isneginf(a) and np.isneginf(b):return 0.
    return float(abs(a-b)) if np.isfinite([a,b]).all() else np.inf


def period_summary(simple_returns):
    """Sufficient log summaries of actual daily-reset simple fund returns.

    Shape (...,sessions). No underflow-to-zero is interpreted as liquidation.
    Fields: final log growth, max log level, min log level, min log drawdown,
    minimum pre-closure daily gross return, absorbing liquidation indicator.
    """
    r=np.asarray(simple_returns,float)
    if not np.isfinite(r).all() or r.ndim<1 or r.shape[-1]<1:raise ValueError('Finite daily returns required')
    failure=r<=-1
    dead=np.maximum.accumulate(failure,axis=-1)
    logs=np.zeros_like(r)
    np.log1p(r,out=logs,where=~dead)
    logs=np.where(dead,-np.inf,logs)
    cumulative=np.cumsum(logs,axis=-1)
    initial=np.zeros((*r.shape[:-1],1))
    peak=np.maximum.accumulate(np.concatenate([initial,cumulative],axis=-1),axis=-1)[...,1:]
    minimum=np.minimum(0.,cumulative.min(axis=-1));maximum=np.maximum(0.,cumulative.max(axis=-1))
    gross=np.where(np.concatenate([np.zeros_like(dead[...,:1]),dead[...,:-1]],axis=-1),np.inf,1+r)
    return np.stack([cumulative[...,-1],maximum,minimum,(cumulative-peak).min(axis=-1),gross.min(axis=-1),dead[...,-1]],axis=-1)


def compose_periods(summary):
    """Exact max drawdown and terminal log NAV from ordered monthly summaries."""
    x=np.asarray(summary,float)
    if x.ndim!=3 or x.shape[-1]!=6 or np.isnan(x).any():raise ValueError('Path/month/summary schema required')
    end=np.cumsum(x[:,:,0],axis=1)
    entering=np.column_stack([np.zeros(len(x)),end[:,:-1]])
    highs=np.maximum.accumulate(np.maximum(0.,entering+x[:,:,1]),axis=1)
    prior_high=np.column_stack([np.zeros(len(x)),highs[:,:-1]])
    worst=np.minimum(x[:,:,3],entering+x[:,:,2]-prior_high)
    drawdown=-np.expm1(np.minimum.accumulate(worst,axis=1))
    liquidated=np.maximum.accumulate(x[:,:,5].astype(bool),axis=1)
    min_gross=np.minimum.accumulate(np.where(np.isneginf(entering),np.inf,x[:,:,4]),axis=1)
    return end,drawdown,liquidated,min_gross


@dataclass
class MonthlyLookup:
    shapes:np.ndarray
    row_offsets:np.ndarray
    lengths:np.ndarray
    rates:np.ndarray
    rate_indices:np.ndarray
    summaries:np.ndarray
    columns:list

    @classmethod
    def build(cls,model,templates,lengths,*,fee=.01,spread=.015,sessions=21):
        if model.kind!='stationary_block':raise ValueError('Only unchanged unconditional stationary reference supported')
        if sessions!=21 or fee<0 or spread<0 or not np.isfinite([fee,spread]).all():raise ValueError('Registered finite cost/clock assumptions required')
        x=model.train_values;j=model.columns.index('equity_log');v=model.columns.index('equity_log_variance')
        rate_column=model.columns.index('short_rate');complete=np.isfinite(x).all(1)
        rates,rate_indices=np.unique(x[complete,rate_column],return_inverse=True)
        full_rate_indices=np.full(len(x),-1,dtype=int);full_rate_indices[complete]=rate_indices
        shapes=[];offsets=np.full(len(x),-1,dtype=int)
        for row in np.flatnonzero(complete):
            if lengths[row]<2:raise ValueError('Missing observed daily template')
            offsets[row]=len(shapes)
            indices=(np.arange(lengths[row])[:,None]+np.arange(sessions))%lengths[row]
            residual=templates[row,indices];residual-=residual.mean(1,keepdims=True)
            ss=(residual**2).sum(1)
            if (ss<=0).any():raise ValueError('Degenerate daily shape; no redraw')
            residual*=np.sqrt(np.exp(x[row,v])/ss)[:,None]
            shapes.extend(residual+x[row,j]/sessions)
        shapes=np.asarray(shapes)
        summaries=np.empty((3,len(shapes),len(rates),6))
        # Bound temporary memory; values are exact daily mechanics, not fitted
        # surrogates or a truncation of severe outcomes.
        for begin in range(0,len(shapes),128):
            underlying=np.expm1(shapes[begin:begin+128])[:,None,:]
            for leverage in (1,2,3):
                r=leverage*underlying-(fee/252 if leverage>1 else 0.)-(leverage-1)*(rates[None,:,None]+spread)/252
                summaries[leverage-1,begin:begin+128]=period_summary(r)
        return cls(shapes,offsets,lengths.copy(),rates,full_rate_indices,summaries,model.columns)

    def evaluate(self,lineage,initial_rate_indices,bridge_seed,*,horizons=(5,10,20,30,40)):
        lineage=np.asarray(lineage)
        if lineage.ndim!=2 or lineage.dtype.kind not in 'iu' or (lineage<0).any() or (lineage>=len(self.lengths)).any():raise ValueError('Invalid lineage')
        if (self.row_offsets[lineage]<0).any():raise ValueError('Incomplete month in forecast')
        p,m=lineage.shape;shape_ids=np.empty_like(lineage)
        for t in range(m):
            rng=np.random.default_rng(np.random.SeedSequence([bridge_seed,t]))
            shape_ids[:,t]=self.row_offsets[lineage[:,t]]+(rng.random(p)*self.lengths[lineage[:,t]]).astype(int)
        rate_ids=np.column_stack([initial_rate_indices,self.rate_indices[lineage[:,:-1]]])
        if (rate_ids<0).any():raise ValueError('Missing previous-close rate')
        results={}
        for leverage in (1,2,3):
            summaries=self.summaries[leverage-1,shape_ids,rate_ids]
            lognav,dd,liquidation,min_gross=compose_periods(summaries)
            for h in horizons:
                if type(h)is not int or h*12>m or h<1:raise ValueError('Horizon outside generated months')
                t=h*12-1
                results[h,leverage]={'log_nav':lognav[:,t],'drawdown':dd[:,t],
                    'liquidated':liquidation[:,t],'minimum_daily_gross':min_gross[:,t]}
        return results,shape_ids
