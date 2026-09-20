from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numba import njit

BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
TD=252; TC=.0003

def read(name):
    d=pd.read_csv(BASE+name,sep='\t')
    d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(d.Close,errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

@njit(cache=True)
def run_variant(rt, base, bull, ddq, conf, thr, boost, hold, band):
    n=len(rt); out=np.zeros(n)
    wealth=1.; wtq=0.; armed=False; active=0
    for i in range(n):
        d=ddq[i]
        if not np.isnan(d) and d<=-thr: armed=True
        if active>0:
            active-=1
            if (not np.isnan(d) and d>-thr/2.) or not bull[i]: active=0
        if armed and bull[i] and conf[i]:
            active=hold; armed=False
        target=base[i]
        if active>0: target=min(1.,target+boost)
        if abs(target-wtq)>=band:
            wealth*=max(1.-TC*abs(target-wtq),0.); wtq=target
        nt=wealth*wtq*(1.+rt[i]); nc=wealth*(1.-wtq)
        nw=nt+nc; out[i]=nw/wealth-1.; wealth=nw
        wtq=nt/max(wealth,1e-15)
    return out

def metrics(r):
    r=np.asarray(r,float); e=np.cumprod(np.maximum(1+r,1e-12))
    yrs=len(r)/TD; dd=e/np.maximum.accumulate(e)-1
    if len(r)>=5*TD:
        lr=np.log(np.maximum(1+r,1e-12)); cs=np.r_[0.,np.cumsum(lr)]
        rr=np.exp((cs[5*TD:]-cs[:-5*TD])/5)-1
        min5=float(np.min(rr))
    else:min5=np.nan
    return float(e[-1]**(1/yrs)-1),float(dd.min()),min5,float(np.std(r)*math.sqrt(TD))

def main():
    q=read('synthetic-qqq.tsv'); tq=read('synthetic-tqqq.tsv'); spy=read('spy.tsv')
    idx=q.index.intersection(tq.index).intersection(spy.index)
    idx=idx[idx>=pd.Timestamp('1985-01-31')]
    q=q.reindex(idx); tq=tq.reindex(idx); spy=spy.reindex(idx)
    rt=tq.pct_change().fillna(0).to_numpy()
    spma=spy.rolling(200).mean()
    bull=((spy>spma).shift(1).fillna(False)).to_numpy(dtype=np.bool_)
    vt=(pd.Series(rt,index=idx).rolling(20).std()*math.sqrt(TD)).shift(1)
    base=(pd.Series(np.where(bull,.35,.12),index=idx)/vt.replace(0,np.nan)).clip(0,1).fillna(0).to_numpy()
    ddq=(q/q.rolling(252,min_periods=63).max()-1).shift(1).to_numpy()
    ma10=q.rolling(10).mean().shift(1); ma20=q.rolling(20).mean().shift(1)
    ma10prev=q.rolling(10).mean().shift(2); ma20prev=q.rolling(20).mean().shift(2)
    low20=q.rolling(20).min().shift(1); mom5=q.shift(1)/q.shift(6)-1
    confs={
      'cross10':((q.shift(1)>ma10)&(q.shift(2)<=ma10prev)).fillna(False).to_numpy(dtype=np.bool_),
      'cross20':((q.shift(1)>ma20)&(q.shift(2)<=ma20prev)).fillna(False).to_numpy(dtype=np.bool_),
      'rebound3':(((q.shift(1)/low20-1)>=.03)&(mom5>0)).fillna(False).to_numpy(dtype=np.bool_),
      'rebound5':(((q.shift(1)/low20-1)>=.05)&(mom5>0)).fillna(False).to_numpy(dtype=np.bool_),
      'mom5ma10':((mom5>0)&(q.shift(1)>ma10)).fillna(False).to_numpy(dtype=np.bool_),
    }
    split=int(np.searchsorted(idx.values,np.datetime64('2010-01-01')))
    # compile
    run_variant(rt[:100],base[:100],bull[:100],ddq[:100],next(iter(confs.values()))[:100],.1,.05,20,.1)
    rows=[]
    # baseline using impossible trigger
    zero=np.zeros(len(rt),dtype=np.bool_)
    rb=run_variant(rt,base,bull,ddq,zero,9.,0.,0,.10)
    for period,rr in [('full',rb),('pre2010',rb[:split]),('holdout2010',rb[split:])]:
        c,d,m,v=metrics(rr);rows.append(dict(strategy='S9_10',period=period,thr=np.nan,boost=0,confirm='base',hold=0,band=.10,cagr=c,max_dd=d,min_5y_cagr=m,vol=v))
    thrs=(.05,.075,.10,.125,.15,.20,.25); boosts=(.025,.05,.075,.10,.125,.15,.20); holds=(10,20,40,60); bands=(.05,.10)
    for thr in thrs:
      for boost in boosts:
       for cn,cf in confs.items():
        for hold in holds:
         for band in bands:
          r=run_variant(rt,base,bull,ddq,cf,thr,boost,hold,band)
          name=f'CONF_t{int(thr*1000):03d}_b{int(boost*1000):03d}_{cn}_h{hold}_rb{int(band*100):02d}'
          for period,rr in [('full',r),('pre2010',r[:split]),('holdout2010',r[split:])]:
            c,d,m,v=metrics(rr);rows.append(dict(strategy=name,period=period,thr=thr,boost=boost,confirm=cn,hold=hold,band=band,cagr=c,max_dd=d,min_5y_cagr=m,vol=v))
    z=pd.DataFrame(rows); z.to_csv('results/confirmed_dip_v2_fast_all.csv',index=False)
    pre=z[z.period=='pre2010']
    elig=pre[(pre.max_dd>=-.75)&(pre.min_5y_cagr>=-.20)].sort_values('cagr',ascending=False)
    names=elig.head(30).strategy.tolist()
    short=z[z.strategy.isin(names+['S9_10'])]
    short.to_csv('results/confirmed_dip_v2_fast_shortlist.csv',index=False)
    print('PRE2010 TOP ELIGIBLE')
    print(elig[['strategy','cagr','max_dd','min_5y_cagr','vol']].head(20).to_string(index=False))
    print('\nFROZEN TOP20 HOLDOUT')
    h=short[(short.period=='holdout2010') & short.strategy.isin(elig.head(20).strategy)]
    print(h[['strategy','cagr','max_dd','min_5y_cagr','vol']].sort_values('cagr',ascending=False).to_string(index=False))
    print('\nS9 BASE')
    print(short[short.strategy=='S9_10'][['period','cagr','max_dd','min_5y_cagr','vol']].to_string(index=False))

if __name__=='__main__':main()
