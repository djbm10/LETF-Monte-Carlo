from __future__ import annotations
import math
import numpy as np
import pandas as pd

BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
TD=252; TC=0.0003

def read(name):
    d=pd.read_csv(BASE+name,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(d.Close,errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

def metrics(r):
    e=(1+r).cumprod(); yrs=len(r)/TD; dd=e/e.cummax()-1
    roll=(1+r).rolling(5*TD).apply(np.prod,raw=True)**(1/5)-1
    return dict(cagr=e.iloc[-1]**(1/yrs)-1,max_dd=dd.min(),min_5y=roll.min(),vol=r.std()*math.sqrt(TD))

def run_port(rt, target, band=.10):
    wealth=1.; w=0.; out=[]
    for i in range(len(target)):
        want=float(target.iloc[i])
        if abs(want-w)>=band:
            wealth*=max(1-TC*abs(want-w),0); w=want
        nw=wealth*(w*(1+rt.iloc[i])+(1-w)); out.append(nw/wealth-1); wealth=nw
        w=(wealth*w*(1+rt.iloc[i]))/max(wealth,1e-15)
    return pd.Series(out,index=target.index)

def main():
    q=read('synthetic-qqq.tsv'); tq=read('synthetic-tqqq.tsv'); spy=read('spy.tsv')
    idx=q.index.intersection(tq.index).intersection(spy.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]
    q=q.reindex(idx); tq=tq.reindex(idx); spy=spy.reindex(idx)
    rq=q.pct_change().fillna(0); rt=tq.pct_change().fillna(0)
    spy200=spy.rolling(200).mean(); q200=q.rolling(200).mean(); q50=q.rolling(50).mean(); q20=q.rolling(20).mean(); q10=q.rolling(10).mean()
    vt=rt.rolling(20).std()*math.sqrt(TD); hi=tq.rolling(252,min_periods=63).max(); dd=tq/hi-1
    r5=q.pct_change(5); r10=q.pct_change(10)
    delta=q.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rsi=100-100/(1+gain/loss)
    # all signals lagged one full day
    dd=dd.shift(1); vt=vt.shift(1); bull_spy=(spy>spy200).shift(1).fillna(False); bull_q=(q>q200).shift(1).fillna(False)
    r5=r5.shift(1); r10=r10.shift(1); ql=q.shift(1); q10=q10.shift(1); q20=q20.shift(1); q50=q50.shift(1); rsi=rsi.shift(1)
    rows=[]; targets={}
    for trend_name,bull in [('SPY',bull_spy),('QQQ',bull_q)]:
      for bulltv in (.35,.40):
       for beartv in (.12,0.0):
        base=pd.Series(np.clip(np.where(bull,bulltv,beartv)/vt.replace(0,np.nan),0,1),index=idx).fillna(0)
        bname=f'BASE_{trend_name}_{int(bulltv*100)}_{int(beartv*100)}'; targets[bname]=base
        for start in (.10,.15,.20,.25):
         levels=[start,start+.10,start+.20,start+.30]
         for step in (.05,.075,.10):
          ladder=np.select([dd<=-levels[3],dd<=-levels[2],dd<=-levels[1],dd<=-levels[0]],[4*step,3*step,2*step,step],default=0.)
          conds={
            'BULL':bull,
            'BULL_R5':bull & (r5>0),
            'BULL_R10':bull & (r10>0),
            'BULL_RECLAIM10':bull & (ql>q10),
            'BULL_RECLAIM20':bull & (ql>q20),
            'BULL_RSI':bull & (rsi>35),
            # true bear-bottom attempts: require a deep dip plus positive reversal evidence
            'REV20':(r5>0)&(ql>q20),
            'REV50':(r10>0)&(ql>q50),
          }
          for cname,cond in conds.items():
            add=np.where(cond,ladder,0.); targets[f'{trend_name}_u{int(bulltv*100)}d{int(beartv*100)}_s{int(start*100)}_b{int(step*1000):03d}_{cname}']=pd.Series(np.clip(base+add,0,1),index=idx)
    # Historical split and crisis windows. Parameters are judged on pre-2010 only; holdout remains untouched.
    events={
      '1987':('1987-08-01','1989-07-31'),
      'dotcom':('2000-03-01','2007-06-30'),
      'gfc':('2007-10-01','2013-03-31'),
      'covid':('2020-02-01','2020-08-31'),
      '2022':('2022-01-01','2024-07-31'),
    }
    for name,t in targets.items():
        r=run_port(rt,t,.10)
        for period,mask in [('pre2010',idx<pd.Timestamp('2010-01-01')),('holdout2010',idx>=pd.Timestamp('2010-01-01')),('full',np.ones(len(idx),dtype=bool))]:
            rr=r.loc[mask]; rows.append({'strategy':name,'period':period,**metrics(rr)})
        for ev,(a,b) in events.items():
            rr=r.loc[(idx>=a)&(idx<=b)]
            if len(rr)>50: rows.append({'strategy':name,'period':ev,**metrics(rr)})
    z=pd.DataFrame(rows); z.to_csv('results/true_dip_screen.csv',index=False)
    p=z[z.period=='pre2010'].copy(); p['score']=p.cagr + .35*p.min_5y + .20*p.max_dd
    frozen=p.sort_values('score',ascending=False).head(15).strategy.tolist()
    print('FROZEN PRE2010 TOP15')
    print(p[p.strategy.isin(frozen)].sort_values('score',ascending=False).to_string(index=False))
    print('\nUNTOUCHED 2010+ FOR FROZEN')
    print(z[(z.period=='holdout2010')&z.strategy.isin(frozen)].sort_values('cagr',ascending=False).to_string(index=False))
    print('\nCRISIS WINDOWS FOR TOP PRE2010 8')
    top8=frozen[:8]
    print(z[z.strategy.isin(top8)&z.period.isin(events)].sort_values(['strategy','period']).to_string(index=False))
    pd.DataFrame({'strategy':frozen}).to_csv('results/true_dip_frozen.csv',index=False)

if __name__=='__main__': main()
