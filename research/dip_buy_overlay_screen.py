from __future__ import annotations
import math
import numpy as np
import pandas as pd

BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
TD=252; TC=0.0003

def read(name):
    d=pd.read_csv(BASE+name,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize();
    return pd.Series(pd.to_numeric(d.Close,errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

def metrics(r):
    e=(1+r).cumprod(); yrs=len(r)/TD; dd=e/e.cummax()-1
    roll=(1+r).rolling(5*TD).apply(np.prod,raw=True)**(1/5)-1
    return dict(cagr=e.iloc[-1]**(1/yrs)-1,max_dd=dd.min(),min_5y_cagr=roll.min(),vol=r.std()*math.sqrt(TD),terminal=e.iloc[-1])

def portfolio(rtq,rq,targets,band=0.0):
    wealth=1.; wtq=0.; wq=0.; out=[]
    for i in range(len(targets)):
        dtq=float(targets.iloc[i,0]); dq=float(targets.iloc[i,1]);
        if abs(dtq-wtq)>=band or abs(dq-wq)>=band:
            traded=abs(dtq-wtq)+abs(dq-wq)
            wealth*=max(1-TC*traded,0); wtq,wq=dtq,dq
        ct=wealth*wtq*(1+rtq.iloc[i]); cq=wealth*wq*(1+rq.iloc[i]); cc=wealth*(1-wtq-wq)
        nw=ct+cq+cc; out.append(nw/wealth-1); wealth=nw
        wtq=ct/max(wealth,1e-15); wq=cq/max(wealth,1e-15)
    return pd.Series(out,index=targets.index)

def main():
    q=read('synthetic-qqq.tsv'); tq=read('synthetic-tqqq.tsv'); spy=read('spy.tsv')
    idx=q.index.intersection(tq.index).intersection(spy.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]
    q=q.reindex(idx); tq=tq.reindex(idx); spy=spy.reindex(idx)
    rq=q.pct_change().fillna(0); rt=tq.pct_change().fillna(0)
    ma=spy.rolling(200).mean(); bull=spy>ma; vol=rt.rolling(20).std()*math.sqrt(TD)
    high252=tq.rolling(252,min_periods=63).max(); dd=tq/high252-1
    bull_s=bull.shift(1).fillna(False); vol_s=vol.shift(1); dd_s=dd.shift(1)
    base=np.where(bull_s,.35,.12)/vol_s.replace(0,np.nan); s9=pd.Series(np.clip(base,0,1),index=idx).fillna(0)
    cands={'QQQ_BH':pd.DataFrame({'tq':0.,'q':1.},index=idx),'TQQQ_BH':pd.DataFrame({'tq':1.,'q':0.},index=idx),'S9_10':pd.DataFrame({'tq':s9,'q':0.},index=idx)}
    ladders=[(0.10,0.20,0.30,0.40),(0.15,0.25,0.35,0.50),(0.20,0.30,0.40,0.50),(0.20,0.35,0.50,0.65)]
    for th in ladders:
        a=np.select([dd_s<=-th[3],dd_s<=-th[2],dd_s<=-th[1],dd_s<=-th[0]],[1,.75,.50,.25],default=0.)
        cands[f'DIP_QQQ_{int(th[0]*100)}_{int(th[1]*100)}_{int(th[2]*100)}_{int(th[3]*100)}']=pd.DataFrame({'tq':a,'q':1-a},index=idx)
        aa=np.where(bull_s,a,0.); qq=np.where(bull_s,1-aa,0.)
        cands[f'DIP_TREND_{int(th[0]*100)}_{int(th[1]*100)}_{int(th[2]*100)}_{int(th[3]*100)}']=pd.DataFrame({'tq':aa,'q':qq},index=idx)
        for boost in (0.05,0.10,0.20):
            b=np.select([dd_s<=-th[3],dd_s<=-th[2],dd_s<=-th[1],dd_s<=-th[0]],[4*boost,3*boost,2*boost,boost],default=0.)
            ta=np.clip(s9+b,0,1)
            cands[f'S9_DIP_ANY_b{int(boost*100)}_{int(th[0]*100)}']=pd.DataFrame({'tq':ta,'q':0.},index=idx)
            tb=np.clip(s9+np.where(bull_s,b,0.),0,1)
            cands[f'S9_DIP_BULL_b{int(boost*100)}_{int(th[0]*100)}']=pd.DataFrame({'tq':tb,'q':0.},index=idx)
    for deep,forced in [(0.30,.50),(0.40,.50),(0.40,.75),(0.50,1.0),(0.60,1.0)]:
        ta=s9.copy(); mask=(~bull_s)&(dd_s<=-deep); ta[mask]=np.maximum(ta[mask],forced)
        cands[f'S9_CATCH_FALLING_{int(deep*100)}_{int(forced*100)}']=pd.DataFrame({'tq':ta,'q':0.},index=idx)
    rows=[]
    for name,targ in cands.items():
        band=.10 if name.startswith('S9') else 0.
        r=portfolio(rt,rq,targ,band)
        for label,mask in [('full',idx>=pd.Timestamp('1985-01-31')),('pre2010',idx<pd.Timestamp('2010-01-01')),('holdout2010',idx>=pd.Timestamp('2010-01-01'))]:
            rr=r.loc[mask]; rows.append({'strategy':name,'period':label,**metrics(rr)})
    z=pd.DataFrame(rows); z.to_csv('results/dip_buy_overlay_screen.csv',index=False)
    print('\nFULL HISTORY TOP\n',z[z.period=='full'].sort_values(['cagr','max_dd'],ascending=[False,False]).head(30).to_string(index=False))
    print('\n2010+ TOP\n',z[z.period=='holdout2010'].sort_values('cagr',ascending=False).head(30).to_string(index=False))
    p=z.pivot(index='strategy',columns='period',values=['cagr','max_dd','min_5y_cagr'])
    p['score']=np.minimum(p[('cagr','pre2010')],p[('cagr','holdout2010')])+0.20*np.minimum(p[('max_dd','pre2010')],p[('max_dd','holdout2010')])
    print('\nROBUST SPLIT SCORE\n',p.sort_values('score',ascending=False).head(30).to_string())
if __name__=='__main__': main()
