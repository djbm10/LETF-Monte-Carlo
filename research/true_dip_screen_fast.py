from __future__ import annotations
import math, numpy as np, pandas as pd
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'; TD=252; TC=.0003

def read(n):
 d=pd.read_csv(BASE+n,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize(); return pd.Series(pd.to_numeric(d.Close,errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()
def met(r):
 e=(1+r).cumprod(); yrs=len(r)/TD; dd=e/e.cummax()-1; r5=(e/e.shift(5*TD))**(1/5)-1
 return {'cagr':e.iloc[-1]**(1/yrs)-1,'max_dd':dd.min(),'min_5y':r5.min(),'vol':r.std()*math.sqrt(TD)}
def port(rt,targ,band=.10):
 w=0.; wealth=1.; out=np.empty(len(rt)); rr=rt.to_numpy(); tt=targ.to_numpy()
 for i in range(len(rr)):
  want=tt[i]
  if abs(want-w)>=band: wealth*=1-TC*abs(want-w); w=want
  old=wealth; risky=old*w*(1+rr[i]); cash=old*(1-w); wealth=risky+cash; out[i]=wealth/old-1; w=risky/wealth
 return pd.Series(out,index=rt.index)
def main():
 q,tq,sp=read('synthetic-qqq.tsv'),read('synthetic-tqqq.tsv'),read('spy.tsv'); idx=q.index.intersection(tq.index).intersection(sp.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]; q,tq,sp=q.reindex(idx),tq.reindex(idx),sp.reindex(idx)
 rt=tq.pct_change().fillna(0); ma200=sp.rolling(200).mean(); bull=(sp>ma200).shift(1).fillna(False); vol=rt.rolling(20).std().shift(1)*math.sqrt(TD); hi=tq.rolling(252,min_periods=63).max(); dd=(tq/hi-1).shift(1)
 ql=q.shift(1); q10=q.rolling(10).mean().shift(1); q20=q.rolling(20).mean().shift(1); q50=q.rolling(50).mean().shift(1); r5=q.pct_change(5).shift(1); r10=q.pct_change(10).shift(1); d=q.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); rsi=(100-100/(1+g/l)).shift(1)
 base=pd.Series(np.clip(np.where(bull,.35,.12)/vol.replace(0,np.nan),0,1),index=idx).fillna(0); c={'S9_10':base}
 cond={'BULL':bull,'BULL_R5':bull&(r5>0),'BULL_R10':bull&(r10>0),'BULL_RECLAIM10':bull&(ql>q10),'BULL_RECLAIM20':bull&(ql>q20),'BULL_RSI35':bull&(rsi>35)}
 for st in (.10,.15,.20,.25):
  levels=[st,st+.10,st+.20,st+.30]
  for step in (.025,.05,.075,.10):
   ladder=np.select([dd<=-levels[3],dd<=-levels[2],dd<=-levels[1],dd<=-levels[0]],[4*step,3*step,2*step,step],default=0.)
   for cn,cc in cond.items(): c[f'DIP_s{int(st*100)}_b{int(step*1000):03d}_{cn}']=pd.Series(np.clip(base+np.where(cc,ladder,0),0,1),idx)
 # deeper bear-market entries only after reversal evidence
 for deep in (.20,.30,.40,.50):
  for add in (.10,.20,.30,.40):
   for typ,cc in [('REV20',(r5>0)&(ql>q20)),('REV50',(r10>0)&(ql>q50))]:
    boost=np.where((dd<=-deep)&(~bull)&cc,add,0); c[f'BEAR_{int(deep*100)}_a{int(add*100)}_{typ}']=pd.Series(np.clip(base+boost,0,1),idx)
 rows=[]; events={'1987':('1987-08-01','1989-07-31'),'dotcom':('2000-03-01','2007-06-30'),'gfc':('2007-10-01','2013-03-31'),'covid':('2020-02-01','2020-08-31'),'2022':('2022-01-01','2024-07-31')}
 for name,t in c.items():
  r=port(rt,t)
  for per,mask in [('pre2010',idx<pd.Timestamp('2010-01-01')),('holdout2010',idx>=pd.Timestamp('2010-01-01')),('full',np.ones(len(idx),bool))]: rows.append({'strategy':name,'period':per,**met(r.loc[mask])})
  for ev,(a,b) in events.items(): rows.append({'strategy':name,'period':ev,**met(r.loc[(idx>=a)&(idx<=b)])})
 z=pd.DataFrame(rows); z.to_csv('results/true_dip_fast.csv',index=False); tr=z[z.period=='pre2010'].copy(); tr['score']=tr.cagr+.35*tr.min_5y+.20*tr.max_dd; frozen=tr.sort_values('score',ascending=False).head(12).strategy.tolist(); pd.DataFrame({'strategy':frozen}).to_csv('results/true_dip_fast_frozen.csv',index=False)
 print('FROZEN PRE2010'); print(tr[tr.strategy.isin(frozen)].sort_values('score',ascending=False).to_string(index=False)); print('\nHOLDOUT'); print(z[(z.period=='holdout2010')&z.strategy.isin(frozen)].sort_values('cagr',ascending=False).to_string(index=False)); print('\nEVENTS'); print(z[z.strategy.isin(frozen[:6])&z.period.isin(events)].sort_values(['strategy','period']).to_string(index=False))
if __name__=='__main__': main()
