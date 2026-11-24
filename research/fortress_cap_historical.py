from __future__ import annotations
import math, numpy as np, pandas as pd
TD=252
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
EVENTS={'1987 crash':('1987-08-01','1989-07-31'),'dot-com':('2000-03-01','2007-06-30'),'GFC':('2007-10-01','2013-03-31'),'COVID':('2020-02-01','2020-08-31'),'2022 inflation':('2022-01-01','2024-07-31')}
def read(name,col='Close'):
 d=pd.read_csv(BASE+name,sep='\t');d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize();return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()
def load():
 q=read('synthetic-qqq.tsv');rr=read('short-rates.tsv','Rate');q=q[q.index>=pd.Timestamp('1985-01-31')];rf=rr.reindex(q.index).ffill().bfill()/100/TD;m=q.pct_change().fillna(0)+.002/TD;m.loc[m.index<pd.Timestamp('1999-03-10')]+=.007/TD;return m.to_numpy(),rf.to_numpy(),q.index
def tv_ret(m,rf,target=.15,cap=1.,band=.30):
 px=np.cumprod(np.maximum(1+m,1e-12));sma=pd.Series(px).rolling(200).mean().to_numpy();vol=pd.Series(m).rolling(20).std().to_numpy()*math.sqrt(TD);e=np.zeros(len(m));hold=0.
 for i in range(1,len(m)):
  if not (np.isfinite(sma[i-1]) and np.isfinite(vol[i-1]) and vol[i-1]>0):continue
  want=min(cap,target/vol[i-1]) if px[i-1]>sma[i-1] else 0.
  if abs(want-hold)>=band:hold=want
  e[i]=hold
 return rf+e*(m-rf)-.00010*np.abs(np.diff(e,prepend=0.))
def stats(r):
 g=np.maximum(1+np.asarray(r,float),1e-12);eq=np.cumprod(g);yrs=len(r)/TD;peak=np.maximum.accumulate(eq);dd=eq/peak-1;cur=mx=0
 for x in dd:
  if x<0:cur+=1;mx=max(mx,cur)
  else:cur=0
 return {'ann_return':eq[-1]**(1/yrs)-1,'total_return':eq[-1]-1,'max_dd':dd.min(),'max_underwater_years':mx/TD}
def main():
 m,rf,idx=load();strategies={'NDX_1X':rf+1*(m-rf),'TV15_0_CAP1':tv_ret(m,rf,.15,1.),'TV15_0_CAP1.5':tv_ret(m,rf,.15,1.5),'TV20_0_CAP1':tv_ret(m,rf,.20,1.)};rows=[]
 for name,r in strategies.items():
  sr=pd.Series(r,index=idx)
  for ev,(a,b) in EVENTS.items():rows.append({'strategy':name,'event':ev,**stats(sr.loc[(sr.index>=a)&(sr.index<=b)])})
 out=pd.DataFrame(rows);out.to_csv('results/fortress_cap_historical.csv',index=False);print(out.to_string(index=False))
if __name__=='__main__':main()
