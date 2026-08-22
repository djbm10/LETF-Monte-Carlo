from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
TD=252; PATHS=10000; WARM=300
H=np.array([5,10,20,40],dtype=np.int64)
NAMES=('SPY60_40','SPY80_20','NDX_1X','VOL15','VOL20','VOL25','TV25_0')
SCEN={
 'baseline':{'block':63},
 'low_premium':{'block':63,'drift_cut':.04},
 'high_rates':{'block':63,'rate_floor':.05},
 'stress_regimes':{'block':126,'stress_weight':4},
 'long_clusters':{'block':252,'stress_weight':2},
 'crash_heavy':{'block':63,'crashes_per_year':.20},
 'combined_hostile':{'block':126,'stress_weight':4,'drift_cut':.04,'rate_floor':.05,'crashes_per_year':.25},
}

def read(n,col='Close'):
 d=pd.read_csv(BASE+n,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize()
 return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

def load():
 q=read('synthetic-qqq.tsv'); sp=read('spy.tsv'); rr=read('short-rates.tsv','Rate')
 idx=q.index.intersection(sp.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]
 q=q.reindex(idx); sp=sp.reindex(idx); rf=rr.reindex(idx).ffill().bfill()/100/TD
 ndx=q.pct_change().fillna(0)+.002/TD; ndx.loc[idx<pd.Timestamp('1999-03-10')]+=.007/TD
 spr=sp.pct_change().fillna(0)
 return ndx.to_numpy(),spr.to_numpy(),rf.to_numpy(),idx

@njit(cache=True,parallel=True)
def sim(M,S,R):
 b,N=M.shape; Z=7; K=len(H); hds=H*TD
 term=np.empty((b,Z,K)); mdd=np.empty((b,Z,K))
 for p in prange(b):
  spx=np.empty(N); spx[0]=1.
  cs=np.zeros(N+1); cm=np.zeros(N+1); cm2=np.zeros(N+1)
  for t in range(1,N): spx[t]=spx[t-1]*max(1.+S[p,t],1e-12)
  for t in range(N):
   cs[t+1]=cs[t]+spx[t]; cm[t+1]=cm[t]+M[p,t]; cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
  wealth=np.ones(Z); peak=np.ones(Z); ddmin=np.zeros(Z); prev=np.zeros(Z); hk=0
  for t in range(WARM,N):
   sig=t-1
   sm=cm[sig+1]-cm[sig+1-20]; sm2=cm2[sig+1]-cm2[sig+1-20]
   var=max((sm2-sm*sm/20.)/19.,0.); vol=math.sqrt(var)*math.sqrt(TD)
   sma=(cs[sig+1]-cs[sig+1-200])/200.; bull=spx[sig]>sma
   e15=min(2.,.15/vol) if vol>0 else 0.; e20=min(2.,.20/vol) if vol>0 else 0.; e25=min(2.,.25/vol) if vol>0 else 0.
   etv=min(2.,.25/vol) if bull and vol>0 else 0.
   # Daily returns. SPY mixes are continuously rebalanced approximations; dynamic NDX exposures pay 1bp per unit notional change.
   r0=.60*S[p,t]+.40*R[p,t]; r1=.80*S[p,t]+.20*R[p,t]; r2=M[p,t]
   es=np.array([0.,0.,1.,e15,e20,e25,etv])
   rs=np.array([r0,r1,r2, R[p,t]+e15*(M[p,t]-R[p,t]), R[p,t]+e20*(M[p,t]-R[p,t]), R[p,t]+e25*(M[p,t]-R[p,t]), R[p,t]+etv*(M[p,t]-R[p,t])])
   for z in range(Z):
    cost=0.
    if z>=3: cost=.0001*abs(es[z]-prev[z]); prev[z]=es[z]
    wealth[z]*=max(1.+rs[z]-cost,1e-12)
    if wealth[z]>peak[z]: peak[z]=wealth[z]
    dd=wealth[z]/peak[z]-1.; ddmin[z]=min(ddmin[z],dd)
   elapsed=t-WARM+1
   if hk<K and elapsed==hds[hk]:
    for z in range(Z): term[p,z,hk]=wealth[z]; mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd

def boot(m,s,r,dates,b,n,rng,sc):
 L=sc.get('block',63); N=len(m); starts=np.arange(0,max(1,N-L)); stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]); w=None
 if sc.get('stress_weight',1)>1:
  w=np.ones(len(starts)); w[stress[:len(starts)]]=sc['stress_weight']; w/=w.sum()
 M=np.empty((b,n)); S=np.empty((b,n)); R=np.empty((b,n))
 for i in range(b):
  pos=0
  while pos<n:
   st=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts)))
   take=min(L,n-pos); M[i,pos:pos+take]=m[st:st+take]; S[i,pos:pos+take]=s[st:st+take]; R[i,pos:pos+take]=r[st:st+take]; pos+=take
 cut=sc.get('drift_cut',0.); M-=cut/TD; S-=cut/TD
 if sc.get('rate_floor',0): R=np.maximum(R,sc['rate_floor']/TD)
 cp=sc.get('crashes_per_year',0.)
 if cp:
  mask=rng.random(M.shape)<cp/TD; shock=rng.uniform(.08,.18,M.shape); M=np.where(mask,M-shock,M); S=np.where(mask,S-shock,S)
 return M,S,R

def main():
 m,s,r,d=load(); total=WARM+40*TD; rng=np.random.default_rng(240826); rows=[]
 sim(np.zeros((1,total)),np.zeros((1,total)),np.zeros((1,total)))
 for sn,ss in SCEN.items():
  TT=[]; DD=[]; done=0
  while done<PATHS:
   b=min(64,PATHS-done); M,S,R=boot(m,s,r,d,b,total,rng,ss); t,x=sim(M,S,R); TT.append(t); DD.append(x); done+=b
  T=np.concatenate(TT); D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1
    rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd60':np.mean(D[:,z,j]<=-.60),'prob_dd90':np.mean(D[:,z,j]<=-.90),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows); out.to_csv('results/friend_safe_frontier.csv',index=False)
 agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd60=('prob_dd60','max'),max_dd90=('prob_dd90','max'),worst_median_dd=('median_dd','min')).reset_index()
 agg.to_csv('results/friend_safe_ranking.csv',index=False)
 print(agg[agg.horizon.isin([20,40])].to_string(index=False)); print('\n20Y BY SCENARIO'); print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__': main()
