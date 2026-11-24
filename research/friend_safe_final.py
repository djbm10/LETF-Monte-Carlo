from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'; TD=252; PATHS=10000; WARM=300
H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=('VOL20','VOL25','VOL30','TV20_0','TV25_0','TV30_0','TV35_0','TV25_5','TV30_5')
TARGET=np.array([.20,.25,.30,.20,.25,.30,.35,.25,.30]); BEAR=np.array([np.nan,np.nan,np.nan,0.,0.,0.,0.,.05,.05])
SCEN={'baseline':{'block':63},'low_premium':{'block':63,'drift_cut':.04},'high_rates':{'block':63,'rate_floor':.05},'stress_regimes':{'block':126,'stress_weight':4},'long_clusters':{'block':252,'stress_weight':2},'crash_heavy':{'block':63,'crashes_per_year':.20},'combined_hostile':{'block':126,'stress_weight':4,'drift_cut':.04,'rate_floor':.05,'crashes_per_year':.25}}
def read(n,col='Close'):
 d=pd.read_csv(BASE+n,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize(); return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()
def load():
 q=read('synthetic-qqq.tsv'); sp=read('spy.tsv'); rr=read('short-rates.tsv','Rate'); idx=q.index.intersection(sp.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]; q=q.reindex(idx);sp=sp.reindex(idx);rf=rr.reindex(idx).ffill().bfill()/100/TD
 m=q.pct_change().fillna(0)+.002/TD; m.loc[idx<pd.Timestamp('1999-03-10')]+=.007/TD; s=sp.pct_change().fillna(0); return m.to_numpy(),s.to_numpy(),rf.to_numpy(),idx
@njit(cache=True,parallel=True)
def sim(M,S,R):
 b,N=M.shape; Z=len(TARGET); K=len(H); term=np.empty((b,Z,K));mdd=np.empty((b,Z,K)); hds=H*TD
 for p in prange(b):
  qpx=np.empty(N);spx=np.empty(N);qpx[0]=spx[0]=1.
  for t in range(1,N): qpx[t]=qpx[t-1]*max(1.+M[p,t],1e-12);spx[t]=spx[t-1]*max(1.+S[p,t],1e-12)
  cs=np.zeros(N+1);cm=np.zeros(N+1);cm2=np.zeros(N+1)
  for t in range(N): cs[t+1]=cs[t]+spx[t];cm[t+1]=cm[t]+M[p,t];cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);hk=0
  for t in range(WARM,N):
   sig=t-1; sm=cm[sig+1]-cm[sig+1-20];sm2=cm2[sig+1]-cm2[sig+1-20];var=max((sm2-sm*sm/20.)/19.,0.);vol=math.sqrt(var)*math.sqrt(TD)
   sma=(cs[sig+1]-cs[sig+1-200])/200.; bull=spx[sig]>sma
   for z in range(Z):
    if vol<=0:e=0.
    else:
     if z<3: tv=TARGET[z]
     else: tv=TARGET[z] if bull else BEAR[z]
     e=min(3.,tv/vol)
    rr=R[p,t]+e*(M[p,t]-R[p,t]); wealth[z]*=max(1.+rr,1e-12)
    if wealth[z]>peak[z]:peak[z]=wealth[z]
    dd=wealth[z]/peak[z]-1.;ddmin[z]=min(ddmin[z],dd)
   elapsed=t-WARM+1
   if hk<K and elapsed==hds[hk]:
    for z in range(Z):term[p,z,hk]=wealth[z];mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd
def boot(m,s,r,dates,b,n,rng,sc):
 L=sc.get('block',63);N=len(m);starts=np.arange(0,max(1,N-L));stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]);w=None
 if sc.get('stress_weight',1)>1:w=np.where(stress[:len(starts)],sc['stress_weight'],1.0);w/=w.sum()
 M=np.empty((b,n));S=np.empty((b,n));R=np.empty((b,n))
 for i in range(b):
  pos=0
  while pos<n:
   st=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts)));take=min(L,n-pos);M[i,pos:pos+take]=m[st:st+take];S[i,pos:pos+take]=s[st:st+take];R[i,pos:pos+take]=r[st:st+take];pos+=take
 cut=sc.get('drift_cut',0.);M-=cut/TD;S-=cut/TD
 if sc.get('rate_floor',0):R=np.maximum(R,sc['rate_floor']/TD)
 cp=sc.get('crashes_per_year',0.)
 if cp:
  mask=rng.random(M.shape)<cp/TD;shock=rng.uniform(.08,.18,M.shape);M=np.where(mask,M-shock,M);S=np.where(mask,S-shock,S)
 return M,S,R
def hist(m,s,r,d):
 M=m[None,:];S=s[None,:];R=r[None,:]; t,x=sim(M,S,R); rows=[]
 for z,n in enumerate(NAMES):
  for j,h in enumerate(H):
   c=max(t[0,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':'historical_path','strategy':n,'horizon':int(h),'median_cagr':c,'p10_cagr':c,'p01_cagr':c,'prob_dd90':float(x[0,z,j]<=-.9),'median_dd':x[0,z,j]})
 return rows
def main():
 m,s,r,d=load();total=WARM+50*TD;rng=np.random.default_rng(260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,S,R=boot(m,s,r,d,b,total,rng,sc);t,x=sim(M,S,R);TT.append(t);DD.append(x);done+=b
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/friend_safe_final.csv',index=False)
 agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/friend_safe_final_ranking.csv',index=False)
 print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False));print('\n40Y');print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
