from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'; TD=252; PATHS=5000; WARM=300
H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=['S9_35_12','BASE_35_0','DIP_35_0_075','BASE_40_0','DIP_40_0_050','DIP_40_0_075','BASE_40_12','DIP_40_12_050']
SCEN={'baseline':{'block':63},'low_premium':{'block':63,'drift_cut':.04},'high_rates':{'block':63,'rate_floor':.05},'stress_regimes':{'block':126,'stress_weight':4},'long_clusters':{'block':252,'stress_weight':2},'crash_heavy':{'block':63,'crashes_per_year':.20},'combined_hostile':{'block':126,'stress_weight':4,'drift_cut':.04,'rate_floor':.05,'crashes_per_year':.25}}
def read(n,col='Close'):
 d=pd.read_csv(BASE+n,sep='\t'); d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize(); return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()
def load():
 q=read('synthetic-qqq.tsv'); sp=read('spy.tsv'); rr=read('short-rates.tsv','Rate'); idx=q.index.intersection(sp.index); idx=idx[idx>=pd.Timestamp('1985-01-31')]; q=q.reindex(idx);sp=sp.reindex(idx);rf=rr.reindex(idx).ffill().bfill()/100/TD
 m=q.pct_change().fillna(0)+.002/TD; m.loc[idx<pd.Timestamp('1999-03-10')]+=.007/TD; s=sp.pct_change().fillna(0); return m.to_numpy(),s.to_numpy(),rf.to_numpy(),idx
@njit(cache=True,parallel=True)
def sim(M,S,R):
 b,N=M.shape; K=len(H); Z=8; term=np.empty((b,Z,K)); mdd=np.empty((b,Z,K)); hds=H*TD
 for p in prange(b):
  qpx=np.empty(N);spx=np.empty(N);tqpx=np.empty(N);tr=np.empty(N);qpx[0]=spx[0]=tqpx[0]=1.;tr[0]=0.
  for t in range(1,N):
   qpx[t]=qpx[t-1]*max(1+M[p,t],1e-12);spx[t]=spx[t-1]*max(1+S[p,t],1e-12);tr[t]=3*M[p,t]-2*R[p,t]-(.0088+2*.0065)/TD;tqpx[t]=tqpx[t-1]*max(1+tr[t],1e-12)
  cs=np.zeros(N+1);cv=np.zeros(N+1);cv2=np.zeros(N+1)
  for t in range(N):cs[t+1]=cs[t]+spx[t];cv[t+1]=cv[t]+tr[t];cv2[t+1]=cv2[t]+tr[t]*tr[t]
  dq=np.empty(N,np.int64);head=tail=0;rm=np.empty(N)
  for t in range(N):
   while head<tail and dq[head]<=t-252:head+=1
   while head<tail and tqpx[dq[tail-1]]<=tqpx[t]:tail-=1
   dq[tail]=t;tail+=1;rm[t]=tqpx[dq[head]]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);w=np.zeros(Z);hk=0
  for t in range(WARM,N):
   sig=t-1;sma=(cs[sig+1]-cs[sig+1-200])/200.;bull=spx[sig]>sma
   sm=cv[sig+1]-cv[sig+1-20];sm2=cv2[sig+1]-cv2[sig+1-20];v=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD)
   b3512=min(1.,(.35 if bull else .12)/v) if v>0 else 0.;b350=min(1.,(.35 if bull else 0.)/v) if v>0 else 0.;b400=min(1.,(.40 if bull else 0.)/v) if v>0 else 0.;b4012=min(1.,(.40 if bull else .12)/v) if v>0 else 0.
   draw=tqpx[sig]/rm[sig]-1.;q10=qpx[sig]/qpx[max(0,sig-10)]-1.;rung=0
   if draw<=-.40:rung=4
   elif draw<=-.30:rung=3
   elif draw<=-.20:rung=2
   elif draw<=-.10:rung=1
   ok=bull and q10>0
   want=np.empty(Z);want[0]=b3512;want[1]=b350;want[2]=min(1.,b350+(.075*rung if ok else 0.));want[3]=b400;want[4]=min(1.,b400+(.05*rung if ok else 0.));want[5]=min(1.,b400+(.075*rung if ok else 0.));want[6]=b4012;want[7]=min(1.,b4012+(.05*rung if ok else 0.))
   for z in range(Z):
    if abs(want[z]-w[z])>=.10:wealth[z]*=max(1-.0003*abs(want[z]-w[z]),0.);w[z]=want[z]
    risky=wealth[z]*w[z]*(1+tr[t]);cash=wealth[z]*(1-w[z]);wealth[z]=risky+cash;w[z]=risky/max(wealth[z],1e-15);peak[z]=max(peak[z],wealth[z]);ddmin[z]=min(ddmin[z],wealth[z]/peak[z]-1)
   elapsed=t-WARM+1
   if hk<K and elapsed==hds[hk]:
    for z in range(Z):term[p,z,hk]=wealth[z];mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd
def boot(m,s,r,dates,b,n,rng,sc):
 L=sc.get('block',63);N=len(m);starts=np.arange(0,max(1,N-L));stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]);w=None
 if sc.get('stress_weight',1)>1:w=np.ones(len(starts));w[stress[:len(starts)]]=sc['stress_weight'];w/=w.sum()
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
def main():
 m,s,r,d=load();total=WARM+50*TD;rng=np.random.default_rng(3260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,S,R=boot(m,s,r,d,b,total,rng,sc);t,x=sim(M,S,R);TT.append(t);DD.append(x);done+=b
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/aggressive_final_hostile.csv',index=False);agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/aggressive_final_ranking.csv',index=False);print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False));print('\n40Y');print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
