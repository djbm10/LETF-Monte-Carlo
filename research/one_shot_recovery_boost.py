from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
import research.nasdaq1985_final_tournament as n
TD=252; PATHS=5000; WARM=300; H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=['S9_35_12','REC15_10_63','REC20_10_63','REC25_15_63','REC20_10_126','REC_TIERED_63']
SCEN=n.SCEN
@njit(cache=True,parallel=True)
def sim(M,S,R):
 b,N=M.shape;Z=6;K=len(H);term=np.empty((b,Z,K));mdd=np.empty((b,Z,K));hd=H*TD
 for p in prange(b):
  q=np.empty(N);sp=np.empty(N);tr=np.empty(N);q[0]=sp[0]=1.;tr[0]=0.
  for t in range(1,N):q[t]=q[t-1]*max(1+M[p,t],1e-12);sp[t]=sp[t-1]*max(1+S[p,t],1e-12);tr[t]=3*M[p,t]-2*R[p,t]-(.0088+2*.0065)/TD
  cs=np.zeros(N+1);cq=np.zeros(N+1);cv=np.zeros(N+1);cv2=np.zeros(N+1)
  for t in range(N):cs[t+1]=cs[t]+sp[t];cq[t+1]=cq[t]+q[t];cv[t+1]=cv[t]+tr[t];cv2[t+1]=cv2[t]+tr[t]*tr[t]
  dq=np.empty(N,np.int64);head=tail=0;rm=np.empty(N)
  for t in range(N):
   while head<tail and dq[head]<=t-252:head+=1
   while head<tail and q[dq[tail-1]]<=q[t]:tail-=1
   dq[tail]=t;tail+=1;rm[t]=q[dq[head]]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);w=np.zeros(Z);expiry=np.full(Z,-1,np.int64);last=np.full(Z,-9999,np.int64);hk=0
  for t in range(WARM,N):
   sig=t-1;sma200=(cs[sig+1]-cs[sig+1-200])/200.;bull=sp[sig]>sma200;ma50=(cq[sig+1]-cq[sig+1-50])/50.;ma50prev=(cq[sig]-cq[sig-50])/50.;cross=q[sig]>ma50 and q[sig-1]<=ma50prev
   sm=cv[sig+1]-cv[sig+1-20];sm2=cv2[sig+1]-cv2[sig+1-20];v=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD);base=min(1.,(.35 if bull else .12)/v) if v>0 else 0.;dd=q[sig]/rm[sig]-1.;q10=q[sig]/q[max(0,sig-10)]-1.
   # one-shot triggers; 126d cooldown, canceled if QQQ loses 50DMA
   if cross and q10>0:
    if dd<=-.15 and t-last[1]>=126:expiry[1]=t+63;last[1]=t
    if dd<=-.20 and t-last[2]>=126:expiry[2]=t+63;last[2]=t
    if dd<=-.25 and t-last[3]>=126:expiry[3]=t+63;last[3]=t
    if dd<=-.20 and t-last[4]>=189:expiry[4]=t+126;last[4]=t
    if dd<=-.15 and t-last[5]>=126:expiry[5]=t+63;last[5]=t
   boosts=np.zeros(Z);boosts[1]=.10 if t<=expiry[1] and q[sig]>ma50 else 0.;boosts[2]=.10 if t<=expiry[2] and q[sig]>ma50 else 0.;boosts[3]=.15 if t<=expiry[3] and q[sig]>ma50 else 0.;boosts[4]=.10 if t<=expiry[4] and q[sig]>ma50 else 0.
   if t<=expiry[5] and q[sig]>ma50:boosts[5]=.15 if dd<=-.25 else .10
   for z in range(Z):
    want=min(1.,base+boosts[z])
    if abs(want-w[z])>=.10:wealth[z]*=max(1-.0003*abs(want-w[z]),0.);w[z]=want
    risky=wealth[z]*w[z]*(1+tr[t]);cash=wealth[z]*(1-w[z]);wealth[z]=risky+cash;w[z]=risky/max(wealth[z],1e-15);peak[z]=max(peak[z],wealth[z]);ddmin[z]=min(ddmin[z],wealth[z]/peak[z]-1)
   elapsed=t-WARM+1
   if hk<K and elapsed==hd[hk]:
    for z in range(Z):term[p,z,hk]=wealth[z];mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd
def boot(m,s,r,dates,b,nlen,rng,sc):
 L=int(sc.get('block',63));N=len(m);starts=np.arange(0,max(1,N-L));stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]);ww=None
 if sc.get('stress_weight',1)>1:ww=np.ones(len(starts));ww[stress[:len(starts)]]=sc['stress_weight'];ww/=ww.sum()
 M=np.empty((b,nlen));S=np.empty((b,nlen));R=np.empty((b,nlen))
 for i in range(b):
  pos=0
  while pos<nlen:
   st=int(rng.choice(starts,p=ww)) if ww is not None else int(rng.integers(0,len(starts)));take=min(L,nlen-pos);M[i,pos:pos+take]=m[st:st+take];S[i,pos:pos+take]=s[st:st+take];R[i,pos:pos+take]=r[st:st+take];pos+=take
 cut=sc.get('drift_cut',0.);M-=cut/TD;S-=cut/TD
 if sc.get('rate_floor',0):R=np.maximum(R,sc['rate_floor']/TD)
 cp=sc.get('crashes_per_year',0.)
 if cp:
  mask=rng.random(M.shape)<cp/TD;shock=rng.uniform(.08,.18,M.shape);M=np.where(mask,M-shock,M);S=np.where(mask,S-shock,S)
 return M,S,R
def main():
 df=n.load_data(.007);m=df.ndx.to_numpy();r=df.rf.to_numpy();dates=df.index;sp=n.read_series('spy.tsv').reindex(df.index).ffill().bfill().pct_change().fillna(0).to_numpy();total=WARM+50*TD;rng=np.random.default_rng(15260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,S,R=boot(m,sp,r,dates,b,total,rng,sc);t,d=sim(M,S,R);TT.append(t);DD.append(d);done+=b
   if done%1000==0:print(sn,done,flush=True)
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/one_shot_recovery_boost.csv',index=False);agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/one_shot_recovery_ranking.csv',index=False);print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False));print('\n40Y');print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
