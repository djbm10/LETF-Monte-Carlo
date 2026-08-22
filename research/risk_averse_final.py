from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
import research.nasdaq1985_final_tournament as n
TD=252; PATHS=5000; WARM=300; H=np.array([10,20,40],dtype=np.int64)
NAMES=['VOL15','VOL20','VOL25','TV25_0','TV30_0','TV35_0','NDX_1x']
SCEN=n.SCEN
@njit(cache=True,parallel=True)
def sim(M,R):
 b,N=M.shape; Z=7; K=3; term=np.empty((b,Z,K)); mdd=np.empty((b,Z,K)); hd=H*TD
 for p in prange(b):
  px=np.empty(N);px[0]=1.
  for t in range(1,N):px[t]=px[t-1]*max(1+M[p,t],1e-12)
  cp=np.zeros(N+1);cm=np.zeros(N+1);cm2=np.zeros(N+1)
  for t in range(N):cp[t+1]=cp[t]+px[t];cm[t+1]=cm[t]+M[p,t];cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);hk=0
  for t in range(WARM,N):
   sig=t-1;sma=(cp[sig+1]-cp[sig+1-200])/200.;bull=px[sig]>sma;sm=cm[sig+1]-cm[sig+1-20];sm2=cm2[sig+1]-cm2[sig+1-20];vol=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD)
   e=np.zeros(Z)
   if vol>0:
    e[0]=min(.15/vol,3.);e[1]=min(.20/vol,3.);e[2]=min(.25/vol,3.);e[3]=min((.25 if bull else 0.)/vol,3.);e[4]=min((.30 if bull else 0.)/vol,3.);e[5]=min((.35 if bull else 0.)/vol,3.)
   e[6]=1.
   for z in range(Z):
    rr=R[p,t]+e[z]*(M[p,t]-R[p,t]);wealth[z]*=max(1+rr,1e-12);peak[z]=max(peak[z],wealth[z]);ddmin[z]=min(ddmin[z],wealth[z]/peak[z]-1)
   elapsed=t-WARM+1
   if hk<K and elapsed==hd[hk]:
    for z in range(Z):term[p,z,hk]=wealth[z];mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd
def boot(m,r,dates,b,nlen,rng,sc):
 L=int(sc.get('block',63));N=len(m);starts=np.arange(0,max(1,N-L));stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]);w=None
 if sc.get('stress_weight',1)>1:w=np.ones(len(starts));w[stress[:len(starts)]]=sc['stress_weight'];w/=w.sum()
 M=np.empty((b,nlen));R=np.empty((b,nlen))
 for i in range(b):
  pos=0
  while pos<nlen:
   st=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts)));take=min(L,nlen-pos);M[i,pos:pos+take]=m[st:st+take];R[i,pos:pos+take]=r[st:st+take];pos+=take
 M-=sc.get('drift_cut',0.)/TD
 if sc.get('rate_floor',0):R=np.maximum(R,sc['rate_floor']/TD)
 cp=sc.get('crashes_per_year',0.)
 if cp:
  mask=rng.random(M.shape)<cp/TD;M=np.where(mask,M-rng.uniform(.08,.18,M.shape),M)
 return M,R
def main():
 df=n.load_data(.007);m=df.ndx.to_numpy();r=df.rf.to_numpy();dates=df.index;total=WARM+40*TD;rng=np.random.default_rng(4260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,R=boot(m,r,dates,b,total,rng,sc);t,d=sim(M,R);TT.append(t);DD.append(d);done+=b
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd50':np.mean(D[:,z,j]<=-.5),'prob_dd70':np.mean(D[:,z,j]<=-.7),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/risk_averse_final.csv',index=False);agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),max_dd50=('prob_dd50','max'),max_dd70=('prob_dd70','max'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/risk_averse_ranking.csv',index=False);print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
