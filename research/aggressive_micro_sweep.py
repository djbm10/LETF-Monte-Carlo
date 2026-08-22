from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
import research.nasdaq1985_final_tournament as n
TD=252; PATHS=10_000; WARM=300; H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=['S9_35_12','S9_37.5_12','S9_40_12','DIP_35_12_025','DIP_37.5_12_025','DIP_37.5_12_050','DIP_40_12_025']
SCEN=n.SCEN
@njit(cache=True,parallel=True)
def sim(M,S,R):
 b,N=M.shape;Z=7;K=len(H);term=np.empty((b,Z,K));mdd=np.empty((b,Z,K));hd=H*TD
 for p in prange(b):
  q=np.empty(N);sp=np.empty(N);tr=np.empty(N);tqp=np.empty(N);q[0]=sp[0]=tqp[0]=1.;tr[0]=0.
  for t in range(1,N):
   q[t]=q[t-1]*max(1+M[p,t],1e-12);sp[t]=sp[t-1]*max(1+S[p,t],1e-12);tr[t]=3*M[p,t]-2*R[p,t]-(.0088+2*.0065)/TD;tqp[t]=tqp[t-1]*max(1+tr[t],1e-12)
  cs=np.zeros(N+1);cv=np.zeros(N+1);cv2=np.zeros(N+1)
  for t in range(N):cs[t+1]=cs[t]+sp[t];cv[t+1]=cv[t]+tr[t];cv2[t+1]=cv2[t]+tr[t]*tr[t]
  dq=np.empty(N,np.int64);head=tail=0;rm=np.empty(N)
  for t in range(N):
   while head<tail and dq[head]<=t-252:head+=1
   while head<tail and tqp[dq[tail-1]]<=tqp[t]:tail-=1
   dq[tail]=t;tail+=1;rm[t]=tqp[dq[head]]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);w=np.zeros(Z);hk=0
  for t in range(WARM,N):
   sig=t-1;sma=(cs[sig+1]-cs[sig+1-200])/200.;bull=sp[sig]>sma
   sm=cv[sig+1]-cv[sig+1-20];sm2=cv2[sig+1]-cv2[sig+1-20];v=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD)
   b35=min(1.,(.35 if bull else .12)/v) if v>0 else 0.;b375=min(1.,(.375 if bull else .12)/v) if v>0 else 0.;b40=min(1.,(.40 if bull else .12)/v) if v>0 else 0.
   draw=tqp[sig]/rm[sig]-1.;q10=q[sig]/q[max(0,sig-10)]-1.;rung=0
   if draw<=-.40:rung=4
   elif draw<=-.30:rung=3
   elif draw<=-.20:rung=2
   elif draw<=-.10:rung=1
   ok=bull and q10>0
   want=np.empty(Z);want[0]=b35;want[1]=b375;want[2]=b40;want[3]=min(1.,b35+(.025*rung if ok else 0.));want[4]=min(1.,b375+(.025*rung if ok else 0.));want[5]=min(1.,b375+(.05*rung if ok else 0.));want[6]=min(1.,b40+(.025*rung if ok else 0.))
   for z in range(Z):
    if abs(want[z]-w[z])>=.10:wealth[z]*=max(1-.0003*abs(want[z]-w[z]),0.);w[z]=want[z]
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
def hist_series(m,s,r,bull_target,dip_step=0.):
 N=len(m);q=np.cumprod(np.maximum(1+m,1e-12));sp=np.cumprod(np.maximum(1+s,1e-12));tr=np.zeros(N);tr[1:]=3*m[1:]-2*r[1:]-(.0088+2*.0065)/TD;tqp=np.cumprod(np.maximum(1+tr,1e-12));sma=pd.Series(sp).rolling(200).mean().to_numpy();vol=pd.Series(tr).rolling(20).std().to_numpy()*math.sqrt(TD);rm=pd.Series(tqp).rolling(252,min_periods=1).max().to_numpy();out=np.zeros(N);w=0.;wealth=1.
 for t in range(1,N):
  sig=t-1
  if not (np.isfinite(sma[sig]) and np.isfinite(vol[sig]) and vol[sig]>0):continue
  bull=sp[sig]>sma[sig];want=min(1.,(bull_target if bull else .12)/vol[sig]);draw=tqp[sig]/rm[sig]-1.;rung=4 if draw<=-.40 else 3 if draw<=-.30 else 2 if draw<=-.20 else 1 if draw<=-.10 else 0
  if dip_step and bull and q[sig]/q[max(0,sig-10)]-1>0:want=min(1.,want+dip_step*rung)
  if abs(want-w)>=.10:wealth*=max(1-.0003*abs(want-w),0.);w=want
  old=wealth;risky=wealth*w*(1+tr[t]);cash=wealth*(1-w);wealth=risky+cash;w=risky/max(wealth,1e-15);out[t]=wealth/old-1
 return out
def met(x):
 g=np.maximum(1+np.asarray(x),1e-12);eq=np.cumprod(g);yrs=len(x)/TD;dd=eq/np.maximum.accumulate(eq)-1;return eq[-1]**(1/yrs)-1,dd.min()
def main():
 df=n.load_data(.007);m=df.ndx.to_numpy();r=df.rf.to_numpy();dates=df.index
 # SPY proxy used in prior true-dip work
 sp=n.read_series('spy.tsv').reindex(df.index).ffill().bfill().pct_change().fillna(0).to_numpy()
 hs=[]
 params=[(.35,0.),(.375,0.),(.40,0.),(.35,.025),(.375,.025),(.375,.05),(.40,.025)]
 split=pd.Timestamp('2010-02-11')
 for name,(bt,ds) in zip(NAMES,params):
  rr=hist_series(m,sp,r,bt,ds);pre=df.index<split;post=~pre;c0,d0=met(rr[pre]);c1,d1=met(rr[post]);hs.append({'strategy':name,'pre2010_cagr':c0,'pre2010_dd':d0,'holdout_cagr':c1,'holdout_dd':d1})
 pd.DataFrame(hs).to_csv('results/aggressive_micro_history.csv',index=False);print('HISTORY');print(pd.DataFrame(hs).to_string(index=False))
 total=WARM+50*TD;rng=np.random.default_rng(14260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,S,R=boot(m,sp,r,dates,b,total,rng,sc);t,d=sim(M,S,R);TT.append(t);DD.append(d);done+=b
   if done%2000==0:print(sn,done,flush=True)
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/aggressive_micro_hostile.csv',index=False);agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/aggressive_micro_ranking.csv',index=False);print('\nRANK');print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False));print('\n40Y');print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
