from __future__ import annotations
import math, numpy as np, pandas as pd
from numba import njit, prange
TD=252; PATHS=5000; WARM=300; H=np.array([20,40],dtype=np.int64)
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
NAMES=['NDX_1X','TV15_0_CAP1','TV15_0_CAP1.5','TV15_0_CAP2','TV15_0_CAP3','TV20_0_CAP1','TV20_0_CAP1.5','TV20_0_CAP2']
TARGET=np.array([0,.15,.15,.15,.15,.20,.20,.20]); CAPS=np.array([1.,1.,1.5,2.,3.,1.,1.5,2.])
SCEN={'baseline':{'block':63},'low_premium':{'block':63,'drift_cut':.04},'high_rates':{'block':63,'rate_floor':.05},'stress_regimes':{'block':126,'stress_weight':4},'long_clusters':{'block':252,'stress_weight':2},'crash_heavy':{'block':63,'crashes_per_year':.20},'combined_hostile':{'block':126,'stress_weight':4,'drift_cut':.04,'rate_floor':.05,'crashes_per_year':.25}}
def read(name,col='Close'):
 d=pd.read_csv(BASE+name,sep='\t');d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize();return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()
def load():
 q=read('synthetic-qqq.tsv');rr=read('short-rates.tsv','Rate');q=q[q.index>=pd.Timestamp('1985-01-31')];rf=rr.reindex(q.index).ffill().bfill()/100/TD;m=q.pct_change().fillna(0)+.002/TD;m.loc[m.index<pd.Timestamp('1999-03-10')]+=.007/TD;return m.to_numpy(),rf.to_numpy(),q.index
@njit(cache=True,parallel=True)
def sim(M,R):
 b,N=M.shape;Z=len(NAMES);K=len(H);term=np.empty((b,Z,K));mdd=np.empty((b,Z,K));hd=H*TD
 for p in prange(b):
  px=np.empty(N);px[0]=1.
  for t in range(1,N):px[t]=px[t-1]*max(1+M[p,t],1e-12)
  cp=np.zeros(N+1);cm=np.zeros(N+1);cm2=np.zeros(N+1)
  for t in range(N):cp[t+1]=cp[t]+px[t];cm[t+1]=cm[t]+M[p,t];cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
  wealth=np.ones(Z);peak=np.ones(Z);ddmin=np.zeros(Z);e=np.zeros(Z);hk=0
  for t in range(WARM,N):
   sig=t-1;sma=(cp[sig+1]-cp[sig+1-200])/200.;bull=px[sig]>sma;sm=cm[sig+1]-cm[sig+1-20];sm2=cm2[sig+1]-cm2[sig+1-20];v=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD)
   for z in range(Z):
    want=1. if z==0 else (min(CAPS[z],TARGET[z]/v) if bull and v>0 else 0.)
    if z>0 and abs(want-e[z])<.30:want=e[z]
    rr=R[p,t]+want*(M[p,t]-R[p,t])-.00010*abs(want-e[z]);e[z]=want;wealth[z]*=max(1+rr,1e-12);peak[z]=max(peak[z],wealth[z]);ddmin[z]=min(ddmin[z],wealth[z]/peak[z]-1)
   elapsed=t-WARM+1
   if hk<K and elapsed==hd[hk]:
    for z in range(Z):term[p,z,hk]=wealth[z];mdd[p,z,hk]=ddmin[z]
    hk+=1
 return term,mdd
def boot(m,r,dates,b,nlen,rng,sc):
 L=int(sc.get('block',63));N=len(m);starts=np.arange(0,max(1,N-L));stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]);ww=None
 if sc.get('stress_weight',1)>1:ww=np.ones(len(starts));ww[stress[:len(starts)]]=sc['stress_weight'];ww/=ww.sum()
 M=np.empty((b,nlen));R=np.empty((b,nlen))
 for i in range(b):
  pos=0
  while pos<nlen:
   st=int(rng.choice(starts,p=ww)) if ww is not None else int(rng.integers(0,len(starts)));take=min(L,nlen-pos);M[i,pos:pos+take]=m[st:st+take];R[i,pos:pos+take]=r[st:st+take];pos+=take
 M-=sc.get('drift_cut',0.)/TD
 if sc.get('rate_floor',0):R=np.maximum(R,sc['rate_floor']/TD)
 cp=sc.get('crashes_per_year',0.)
 if cp:
  mask=rng.random(M.shape)<cp/TD;M=np.where(mask,M-rng.uniform(.08,.18,M.shape),M)
 return M,R
def main():
 m,r,d=load();total=WARM+40*TD;rng=np.random.default_rng(16260822);rows=[];sim(np.zeros((1,total)),np.zeros((1,total)))
 for sn,sc in SCEN.items():
  TT=[];DD=[];done=0
  while done<PATHS:
   b=min(64,PATHS-done);M,R=boot(m,r,d,b,total,rng,sc);t,x=sim(M,R);TT.append(t);DD.append(x);done+=b
  T=np.concatenate(TT);D=np.concatenate(DD)
  for z,name in enumerate(NAMES):
   for j,h in enumerate(H):
    c=np.maximum(T[:,z,j],1e-300)**(1/int(h))-1;rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_dd50':np.mean(D[:,z,j]<=-.5),'prob_dd70':np.mean(D[:,z,j]<=-.7),'prob_dd90':np.mean(D[:,z,j]<=-.9),'median_dd':np.median(D[:,z,j])})
 out=pd.DataFrame(rows);out.to_csv('results/fortress_cap_sweep.csv',index=False);agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd50=('prob_dd50','max'),max_dd70=('prob_dd70','max'),max_dd90=('prob_dd90','max')).reset_index();agg.to_csv('results/fortress_cap_ranking.csv',index=False);print(agg.to_string(index=False));print('\n20Y');print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False));print('\n40Y');print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__':main()
