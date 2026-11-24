from __future__ import annotations
import math
import numpy as np, pandas as pd
from numba import njit
import research.nasdaq1985_final_tournament as n
TD=252;PATHS=10000;WARM=300;H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=['NDX_1x','S9_base','DIP10_rebound3_b20_h10','DIP5_rebound5_b20_h60','DIP5_cross20_b20_h60']

@njit(cache=True)
def sim(M,R):
    b,N=M.shape;S=5;K=len(H)
    term=np.empty((b,S,K));mdd=np.empty((b,S,K));rec=np.empty((b,S,K));hds=H*TD
    for p in range(b):
        px=np.empty(N);tq=np.empty(N);px[0]=1.
        for t in range(N):
            if t>0:px[t]=px[t-1]*max(1.+M[p,t],1e-12)
            tq[t]=3*M[p,t]-2*R[p,t]-(.0088+2*.0065)/TD
        # prefix arrays
        ppx=np.empty(N+1);pt=np.empty(N+1);pt2=np.empty(N+1)
        ppx[0]=pt[0]=pt2[0]=0.
        for t in range(N):
            ppx[t+1]=ppx[t]+px[t];pt[t+1]=pt[t]+tq[t];pt2[t+1]=pt2[t]+tq[t]*tq[t]
        # rolling max252 / min20 O(N)
        hi=np.empty(N);lo=np.empty(N);qmax=np.empty(N,np.int64);qmin=np.empty(N,np.int64)
        h1=0;t1=0;h2=0;t2=0
        for t in range(N):
            while t1>h1 and px[qmax[t1-1]]<=px[t]:t1-=1
            qmax[t1]=t;t1+=1
            while h1<t1 and qmax[h1]<t-251:h1+=1
            hi[t]=px[qmax[h1]]
            while t2>h2 and px[qmin[t2-1]]>=px[t]:t2-=1
            qmin[t2]=t;t2+=1
            while h2<t2 and qmin[h2]<t-19:h2+=1
            lo[t]=px[qmin[h2]]
        wealth=np.ones(S);peak=np.ones(S);ddmin=np.zeros(S);cur=np.zeros(S,np.int64);mx=np.zeros(S,np.int64);holdw=np.zeros(S)
        armed10=False;armed5r=False;armed5c=False;act10=0;act5r=0;act5c=0;hk=0
        for t in range(WARM,N):
            s=t-1
            ma200=(ppx[s+1]-ppx[s+1-200])/200.
            bull=px[s]>ma200
            sm=pt[s+1]-pt[s+1-20];sm2=pt2[s+1]-pt2[s+1-20];nn=20.
            vt=math.sqrt(max((sm2-sm*sm/nn)/(nn-1.),0.))*math.sqrt(TD)
            base=min(1.,(.35 if bull else .12)/vt) if vt>0 else 0.
            dd=px[s]/hi[s]-1.;reb=px[s]/lo[s]-1.;mom5=px[s]/px[s-5]-1.
            ma20=(ppx[s+1]-ppx[s+1-20])/20.; ma20prev=(ppx[s]-ppx[s-20])/20.
            cross20=px[s]>ma20 and px[s-1]<=ma20prev
            if dd<=-.10:armed10=True
            if dd<=-.05:armed5r=True;armed5c=True
            if act10>0:
                act10-=1
                if dd>-.05 or not bull:act10=0
            if act5r>0:
                act5r-=1
                if dd>-.025 or not bull:act5r=0
            if act5c>0:
                act5c-=1
                if dd>-.025 or not bull:act5c=0
            if armed10 and bull and reb>=.03 and mom5>0:act10=10;armed10=False
            if armed5r and bull and reb>=.05 and mom5>0:act5r=60;armed5r=False
            if armed5c and bull and cross20:act5c=60;armed5c=False
            wants=np.empty(S);wants[0]=1.;wants[1]=base;wants[2]=min(1.,base+(.20 if act10>0 else 0));wants[3]=min(1.,base+(.20 if act5r>0 else 0));wants[4]=min(1.,base+(.20 if act5c>0 else 0))
            for j in range(S):
                band=0. if j==0 else (.05 if j==2 else .10)
                if abs(wants[j]-holdw[j])>=band:holdw[j]=wants[j]
                rr=M[p,t] if j==0 else holdw[j]*tq[t]
                wealth[j]*=max(1.+rr,1e-12)
                if wealth[j]>=peak[j]:peak[j]=wealth[j];cur[j]=0
                else:
                    cur[j]+=1
                    if cur[j]>mx[j]:mx[j]=cur[j]
                d=wealth[j]/peak[j]-1.
                if d<ddmin[j]:ddmin[j]=d
            elapsed=t-WARM+1
            if hk<K and elapsed==hds[hk]:
                for j in range(S):term[p,j,hk]=wealth[j];mdd[p,j,hk]=ddmin[j];rec[p,j,hk]=mx[j]/TD
                hk+=1
    return term,mdd,rec

def main():
    df=n.load_data(.007);ndx=df.ndx.to_numpy();rf=df.rf.to_numpy();dates=df.index
    total=WARM+50*TD;rng=np.random.default_rng(20260920);rows=[]
    sim(np.zeros((1,total)),np.zeros((1,total)))
    for sn,sc in n.SCEN.items():
        Ts=[];Ds=[];Rs=[];done=0
        while done<PATHS:
            b=min(128,PATHS-done);M,R=n.bootstrap(ndx,rf,dates,b,total,rng,sc);t,d,r=sim(M,R);Ts.append(t);Ds.append(d);Rs.append(r);done+=b
            if done%2000==0:print(sn,done,flush=True)
        T=np.concatenate(Ts);D=np.concatenate(Ds);RC=np.concatenate(Rs)
        for s,name in enumerate(NAMES):
            for j,h in enumerate(H):
                c=np.maximum(T[:,s,j],1e-300)**(1/int(h))-1
                rows.append(dict(scenario=sn,strategy=name,horizon=int(h),median_cagr=np.median(c),p10_cagr=np.quantile(c,.1),p01_cagr=np.quantile(c,.01),prob_dd90=np.mean(D[:,s,j]<=-.9),median_dd=np.median(D[:,s,j]),p90_recovery=np.quantile(RC[:,s,j],.9)))
    z=pd.DataFrame(rows);z.to_csv('results/confirmed_dip_hostile_fast.csv',index=False)
    agg=z.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max'),worst_median_dd=('median_dd','min')).reset_index()
    agg.to_csv('results/confirmed_dip_hostile_fast_ranking.csv',index=False)
    print('\nROBUST 20/40/50');print(agg[agg.horizon.isin([20,40,50])].sort_values(['horizon','worst_p10'],ascending=[True,False]).to_string(index=False))
    print('\nBASE 40');print(z[(z.scenario=='baseline')&(z.horizon==40)].sort_values('median_cagr',ascending=False).to_string(index=False))
    print('\nHOSTILE 40');print(z[(z.scenario=='combined_hostile')&(z.horizon==40)].sort_values('p10_cagr',ascending=False).to_string(index=False))
if __name__=='__main__':main()
