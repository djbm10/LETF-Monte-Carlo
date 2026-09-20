from __future__ import annotations
import math, json
import numpy as np, pandas as pd
from numba import njit
import research.nasdaq1985_final_tournament as n

TD=252; PATHS=10000; WARM=300; H=(5,10,20,30,40,50)
NAMES=['NDX_1x','S9_base','DIP10_rebound3_b20_h10','DIP5_rebound5_b20_h60','DIP5_cross20_b20_h60']

@njit(cache=True)
def rolling_mean(px,t,w):
    if t+1<w:return np.nan
    s=0.
    for j in range(t-w+1,t+1):s+=px[j]
    return s/w
@njit(cache=True)
def rolling_vol(r,t,w):
    if t+1<w:return np.nan
    s=0.;ss=0.
    for j in range(t-w+1,t+1):s+=r[j];ss+=r[j]*r[j]
    m=s/w;v=(ss-w*m*m)/(w-1)
    return math.sqrt(max(v,0.))*math.sqrt(TD)
@njit(cache=True)
def rolling_max(px,t,w):
    a=max(0,t-w+1);m=px[a]
    for j in range(a+1,t+1):
        if px[j]>m:m=px[j]
    return m
@njit(cache=True)
def rolling_min(px,t,w):
    a=max(0,t-w+1);m=px[a]
    for j in range(a+1,t+1):
        if px[j]<m:m=px[j]
    return m

@njit(cache=True)
def simulate(M,R):
    b,N=M.shape; S=5; K=6
    term=np.empty((b,S,K));mdd=np.empty((b,S,K));rec=np.empty((b,S,K))
    hds=np.array([5,10,20,30,40,50])*TD
    for p in range(b):
        px=np.empty(N); tq=np.empty(N);px[0]=1.
        for t in range(N):
            if t>0:px[t]=px[t-1]*max(1+M[p,t],1e-12)
            tq[t]=3*M[p,t]-2*R[p,t]-(.0088+2*.0065)/TD
        wealth=np.ones(S);peak=np.ones(S);ddmin=np.zeros(S);cur=np.zeros(S,np.int64);mx=np.zeros(S,np.int64)
        hold=np.zeros(S)
        armed10=False;armed5a=False;armed5b=False
        active10=0;active5a=0;active5b=0
        hk=0
        for t in range(WARM,N):
            sig=t-1
            ma=rolling_mean(px,sig,200); vt=rolling_vol(tq,sig,20)
            bull=(not math.isnan(ma)) and px[sig]>ma
            base=0.
            if not math.isnan(vt) and vt>0:
                base=min(1.,(.35 if bull else .12)/vt)
            # drawdown and rebound diagnostics on underlying
            hi=rolling_max(px,sig,252); dd=px[sig]/hi-1 if hi>0 else 0.
            lo=rolling_min(px,sig,20); reb=px[sig]/lo-1 if lo>0 else 0.
            mom5=px[sig]/px[sig-5]-1 if sig>=5 else 0.
            ma20=rolling_mean(px,sig,20)
            ma20prev=rolling_mean(px,sig-1,20) if sig>=1 else np.nan
            cross20=(not math.isnan(ma20)) and (not math.isnan(ma20prev)) and px[sig]>ma20 and px[sig-1]<=ma20prev
            if dd<=-.10:armed10=True
            if dd<=-.05:armed5a=True;armed5b=True
            if active10>0:
                active10-=1
                if dd>-.05 or not bull:active10=0
            if active5a>0:
                active5a-=1
                if dd>-.025 or not bull:active5a=0
            if active5b>0:
                active5b-=1
                if dd>-.025 or not bull:active5b=0
            if armed10 and bull and reb>=.03 and mom5>0:active10=10;armed10=False
            if armed5a and bull and reb>=.05 and mom5>0:active5a=60;armed5a=False
            if armed5b and bull and cross20:active5b=60;armed5b=False
            wants=np.empty(S);wants[0]=1.;wants[1]=base;wants[2]=min(1.,base+(.20 if active10>0 else 0));wants[3]=min(1.,base+(.20 if active5a>0 else 0));wants[4]=min(1.,base+(.20 if active5b>0 else 0))
            bands=np.array([0.,.10,.05,.10,.10])
            for s in range(S):
                if abs(wants[s]-hold[s])>=bands[s]:hold[s]=wants[s]
                rr=M[p,t] if s==0 else hold[s]*tq[t]
                wealth[s]*=max(1+rr,1e-12)
                if wealth[s]>=peak[s]:peak[s]=wealth[s];cur[s]=0
                else:
                    cur[s]+=1
                    if cur[s]>mx[s]:mx[s]=cur[s]
                d=wealth[s]/peak[s]-1
                if d<ddmin[s]:ddmin[s]=d
            elapsed=t-WARM+1
            if hk<K and elapsed==hds[hk]:
                for s in range(S):term[p,s,hk]=wealth[s];mdd[p,s,hk]=ddmin[s];rec[p,s,hk]=mx[s]/TD
                hk+=1
    return term,mdd,rec

def main():
    df=n.load_data(.007);ndx=df.ndx.to_numpy();rf=df.rf.to_numpy();dates=df.index
    total=WARM+50*TD;rng=np.random.default_rng(20260920);rows=[]
    simulate(np.zeros((1,total)),np.zeros((1,total)))
    for sn,sc in n.SCEN.items():
        Ts=[];Ds=[];Rs=[];done=0
        while done<PATHS:
            b=min(64,PATHS-done);M,R=n.bootstrap(ndx,rf,dates,b,total,rng,sc);t,d,r=simulate(M,R);Ts.append(t);Ds.append(d);Rs.append(r);done+=b
        T=np.concatenate(Ts);D=np.concatenate(Ds);RC=np.concatenate(Rs)
        for s,name in enumerate(NAMES):
            for j,h in enumerate(H):
                c=np.maximum(T[:,s,j],1e-300)**(1/h)-1
                rows.append(dict(scenario=sn,strategy=name,horizon=h,median_cagr=np.median(c),p10_cagr=np.quantile(c,.1),p01_cagr=np.quantile(c,.01),prob_dd90=np.mean(D[:,s,j]<=-.9),median_dd=np.median(D[:,s,j]),p90_recovery=np.quantile(RC[:,s,j],.9)))
    z=pd.DataFrame(rows);z.to_csv('results/confirmed_dip_hostile.csv',index=False)
    agg=z.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max'),worst_median_dd=('median_dd','min')).reset_index()
    agg.to_csv('results/confirmed_dip_hostile_ranking.csv',index=False)
    print(agg[agg.horizon.isin([20,40,50])].sort_values(['horizon','worst_p10'],ascending=[True,False]).to_string(index=False))
    print('\nBASELINE SCENARIO 40Y')
    print(z[(z.scenario=='baseline')&(z.horizon==40)].sort_values('median_cagr',ascending=False).to_string(index=False))
    print('\nCOMBINED HOSTILE 40Y')
    print(z[(z.scenario=='combined_hostile')&(z.horizon==40)].sort_values('p10_cagr',ascending=False).to_string(index=False))
if __name__=='__main__':main()
