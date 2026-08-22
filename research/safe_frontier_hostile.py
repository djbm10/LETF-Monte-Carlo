from __future__ import annotations
import math
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
import research.nasdaq1985_final_tournament as n

TD=252; PATHS=20_000; WARM=300
H=np.array([5,10,20,30,40,50],dtype=np.int64)
NAMES=('NDX_1X','VOL20','VOL25','TV15_0','TV20_0','TV25_0','TV30_0','TV35_0')
SCOUNT=len(NAMES); HCOUNT=len(H)
SCEN=n.SCEN
OUT=Path('results/safe_frontier_hostile'); OUT.mkdir(parents=True,exist_ok=True)

@njit(cache=True)
def sim_batch(M,R):
    b,N=M.shape; term=np.empty((b,SCOUNT,HCOUNT)); mdd=np.empty((b,SCOUNT,HCOUNT)); rec=np.empty((b,SCOUNT,HCOUNT)); hds=H*TD
    for p in range(b):
        px=np.empty(N); px[0]=1.
        for t in range(1,N): px[t]=px[t-1]*max(1.+M[p,t],1e-12)
        cpx=np.empty(N+1); cm=np.empty(N+1); cm2=np.empty(N+1); cpx[0]=cm[0]=cm2[0]=0.
        for t in range(N):
            cpx[t+1]=cpx[t]+px[t]; cm[t+1]=cm[t]+M[p,t]; cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
        wealth=np.ones(SCOUNT); peak=np.ones(SCOUNT); ddmin=np.zeros(SCOUNT); cur=np.zeros(SCOUNT,np.int64); mx=np.zeros(SCOUNT,np.int64)
        last=np.zeros(SCOUNT); holds=np.zeros(5); hk=0
        for t in range(WARM,N):
            sig=t-1; sma=(cpx[sig+1]-cpx[sig+1-200])/200.
            sm=cm[sig+1]-cm[sig+1-20]; sm2=cm2[sig+1]-cm2[sig+1-20]; vm=math.sqrt(max((sm2-sm*sm/20.)/19.,0.))*math.sqrt(TD)
            E=np.zeros(SCOUNT); E[0]=1.
            if vm>0:
                E[1]=min(.20/vm,3.); E[2]=min(.25/vm,3.)
                bull=px[sig]>sma
                targets=(.15,.20,.25,.30,.35)
                for k in range(5):
                    want=min((targets[k] if bull else 0.)/vm,3.)
                    if abs(want-holds[k])>=.30: holds[k]=want
                    E[3+k]=holds[k]
            for s in range(SCOUNT):
                rr=R[p,t]+E[s]*(M[p,t]-R[p,t])-0.00010*abs(E[s]-last[s]); last[s]=E[s]
                wealth[s]*=max(1.+rr,1e-12)
                if wealth[s]>=peak[s]: peak[s]=wealth[s]; cur[s]=0
                else:
                    cur[s]+=1
                    if cur[s]>mx[s]: mx[s]=cur[s]
                dd=wealth[s]/peak[s]-1.
                if dd<ddmin[s]: ddmin[s]=dd
            elapsed=t-WARM+1
            if hk<HCOUNT and elapsed==hds[hk]:
                for s in range(SCOUNT): term[p,s,hk]=wealth[s]; mdd[p,s,hk]=ddmin[s]; rec[p,s,hk]=mx[s]/TD
                hk+=1
    return term,mdd,rec

def bootstrap(m,r,dates,b,nlen,rng,sc):
    L=int(sc.get('block',63)); N=len(m); starts=np.arange(0,max(1,N-L)); stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]); w=None
    if sc.get('stress_weight',1)>1:
        w=np.ones(len(starts)); w[stress[:len(starts)]]=sc['stress_weight']; w/=w.sum()
    M=np.empty((b,nlen)); R=np.empty((b,nlen))
    for i in range(b):
        pos=0
        while pos<nlen:
            st=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts))); take=min(L,nlen-pos)
            M[i,pos:pos+take]=m[st:st+take]; R[i,pos:pos+take]=r[st:st+take]; pos+=take
    M-=sc.get('drift_cut',0.)/TD
    floor=sc.get('rate_floor',0.)
    if floor: R=np.maximum(R,floor/TD)
    cp=sc.get('crashes_per_year',0.)
    if cp:
        mask=rng.random(M.shape)<cp/TD; M=np.where(mask,M-rng.uniform(.08,.18,M.shape),M)
    return M,R

def main():
    df=n.load_data(.007); m=df.ndx.to_numpy(); rf=df.rf.to_numpy(); dates=df.index; total=WARM+50*TD; rng=np.random.default_rng(12260822); rows=[]
    sim_batch(np.zeros((1,total)),np.zeros((1,total)))
    for sn,sc in SCEN.items():
        Ts=[];Ds=[];Rs=[];done=0
        while done<PATHS:
            b=min(64,PATHS-done);M,R=bootstrap(m,rf,dates,b,total,rng,sc);t,d,r=sim_batch(M,R);Ts.append(t);Ds.append(d);Rs.append(r);done+=b
            if done%4000==0: print(sn,done,flush=True)
        T=np.concatenate(Ts);D=np.concatenate(Ds);RC=np.concatenate(Rs);B=T[:,0,:]
        for s,name in enumerate(NAMES):
            for j,h in enumerate(H):
                c=np.maximum(T[:,s,j],1e-300)**(1/int(h))-1
                rows.append({'scenario':sn,'strategy':name,'horizon':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),'prob_beat_1x':np.nan if s==0 else np.mean(T[:,s,j]>B[:,j]),'prob_dd90':np.mean(D[:,s,j]<=-.9),'median_dd':np.median(D[:,s,j]),'p90_recovery':np.quantile(RC[:,s,j],.9)})
    out=pd.DataFrame(rows); out.to_csv(OUT/'scenario_results.csv',index=False)
    agg=out.groupby(['strategy','horizon']).agg(worst_median=('median_cagr','min'),worst_p10=('p10_cagr','min'),worst_p01=('p01_cagr','min'),max_dd90=('prob_dd90','max'),worst_median_dd=('median_dd','min'),max_p90_recovery=('p90_recovery','max')).reset_index()
    agg.to_csv(OUT/'robust_ranking.csv',index=False)
    print(agg.sort_values(['horizon','max_dd90','worst_p10'],ascending=[True,True,False]).to_string(index=False))
    print('\n20Y BY SCENARIO'); print(out[out.horizon==20].sort_values(['scenario','strategy']).to_string(index=False))
    print('\n40Y BY SCENARIO'); print(out[out.horizon==40].sort_values(['scenario','strategy']).to_string(index=False))
if __name__=='__main__': main()
