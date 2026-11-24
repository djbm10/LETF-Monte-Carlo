from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
import research.nasdaq1985_final_tournament as n

TD=252; OUT=Path('results/nasdaq1985_phase2'); OUT.mkdir(parents=True,exist_ok=True)
PATHS=20_000; H=np.array([5,10,20,30,40,50],dtype=np.int64); WARM=300; SCOUNT=13; HCOUNT=6
# Frozen BEFORE hostile confirmation. First five are strictly pre-2010-selected representatives.
NAMES=(
 'ndx_1x',
 'fut_sma100_2x_b0_buf1',
 'fut_sma100_2.5x_b0_buf1',
 'fut_mom189_2x',
 'fut_mom189_2.5x',
 'fut_sma200_2x_b0_buf0',
 'fut_sma200_3x_b0_buf0',
 'fut_tv_w200_u35_d12_b30',
 'fut_tv_w200_u35_d0_b30',
 'fut_vol25',
 'fut_mom252_2x',
 'qld_s9_w200_u35_d12_b10',
 'tqqq_s9_w200_u35_d12_b10',
)
SCEN=n.SCEN

@njit(cache=True)
def sim_batch(M,R):
    b,N=M.shape; S=SCOUNT; K=HCOUNT; term=np.empty((b,S,K)); mdd=np.empty((b,S,K)); rec=np.empty((b,S,K))
    horizon_days=H*TD
    for p in range(b):
        px=np.empty(N); qld=np.empty(N); tq=np.empty(N)
        px[0]=1.; qld[0]=0.; tq[0]=0.
        for t in range(N):
            if t>0: px[t]=px[t-1]*max(1.+M[p,t],1e-12)
            qld[t]=2*M[p,t]-R[p,t]-(0.0095+0.0050)/TD
            tq[t]=3*M[p,t]-2*R[p,t]-(0.0088+2*0.0065)/TD
        cpx=np.empty(N+1); cpx[0]=0.
        cm=np.empty(N+1); cm2=np.empty(N+1); cq=np.empty(N+1); cq2=np.empty(N+1); ct=np.empty(N+1); ct2=np.empty(N+1)
        cm[0]=cm2[0]=cq[0]=cq2[0]=ct[0]=ct2[0]=0.
        for t in range(N):
            cpx[t+1]=cpx[t]+px[t]
            cm[t+1]=cm[t]+M[p,t]; cm2[t+1]=cm2[t]+M[p,t]*M[p,t]
            cq[t+1]=cq[t]+qld[t]; cq2[t+1]=cq2[t]+qld[t]*qld[t]
            ct[t+1]=ct[t]+tq[t]; ct2[t+1]=ct2[t]+tq[t]*tq[t]
        wealth=np.ones(S); peak=np.ones(S); ddmin=np.zeros(S); cur=np.zeros(S,dtype=np.int64); mx=np.zeros(S,dtype=np.int64)
        sma100_2=0.; sma100_25=0.; tv12=0.; tv0=0.; qhold=0.; thold=0.; hk=0
        lastE=np.zeros(S)
        for t in range(WARM,N):
            sig=t-1; sma100=np.nan; sma200=np.nan
            if sig>=99: sma100=(cpx[sig+1]-cpx[sig+1-100])/100.
            if sig>=199: sma200=(cpx[sig+1]-cpx[sig+1-200])/200.
            vm=np.nan; vq=np.nan; vt=np.nan
            if sig>=19:
                nn=20.; sm=cm[sig+1]-cm[sig+1-20]; sm2=cm2[sig+1]-cm2[sig+1-20]
                sq=cq[sig+1]-cq[sig+1-20]; sq2=cq2[sig+1]-cq2[sig+1-20]
                st=ct[sig+1]-ct[sig+1-20]; st2=ct2[sig+1]-ct2[sig+1-20]
                vm=math.sqrt(max((sm2-sm*sm/nn)/(nn-1.),0.))*math.sqrt(TD)
                vq=math.sqrt(max((sq2-sq*sq/nn)/(nn-1.),0.))*math.sqrt(TD)
                vt=math.sqrt(max((st2-st*st/nn)/(nn-1.),0.))*math.sqrt(TD)
            E=np.zeros(S); E[0]=1.
            if not math.isnan(sma100):
                ratio=px[sig]/sma100-1.
                if sma100_2<=1e-12:
                    if ratio>.01:sma100_2=2.
                elif ratio<-.01:sma100_2=0.
                if sma100_25<=1e-12:
                    if ratio>.01:sma100_25=2.5
                elif ratio<-.01:sma100_25=0.
            E[1]=sma100_2;E[2]=sma100_25
            if sig>=189:
                E[3]=2. if px[sig]>px[sig-189] else 0.
                E[4]=2.5 if px[sig]>px[sig-189] else 0.
            if not math.isnan(sma200):
                E[5]=2. if px[sig]>sma200 else 0.
                E[6]=3. if px[sig]>sma200 else 0.
            if not math.isnan(sma200) and not math.isnan(vm) and vm>0:
                want=(.35 if px[sig]>sma200 else .12)/vm; want=min(max(want,0.),3.)
                if abs(want-tv12)>=.30:tv12=want
                want0=(.35 if px[sig]>sma200 else 0.)/vm; want0=min(max(want0,0.),3.)
                if abs(want0-tv0)>=.30:tv0=want0
            E[7]=tv12;E[8]=tv0
            if not math.isnan(vm) and vm>0:E[9]=min(.25/vm,3.)
            if sig>=252:E[10]=2. if px[sig]>px[sig-252] else 0.
            if not math.isnan(sma200) and not math.isnan(vq) and vq>0:
                wantq=min((.35 if px[sig]>sma200 else .12)/vq,1.)
                if abs(wantq-qhold)>=.10:qhold=wantq
            if not math.isnan(sma200) and not math.isnan(vt) and vt>0:
                wantt=min((.35 if px[sig]>sma200 else .12)/vt,1.)
                if abs(wantt-thold)>=.10:thold=wantt
            rr=np.empty(S)
            for s in range(11): rr[s]=R[p,t]+E[s]*(M[p,t]-R[p,t])-0.00010*abs(E[s]-lastE[s])
            rr[11]=qhold*qld[t]-0.0003*abs(qhold-lastE[11]); rr[12]=thold*tq[t]-0.0003*abs(thold-lastE[12])
            E[11]=qhold;E[12]=thold
            for s in range(S):
                lastE[s]=E[s]; wealth[s]*=max(1.+rr[s],1e-12)
                if wealth[s]>=peak[s]:peak[s]=wealth[s];cur[s]=0
                else:
                    cur[s]+=1
                    if cur[s]>mx[s]:mx[s]=cur[s]
                dd=wealth[s]/peak[s]-1.
                if dd<ddmin[s]:ddmin[s]=dd
            elapsed=t-WARM+1
            if hk<K and elapsed==horizon_days[hk]:
                for s in range(S):term[p,s,hk]=wealth[s];mdd[p,s,hk]=ddmin[s];rec[p,s,hk]=mx[s]/TD
                hk+=1
    return term,mdd,rec

def bootstrap(arr_m,arr_r,dates,b,nlen,rng,sc):
    L=int(sc.get('block',63)); N=len(arr_m); starts=np.arange(0,max(1,N-L))
    stress=np.isin(pd.DatetimeIndex(dates).year,[1987,2000,2001,2002,2008,2009,2020,2022]); w=None
    if sc.get('stress_weight',1)>1:
        w=np.ones(len(starts)); w[stress[:len(starts)]]=sc['stress_weight'];w/=w.sum()
    M=np.empty((b,nlen));R=np.empty((b,nlen))
    for i in range(b):
        pos=0
        while pos<nlen:
            s=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts)))
            take=min(L,nlen-pos);M[i,pos:pos+take]=arr_m[s:s+take];R[i,pos:pos+take]=arr_r[s:s+take];pos+=take
    M-=sc.get('drift_cut',0.)/TD
    floor=sc.get('rate_floor',0.)
    if floor:R=np.maximum(R,floor/TD)
    cp=sc.get('crashes_per_year',0.)
    if cp:
        mask=rng.random(M.shape)<cp/TD;M=np.where(mask,M-rng.uniform(.08,.18,M.shape),M)
    return M,R

def main():
    df=n.load_data(.007); m=df.ndx.to_numpy();rf=df.rf.to_numpy(); dates=df.index
    total=WARM+50*TD; rows=[]; rng=np.random.default_rng(9260822)
    sim_batch(np.zeros((1,total)),np.zeros((1,total)))
    for sn,sc in SCEN.items():
        Ts=[];Ds=[];Rs=[];done=0
        while done<PATHS:
            b=min(64,PATHS-done);M,R=bootstrap(m,rf,dates,b,total,rng,sc);t,d,r=sim_batch(M,R);Ts.append(t);Ds.append(d);Rs.append(r);done+=b
            if done%2000==0:print(sn,done,flush=True)
        T=np.concatenate(Ts);D=np.concatenate(Ds);RC=np.concatenate(Rs);B=T[:,0,:]
        for s,name in enumerate(NAMES):
            for j,h in enumerate(H):
                c=np.maximum(T[:,s,j],1e-300)**(1/int(h))-1
                rows.append({'scenario':sn,'strategy':name,'horizon_years':int(h),'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'p01_cagr':np.quantile(c,.01),
                             'prob_beat_1x':np.nan if s==0 else np.mean(T[:,s,j]>B[:,j]),'prob_dd_gt_90':np.mean(D[:,s,j]<=-.9),'median_drawdown':np.median(D[:,s,j]),'p90_recovery_years':np.quantile(RC[:,s,j],.9)})
    out=pd.DataFrame(rows);agg=out.groupby(['strategy','horizon_years']).agg(worst_median_cagr=('median_cagr','min'),worst_p10_cagr=('p10_cagr','min'),worst_p01_cagr=('p01_cagr','min'),min_prob_beat_1x=('prob_beat_1x','min'),max_prob_dd_gt_90=('prob_dd_gt_90','max'),worst_median_drawdown=('median_drawdown','min'),worst_p90_recovery=('p90_recovery_years','max')).reset_index()
    out.to_csv(OUT/'scenario_results.csv',index=False);agg.to_csv(OUT/'robust_ranking.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'paths_per_scenario':PATHS,'scenarios':SCEN,'horizons':H.tolist(),'strategies':NAMES,'selection':'Frozen from strict pre-2010 phase1 representatives + predeclared benchmarks; 2010+ holdout not used to choose parameters','seed':9260822},indent=2))
    print(agg.sort_values(['horizon_years','worst_p10_cagr'],ascending=[True,False]).groupby('horizon_years').head(13).to_string(index=False))

if __name__=='__main__':main()
