from __future__ import annotations
import sys, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from numba import njit
import research.hostile_strategy_tournament as base

@njit(cache=True)
def pathstats_fast(r,hds):
    b,n=r.shape; k=len(hds)
    term=np.empty((b,k)); mdd=np.empty((b,k)); rec=np.empty((b,k))
    for i in range(b):
        wealth=1.0; peak=1.0; ddmin=0.0; cur=0; mx=0; j=0
        for t in range(n):
            g=1.0+r[i,t]
            if g<0.0: g=0.0
            wealth*=g
            if wealth>=peak:
                peak=wealth; cur=0
            else:
                cur+=1
                if cur>mx: mx=cur
            if peak>0:
                dd=wealth/peak-1.0
                if dd<ddmin: ddmin=dd
            if j<k and t+1==hds[j]:
                term[i,j]=wealth; mdd[i,j]=ddmin; rec[i,j]=mx/base.TD; j+=1
    return term,mdd,rec

@njit(cache=True)
def hysteresis_fast(desired,threshold=.05):
    b,n=desired.shape; a=np.zeros_like(desired)
    for t in range(1,n):
        for i in range(b):
            prev=a[i,t-1]; want=desired[i,t]
            if abs(want-prev)>=threshold: a[i,t]=want
            else: a[i,t]=prev
    return a

@njit(cache=True)
def weekly_fast(desired,step=5):
    b,n=desired.shape; a=np.zeros_like(desired)
    for t in range(1,n):
        for i in range(b):
            if t%step==0: a[i,t]=desired[i,t]
            else: a[i,t]=a[i,t-1]
    return a

@njit(cache=True)
def true_band_state(px,ma,upper=.02,lower=.02):
    b,n=px.shape; state=np.zeros((b,n),np.int8)
    for t in range(1,n):
        for i in range(b):
            prev=state[i,t-1]; p=px[i,t-1]; m=ma[i,t-1]
            if not np.isfinite(m): state[i,t]=0
            elif prev==0 and p>=m*(1+upper): state[i,t]=1
            elif prev==1 and p<m*(1-lower): state[i,t]=0
            else: state[i,t]=prev
    return state

def state_returns(state,risk,defensive):
    sw=np.abs(np.diff(state.astype(float),axis=1,prepend=np.zeros((state.shape[0],1))))
    return np.where(state>0,risk,defensive)-base.TC_SWITCH*sw

def s8_composite(px,r1,r3,rf):
    ma=base.rm2(px,200); delta=np.diff(px,axis=1,prepend=px[:,:1]); gain=np.maximum(delta,0); loss=np.maximum(-delta,0)
    ag=base.rm2(gain,14); al=base.rm2(loss,14); rs=ag/np.maximum(al,1e-12); rsi=100-100/(1+rs)
    vix=base.rs2(r1,20)*math.sqrt(base.TD)*100; score=np.zeros_like(px,dtype=np.int8)
    good=np.isfinite(ma[:,:-1])&np.isfinite(rsi[:,:-1])&np.isfinite(vix[:,:-1])
    score[:,1:]=good*((px[:,:-1]>ma[:,:-1]).astype(np.int8)+((rsi[:,:-1]>40)&(rsi[:,:-1]<80)).astype(np.int8)+(vix[:,:-1]<25).astype(np.int8))
    state=np.where(score==3,2,np.where(score==2,1,0)).astype(np.int8); ret=np.where(state==2,r3,np.where(state==1,r1,rf)); sw=(np.diff(state,axis=1,prepend=np.zeros((state.shape[0],1)))!=0)
    return ret-base.TC_SWITCH*sw

def s19_roth(px,r1,r3,rf):
    mom=base.rm2(r1,126)*126; neg=np.where(r3<0,r3,0); dvol=base.rs2(neg,20)*math.sqrt(base.TD)
    vf=base.rs2(r3,5)*math.sqrt(base.TD); vs=base.rs2(r3,60)*math.sqrt(base.TD); sma=base.rm2(px,100); want=np.zeros_like(px)
    mm=mom[:,:-1]; dv=dvol[:,:-1]; vfast=vf[:,:-1]; vslow=vs[:,:-1]; p=px[:,:-1]; sm=sma[:,:-1]
    valid=np.isfinite(mm)&np.isfinite(dv)&np.isfinite(vfast)&np.isfinite(vslow)&np.isfinite(sm)&(dv>.001)&(vslow>.001)
    momscore=np.where(mm>.15,1.0,np.where(mm>.05,.7,np.where(mm>0,.4,0))); trend=np.where(p>sm,.5,0); vr=vfast/np.maximum(vslow,1e-12); volscore=np.where(vr<.9,.5,np.where(vr<1.2,.3,0))
    conviction=momscore+trend+volscore; mult=.3+conviction*.55; alloc=np.clip((.32/np.maximum(dv,1e-12))*mult,0,1.0); want[:,1:]=np.where(valid,alloc,0)
    a=hysteresis_fast(want,.05); return base.alloc_returns(a,r3,rf)

orig_build=base.build_strategies

def build_extended(bm,br):
    out=orig_build(bm,br); r1=base.levret(bm,br,1); r3=base.levret(bm,br,3); px=np.cumprod(np.maximum(1+r1,1e-12),axis=1); ma=base.rm2(px,200)
    sig=np.zeros_like(px,dtype=np.int8); sig[:,1:]=(np.isfinite(ma[:,:-1])&(px[:,:-1]>=ma[:,:-1]*.98)).astype(np.int8)
    out['repo_s5_minus2_threshold']=state_returns(sig,r3,br)
    out['true_sma_plus2_minus2_band']=state_returns(true_band_state(px,ma,.02,.02),r3,br)
    out['repo_s8_composite']=s8_composite(px,r1,r3,br)
    out['s19_roth_cap100']=s19_roth(px,r1,r3,br)
    return out

base.pathstats=pathstats_fast; base.hysteresis_alloc=hysteresis_fast; base.weekly_alloc=weekly_fast; base.build_strategies=build_extended
# Every repo strategy was already scored in the 1926 historical screen. Spend the 10k-path hostile budget only on non-dominated, Roth-feasible finalists and strong external challengers.
base.STRATEGIES=(
    '1x_buy_hold','3x_sma200_cash','repo_s5_minus2_threshold','true_sma_plus2_minus2_band','repo_s8_composite',
    'tsmom_12m_3x_cash','vol30_tqqq_cash','s9_200_35_12','s9_200_35_12_lag1','s9_200_35_12_band5','s19_roth_cap100'
)

if __name__=='__main__':
    x=np.zeros((2,20)); h=np.array([5,10],dtype=np.int64); pathstats_fast(x,h); hysteresis_fast(x,.05); weekly_fast(x,5); true_band_state(x,x+1,.02,.02); base.main()
