from __future__ import annotations
import sys
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

base.pathstats=pathstats_fast
base.hysteresis_alloc=hysteresis_fast
base.weekly_alloc=weekly_fast

if __name__=='__main__':
    # Trigger JIT on tiny arrays before the serious run so compile failures fail fast.
    x=np.zeros((2,20)); h=np.array([5,10],dtype=np.int64)
    pathstats_fast(x,h); hysteresis_fast(x,.05); weekly_fast(x,5)
    base.main()
