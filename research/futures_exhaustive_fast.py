from __future__ import annotations
import numpy as np
from numba import njit
import research.futures_exhaustive as f

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
                term[i,j]=wealth; mdd[i,j]=ddmin; rec[i,j]=mx/f.TD; j+=1
    return term,mdd,rec

# Same exact futures study, only replacing the pure-Python path metric loop.
f.base.pathstats=pathstats_fast

if __name__=='__main__':
    x=np.zeros((2,20)); h=np.array([5,10],dtype=np.int64); pathstats_fast(x,h)
    f.main()
