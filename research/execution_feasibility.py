from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

TD=252
OUT=Path('results/execution_feasibility')
OUT.mkdir(parents=True,exist_ok=True)

def series(x,f,t):
    s=x[(f,t)] if isinstance(x.columns,pd.MultiIndex) else x[f]
    s=pd.to_numeric(s,errors='coerce');s.index=pd.to_datetime(s.index).tz_localize(None);return s

def alloc_s9(spy,tq,sma=200,bull=.35,bear=.12):
    d=pd.concat([spy.rename('spy'),tq.rename('tq')],axis=1).dropna()
    px=(1+d.spy).cumprod(); ma=px.rolling(sma).mean(); vol=d.tq.rolling(20).std()*math.sqrt(TD)
    sig=(px>ma).shift(1); vv=vol.shift(1)
    a=pd.Series(np.where(vv.notna()&(vv>0),np.clip(np.where(sig,bull,bear)/vv,0,1),0),index=d.index)
    return d,a

def band(a,th=.05):
    z=pd.Series(0.,index=a.index)
    for i in range(1,len(a)):
        z.iloc[i]=a.iloc[i] if abs(a.iloc[i]-z.iloc[i-1])>=th else z.iloc[i-1]
    return z

def weekly(a):
    z=pd.Series(0.,index=a.index)
    for i in range(1,len(a)): z.iloc[i]=a.iloc[i] if i%5==0 else z.iloc[i-1]
    return z

def summarize(name,a):
    ch=a.diff().fillna(a).abs(); active=ch>1e-6
    years=len(a)/TD
    return {'variant':name,'start':str(a.index.min().date()),'end':str(a.index.max().date()),'years':years,
            'allocation_change_days':int(active.sum()),'change_days_per_year':float(active.sum()/years),
            'one_way_turnover_per_year':float(ch.sum()/years),'median_abs_change_when_trading':float(ch[active].median() if active.any() else 0),
            'mean_allocation':float(a.mean()),'median_allocation':float(a.median()),'pct_days_at_100pct':float((a>=.999).mean()),
            'pct_days_below_25pct':float((a<.25).mean())}

def main():
    x=yf.download(['SPY','TQQQ'],start='2010-01-01',end='2026-08-22',auto_adjust=True,progress=False)
    sp=series(x,'Close','SPY').pct_change(); tq=series(x,'Close','TQQQ').pct_change(); d,a=alloc_s9(sp,tq)
    variants={'daily':a,'band_2pct':band(a,.02),'band_5pct':band(a,.05),'band_10pct':band(a,.10),'weekly':weekly(a)}
    pd.DataFrame([summarize(k,v) for k,v in variants.items()]).to_csv(OUT/'turnover.csv',index=False)
    pd.DataFrame(variants).to_csv(OUT/'allocations.csv')
    print(pd.DataFrame([summarize(k,v) for k,v in variants.items()]).to_string(index=False))
if __name__=='__main__': main()
