from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
TD=252; TC=.0003; OUT=Path('results/s9_vehicle_comparison');OUT.mkdir(parents=True,exist_ok=True)

def fetch_ret(t):
    last=None
    for k in range(3):
        try:
            x=yf.download(t,start='2006-01-01',end='2026-08-22',auto_adjust=True,progress=False,threads=False)
            if x.empty: raise RuntimeError(f'empty {t}')
            s=x['Close']
            if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
            s=pd.to_numeric(s,errors='coerce');s.index=pd.to_datetime(s.index).tz_localize(None)
            r=s.pct_change().dropna()
            if len(r)<100: raise RuntimeError(f'too short {t}: {len(r)}')
            return r
        except Exception as e:
            last=e;time.sleep(2*(k+1))
    raise last

def s9(spy,risk,b=.35,d=.12,w=200):
    x=pd.concat([spy.rename('spy'),risk.rename('risk')],axis=1).dropna();p=(1+x.spy).cumprod();ma=p.rolling(w).mean();v=x.risk.rolling(20).std()*math.sqrt(TD);sig=(p>ma).shift(1);vv=v.shift(1);a=pd.Series(np.where(vv.notna()&(vv>0),np.clip(np.where(sig,b,d)/vv,0,1),0),index=x.index);return a*x.risk-TC*a.diff().abs().fillna(a.abs())
def met(r):
    r=pd.Series(r).dropna();e=(1+r).cumprod();y=len(r)/TD;dd=e/e.cummax()-1;return {'start':str(r.index.min().date()),'end':str(r.index.max().date()),'years':y,'cagr':e.iloc[-1]**(1/y)-1,'max_dd':dd.min(),'vol':r.std()*math.sqrt(TD)}
def main():
    tick=['SPY','SSO','UPRO','QLD','TQQQ'];R={}
    for t in tick:
        R[t]=fetch_ret(t); print(t,len(R[t]),flush=True)
    rows=[]
    for t in ['SSO','UPRO','QLD','TQQQ']:
        rows.append({'vehicle':t,'strategy':'buy_hold',**met(R[t])}); rows.append({'vehicle':t,'strategy':'S9_200_35_12',**met(s9(R['SPY'],R[t]))})
    ix=pd.concat([R[t] for t in tick],axis=1).dropna().index
    for t in ['SSO','UPRO','QLD','TQQQ']:
        rows.append({'vehicle':t,'strategy':'S9_common_2010plus',**met(s9(R['SPY'].reindex(ix),R[t].reindex(ix)))})
    pd.DataFrame(rows).to_csv(OUT/'vehicle_results.csv',index=False);(OUT/'manifest.json').write_text(json.dumps({'source':'Yahoo Finance via yfinance, sequential downloads with retries','signal':'SPY 200SMA + 20d realized vehicle vol; 35% bull/12% bear target; prior-day information only','cost':'3bp per 100% allocation turnover'},indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
