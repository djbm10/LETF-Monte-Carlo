from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

TD=252
TC=.0003

def dl(tickers):
    x=yf.download(tickers,start='1993-01-01',end='2026-08-22',auto_adjust=False,actions=True,progress=False,group_by='column')
    if x.empty: raise RuntimeError('yfinance returned no data')
    return x

def field(x,f,t):
    if isinstance(x.columns,pd.MultiIndex):
        # yfinance normally (Price,Ticker)
        if (f,t) in x.columns: s=x[(f,t)]
        elif (t,f) in x.columns: s=x[(t,f)]
        else: raise KeyError((f,t))
    else: s=x[f]
    s=pd.to_numeric(s,errors='coerce'); s.index=pd.to_datetime(s.index).tz_localize(None); return s

def adj_series(x,t):
    c=field(x,'Close',t); a=field(x,'Adj Close',t) if ('Adj Close',t) in x.columns else c
    o=field(x,'Open',t); fac=(a/c).replace([np.inf,-np.inf],np.nan)
    return pd.DataFrame({'open':o*fac,'close':c*fac,'adj':a}).dropna()

def ret(x,t): return adj_series(x,t).adj.pct_change().dropna()
def equity(r): return (1+pd.Series(r).fillna(0)).cumprod()
def met(r):
    r=pd.Series(r).dropna(); e=equity(r); y=len(r)/TD; p=e.cummax(); dd=e/p-1
    cur=mx=0
    for is_under in (e<p).to_numpy():
        cur=cur+1 if is_under else 0; mx=max(mx,cur)
    return {'start':str(r.index.min().date()),'end':str(r.index.max().date()),'years':y,'cagr':e.iloc[-1]**(1/y)-1,'max_drawdown':dd.min(),'vol':r.std()*math.sqrt(TD),'max_underwater_years':mx/TD}

def port_rebal(returns,weights,every):
    d=pd.concat(returns,axis=1).dropna(); names=list(d.columns); w=np.asarray(weights,float); pos=w.copy(); out=[]
    for i,(_,row) in enumerate(d.iterrows()):
        rr=row.to_numpy(); pr=float(np.dot(pos,rr)); out.append(pr)
        pos=pos*(1+rr); pos=pos/pos.sum()
        if (i+1)%every==0: pos=w.copy()
    return pd.Series(out,index=d.index)

def s9(spy,tqqq,bull=.35,bear=.12,sma=200,lag=0,band=None,weekly=False):
    d=pd.concat([spy.rename('spy'),tqqq.rename('tq')],axis=1).dropna(); px=(1+d.spy).cumprod(); ma=px.rolling(sma).mean(); vol=d.tq.rolling(20).std()*math.sqrt(TD)
    want=pd.Series(0.0,index=d.index); shift=1+lag
    bullsig=(px>ma).shift(shift); vv=vol.shift(shift); target=np.where(bullsig,bull,bear); want[:]=np.where(vv.notna()&(vv>0),np.clip(target/vv,0,1),0)
    a=pd.Series(0.0,index=d.index)
    if band is not None:
        for i in range(1,len(d)):
            a.iloc[i]=want.iloc[i] if abs(want.iloc[i]-a.iloc[i-1])>=band else a.iloc[i-1]
    elif weekly:
        for i in range(1,len(d)): a.iloc[i]=want.iloc[i] if i%5==0 else a.iloc[i-1]
    else: a=want
    turn=a.diff().abs().fillna(a.abs()); return a*d.tq+(1-a)*0-TC*turn

def trend(spy,risk,window=200):
    d=pd.concat([spy.rename('spy'),risk.rename('risk')],axis=1).dropna(); px=(1+d.spy).cumprod(); sig=(px>px.rolling(window).mean()).shift(1).fillna(False); turn=sig.astype(float).diff().abs().fillna(0); return sig*d.risk-TC*turn

def overnight(df,cost_side=.0001):
    close=df.close; next_open=df.open.shift(-1); r=next_open/close-1; return ((1+r)*(1-cost_side)**2-1).dropna()
def intraday(df,cost_side=.0001): return ((df.close/df.open)*(1-cost_side)**2-1).dropna()

def hybrid_overnight_day(lev,base,cost_side=.0001):
    ov=(lev.open.shift(-1)/lev.close-1).rename('ov'); day=(base.close/base.open-1).shift(-1).rename('day')
    d=pd.concat([ov,day],axis=1).dropna(); return ((1+d.ov)*(1+d.day)*(1-cost_side)**4-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('results/modern_real_etf_validation'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tick=['SPY','QQQ','SSO','UPRO','TQQQ','TMF'];x=dl(tick); R={t:ret(x,t) for t in tick}; A={t:adj_series(x,t) for t in tick}; rows=[]
    def add(name,r,kind):
        try: rows.append({'strategy':name,'kind':kind,**met(r)})
        except Exception as e: rows.append({'strategy':name,'kind':kind,'error':repr(e)})
    common=pd.concat([R['SPY'],R['TQQQ']],axis=1).dropna().index
    add('SPY_BH_TQQQ_period',R['SPY'].reindex(common).dropna(),'real ETF')
    add('TQQQ_BH',R['TQQQ'],'real ETF')
    add('UPRO_BH',R['UPRO'],'real ETF')
    add('S9_200_35_12_real_TQQQ',s9(R['SPY'],R['TQQQ']),'real ETF')
    add('S9_200_35_12_lag1_real',s9(R['SPY'],R['TQQQ'],lag=1),'execution stress')
    add('S9_200_35_12_band5_real',s9(R['SPY'],R['TQQQ'],band=.05),'execution stress')
    add('S9_200_35_12_weekly_real',s9(R['SPY'],R['TQQQ'],weekly=True),'execution stress')
    add('TQQQ_SPY_SMA200_cash',trend(R['SPY'],R['TQQQ']),'real ETF')
    add('UPRO_SPY_SMA200_cash',trend(R['SPY'],R['UPRO']),'real ETF')
    add('repo_S6_60TQQQ_40TMF_monthly',port_rebal([R['TQQQ'].rename('TQQQ'),R['TMF'].rename('TMF')],[.6,.4],21),'real ETF portfolio')
    add('HFEA_55UPRO_45TMF_quarterly',port_rebal([R['UPRO'].rename('UPRO'),R['TMF'].rename('TMF')],[.55,.45],63),'web candidate')
    add('HFEA_55UPRO_45TMF_monthly',port_rebal([R['UPRO'].rename('UPRO'),R['TMF'].rename('TMF')],[.55,.45],21),'web candidate')
    add('60UPRO_40TMF_quarterly',port_rebal([R['UPRO'].rename('UPRO'),R['TMF'].rename('TMF')],[.6,.4],63),'web candidate')
    for cs in (.0001,.0002):
        lab=f'{cs*10000:.0f}bp_side'
        add(f'SPY_overnight_{lab}',overnight(A['SPY'],cs),'MOC-MOO approximation')
        add(f'UPRO_overnight_{lab}',overnight(A['UPRO'],cs),'MOC-MOO approximation')
        add(f'TQQQ_overnight_{lab}',overnight(A['TQQQ'],cs),'MOC-MOO approximation')
        add(f'UPRO_overnight_SPY_day_{lab}',hybrid_overnight_day(A['UPRO'],A['SPY'],cs),'overnight/day hybrid')
        add(f'TQQQ_overnight_QQQ_day_{lab}',hybrid_overnight_day(A['TQQQ'],A['QQQ'],cs),'overnight/day hybrid')
    pd.DataFrame(rows).to_csv(a.output/'modern_results.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'tickers':tick,'source':'Yahoo Finance via yfinance','end_requested':'2026-08-22','overnight_note':'Adjusted regular-session Open/Close approximate MOO/MOC auction fills; not tick-level auction data.','costs':'1bp and 2bp per order side for overnight variants; 3bp per 100% allocation turnover for tactical variants.'},indent=2))
    print(pd.DataFrame(rows).sort_values('cagr',ascending=False).to_string(index=False))
if __name__=='__main__':main()
