from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252
OUT=Path("results/actual_futures_continuous_validation_20260921")
OUT.mkdir(parents=True,exist_ok=True)
SPECS={
    "NQ=F":{"family":"NASDAQ","multiplier":20.0,"benchmark":"QQQ"},
    "MNQ=F":{"family":"NASDAQ","multiplier":2.0,"benchmark":"QQQ"},
    "ES=F":{"family":"SP500","multiplier":50.0,"benchmark":"SPY"},
    "MES=F":{"family":"SP500","multiplier":5.0,"benchmark":"SPY"},
}
TC_PER_1X_TURNOVER=.0002

def dl(ticker,start="1998-01-01"):
    x=yf.download(ticker,start=start,end="2026-09-22",auto_adjust=False,progress=False,threads=False)
    if x.empty:return pd.DataFrame()
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    x.index=pd.to_datetime(x.index).tz_localize(None)
    return x.sort_index()

def met(r):
    r=pd.Series(r).dropna()
    if len(r)<252:return {}
    e=(1+r).cumprod();y=len(r)/TD;dd=e/e.cummax()-1
    return dict(start=str(r.index.min().date()),end=str(r.index.max().date()),years=y,
        cagr=e.iloc[-1]**(1/y)-1,max_dd=dd.min(),vol=r.std()*math.sqrt(TD),
        sharpe=(r.mean()*TD)/(r.std()*math.sqrt(TD)) if r.std()>0 else np.nan,
        terminal=e.iloc[-1])

def bt(px,rf,mode):
    ret=px.pct_change()
    vol=ret.rolling(20).std()*math.sqrt(TD)
    ma=px.rolling(200).mean()
    e=pd.Series(0.0,index=px.index)
    for i in range(1,len(px)):
        j=i-1
        if not np.isfinite(vol.iloc[j]) or vol.iloc[j]<=0 or not np.isfinite(ma.iloc[j]):continue
        if mode=="35_0":
            target=.35 if px.iloc[j]>ma.iloc[j] else 0.
            e.iloc[i]=np.clip(target/vol.iloc[j],0,3)
        elif mode=="35_12":
            target=.35 if px.iloc[j]>ma.iloc[j] else .12
            e.iloc[i]=np.clip(target/vol.iloc[j],0,3)
        elif mode=="vol25":
            e.iloc[i]=np.clip(.25/vol.iloc[j],0,3)
        elif mode=="2x_200":
            e.iloc[i]=2.0 if px.iloc[j]>ma.iloc[j] else 0.
        elif mode=="1x":
            e.iloc[i]=1.0
        else:raise ValueError(mode)
    rr=rf.reindex(px.index).ffill().fillna(0)+e*ret.fillna(0)-TC_PER_1X_TURNOVER*e.diff().abs().fillna(e.abs())
    return rr,e

def main():
    irx=dl("^IRX")
    if irx.empty:raise RuntimeError("^IRX unavailable")
    rf=(irx["Close"]/100/TD).rename("rf")
    rows=[];track=[];expo=[]
    for tick,spec in SPECS.items():
        d=dl(tick)
        if d.empty:
            rows.append({"ticker":tick,"status":"unavailable"});continue
        px=d["Close"].dropna()
        bench=dl(spec["benchmark"])["Adj Close"].dropna()
        ix=px.index.intersection(bench.index)
        fr=px.reindex(ix).pct_change()
        br=bench.reindex(ix).pct_change()
        valid=pd.concat([fr.rename("fut"),br.rename("bench")],axis=1).dropna()
        track.append({"ticker":tick,"start":str(valid.index.min().date()),"end":str(valid.index.max().date()),
                      "n":len(valid),"corr_daily":valid.corr().iloc[0,1],
                      "mean_fut_minus_bench_ann":(valid.fut-valid.bench).mean()*TD,
                      "tracking_diff_vol_ann":(valid.fut-valid.bench).std()*math.sqrt(TD),
                      "largest_abs_daily_gap":(valid.fut-valid.bench).abs().max()})
        for mode in ["1x","vol25","35_12","35_0","2x_200"]:
            rr,e=bt(px,rf,mode)
            m=met(rr)
            rows.append({"ticker":tick,"family":spec["family"],"mode":mode,"status":"scored",**m,
                         "avg_exposure":e.mean(),"max_exposure":e.max(),"turnover_per_year":e.diff().abs().sum()/(len(e)/TD)})
        expo.append(pd.DataFrame({"date":px.index,"ticker":tick,"close":px.values}))
    pd.DataFrame(rows).to_csv(OUT/"strategy_results.csv",index=False)
    pd.DataFrame(track).to_csv(OUT/"tracking_checks.csv",index=False)
    manifest={
      "source":"Yahoo Finance continuous futures tickers NQ=F, MNQ=F, ES=F, MES=F; ^IRX for collateral yield",
      "important_limitation":"These are real futures-derived continuous price series, but Yahoo does not document a CME-official roll/back-adjustment methodology. Roll gaps can bias close-to-close returns. Treat this as an independent confirmation layer, not the final settlement-grade truth.",
      "official_gold_standard":"CME Continuous Price Series/DataMine official settlement data, or QuantConnect/AlgoSeek mapped contract histories with explicit roll events.",
      "account_return_model":"daily collateral RF + prior-day exposure * continuous-futures close return - 2bp per 1.0 exposure turnover",
      "signals":"prior-day 200DMA and 20d annualized realized futures volatility",
      "modes":{"35_0":"35% target vol above 200DMA, 0 below","35_12":"35%/12%","vol25":"25% vol all regimes","2x_200":"2x above 200DMA/cash below","1x":"fully collateralized 1x futures"},
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(track).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
