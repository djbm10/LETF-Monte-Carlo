from __future__ import annotations
import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
warnings.filterwarnings("ignore")
OUT=Path("results/cross_strategy_tournament_20261106")
OUT.mkdir(parents=True,exist_ok=True)
SECTORS="XLK XLF XLE XLV XLI XLP XLU XLY XLB".split()
MACRO="SPY TLT GLD UUP DBC".split()
SURVIVORS="AAPL MSFT AMZN NVDA GOOGL META TSLA NFLX CRM ADBE AMD AVGO".split()
NEEDED=list(dict.fromkeys(["QQQ","SPY","VOO","TQQQ","^IRX"]+SECTORS+MACRO+SURVIVORS))
RF_FALLBACK=0.025/252

def download():
    frames={}
    for chunk in [NEEDED[i:i+7] for i in range(0,len(NEEDED),7)]:
        try:
            x=yf.download(chunk,start="1998-01-01",end="2026-11-07",auto_adjust=True,progress=False,threads=False,group_by="ticker")
        except Exception as e:
            print("DOWNLOAD-ERROR",chunk,str(e));continue
        for t in chunk:
            try:
                col=x[t]["Close"] if isinstance(x.columns,pd.MultiIndex) else x["Close"]
                if isinstance(col,pd.DataFrame):col=col.iloc[:,0]
                col=pd.to_numeric(col,errors="coerce").dropna()
                if len(col)>100:
                    col.index=pd.to_datetime(col.index).tz_localize(None)
                    frames[t]=col[~col.index.duplicated()].sort_index()
                    print("data",t,str(col.index.min().date()),str(col.index.max().date()),len(col))
            except Exception as e:print("MISSING",t,str(e))
    if "QQQ" not in frames or "SPY" not in frames:raise RuntimeError("QQQ/SPY adjusted history unavailable; fail closed.")
    px=pd.DataFrame(frames).sort_index()
    px=px.loc[px.index>=pd.Timestamp("2000-01-01")]
    return px

def summary(r):
    r=pd.Series(r).dropna()
    if len(r)<252:return None
    e=(1+r).cumprod();peak=e.cummax();dd=e/peak-1;n=len(e)
    one=r.rolling(252).apply(lambda z:np.prod(1+z)-1,raw=True).dropna()
    three=r.rolling(756).apply(lambda z:np.prod(1+z)**(1/3)-1,raw=True).dropna()
    loss=r[r<r.quantile(.05)]
    return dict(years=round(n/252,2),cagr=float(e.iloc[-1]**(252/n)-1),
        max_dd=float(dd.min()),annual_vol=float(r.std()*np.sqrt(252)),
        sharpe=float(np.sqrt(252)*r.mean()/r.std()) if r.std()>0 else None,
        terminal_multiple=float(e.iloc[-1]),worst_day=float(r.min()),
        daily_es5=float(loss.mean()) if len(loss) else 0.0,
        p10_1y=float(one.quantile(.1)) if len(one) else None,
        p10_3y_cagr=float(three.quantile(.1)) if len(three) else None,
        pct_days_underwater=float((dd<-.00001).mean()))

def bsm_call(s,k,t,v,rf):
    t=max(t,1/365);v=max(v,.08)
    d1=(math.log(s/k)+(rf+.5*v*v)*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
    return s*norm.cdf(d1)-k*math.exp(-rf*t)*norm.cdf(d2)

def leaps_synthetic(px,rf,alloc,delta=.7):
    x=px["QQQ"].dropna().loc["2012-01-01":]
    if len(x)<300:return pd.Series(dtype=float)
    ret=x.pct_change(fill_method=None)
    vol=(ret.rolling(20).std()*np.sqrt(252)+.10).clip(.16,.85).shift(1).ffill()
    rv=(rf.reindex(x.index).ffill().fillna(RF_FALLBACK))*252
    wealth=1.;cash=1.;units=0.;strike=0.;dte=0;rows=[];last=-999
    for i,(date,s) in enumerate(x.items()):
        sigma=float(vol.iloc[i]) if pd.notna(vol.iloc[i]) else .30
        rate=float(rv.iloc[i])
        if i>0:
            dte=max(dte-1,0)
            mark=bsm_call(s,strike,dte/252,sigma,rate) if units else 0.
            total=cash+units*mark
            if total<=0:raise RuntimeError("negative modeled option wealth")
            cash*=1+rate/252
            total=cash+units*mark
            wealth=total
        else:wealth=cash
        if i>=30 and (i-last>=126 or (units and dte<126) or last==-999):
            mark=bsm_call(s,strike,dte/252,sigma,rate) if units else 0.
            cash+=units*mark*.99
            wealth=cash
            dte=378  # 1.5y in trading sessions
            T=dte/252
            d1=float(norm.ppf(delta))
            strike=s*math.exp((rate+.5*sigma*sigma)*T-d1*sigma*math.sqrt(T))
            ask=bsm_call(s,strike,T,sigma,rate)*1.01
            units=(wealth*alloc)/ask if ask>0 else 0.
            cash-=units*ask
            last=i
            wealth=cash+units*bsm_call(s,strike,T,sigma,rate)
        rows.append((date,wealth))
    eq=pd.Series(dict(rows)).sort_index()
    return eq.pct_change(fill_method=None).dropna()

def daily_return(expo,qr,rf,fee=.0002):
    ex=expo.fillna(0).clip(0,3)
    cost=ex.diff().abs().fillna(ex.abs())*fee
    return rf+ex*(qr-rf)-cost

def sector_rotation(px,rf,lookback,top_n,trend):
    available=[k for k in SECTORS if k in px]
    sp=px[available].copy()
    ret=sp.pct_change(fill_method=None)
    rank=sp/sp.shift(lookback)-1
    spy=px["SPY"];ma=spy.rolling(200).mean()
    weights=pd.DataFrame(0.,index=px.index,columns=available)
    prev=np.zeros(len(available))
    months=px.index.to_period("M")
    for i,date in enumerate(px.index):
        if i<lookback+1:continue
        if i==0 or months[i]!=months[i-1]:
            candidates=rank.iloc[i-1].dropna().sort_values(ascending=False)
            if len(candidates)>=len(available):
                prev=np.zeros(len(available))
                for t in candidates.head(top_n).index:prev[available.index(t)]=1/top_n
        weights.iloc[i]=prev
        if trend and not (spy.iloc[i-1]>ma.iloc[i-1]):weights.iloc[i]=0
    turnover=weights.diff().abs().sum(axis=1).fillna(0)
    out=(ret*weights).sum(axis=1)-turnover*.001
    return out

def crossasset(px,rf):
    available=[t for t in MACRO if t in px]
    p=px[available];ret=p.pct_change(fill_method=None)
    vol=ret.rolling(63).std()*np.sqrt(252)
    mom=p/p.shift(252)-1
    weights=pd.DataFrame(0.,index=px.index,columns=available)
    prev=np.zeros(len(available));months=px.index.to_period("M")
    for i,date in enumerate(px.index):
        if i<253:continue
        if months[i]!=months[i-1]:
            z=(np.sign(mom.iloc[i-1])*.10/np.sqrt(5)/vol.iloc[i-1].clip(lower=.08)).clip(-.9,.9)
            z=z.reindex(available).fillna(0)
            if z.abs().sum()>2:z=z*2/z.abs().sum()
            prev=z.to_numpy()
        weights.iloc[i]=prev
    gross=weights.abs().sum(axis=1)
    costs=weights.diff().abs().sum(axis=1)*.0005
    short_cost=weights.clip(upper=0).abs().sum(axis=1)*.01/252
    return rf+(weights*(ret.sub(rf,axis=0))).sum(axis=1)-costs-short_cost

def survivor_top1(px,rf):
    available=[t for t in SURVIVORS if t in px]
    p=px[available];rets=p.pct_change(fill_method=None)
    mom=p/p.shift(126)-1
    ma=p.rolling(200).mean()
    out=pd.Series(index=px.index,dtype=float)
    current=None;months=px.index.to_period("M")
    for i,date in enumerate(px.index):
        if i<201:continue
        if months[i]!=months[i-1]:
            e=(mom.iloc[i-1].where(p.iloc[i-1]>ma.iloc[i-1])).dropna()
            current=e.idxmax() if len(e)>=8 else None
        out.iloc[i]=(rets.at[date,current] if current and np.isfinite(rets.at[date,current]) else rf.loc[date])
    return out

def bootstrap(r,year_horizon=3,npaths=3000,block=21,seed=45):
    r=pd.Series(r).dropna().to_numpy(dtype=float)
    if len(r)<260:return {}
    rng=np.random.default_rng(seed)
    steps=252*year_horizon;k=int(np.ceil(steps/block));starts=rng.integers(0,len(r)-block,size=(npaths,k))
    idx=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(npaths,-1)[:,:steps]
    draws=r[idx]
    equity=np.cumprod(1+draws,axis=1)
    peak=np.maximum.accumulate(np.concatenate([np.ones((npaths,1)),equity],axis=1),axis=1)[:,1:]
    dd=(equity/peak-1).min(axis=1)
    cagr=equity[:,-1]**(1/year_horizon)-1
    return dict(p10_cagr=float(np.quantile(cagr,.1)),median_cagr=float(np.median(cagr)),
        p10_terminal=float(np.quantile(equity[:,-1],.1)),
        prob_dd50=float(np.mean(dd<=-.50)),prob_dd70=float(np.mean(dd<=-.70)),
        p10_maxdd=float(np.quantile(dd,.1)),npaths=npaths,block_days=block)

def run():
    px=download();px.to_csv(OUT/"downloaded_adjusted_closes.csv")
    dates=px.index
    irx=px["^IRX"].ffill() if "^IRX" in px else pd.Series(np.nan,index=dates)
    rf=(irx/100/252).clip(0,.001).fillna(RF_FALLBACK)
    q=px["QQQ"];qr=q.pct_change(fill_method=None)
    ex=qr-rf;vol20=ex.rolling(20).std()*np.sqrt(252)
    vol5=ex.rolling(5).std()*np.sqrt(252)
    ma=q.rolling(200).mean();mult=.35/vol20.clip(lower=.04)
    cap=mult.clip(upper=3)
    base=(q>ma).shift(1).fillna(False).astype(float)*cap.shift(1).fillna(0)
    shock=(q>ma).shift(1).fillna(False).astype(float)*(.35/pd.concat([vol5,vol20],axis=1).max(axis=1).clip(lower=.04)).clip(upper=3).shift(1).fillna(0)
    ratio=(q/ma).shift(1)
    graduated=cap.shift(1).fillna(0)*np.select([ratio>1.02,ratio<.98],[1.,0.],default=.5)
    # Return-sign trend ensemble is separate from the 200DMA baseline
    trend=pd.concat([(q/q.shift(w)-1)>0 for w in (63,126,252)],axis=1).mean(axis=1)
    ensemble=cap.shift(1).fillna(0)*trend.shift(1).fillna(0)
    credit=px["HYG"] if "HYG" in px else None
    # HYG is not required; this benchmark remains absent if input unavailable
    out={}
    out["35_0_QQQ_proxy"]=daily_return(base,qr,rf)
    out["shock_5d20d"]=daily_return(shock,qr,rf)
    out["graduated_2pct"]=daily_return(graduated,qr,rf)
    out["trend_63_126_252"]=daily_return(ensemble,qr,rf)
    out["SPY_buyhold"]=px["SPY"].pct_change(fill_method=None)
    if "VOO" in px:out["VOO_buyhold"]=px["VOO"].pct_change(fill_method=None)
    if "TQQQ" in px:out["TQQQ_buyhold"]=px["TQQQ"].pct_change(fill_method=None)
    for w in (126,252):
        for n in (1,3):
            for gate in (False,True):
                out[f"sector_{w}_{n}_{int(gate)}"]=sector_rotation(px,rf,w,n,gate)
    out["crossasset_5proxy_TSMOM"]=crossasset(px,rf)
    out["survivor_biased_top1"]=survivor_top1(px,rf)
    for frac in (.2,.4,.6):
        out[f"reserve_{int(frac*100)}_cash"]=frac*rf+(1-frac)*out["35_0_QQQ_proxy"]
    out["reserve_40_cash_zero_yield"]=.6*out["35_0_QQQ_proxy"]
    out["synthetic_QQQ_LEAPS_25pct"]=leaps_synthetic(px,rf,.25)
    out["synthetic_QQQ_LEAPS_50pct"]=leaps_synthetic(px,rf,.5)
    out["synthetic_QQQ_LEAPS_100pct"]=leaps_synthetic(px,rf,1.)
    df=pd.DataFrame(out,index=dates).sort_index().replace([np.inf,-np.inf],np.nan)
    df.to_csv(OUT/"daily_returns.csv")
    windows={"2008plus":("2008-01-01","2026-11-06"),"2012plus":("2012-01-01","2026-11-06"),
        "2019plus":("2019-01-01","2026-11-06"),"2023plus_descriptive":("2023-01-01","2026-11-06")}
    results=[]
    for cohort,(start,end) in windows.items():
        for name in df.columns:
            s=df.loc[start:end,name].dropna()
            if len(s)<252:continue
            # exclude returns dated before instrument inception
            if name.startswith("synthetic_") and cohort=="2008plus":continue
            m=summary(s)
            if m:results.append({"window":cohort,"strategy":name,**m})
    result=pd.DataFrame(results)
    result.to_csv(OUT/"window_metrics.csv",index=False)
    stress=[]
    for window in ("2012plus","2019plus"):
        for name in ["35_0_QQQ_proxy","shock_5d20d","graduated_2pct","trend_63_126_252","SPY_buyhold","crossasset_5proxy_TSMOM","reserve_20_cash","reserve_40_cash","reserve_60_cash","survivor_biased_top1","synthetic_QQQ_LEAPS_25pct","synthetic_QQQ_LEAPS_50pct","synthetic_QQQ_LEAPS_100pct"]:
            s=df.loc[windows[window][0]:windows[window][1],name].dropna()
            for block in (10,21,63):
                if len(s)<500:continue
                a=bootstrap(s,3,3000,block,45)
                stress.append({"window":window,"strategy":name,"horizon_years":3,**a})
    pd.DataFrame(stress).to_csv(OUT/"bootstrap_3y.csv",index=False)
    exp=pd.DataFrame({"base_exposure":base,"shock_exposure":shock,"graduated_exposure":graduated,"ensemble_exposure":ensemble})
    exp.to_csv(OUT/"daily_exposure.csv")
    manifest=dict(run_date="2026-11-06",source="Yahoo Finance adjusted close via yfinance; ^IRX simple interest approximation",
        status="Exploratory model-comparison replay; not new untouched holdout; never claim live alpha or official futures execution",
        baseline="Frozen 35/0 QQQ 20d rolling excess-return volatility, 200DMA previous-day, 3x max, risk-free collateral, 2bp per 1x turnover",
        timing="Previous-session close determines current-session close-to-close position; assumes exposure executed before current close-to-close return, without overnight execution premium; risk of market-on-open slippage omitted",
        sector="Original nine SPDR sectors, monthly previous close ranks, 6/12m top1/top3, 10bps per dollar turnover; sector historical inception gaps",
        crossasset="SPY TLT GLD UUP DBC ETF total-return proxies only, not true dated futures; approximate borrow 1%/year on shorts and 5bps turnover; cannot validate actual managed futures",
        concentrated="CURRENT surviving mega-cap cohort intentionally biased. Informational stress example only; NOT point-in-time stock selection or robust alpha evidence",
        leaps="Black-Scholes synthetic LEAPS: rolling 548 calendar-ish days via 378 sessions, vol=prior20d realized vol +10pp capped, 1% spread; NO observed historical options quote. Exclude from validated ranking",
        monte_carlo="3,000 stationary contiguous-block return resamples per strategy/window/block; not dynamic regime re-simulation. Combined blocks do not reconstruct strategy signals.",
        holdout="Previously reviewed periods remain contaminated by earlier research; sector frozen 2017-2026 holdout deliberately not opened for model promotion; Bottleneck 2020-2023 untouched holdout not accessed.",
        limitations=["ETF vs futures carry, roll, funding, margin and borrow costs not fully matched","Volatility timing and turnover assume frictionless rebalancing at previous close","No real option implied-vol surface, bid-ask history, exercise, tax accounting","Survivorship bias in single-name screen","Cash needs and taxes require individual circumstances"])
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("SUMMARY 2012plus")
    z=result[result.window=="2012plus"].sort_values("cagr",ascending=False)
    print(z[["strategy","cagr","max_dd","p10_3y_cagr","annual_vol"]].to_string(index=False))
    print("STRESS:",len(stress),"cases")
    print("FILES:",[p.name for p in OUT.glob("*")])
if __name__=="__main__":run()
