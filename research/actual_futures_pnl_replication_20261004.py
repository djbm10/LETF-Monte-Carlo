from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252
OUT=Path("results/actual_futures_pnl_replication_20261004")
OUT.mkdir(parents=True,exist_ok=True)
TC_PER_1X=.0002
BULL=.35
BEAR=0.0
CAP=3.0

def dl(t,start):
    x=yf.download(t,start=start,end="2026-10-03",auto_adjust=False,progress=False,threads=False)
    if x.empty: return pd.DataFrame()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.index=pd.to_datetime(x.index).tz_localize(None)
    return x.sort_index()

def series(df,col):
    return pd.to_numeric(df[col],errors="coerce").dropna()

def metrics(r):
    r=pd.Series(r).dropna()
    eq=(1+r).cumprod(); yrs=len(r)/TD; peak=eq.cummax(); dd=eq/peak-1
    return {"start":str(r.index.min().date()),"end":str(r.index.max().date()),"years":yrs,
            "cagr":float(eq.iloc[-1]**(1/yrs)-1),"terminal":float(eq.iloc[-1]),
            "max_dd":float(dd.min()),"vol":float(r.std()*math.sqrt(TD)),
            "sharpe":float((r.mean()*TD)/(r.std()*math.sqrt(TD))) if r.std()>0 else np.nan,
            "worst_day":float(r.min())}

def rolling(r,years):
    r=pd.Series(r).dropna(); n=int(years*TD); vals=[]; dds=[]
    for s in range(0,len(r)-n+1,21):
        x=r.iloc[s:s+n]; eq=(1+x).cumprod()
        vals.append(float(eq.iloc[-1]**(1/years)-1))
        dds.append(float((eq/eq.cummax()-1).min()))
    if not vals:return {}
    return {"windows":len(vals),"median_cagr":float(np.median(vals)),"p10_cagr":float(np.quantile(vals,.10)),
            "worst_cagr":float(np.min(vals)),"median_max_dd":float(np.median(dds)),
            "p10_max_dd":float(np.quantile(dds,.10)),"worst_max_dd":float(np.min(dds))}

def desired_exposure(qqq,rf):
    qret=qqq.pct_change()
    ex=qret-rf.reindex(qqq.index).ffill().fillna(0)
    vol=ex.rolling(20).std()*math.sqrt(TD)
    ma=qqq.rolling(200).mean()
    e=pd.Series(0.0,index=qqq.index)
    for i in range(1,len(qqq)):
        j=i-1
        if np.isfinite(vol.iloc[j]) and vol.iloc[j]>0 and np.isfinite(ma.iloc[j]):
            target=BULL if qqq.iloc[j]>ma.iloc[j] else BEAR
            e.iloc[i]=float(np.clip(target/vol.iloc[j],0,CAP))
    return e

def quarter_roll_dates(index):
    out=[]
    years=sorted(set(index.year))
    for y in years:
        for m in (3,6,9,12):
            first=pd.Timestamp(y,m,1)
            fridays=pd.date_range(first,first+pd.offsets.MonthEnd(0),freq="W-FRI")
            if len(fridays)<3: continue
            expiry=fridays[2]
            candidates=index[index < expiry]
            if len(candidates)>=5:
                out.append(candidates[-5])
    return set(out)

def discrete_mnq(px, e, rf, capital, fee_side, slippage_ticks, margin_frac, band):
    mult=2.0; tick=0.25; tick_value=.50
    idx=px.index; rollset=quarter_roll_dates(idx)
    wealth=float(capital); peak=wealth; mdd=0.; n=0; target_hold=0.; sides=0; roll_sides=0
    exposures=[]; daily=[]
    for i in range(1,len(idx)):
        prior_notional=float(px.iloc[i-1])*mult
        actual=n*prior_notional/max(wealth,1e-12)
        want_exp=float(e.reindex(idx).iloc[i]) if np.isfinite(e.reindex(idx).iloc[i]) else actual
        if abs(want_exp-actual)>=band: target_hold=want_exp
        want=int(np.round(target_hold*wealth/max(prior_notional,1e-12)))
        maxn=int(np.floor(wealth/max(prior_notional*margin_frac,1e-12)))
        want=max(0,min(want,maxn))
        if want!=n:
            traded=abs(want-n); wealth-=traded*(fee_side+slippage_ticks*tick_value);sides+=traded;n=want
        if idx[i] in rollset and n:
            wealth-=2*abs(n)*(fee_side+slippage_ticks*tick_value);roll_sides+=2*abs(n)
        exp=n*prior_notional/max(wealth,1e-12)
        futret=float(px.iloc[i]/px.iloc[i-1]-1)
        day=float(rf.reindex(idx).ffill().fillna(0).iloc[i])+exp*futret
        wealth*=max(1+day,0)
        peak=max(peak,wealth);mdd=min(mdd,wealth/peak-1);daily.append(day);exposures.append(exp)
    yrs=(len(idx)-1)/TD
    return {"capital":capital,"fee_side":fee_side,"slippage_ticks":slippage_ticks,"margin_frac":margin_frac,"band":band,
            "cagr":float((wealth/capital)**(1/yrs)-1),"terminal_dollars":wealth,"max_dd":mdd,
            "contract_sides_per_year":(sides+roll_sides)/yrs,"roll_sides_per_year":roll_sides/yrs,
            "avg_exposure":float(np.mean(exposures)),"max_exposure":float(np.max(exposures)),
            "tracking_error_to_target_mae":float(np.mean(np.abs(np.asarray(exposures)-e.reindex(idx).iloc[1:].to_numpy())))}

def main():
    q=dl("QQQ","2000-01-01"); nq=dl("NQ=F","2000-01-01"); mnq=dl("MNQ=F","2019-05-06"); irx=dl("^IRX","2000-01-01")
    if q.empty or nq.empty or irx.empty: raise RuntimeError("required series unavailable")
    qpx=series(q,"Adj Close"); npx=series(nq,"Close")
    rf=(series(irx,"Close")/100/TD).rename("rf")
    ix=qpx.index.intersection(npx.index)
    qpx=qpx.reindex(ix);npx=npx.reindex(ix);rf0=rf.reindex(ix).ffill().fillna(0)
    e=desired_exposure(qpx,rf0)
    qret=qpx.pct_change().fillna(0); nret=npx.pct_change().fillna(0)
    turnover=e.diff().abs().fillna(e.abs())
    proxy=rf0+e*(qret-rf0)-TC_PER_1X*turnover
    actual=rf0+e*nret-TC_PER_1X*turnover
    rows=[]
    for name,r in [("QQQ_proxy",proxy),("NQ_continuous_actual_futures_derived",actual)]:
        row={"series":name,**metrics(r),"avg_exposure":float(e.mean()),"max_exposure":float(e.max()),
             "turnover_per_year":float(turnover.sum()/(len(e)/TD))}
        for y in (3,5,10):
            for k,v in rolling(r,y).items():row[f"rolling_{y}y_{k}"]=v
        rows.append(row)
    common=pd.DataFrame({"qqq":qret,"nq":nret}).dropna()
    tracking={"rows":len(common),"corr_daily":float(common.corr().iloc[0,1]),
              "mean_nq_minus_qqq_ann":float((common.nq-common.qqq).mean()*TD),
              "tracking_diff_vol_ann":float((common.nq-common.qqq).std()*math.sqrt(TD)),
              "largest_abs_daily_gap":float((common.nq-common.qqq).abs().max())}
    pd.DataFrame(rows).to_csv(OUT/"continuous_comparison.csv",index=False)
    (OUT/"tracking.json").write_text(json.dumps(tracking,indent=2))

    discrete=[]
    if not mnq.empty:
        mpx=series(mnq,"Close")
        mix=mpx.index.intersection(qpx.index)
        mpx=mpx.reindex(mix); em=e.reindex(mix).ffill().fillna(0); rfm=rf.reindex(mix).ffill().fillna(0)
        # Frozen implementation sensitivity grid from prior discrete validation; report all cells, do not optimize.
        for cap in (25_000,50_000,100_000,250_000):
            for fee in (2.5,4.0):
                for ticks in (0.0,0.5,1.0):
                    for margin in (.20,.30):
                        for band in (.15,.30,.45):
                            discrete.append(discrete_mnq(mpx,em,rfm,cap,fee,ticks,margin,band))
    ddf=pd.DataFrame(discrete);ddf.to_csv(OUT/"mnq_integer_sensitivity.csv",index=False)
    if not ddf.empty:
        robust=(ddf.groupby("capital").agg(median_cagr=("cagr","median"),worst_cagr=("cagr","min"),best_cagr=("cagr","max"),
                worst_max_dd=("max_dd","min"),median_max_dd=("max_dd","median"),
                median_tracking_mae=("tracking_error_to_target_mae","median"),
                max_tracking_mae=("tracking_error_to_target_mae","max"),
                median_sides_per_year=("contract_sides_per_year","median")).reset_index())
        robust.to_csv(OUT/"mnq_integer_robustness.csv",index=False)
        print(robust.to_string(index=False))
    manifest={
      "strategy_frozen":"35% target volatility when prior-day QQQ adjusted close is above prior-day 200DMA; 0 otherwise; 20-day prior-data volatility; cap 3x. No strategy parameter was changed for this replication.",
      "comparison_design":"The same QQQ-derived desired exposure drives both proxy and futures P&L. Proxy uses QQQ total-return excess return plus collateral. Futures uses Yahoo NQ=F continuous futures price return plus collateral.",
      "actual_futures_source":"Yahoo Finance NQ=F / MNQ=F. These are futures-derived traded-price continuous series, not ETF returns.",
      "critical_limit":"Yahoo does not document contract mapping, settlement field provenance, or roll/back-adjustment methodology. Therefore this is stronger than SPY/QQQ proxy P&L but NOT exchange-settlement-grade contract-chain validation.",
      "failed_contract_probe":"Expired symbols NQZ24.CME, NQH20.CME and NQZ10.CME returned no Yahoo history; Stooq CSV path was blocked by browser verification.",
      "qc_limit":"Repository has no QC_USER_ID, QC_API_TOKEN, or QC_ORGANIZATION_ID secrets, so QuantConnect contract-chain validation could not be launched.",
      "costs":"continuous comparison retains frozen 2bp per 1x exposure turnover. MNQ integer grid adds per-side commission and 0/0.5/1 tick slippage plus explicit quarterly roll sides five trading sessions before third Friday.",
      "cash_collateral":"^IRX annual yield / 252 on full collateral.",
      "margin":"MNQ sensitivity uses 20% and 30% of notional as conservative scenario constraints, not claims about current broker/CME margin.",
      "decision_rule":"No optimization on these results. Evidence can validate implementation, leave it MONITOR, or fail implementation."
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(tracking,indent=2))

if __name__=="__main__": main()
