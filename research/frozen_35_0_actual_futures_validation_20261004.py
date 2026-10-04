from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

TD=252
OUT=Path("results/frozen_35_0_actual_futures_validation_20261004")
OUT.mkdir(parents=True,exist_ok=True)

# Frozen scientific rule from prior PROMOTE work. These are not tuned here.
BULL_VOL_TARGET=.35
BEAR_VOL_TARGET=0.0
SMA_DAYS=200
VOL_DAYS=20
EXPOSURE_CAP=3.0
FROZEN_FRACTIONAL_TURNOVER_COST=.0002
STARTING_CAPITAL=100_000.0

def dl(ticker,start="1999-01-01"):
    x=yf.download(ticker,start=start,end="2026-10-05",auto_adjust=False,progress=False,threads=False)
    if x.empty: raise RuntimeError(f"no data for {ticker}")
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.index=pd.to_datetime(x.index).tz_localize(None)
    if ticker in ("QQQ","SPY") and "Adj Close" in x:
        s=x["Adj Close"]
    else:
        s=x["Close"]
    return pd.to_numeric(s,errors="coerce").dropna().sort_index()

def metrics(r):
    r=pd.Series(r).dropna()
    eq=(1+r).cumprod()
    yrs=len(r)/TD
    dd=eq/eq.cummax()-1
    annvol=r.std()*math.sqrt(TD)
    roll252=(1+r).rolling(252).apply(np.prod,raw=True)-1
    return {
      "start":str(r.index.min().date()),"end":str(r.index.max().date()),"days":int(len(r)),"years":float(yrs),
      "cagr":float(eq.iloc[-1]**(1/yrs)-1),"terminal_multiple":float(eq.iloc[-1]),
      "terminal_wealth_100k":float(STARTING_CAPITAL*eq.iloc[-1]),
      "max_dd":float(dd.min()),"ann_vol":float(annvol),
      "sharpe_zero_rf":float(r.mean()*TD/annvol) if annvol>0 else None,
      "worst_day":float(r.min()),
      "rolling_1y_p10":float(roll252.quantile(.10)) if roll252.notna().any() else None,
      "rolling_1y_min":float(roll252.min()) if roll252.notna().any() else None,
      "rolling_1y_negative_pct":float((roll252.dropna()<0).mean()) if roll252.notna().any() else None,
    }

def rf_daily(index):
    irx=dl("^IRX","1999-01-01")
    # ^IRX is an annualized percent yield quote.
    return (irx/100/TD).reindex(index).ffill().fillna(0)

def frozen_exposure(qqq,rf):
    qret=qqq.pct_change()
    excess=qret-rf.reindex(qqq.index).fillna(0)
    vol=excess.rolling(VOL_DAYS).std()*math.sqrt(TD)
    ma=qqq.rolling(SMA_DAYS).mean()
    e=pd.Series(0.0,index=qqq.index)
    for i in range(1,len(qqq)):
        j=i-1
        if pd.notna(vol.iloc[j]) and vol.iloc[j]>0 and pd.notna(ma.iloc[j]):
            target=BULL_VOL_TARGET if qqq.iloc[j]>ma.iloc[j] else BEAR_VOL_TARGET
            e.iloc[i]=float(np.clip(target/vol.iloc[j],0,EXPOSURE_CAP))
    return e

def fractional_proxy(qqq,rf,e,cost=FROZEN_FRACTIONAL_TURNOVER_COST):
    # Original proxy convention: cash collateral + exposure to equity excess return.
    qret=qqq.pct_change().fillna(0)
    return rf + e*(qret-rf)-cost*e.diff().abs().fillna(e.abs())

def fractional_futures(fut,rf,e,cost=FROZEN_FRACTIONAL_TURNOVER_COST):
    # Actual futures-derived P/L convention: cash collateral + notional futures price P/L.
    fret=fut.pct_change().fillna(0)
    return rf + e*fret-cost*e.diff().abs().fillna(e.abs())

def quarter_roll_dates(index):
    out=[]
    years=sorted(set(index.year))
    for y in years:
        for m in (3,6,9,12):
            first=pd.Timestamp(y,m,1)
            fridays=pd.date_range(first,first+pd.offsets.MonthEnd(0),freq="W-FRI")
            if len(fridays)<3: continue
            expiry=fridays[2]
            target=expiry-pd.offsets.BDay(5)
            eligible=index[index<=target]
            if len(eligible): out.append(eligible[-1])
    return set(out)

def discrete_mnq(px,qqq,rf,e,margin_fraction=.20,commission_per_side=2.50,
                 slippage_ticks_per_side=1.0,cash_yield_fraction=1.0):
    ix=px.index.intersection(qqq.index).intersection(rf.index).intersection(e.index)
    px=px.reindex(ix); rf=rf.reindex(ix); e=e.reindex(ix)
    rolls=quarter_roll_dates(ix)
    tick_value=.25*2.0
    side_cost=commission_per_side+slippage_ticks_per_side*tick_value
    mult=2.0

    wealth=STARTING_CAPITAL
    peak=wealth
    min_dd=0.0
    n=0
    trade_sides=0
    roll_sides=0
    desired_hist=[]; actual_hist=[]; ret_hist=[]
    for i in range(1,len(ix)):
        prev_px=float(px.iloc[i-1]); cur_px=float(px.iloc[i])
        desired=float(e.iloc[i]) if np.isfinite(e.iloc[i]) else 0.0
        notional=prev_px*mult
        maxn=int(np.floor(wealth/max(notional*margin_fraction,1e-12)))
        want=int(np.round(desired*wealth/max(notional,1e-12)))
        want=max(0,min(want,maxn))
        if want!=n:
            sides=abs(want-n)
            wealth-=sides*side_cost
            trade_sides+=sides
            n=want
        if ix[i-1] in rolls and n:
            wealth-=2*abs(n)*side_cost
            roll_sides+=2*abs(n)
        actual=n*notional/max(wealth,1e-12)
        prev_wealth=wealth
        pnl=n*mult*(cur_px-prev_px)
        interest=cash_yield_fraction*wealth*float(rf.iloc[i])
        wealth=max(wealth+pnl+interest,0.0)
        ret=(wealth/prev_wealth-1) if prev_wealth>0 else -1.0
        ret_hist.append((ix[i],ret))
        desired_hist.append(desired);actual_hist.append(actual)
        peak=max(peak,wealth); min_dd=min(min_dd,wealth/peak-1 if peak>0 else -1)
        if wealth<=0: break
    r=pd.Series({d:v for d,v in ret_hist}).sort_index()
    m=metrics(r)
    years=max(len(r)/TD,1e-12)
    err=np.abs(np.asarray(actual_hist)-np.asarray(desired_hist))
    m.update({
      "margin_fraction":margin_fraction,"commission_per_side":commission_per_side,
      "slippage_ticks_per_side":slippage_ticks_per_side,"cash_yield_fraction":cash_yield_fraction,
      "contract_trade_sides_per_year":trade_sides/years,
      "roll_sides_per_year":roll_sides/years,
      "total_contract_sides":trade_sides+roll_sides,
      "mean_abs_exposure_rounding_error":float(np.mean(err)) if len(err) else None,
      "p95_abs_exposure_rounding_error":float(np.quantile(err,.95)) if len(err) else None,
      "pct_positive_desired_but_zero_contract":float(np.mean((np.asarray(desired_hist)>0)&(np.asarray(actual_hist)==0))) if len(err) else None,
      "max_dd_path":float(min_dd),
    })
    return r,m

def aligned_inputs():
    qqq=dl("QQQ","1999-01-01")
    nq=dl("NQ=F","1999-01-01")
    mnq=dl("MNQ=F","2019-01-01")
    full_ix=qqq.index.intersection(nq.index)
    rf_full=rf_daily(full_ix)
    q_full=qqq.reindex(full_ix); n_full=nq.reindex(full_ix)
    e_full=frozen_exposure(q_full,rf_full)

    micro_ix=qqq.index.intersection(mnq.index)
    rf_micro=rf_daily(micro_ix)
    q_micro=qqq.reindex(micro_ix); m_micro=mnq.reindex(micro_ix)
    e_micro=frozen_exposure(q_micro,rf_micro)
    return (q_full,n_full,rf_full,e_full),(q_micro,m_micro,rf_micro,e_micro)

def main():
    (q,nq,rf,e),(qm,mnq,rfm,em)=aligned_inputs()
    rows=[]

    # Same frozen exposures; only P/L vehicle changes.
    proxy=fractional_proxy(q,rf,e)
    actual=fractional_futures(nq,rf,e)
    rows += [
      {"series":"QQQ_proxy_35_0","vehicle":"QQQ total-return proxy","sample":"NQ common",**metrics(proxy),
       "avg_exposure":float(e.mean()),"turnover_per_year":float(e.diff().abs().sum()/(len(e)/TD))},
      {"series":"NQ_continuous_35_0","vehicle":"NQ=F futures-derived close","sample":"NQ common",**metrics(actual),
       "avg_exposure":float(e.mean()),"turnover_per_year":float(e.diff().abs().sum()/(len(e)/TD))},
    ]

    proxy_m=fractional_proxy(qm,rfm,em)
    actual_m=fractional_futures(mnq,rfm,em)
    rows += [
      {"series":"QQQ_proxy_35_0","vehicle":"QQQ total-return proxy","sample":"MNQ common",**metrics(proxy_m),
       "avg_exposure":float(em.mean()),"turnover_per_year":float(em.diff().abs().sum()/(len(em)/TD))},
      {"series":"MNQ_continuous_35_0","vehicle":"MNQ=F futures-derived close","sample":"MNQ common",**metrics(actual_m),
       "avg_exposure":float(em.mean()),"turnover_per_year":float(em.diff().abs().sum()/(len(em)/TD))},
    ]

    discrete=[]
    # Predeclared implementation sensitivities, not strategy tuning.
    for margin in (.20,.30):
      for slip in (.5,1.0,2.0):
       for cash in (0.0,.5,1.0):
        _,m=discrete_mnq(mnq,qm,rfm,em,margin_fraction=margin,commission_per_side=2.50,
                         slippage_ticks_per_side=slip,cash_yield_fraction=cash)
        discrete.append(m)
    ddf=pd.DataFrame(discrete)
    # Fixed base case declared before viewing output.
    base=ddf[(ddf.margin_fraction==.20)&(ddf.slippage_ticks_per_side==1.0)&(ddf.cash_yield_fraction==1.0)].iloc[0].to_dict()

    # Tracking/roll-artifact diagnostics; no data are deleted based on this.
    track=pd.DataFrame({"nq":nq.pct_change(),"qqq":q.pct_change()}).dropna()
    diff=track.nq-track.qqq
    diagnostics={
      "nq_qqq_daily_corr":float(track.corr().iloc[0,1]),
      "nq_minus_qqq_mean_ann":float(diff.mean()*TD),
      "nq_minus_qqq_tracking_vol_ann":float(diff.std()*math.sqrt(TD)),
      "largest_abs_daily_return_gap":float(diff.abs().max()),
      "dates_largest_10_gaps":[str(x.date()) for x in diff.abs().nlargest(10).index],
      "mnq_discrete_base_case":base,
      "mnq_discrete_sensitivity":{
        "cagr_min":float(ddf.cagr.min()),"cagr_median":float(ddf.cagr.median()),"cagr_max":float(ddf.cagr.max()),
        "max_dd_worst":float(ddf.max_dd.min()),"max_dd_median":float(ddf.max_dd.median()),
        "rolling_1y_p10_min":float(ddf.rolling_1y_p10.min()),"rolling_1y_p10_median":float(ddf.rolling_1y_p10.median()),
        "terminal_wealth_min":float(ddf.terminal_wealth_100k.min()),"terminal_wealth_median":float(ddf.terminal_wealth_100k.median()),
      }
    }

    pd.DataFrame(rows).to_csv(OUT/"fractional_proxy_vs_actual.csv",index=False)
    ddf.to_csv(OUT/"mnq_integer_sensitivity.csv",index=False)
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,default=float))
    (OUT/"manifest.json").write_text(json.dumps({
      "scientific_rule_status":"FROZEN_NO_RETUNING",
      "rule":"35% annualized target volatility when QQQ total-return price is above its 200-session SMA; 0 below; 20-session realized excess-return volatility; 3x cap; all signals lagged one session.",
      "frozen_fractional_cost":"20 bp per 1.0 exposure turnover, unchanged from prior futures research.",
      "actual_pnl":"NQ=F/MNQ=F daily futures-derived close-to-close returns generate P/L; QQQ is used only to preserve the frozen signal/exposure calculation.",
      "discrete_base_case":"MNQ; $100k start; 20% notional margin constraint; $2.50 commission per contract side + 1 tick slippage per side; 100% cash-yield credit; explicit quarterly two-side roll cost proxy.",
      "sensitivity":"margin 20/30%; slippage 0.5/1/2 ticks per side; cash-yield credit 0/50/100%. These are implementation sensitivities, not strategy parameter selection.",
      "critical_data_limitation":"Yahoo NQ=F/MNQ=F are real futures-derived continuous close series, but Yahoo does not document exchange-settlement or splice/roll methodology. This closes the proxy-P&L gap substantially but is not CME settlement-grade proof.",
      "settlement_grade_adapter":"research/cme_datamine_settlement_backtest.py remains the exchange-grade adapter and requires verified contract-level CME settlement inputs.",
      "decision_rule":"Do not retune 35/0. If real-futures P/L materially destroys the proxy result, FAIL IMPLEMENTATION. If it survives but settlement provenance remains undocumented, MONITOR. Only exchange/contract-settlement validation can earn IMPLEMENTATION-VALIDATED."
    },indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMNQ DISCRETE BASE\n",json.dumps(base,indent=2,default=float))
    print("\nSENSITIVITY\n",json.dumps(diagnostics["mnq_discrete_sensitivity"],indent=2))
    print("\nTRACKING\n",json.dumps({k:v for k,v in diagnostics.items() if k not in ("mnq_discrete_base_case","mnq_discrete_sensitivity")},indent=2))

if __name__=="__main__":
    main()
