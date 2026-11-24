from __future__ import annotations
"""
Real OptionMetrics LEAPS backtest engine.

This module deliberately distinguishes:
1) parser/selection smoke tests using a published real OptionMetrics snapshot, and
2) performance backtests, which require a licensed longitudinal IvyDB US/ETF export.

Expected flattened export columns:
secid,date,exdate,cp_flag,strike_price,best_bid,best_offer,optionid,ticker
Optional but strongly recommended:
delta,impl_volatility,volume,open_interest,underlying_price,rf_daily,ss_flag,am_settlement

Strike convention: OptionMetrics raw strike_price is commonly scaled by 1000.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

MULT=100
TARGET_DELTAS=(0.70,0.80,0.90)
TARGET_DTE=(365,548,730)
ALLOCATIONS=(0.25,0.50,1.00)

def load(path, *, strike_scale=1000.):
    x=pd.read_csv(path)
    for c in ("date","exdate"):
        x[c]=pd.to_datetime(x[c])
    for c in ("strike_price","best_bid","best_offer","delta","impl_volatility","volume","open_interest","underlying_price","rf_daily"):
        if c in x:x[c]=pd.to_numeric(x[c],errors="coerce")
    if not np.isfinite(strike_scale) or strike_scale <= 0:
        raise ValueError('Explicit positive strike scale required')
    x["strike"]=x.strike_price/strike_scale
    if x.duplicated(['date','optionid']).any():
        raise ValueError('Duplicate option/date quote; resolve timestamp/vendor revisions explicitly')
    x["mid"]=(x.best_bid+x.best_offer)/2
    x["spread"]=x.best_offer-x.best_bid
    x["dte"]=(x.exdate-x.date).dt.days
    return x

def quality_filter(x):
    y=x.copy()
    y=y[(y.cp_flag=="C")&(y.best_bid>=0)&(y.best_offer>0)&(y.best_offer>=y.best_bid)]
    if "ss_flag" in y:y=y[y.ss_flag.astype(str).isin(["0","0.0"])]
    if "cfadj" in y:y=y[pd.to_numeric(y.cfadj,errors='coerce').eq(1)]
    return y

def sample_smoke(x):
    q=quality_filter(x)
    long=q[q.dte>=365].copy()
    return {
        "rows_total":int(len(x)),
        "rows_clean_calls":int(len(q)),
        "rows_long_calls_365d_plus":int(len(long)),
        "dates":sorted(x.date.dt.strftime("%Y-%m-%d").unique().tolist()),
        "tickers":sorted(x.ticker.dropna().astype(str).unique().tolist()) if "ticker" in x else [],
        "max_dte":int(long.dte.max()) if len(long) else None,
        "min_long_strike":float(long.strike.min()) if len(long) else None,
        "max_long_strike":float(long.strike.max()) if len(long) else None,
        "example_long_calls":long[["secid","date","exdate","strike","best_bid","best_offer","optionid"]].head(10).to_dict("records"),
        "performance_testable":False,
        "longitudinal_schema_available":bool(x.date.nunique()>=252 and "delta" in x and "underlying_price" in x),
        "performance_blocker":"EOD rows alone do not establish quote timestamps/depth, contract actions, permissions or account taxation",
    }

def choose_contract(chain,target_delta,target_dte):
    c=quality_filter(chain)
    c=c[(c.dte>=270)&(c.dte<=900)]
    if "delta" not in c or c.delta.notna().sum()==0:return None
    c=c[(c.delta>0.45)&(c.delta<0.995)]
    if "volume" in c and c.volume.notna().any():
        oi=c.open_interest.fillna(0) if 'open_interest' in c else pd.Series(0,index=c.index)
        c=c[(c.volume.fillna(0)>0)|(oi>0)]
    if c.empty:return None
    # prioritize delta fit, then maturity fit, then spread as percentage of mid
    c=c.copy()
    c["sel_score"]=abs(c.delta-target_delta)+0.20*abs(c.dte-target_dte)/365+0.10*(c.spread/c.mid.replace(0,np.nan)).fillna(1)
    return c.sort_values(["sel_score","open_interest" if "open_interest" in c else "dte"],ascending=[True,False]).iloc[0]

def monthly_dates(x):
    d=pd.Series(sorted(x.date.unique()))
    return pd.to_datetime(d.groupby(pd.to_datetime(d).dt.to_period("M")).max().values)

def option_mark(row,side):
    if side=="buy":return float(row.best_offer)
    if side=="sell":return float(row.best_bid)
    return float(row.mid)

def backtest_one(x,ticker,target_delta,target_dte,allocation,rebalance_months=6,start_capital=1_000_000,
                 commission_per_contract=.65):
    """Legacy EOD scenario only; never account/performance certification."""
    if not 0 < allocation <= 1 or commission_per_contract < 0:
        raise ValueError('Cash-funded allocation and nonnegative commission required')
    z=x[x.ticker.astype(str)==ticker].copy().sort_values(["date","optionid"])
    if z.empty:return None
    if "delta" not in z or "underlying_price" not in z:return None
    dates=monthly_dates(z)
    if len(dates)<24:return None
    cash=float(start_capital); contracts=0; oid=None; entry_cost=0.; rows=[]; last_rebal_i=-999
    for mi,d in enumerate(dates):
        day=z[z.date==d]
        rf=float(day.rf_daily.dropna().iloc[0]) if "rf_daily" in day and day.rf_daily.notna().any() else 0.
        # A single month-end rate cannot determine the elapsed daily cash return.
        if rf:
            raise ValueError('Nonzero cash yield requires an aligned elapsed daily rate series, not 21 invented days')
        cur=day[day.optionid==oid] if oid is not None else pd.DataFrame()
        if contracts and len(cur)==0:
            raise ValueError(f'Missing held option {oid} quote on {d}; cannot erase position or invent a sale')
        mark=option_mark(cur.iloc[0],"mid") if len(cur) else 0.
        opt_value=contracts*MULT*mark
        wealth=cash+opt_value
        need_roll=(oid is None or len(cur)==0 or (cur.iloc[0].dte<180 if len(cur) else True) or mi-last_rebal_i>=rebalance_months)
        if need_roll:
            if contracts and len(cur):
                cash+=contracts*(MULT*option_mark(cur.iloc[0],"sell")-commission_per_contract)
            contracts=0;oid=None
            pick=choose_contract(day,target_delta,target_dte)
            if pick is not None:
                premium=option_mark(pick,"buy")*MULT+commission_per_contract
                budget=min(max(wealth*allocation,0),cash)
                contracts=int(budget//premium) if premium>0 else 0
                if contracts>0:
                    cash-=contracts*premium;oid=pick.optionid;entry_cost=premium;last_rebal_i=mi
            cur=day[day.optionid==oid] if oid is not None else pd.DataFrame()
            mark=option_mark(cur.iloc[0],"mid") if len(cur) else 0.
            opt_value=contracts*MULT*mark;wealth=cash+opt_value
        spot=float(day.underlying_price.dropna().iloc[0]) if day.underlying_price.notna().any() else np.nan
        eff_delta=(contracts*MULT*float(cur.iloc[0].delta)*spot/wealth) if len(cur) and wealth>0 and np.isfinite(spot) else 0.
        rows.append((d,wealth,cash,opt_value,contracts,oid,eff_delta))
    out=pd.DataFrame(rows,columns=["date","wealth","cash","option_value","contracts","optionid","effective_delta_exposure"]).set_index("date")
    ret=out.wealth.pct_change().dropna()
    if len(ret)<12:return None
    yrs=(out.index[-1]-out.index[0]).days/365.2425
    dd=out.wealth/np.maximum(out.wealth.cummax(),start_capital)-1
    return out,{"ticker":ticker,"target_delta":target_delta,"target_dte":target_dte,"allocation":allocation,
                "evidence":"HYPOTHETICAL_STRESS","investment_performance_certification":"BLOCKED",
                "commission_per_contract":commission_per_contract,
                "months":len(ret),"cagr":(out.wealth.iloc[-1]/start_capital)**(1/yrs)-1,
                "max_dd":dd.min(),"vol":ret.std()*math.sqrt(12),"terminal":out.wealth.iloc[-1],
                "avg_effective_delta_exposure":out.effective_delta_exposure.mean()}

def full_tournament(x,outdir):
    rows=[];curves=[]
    for ticker in sorted(x.ticker.dropna().astype(str).unique()):
      for delta in TARGET_DELTAS:
       for dte in TARGET_DTE:
        for alloc in ALLOCATIONS:
         r=backtest_one(x,ticker,delta,dte,alloc)
         if r is None:continue
         curve,met=r;rows.append(met);q=curve.reset_index();q["ticker"]=ticker;q["target_delta"]=delta;q["target_dte"]=dte;q["allocation"]=alloc;curves.append(q)
    pd.DataFrame(rows).to_csv(outdir/"leaps_results.csv",index=False)
    if curves:pd.concat(curves,ignore_index=True).to_csv(outdir/"leaps_monthly_curves.csv",index=False)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",default="results/optionmetrics_leaps")
    ap.add_argument("--smoke-only",action="store_true")
    ap.add_argument('--strike-scale',type=float,default=1000.)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);x=load(a.input,strike_scale=a.strike_scale)
    smoke=sample_smoke(x);(out/"smoke.json").write_text(json.dumps(smoke,indent=2,default=str))
    manifest={
      "input":a.input,
      "execution":"buy selected LEAPS at ask; sell at bid on roll; mark at midpoint between transactions",
      "selection":"calls 270-900 DTE, target delta 0.70/0.80/0.90, target maturity 365/548/730 days, spread and liquidity tie-breakers",
      "portfolio_allocations":[.25,.50,1.0],"roll":"every six months or when DTE<180",
      "required_for_performance":["multiple historical dates","delta","underlying_price"],
      "limitations":["End-of-day implementation only.","Full decision-grade test requires licensed longitudinal IvyDB data.",
                     "Public sample snapshot can validate schema/LEAPS selection but cannot estimate strategy returns.",
                     "Cash return requires rf_daily in the flattened export; otherwise assumed zero."]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if not a.smoke_only:
      res=full_tournament(x,out);print(res.to_string(index=False) if len(res) else "NO_PERFORMANCE_TEST: longitudinal licensed data required")
    print(json.dumps(smoke,indent=2,default=str))

if __name__=="__main__":main()
