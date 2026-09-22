from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252; OUT=Path("results/discrete_actual_futures_validation_20260921");OUT.mkdir(parents=True,exist_ok=True)
# use actual micro data when available; use NQ continuous as a counterfactual MNQ-sized instrument before MNQ launch
SERIES={
 "MNQ_actual_2019plus":{"ticker":"MNQ=F","mult":2.0,"actual_contract":True,"start":"2019-05-06"},
 "MNQ_sized_NQ_history":{"ticker":"NQ=F","mult":2.0,"actual_contract":False,"start":"2000-09-18"},
 "MES_actual_2019plus":{"ticker":"MES=F","mult":5.0,"actual_contract":True,"start":"2019-05-06"},
 "MES_sized_ES_history":{"ticker":"ES=F","mult":5.0,"actual_contract":False,"start":"2000-09-18"},
}
CAPS=[10_000,15_000,25_000,50_000,100_000]
FEES=[2.5,4.0]
BANDS=[.15,.30,.45]
MARGIN_RATES=[.20,.30]  # conservative scenario fractions of notional; not broker quotes

def dl(t,start):
 x=yf.download(t,start=start,end="2026-09-22",auto_adjust=False,progress=False,threads=False)
 if x.empty:return pd.Series(dtype=float)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 s=x["Close"].dropna();s.index=pd.to_datetime(s.index).tz_localize(None);return pd.to_numeric(s,errors="coerce").dropna()

def desired(px):
 r=px.pct_change();ma=px.rolling(200).mean();vol=r.rolling(20).std()*math.sqrt(TD)
 e=pd.Series(np.nan,index=px.index)
 for i in range(1,len(px)):
  j=i-1
  if np.isfinite(ma.iloc[j]) and np.isfinite(vol.iloc[j]) and vol.iloc[j]>0:
   target=.35 if px.iloc[j]>ma.iloc[j] else 0.
   e.iloc[i]=float(np.clip(target/vol.iloc[j],0,3))
 return e

def simulate(name,px,cap,fee,band,margin_rate,cash_yield=True):
 irx=dl("^IRX",str(px.index.min().date()))/100/TD
 irx=irx.reindex(px.index).ffill().fillna(0)
 sig=desired(px);wealth=float(cap);peak=wealth;mdd=0.;n=0;sides=roll_sides=0;target_hold=0.;days_in=0;zero_days=0
 ret=px.pct_change().fillna(0)
 mult=SERIES[name]["mult"]
 for i in range(1,len(px)):
  notional=float(px.iloc[i-1])*mult
  actual=n*notional/max(wealth,1e-12)
  s=float(sig.iloc[i]) if np.isfinite(sig.iloc[i]) else actual
  if abs(s-actual)>=band:target_hold=s
  want=int(np.round(target_hold*wealth/max(notional,1e-12)))
  maxn=int(np.floor(wealth/max(notional*margin_rate,1e-12)))
  want=max(0,min(want,maxn))
  if want!=n:
   wealth-=abs(want-n)*fee;sides+=abs(want-n);n=want
  if i%63==0 and n:
   wealth-=2*n*fee;roll_sides+=2*n
  exposure=n*notional/max(wealth,1e-12)
  day=(float(irx.iloc[i]) if cash_yield else 0.) + exposure*float(ret.iloc[i])
  wealth*=max(1+day,0)
  peak=max(peak,wealth);mdd=min(mdd,wealth/peak-1);days_in+=int(n>0);zero_days+=int(n==0)
 years=(len(px)-1)/TD
 return {"series":name,"ticker":SERIES[name]["ticker"],"actual_contract_history":SERIES[name]["actual_contract"],
  "start":str(px.index.min().date()),"end":str(px.index.max().date()),"starting_capital":cap,"fee_per_side":fee,
  "band":band,"margin_fraction_of_notional":margin_rate,"years":years,"cagr":(wealth/cap)**(1/years)-1,
  "max_dd":mdd,"terminal":wealth,"contract_sides_per_year":(sides+roll_sides)/years,
  "pct_days_with_contract":days_in/(len(px)-1),"pct_days_zero_contract":zero_days/(len(px)-1)}

def main():
 rows=[]
 for name,spec in SERIES.items():
  px=dl(spec["ticker"],spec["start"])
  if px.empty:continue
  for cap in CAPS:
   for fee in FEES:
    for band in BANDS:
     for mr in MARGIN_RATES:rows.append(simulate(name,px,cap,fee,band,mr,True))
 df=pd.DataFrame(rows);df.to_csv(OUT/"results.csv",index=False)
 # best only by CAGR among each series/cap, plus robustness under all fee/band/margin assumptions
 best=df.sort_values("cagr",ascending=False).groupby(["series","starting_capital"]).head(1)
 robust=(df.groupby(["series","starting_capital"]).agg(median_cagr=("cagr","median"),worst_cagr=("cagr","min"),
         best_cagr=("cagr","max"),worst_max_dd=("max_dd","min"),median_max_dd=("max_dd","median"),
         min_days_in=("pct_days_with_contract","min"),max_days_in=("pct_days_with_contract","max")).reset_index())
 best.to_csv(OUT/"best_by_capital.csv",index=False);robust.to_csv(OUT/"robustness_by_capital.csv",index=False)
 (OUT/"manifest.json").write_text(json.dumps({
  "purpose":"Confirm 35/0 with real futures-derived continuous returns plus integer contract sizing.",
  "important_caveat":"Yahoo Finance continuous futures are not exchange settlement-grade continuous series; roll methodology is undocumented. NQ/ES pre-micro histories are counterfactual micro-sized implementations using actual large-contract continuous returns and the later micro multiplier.",
  "actual_micro_histories":"MNQ and MES rows labeled actual_contract_history=True begin at micro contract launch-era data in 2019.",
  "signal":"prior-day 200DMA + prior-day 20d realized futures vol; 35% target vol above trend, zero below; cap 3x.",
  "costs":"explicit per-side fees plus quarterly 2-side roll proxy; portfolio collateral earns ^IRX and futures P/L is added separately; margin fractions are scenario constraints, not broker quotes."
 },indent=2))
 print(robust.to_string(index=False))
 print(best[["series","starting_capital","cagr","max_dd","band","margin_fraction_of_notional","fee_per_side","pct_days_with_contract"]].to_string(index=False))
if __name__=="__main__":main()
