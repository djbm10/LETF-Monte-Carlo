from __future__ import annotations
import io,json,zipfile,warnings,math
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf,requests
warnings.filterwarnings('ignore')
OUT=Path("results/growth20_50_pt_membership_20261106");OUT.mkdir(parents=True,exist_ok=True)
SOURCE="https://github.com/pierrebrunelle/sp500-historical-constituents/archive/refs/heads/main.zip"
def load_membership():
 r=requests.get(SOURCE,timeout=90);r.raise_for_status()
 z=zipfile.ZipFile(io.BytesIO(r.content));members={}
 for name in z.namelist():
  base=Path(name).name
  if not (base.startswith("spy") and len(base)==9 and base[3:].isdigit()):continue
  ym=base[3:]
  if "201601"<=ym<="202604":
   members[ym]=sorted(set(t.split("\t")[0].strip().replace("/","-") for t in z.read(name).decode("utf-8-sig").splitlines() if t.strip()))
 if len(members)<120:raise RuntimeError("Historical S&P membership source incomplete")
 (OUT/"monthly_point_in_time_membership.json").write_text(json.dumps(members))
 return members
def download_prices(members):
 syms=sorted(set(sum(members.values(),[])))
 universe=sorted(set(syms+["SPY","QQQ","VOO","^IRX"]))
 print("unique historical stocks",len(syms),flush=True)
 histories={};errors=[]
 for j in range(0,len(universe),22):
  chunk=universe[j:j+22]
  for attempt in range(2):
   try:
    d=yf.download(chunk,start="2015-01-01",end="2026-05-01",auto_adjust=True,progress=False,threads=False,group_by="ticker",timeout=45)
    ok=0
    for t in chunk:
     try:
      s=d[t]["Close"] if isinstance(d.columns,pd.MultiIndex) else d["Close"]
      if isinstance(s,pd.DataFrame):s=s.iloc[:,0]
      s=pd.to_numeric(s,errors="coerce").dropna()
      if len(s)>45:
       s.index=pd.to_datetime(s.index).tz_localize(None)
       histories[t]=s[~s.index.duplicated()]
       ok+=1
     except:pass
    if ok:break
   except Exception as e:
    if attempt==1:errors.append([chunk,str(e)[:150]])
  print("price_batch",j+len(chunk),len(universe),"successful",len(histories),flush=True)
 if not {"SPY","QQQ"}.issubset(histories):raise RuntimeError("Missing SPY or QQQ, cannot benchmark")
 p=pd.DataFrame(histories).sort_index()
 p.to_pickle(OUT/"downloaded_adjusted_prices.pkl")
 (OUT/"download_errors.json").write_text(json.dumps(errors))
 return p,syms
def metrics(s):
 s=s.dropna()
 if len(s)<250:return {}
 eq=(1+s).cumprod();years=len(s)/252
 return dict(cagr=float(eq.iloc[-1]**(1/years)-1),max_dd=float((eq/eq.cummax()-1).min()),annual_vol=float(s.std()*np.sqrt(252)),worst_day=float(s.min()),terminal=float(eq.iloc[-1]),years=float(years))
def run():
 members=load_membership();px,stocks=download_prices(members)
 dates=px.loc["2016-01-01":"2026-04-30"].index
 px=px.reindex(columns=stocks)
 returns=px.pct_change(fill_method=None)
 ma=px.rolling(200).mean()
 mom6=px/px.shift(126)-1
 mom12_1=px.shift(21)/px.shift(252)-1
 v=returns.rolling(63).std()*np.sqrt(252)
 high=px/px.rolling(252).max()
 # All screens use price only. No fundamental high-growth proof, no point-in-time sales/earnings data.
 signals={"mom6":mom6,"mom12_1":mom12_1,"mom_per_risk":mom12_1/v.clip(lower=.1),"near_52w_high":high,
          "mom12_trend":mom12_1.where(px>ma)}
 values={};trade_records=[];coverage=[];risk_records=[]
 dateindex=px.index;months=dates.to_period("M");quarters=dates.to_period("Q")
 bench=px.reindex(columns=["SPY"]) if "SPY" in px else None
 for screen,scores in signals.items():
  for count in (20,50):
   for cadence in ("M","Q"):
    for missing in ("optimistic_zero","conservative_full_loss"):
     cash=1.;holdings={};equity=1.;last_px={};series=[];gap_days=0;missed=0;reb=0;active=0;selected_count=[];sum_turn=0.
     for j,date in enumerate(dates):
      signaldate=dateindex[dateindex.get_loc(date)-1] if dateindex.get_loc(date)>0 else None
      monthprev=(date-pd.offsets.MonthBegin(1)).strftime("%Y%m")
      membership_month=monthprev if monthprev in members else max((m for m in members if m<=monthprev),default=None)
      trigger=j==0 or ((months[j]!=months[j-1]) if cadence=="M" else (quarters[j]!=quarters[j-1]))
      if trigger and signaldate is not None and membership_month:
       avail=members[membership_month]
       known=[s for s in avail if s in px and pd.notna(px.at[signaldate,s])]
       frac=len(known)/len(avail)
       coverage.append({"date":str(date.date()),"screen":screen,"n":count,"cadence":cadence,"coverage":frac,"members":len(avail),"price_known":len(known)})
       # If too many unavailable historical names, fail-close rather than silently select survivors.
       if frac>=.80:
        ranked=scores.loc[signaldate,known].dropna().sort_values(ascending=False)
        picked=ranked.head(count).index.tolist()
        if len(picked)==count:
         # Positions are rebalanced at the prior close, then the current day's close-to-close move is applied.
         equity=cash+sum(holdings.values())
         curr={k:z/equity for k,z in holdings.items() if equity>0}
         desired={t:1/count for t in picked}
         turn=sum(abs(desired.get(t,0)-curr.get(t,0)) for t in set(curr)|set(desired))
         sum_turn+=turn
         equity*=1-turn*.001
         holdings={t:equity/count for t in picked}
         cash=0.;reb+=1;selected_count.append(count)
         trade_records.append({"date":str(date.date()),"strategy":f"{screen}_{count}_{cadence}_{missing}","selected":",".join(picked),"turnover":turn})
       else:missed+=1
      rf=.025/252
      if "^IRX" in px:
       rv=px["^IRX"].loc[date] if date in px.index else np.nan
       if np.isfinite(rv):rf=max(0,float(rv)/100/252)
      cash*=1+rf
      next_hold={}
      had_gap=False
      for t,dollars in holdings.items():
       daily=returns.at[date,t] if t in returns.columns else np.nan
       if pd.notna(daily):
        next_hold[t]=max(0,float(dollars)*(1+float(daily)))
       else:
        had_gap=True
        if missing=="optimistic_zero":next_hold[t]=dollars
        else:next_hold[t]=0.
      if had_gap:gap_days+=1
      holdings=next_hold
      wealth=cash+sum(holdings.values())
      if wealth<0:raise RuntimeError("negative simulated wealth")
      dailyret=wealth/equity-1 if equity>0 else -1
      equity=wealth
      series.append((date,dailyret))
      if holdings:active+=1
     nm=f"growth_PRICEONLY_{screen}_N{count}_{cadence}_{missing}"
     vals=pd.Series(dict(series));values[nm]=vals
     risk_records.append(dict(strategy=nm,selected_rebalances=reb,failed_coverage_rebalances=missed,days_with_selected_missing_price=gap_days,active_days=active,number_of_membership_snapshots=len(members),oneway_turnover_annualized=sum_turn/(len(dates)/252),**metrics(vals)))
 f=pd.DataFrame(values);f.to_csv(OUT/"growth_daily_returns.csv")
 k=pd.DataFrame(risk_records);k.to_csv(OUT/"strategy_comparison.csv",index=False)
 pd.DataFrame(trade_records).to_csv(OUT/"selected_stock_trade_audit.csv",index=False)
 cv=pd.DataFrame(coverage);cv.to_csv(OUT/"membership_price_coverage.csv",index=False)
 (OUT/"manifest.json").write_text(json.dumps({
  "source_universe":"Monthly S&P 500 historical membership reconstructed from Wikipedia additions/deletions, source pierrebrunelle/sp500-historical-constituents, January 2016 to April 2026",
  "source_prices":"Yahoo adjusted closing prices downloaded 2015-2026. Some delisted symbols do not resolve; missing-price bounds shown.",
  "asof":"2026-04-30, historical membership public source stops April 2026",
  "strategy":"Five PRICE-ONLY growth-style screens. Absolutely not actual point-in-time revenue or earnings growth. These screens cannot validate fundamental stock growth.",
  "holdings":"20 positions 5pct each or 50 positions 2pct each, monthly/quarterly reselection; drift between rebalance; 10bp transaction cost per dollar one-way",
  "missing_returns":"Compare optimistic missing stock return=0 vs conservative -100pct for affected position on first missing day; incomplete options include ticker changes, acquisitions, bankruptcies, unresolved price history.",
  "execution":"Prior close signal applies to next close-to-close return; no next open fill prices or taxes",
  "disposition":"Exploratory sensitivity only; PIT membership data not equivalent to delisting-safe CRSP returns or true PIT corporate fundamentals; no holdout unexamined"},indent=2))
 print("GROWTH COMPLETE",len(k),"variants",len(stocks),"historical tickers",flush=True)
 print(k.sort_values("cagr",ascending=False).head(15)[["strategy","cagr","max_dd","days_with_selected_missing_price","selected_rebalances"]].to_string(index=False),flush=True)
if __name__=="__main__":run()
