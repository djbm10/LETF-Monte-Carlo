from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252; OUT=Path('results/s9_self_financing'); OUT.mkdir(parents=True,exist_ok=True)

def dl(t):
 x=yf.download(t,start='2010-02-01',end='2026-08-22',auto_adjust=True,progress=False,threads=False)
 if x.empty: raise RuntimeError(t)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(t,axis=1,level=1) if t in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.index=pd.to_datetime(x.index).tz_localize(None); return x[['Open','Close']].apply(pd.to_numeric,errors='coerce').dropna()

def targets(spy,tq):
 ix=spy.index.intersection(tq.index); spy=spy.reindex(ix); tq=tq.reindex(ix)
 c=spy.Close; tr=tq.Close.pct_change(); ma=c.rolling(200).mean(); vol=tr.rolling(20).std()*math.sqrt(TD); out=pd.Series(np.nan,index=ix)
 for i in range(len(ix)):
  if pd.isna(ma.iloc[i]) or pd.isna(vol.iloc[i]) or vol.iloc[i]<=0: continue
  tv=.35 if c.iloc[i]>ma.iloc[i] else .12; out.iloc[i]=float(np.clip(tv/vol.iloc[i],0,1))
 return out

def run(spy,tq,rf,band=.10,cash_frac=0.0,cost=.0003,extra_signal_lag=0,whole=False):
 ix=spy.index.intersection(tq.index); spy=spy.reindex(ix); tq=tq.reindex(ix); rf=rf.reindex(ix).ffill().fillna(0); tg=targets(spy,tq)
 cash=1.0; shares=0.; equity=[]; trades=0; turnover=0.; weights=[]; targets_used=[]
 # signal from prior close trades at today's open. extra_signal_lag=1 means two closes old.
 for i,d in enumerate(ix):
  op=float(tq.Open.iloc[i]); cl=float(tq.Close.iloc[i]);
  # cash accrues daily before open, approximate evenly over calendar/trading day
  cash*=1+cash_frac*float(rf.iloc[i])
  wealth_open=cash+shares*op
  actual=shares*op/max(wealth_open,1e-15)
  sig_i=i-1-extra_signal_lag
  desired=float(tg.iloc[sig_i]) if sig_i>=0 and pd.notna(tg.iloc[sig_i]) else actual
  if abs(desired-actual)>=band:
   target_value=desired*wealth_open; delta_value=target_value-shares*op
   q=delta_value/op
   if whole:
    # nearest whole-share target without borrowing cash
    target_shares=max(0,math.floor(target_value/op+1e-12)); q=target_shares-shares; delta_value=q*op
   fee=abs(delta_value)*cost
   # sell/buy at open, fee from cash; if fee makes cash slightly negative, trim buy
   cash-=delta_value+fee; shares+=q
   if cash<0 and shares>0:
    trim=min(shares,-cash/op); shares-=trim; cash+=trim*op
   trades+=1; turnover+=abs(delta_value)/max(wealth_open,1e-15)
  wealth_close=cash+shares*cl; equity.append(wealth_close); weights.append(shares*cl/max(wealth_close,1e-15)); targets_used.append(desired)
 e=pd.Series(equity,index=ix); ret=e.pct_change().dropna(); years=len(ret)/TD; dd=e/e.cummax()-1
 return dict(cagr=(e.iloc[-1]/e.iloc[0])**(1/years)-1,max_dd=float(dd.min()),vol=float(ret.std()*math.sqrt(TD)),terminal=float(e.iloc[-1]),trades_per_year=trades/years,turnover_per_year=turnover/years,median_weight=float(np.median(weights)),mean_weight=float(np.mean(weights))), pd.DataFrame({'equity':e,'actual_close_weight':weights,'desired_target':targets_used},index=ix)

def ideal(spy,tq,band=.10,cost=.0003):
 ix=spy.index.intersection(tq.index); r=tq.Close.reindex(ix).pct_change().fillna(0); tg=targets(spy,tq).shift(1); a=pd.Series(0.,index=ix)
 for i in range(1,len(ix)):
  want=tg.iloc[i] if pd.notna(tg.iloc[i]) else a.iloc[i-1]; a.iloc[i]=want if abs(want-a.iloc[i-1])>=band else a.iloc[i-1]
 rr=a*r-cost*a.diff().abs().fillna(a.abs()); e=(1+rr).cumprod(); y=(len(rr)-1)/TD; dd=e/e.cummax()-1
 return dict(cagr=e.iloc[-1]**(1/y)-1,max_dd=float(dd.min()),terminal=float(e.iloc[-1]),trades_per_year=float((a.diff().abs()>1e-12).sum()/y),turnover_per_year=float(a.diff().abs().sum()/y))

def main():
 spy=dl('SPY'); tq=dl('TQQQ'); irx=dl('^IRX').Close/100/TD
 rows=[]
 for band in (.05,.10,.15):
  for cash_frac in (0,.5,1):
   for lag in (0,1):
    m,_=run(spy,tq,irx,band,cash_frac,.0003,lag,False); rows.append({'mode':'fractional_next_open','band':band,'cash_yield_fraction':cash_frac,'extra_signal_lag':lag,**m})
 # whole share sensitivity at preferred 10pp band
 for cash_frac in (0,1):
  m,_=run(spy,tq,irx,.10,cash_frac,.0003,0,True); rows.append({'mode':'whole_share_next_open','band':.10,'cash_yield_fraction':cash_frac,'extra_signal_lag':0,**m})
 rows.append({'mode':'old_idealized_close_to_close','band':.10,'cash_yield_fraction':0,'extra_signal_lag':0,**ideal(spy,tq,.10,.0003)})
 df=pd.DataFrame(rows); df.to_csv(OUT/'results.csv',index=False)
 _,trace=run(spy,tq,irx,.10,0,.0003,0,False); trace.to_csv(OUT/'preferred_trace.csv')
 (OUT/'manifest.json').write_text(json.dumps({'period':[str(spy.index.min().date()),str(spy.index.max().date())],'source':'Yahoo Finance adjusted OHLC via yfinance','execution':'signal computed from completed close; trade next regular-session open; actual portfolio weight drifts naturally between trades; trade only when actual open weight differs from target by threshold','cost':'3bp per dollar traded','cash_yield':'0/50/100% of 13-week T-bill proxy (^IRX)','fractional':'fractional shares primary; whole-share sensitivity included'},indent=2))
 print(df.to_string(index=False))
if __name__=='__main__': main()
