from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252; TC=.0003; OUT=Path('results/sector_vehicle_s9'); OUT.mkdir(parents=True,exist_ok=True)
PAIRS={
 'UPRO':'SPY','SPXL':'SPY','TQQQ':'QQQ','TECL':'XLK','SOXL':'SOXX','FAS':'XLF','CURE':'XLV',
 'TNA':'IWM','MIDU':'MDY','UDOW':'DIA','DRN':'VNQ','RETL':'XRT','LABU':'XBI','NAIL':'XHB',
 'UTSL':'XLU','WANT':'XLY','WEBL':'FDN'
}

def dl(t):
 x=yf.download(t,start='2006-01-01',end='2026-08-22',auto_adjust=True,progress=False,threads=False)
 if x.empty:return pd.Series(dtype=float)
 s=x['Close']; s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s; s.index=pd.to_datetime(s.index).tz_localize(None); return pd.to_numeric(s,errors='coerce').dropna()

def target_alloc(signal,lev,band=.10):
 ix=signal.index.intersection(lev.index); signal=signal.reindex(ix); lev=lev.reindex(ix); lr=lev.pct_change(); ma=signal.rolling(200).mean(); v=lr.rolling(20).std()*math.sqrt(TD)
 want=pd.Series(0.,index=ix)
 for i in range(1,len(ix)):
  if pd.isna(ma.iloc[i-1]) or pd.isna(v.iloc[i-1]) or v.iloc[i-1]<=0:continue
  tv=.35 if signal.iloc[i-1]>ma.iloc[i-1] else .12
  want.iloc[i]=float(np.clip(tv/v.iloc[i-1],0,1))
 a=pd.Series(0.,index=ix)
 for i in range(1,len(ix)): a.iloc[i]=want.iloc[i] if abs(want.iloc[i]-a.iloc[i-1])>=band else a.iloc[i-1]
 return a

def met(r):
 r=pd.Series(r).dropna(); e=(1+r).cumprod(); y=len(r)/TD; dd=e/e.cummax()-1
 return {'years':y,'cagr':e.iloc[-1]**(1/y)-1,'max_dd':dd.min(),'vol':r.std()*math.sqrt(TD),'terminal_multiple':e.iloc[-1]}

def run(lev,proxy,spy):
 ix=lev.index.intersection(proxy.index).intersection(spy.index); lev=lev.reindex(ix); proxy=proxy.reindex(ix); spy=spy.reindex(ix); lr=lev.pct_change().fillna(0)
 out=[]
 for sig_name,sig in [('SPY_signal',spy),('underlying_signal',proxy)]:
  a=target_alloc(sig,lev,.10); rr=a*lr-TC*a.diff().abs().fillna(a.abs()); out.append((sig_name,rr,a))
 return out

def main():
 spy=dl('SPY'); cache={'SPY':spy}; rows=[]; allocrows=[]
 for lev,proxy in PAIRS.items():
  L=cache.setdefault(lev,dl(lev)); P=cache.setdefault(proxy,dl(proxy))
  if L.empty or P.empty: continue
  for sig,rr,a in run(L,P,spy):
   m=met(rr); rows.append({'leveraged_etf':lev,'proxy':proxy,'signal':sig,'start':str(rr.index.min().date()),'end':str(rr.index.max().date()),**m,
      'changes_per_year':float((a.diff().abs()>1e-12).sum()/(len(a)/TD)),'turnover_per_year':float(a.diff().abs().sum()/(len(a)/TD))})
  # buy & hold
  r=L.pct_change().dropna(); rows.append({'leveraged_etf':lev,'proxy':proxy,'signal':'buy_hold','start':str(r.index.min().date()),'end':str(r.index.max().date()),**met(r),'changes_per_year':0,'turnover_per_year':0})
 df=pd.DataFrame(rows); df.to_csv(OUT/'vehicle_results.csv',index=False)
 # Common 2018+ comparison to reduce inception-date bias
 common=[]
 for _,z in df.iterrows():
  lev=z.leveraged_etf; proxy=z.proxy; sig=z.signal; L=cache[lev].loc['2018-01-01':]; P=cache[proxy].loc['2018-01-01':]; S=spy.loc['2018-01-01':]
  if len(L)<500: continue
  if sig=='buy_hold': rr=L.pct_change().dropna()
  else:
   rr,a=next((r,a) for name,r,a in run(L,P,S) if name==sig)
  common.append({'leveraged_etf':lev,'proxy':proxy,'signal':sig,**met(rr)})
 pd.DataFrame(common).to_csv(OUT/'common_2018plus.csv',index=False)
 (OUT/'manifest.json').write_text(json.dumps({'source':'Yahoo Finance via yfinance adjusted closes','rule':'S9-10: 35/12 target vol, 20d realized leveraged ETF vol, 200DMA signal, 10 percentage point allocation band, prior-day information','cost':'3bp per 100% allocation turnover','pairs':PAIRS},indent=2))
 print(df.sort_values('cagr',ascending=False).head(40).to_string(index=False)); print('\nCOMMON 2018+\n',pd.DataFrame(common).sort_values('cagr',ascending=False).head(40).to_string(index=False))
if __name__=='__main__':main()
