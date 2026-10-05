from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

TD=252; OUT=Path('results/futures_micro_actual'); OUT.mkdir(parents=True,exist_ok=True)
SPECS={'MES':{'proxy':'SPY','index':'^GSPC','mult':5.0},'MNQ':{'proxy':'QQQ','index':'^NDX','mult':2.0}}

def dl(t):
 x=yf.download(t,start='2019-05-06',end='2026-08-22',auto_adjust=True,progress=False,threads=False)
 if x.empty:return pd.Series(dtype=float)
 s=x['Close']; s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s; s.index=pd.to_datetime(s.index).tz_localize(None); return pd.to_numeric(s,errors='coerce').dropna()

def desired(proxy):
 r=proxy.pct_change(); ma=proxy.rolling(200).mean(); v=r.rolling(20).std()*math.sqrt(TD); out=pd.Series(np.nan,index=proxy.index)
 for i in range(len(proxy)):
  if pd.isna(ma.iloc[i]) or pd.isna(v.iloc[i]) or v.iloc[i]<=0:continue
  tv=.35 if proxy.iloc[i]>ma.iloc[i] else .12; out.iloc[i]=float(np.clip(tv/v.iloc[i],0,3))
 return out

def simulate(kind,capital,fee,cashfrac,band):
 sp=SPECS[kind]; p=dl(sp['proxy']); idx=dl(sp['index']); irx=dl('^IRX')/100/TD; ix=p.index.intersection(idx.index); p=p.reindex(ix); idx=idx.reindex(ix); irx=irx.reindex(ix).ffill().fillna(0); tg=desired(p)
 wealth=float(capital); peak=wealth; mdd=0.; nprev=0; exposure_target=0.; sides=0; roll_sides=0
 # P/L uses proxy total return excess of RF; contract notional uses actual index multiplier.
 ret=p.pct_change().fillna(0)
 for i in range(1,len(ix)):
  signal=float(tg.iloc[i-1]) if pd.notna(tg.iloc[i-1]) else exposure_target
  # actual exposure at prior close from current contract count
  notional=float(idx.iloc[i-1])*sp['mult']; actual=nprev*notional/max(wealth,1e-12)
  if abs(signal-actual)>=band: exposure_target=signal
  n=int(np.round(exposure_target*wealth/max(notional,1e-12)))
  # Conservative stressed IRA capacity: 25% notional initial margin * 125% IRA requirement.
  while n>0 and n*notional*.25*1.25>wealth: n-=1
  if n!=nprev: wealth-=abs(n-nprev)*fee; sides+=abs(n-nprev); nprev=n
  # quarterly roll proxy: close old + open new = 2 contract sides, every 63 sessions
  if i%63==0 and nprev: wealth-=2*abs(nprev)*fee; roll_sides+=2*abs(nprev)
  exposure=nprev*notional/max(wealth,1e-12)
  day=cashfrac*float(irx.iloc[i])+exposure*(float(ret.iloc[i])-float(irx.iloc[i])); wealth*=max(1+day,0)
  peak=max(peak,wealth); mdd=min(mdd,wealth/peak-1)
 years=(len(ix)-1)/TD
 return {'kind':kind,'starting_capital':capital,'fee_per_contract_side':fee,'cash_yield_fraction':cashfrac,'band_exposure':band,'years':years,'cagr':(wealth/capital)**(1/years)-1,'max_dd':mdd,'terminal':wealth,'contract_sides_per_year':(sides+roll_sides)/years}

def main():
 rows=[]
 for kind in SPECS:
  for cap in (25_000,50_000,100_000,250_000,500_000):
   for fee in (2.50,3.00,4.00,5.00):
    for cf in (0,.5,1):
     for b in (.15,.30,.45): rows.append(simulate(kind,cap,fee,cf,b))
 df=pd.DataFrame(rows); df.to_csv(OUT/'results.csv',index=False)
 (OUT/'manifest.json').write_text(json.dumps({'source':'Yahoo Finance adjusted closes; SPY/^GSPC for MES, QQQ/^NDX for MNQ, ^IRX cash proxy','MES_multiplier':5,'MNQ_multiplier':2,'period':'2019-05-06 through 2026-08-21','signal':'underlying 200DMA + 20d underlying realized vol; 35% bull/12% bear target; direct index exposure cap 0..3x; previous-close signal','IRA_constraint':'25% stressed initial margin * 125%; deliberately conservative proxy, not a claim about Schwab current per-contract margin','roll':'quarterly proxy every 63 sessions, 2 contract sides','fee_sensitivity':'all-in per-side scenario, not broker commission alone'},indent=2))
 print(df.sort_values(['kind','starting_capital','cagr'],ascending=[True,True,False]).groupby(['kind','starting_capital']).head(5).to_string(index=False))
if __name__=='__main__':main()
