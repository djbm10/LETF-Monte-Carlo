from __future__ import annotations
import json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import norm
warnings.filterwarnings('ignore')
OUT=Path("results/sector_macro_optionexp_20261106");OUT.mkdir(parents=True,exist_ok=True)
E3={"XLK":"TECL","SOXX":"SOXL","XLF":"FAS","XLV":"CURE","XBI":"LABU","VNQ":"DRN","XHB":"NAIL","XLU":"UTSL","XLY":"WANT","XRT":"RETL","XLI":"DUSL","IWM":"TNA"}
E2={"XLK":"ROM","SOXX":"USD","XLF":"UYG","XLV":"RXL","XLE":"DIG","XLY":"UCC","XLB":"UYM","XLI":"UXI","XLP":"UGE","VNQ":"URE"}
ASSET={"equity":["SPY","IWM","EFA","EEM"],"rates":["IEF","TLT","SHY"],"currency":["UUP","FXE","FXY","FXB","FXA"],"commodities":["GLD","SLV","DBC","DBA","USO","CPER"]}
OPTIONFUNDS=["QYLD","XYLD","RYLD","JEPI","JEPQ","DIVO","PBP","PUTW","QYLG","XYLG","TAIL"]
TICKERS=sorted(set(["SPY","QQQ","VOO","^IRX","^VIX","^VXN"]+list(E3)+list(E3.values())+list(E2)+list(E2.values())+sum(ASSET.values(),[])+OPTIONFUNDS))
def dl():
 d={}
 for j in range(0,len(TICKERS),12):
  b=TICKERS[j:j+12]
  try:
   z=yf.download(b,start="2003-01-01",end="2026-11-06",auto_adjust=True,progress=False,threads=False,group_by="ticker",timeout=40)
   for t in b:
    try:
     p=z[t]["Close"] if isinstance(z.columns,pd.MultiIndex) else z["Close"]
     if isinstance(p,pd.DataFrame):p=p.iloc[:,0]
     p=pd.to_numeric(p,errors="coerce").dropna()
     if len(p)>250:
      p.index=pd.to_datetime(p.index).tz_localize(None)
      d[t]=p[~p.index.duplicated()]
    except:pass
  except Exception as e:print("batch-fail",b,str(e)[:100],flush=True)
  print("downloaded",j+len(b),len(TICKERS),"okay",len(d),flush=True)
 if "SPY" not in d or "QQQ" not in d:raise RuntimeError("SPY/QQQ missing")
 p=pd.DataFrame(d).sort_index().loc[:"2026-11-05"];p.to_csv(OUT/"prices.csv")
 return p
def met(s):
 s=s.dropna()
 if len(s)<250:return None
 eq=(1+s).cumprod(); y=len(s)/252
 if eq.iloc[-1]<=0:return None
 return dict(start=str(s.index[0].date()),end=str(s.index[-1].date()),years=round(y,3),cagr=float(eq.iloc[-1]**(1/y)-1),max_dd=float((eq/eq.cummax()-1).min()),vol=float(s.std()*np.sqrt(252)),sharpe=float(s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else None,worst_day=float(s.min()),terminal=float(eq.iloc[-1]))
def sector(px,rf):
 ix=px.loc["2015-01-01":].index;month=ix.to_period("M");wk=ix.to_period("W")
 ret=px.reindex(ix).pct_change(fill_method=None)
 sp=px["SPY"].reindex(ix);spma=sp.rolling(200).mean()
 histories={};notes=[];sectorprices=[]
 for lev,pairs in [("3x",E3),("2x",E2)]:
  p={u:l for u,l in pairs.items() if u in px and l in px}
  under=px[list(p)].reindex(ix);lever=px[list(p.values())].reindex(ix)
  lr=ret[list(p.values())].rename(columns={v:k for k,v in p.items()})
  v20=lr.rolling(20).std()*np.sqrt(252)
  for cadence in ("M","W"):
   changer=(month!=month.shift()) if cadence=="M" else (wk!=wk.shift())
   for L in (63,126,252):
    score=(under/under.shift(L)-1)
    for n in (1,3,5):
     for gate in ("none","spy","self"):
      for size in ("full","vol25","vol35"):
       a=np.zeros((len(ix),len(p)),dtype="float32")
       keys=list(p)
       for i in range(254,len(ix)):
        if not changer[i]:a[i]=a[i-1];continue
        row=score.iloc[i-1]
        elig=[u for u in keys if pd.notna(row[u]) and pd.notna(v20.iloc[i-1][u]) and
              pd.notna(lever[p[u]].iloc[i-1]) and pd.notna(lever[p[u]].iloc[i-42])]
        if gate=="spy" and not (sp.iloc[i-1]>spma.iloc[i-1]):elig=[]
        if gate=="self":
         elig=[u for u in elig if under[u].iloc[i-1]>under[u].iloc[i-200:i].mean()]
        winners=sorted(elig,key=lambda u:row[u],reverse=True)[:n]
        if len(winners)<n:continue
        for u in winners:
         v=(1/n) if size=="full" else (1/n)*min(1,(.25 if size=="vol25" else .35)/max(float(v20.iloc[i-1][u]),.08))
         a[i,keys.index(u)]=v
       w=pd.DataFrame(a,index=ix,columns=keys)
       missing=((w>0)&lr.isna()).any(axis=1)
       if missing.any():continue
       tr=w.diff().abs().sum(axis=1).fillna(0)
       # 7bp transaction cost per dollar traded. Interest on cash only.
       rs=(w*lr).sum(axis=1)+rf.reindex(ix).fillna(0)*(1-w.sum(axis=1))-.0007*tr
       nm=f"sector_{lev}_{cadence}_{L}_{n}_{gate}_{size}"
       histories[nm]=rs
       notes.append(dict(strategy=nm,avg_exposure=float(w.sum(axis=1).loc["2019":].mean()),turnover_yr=float(tr.loc["2019":].sum()/(len(tr.loc["2019":])/252)),active_fraction=float((w.loc["2019":].sum(axis=1)>0).mean()),n_instruments=len(p)))
 return histories,pd.DataFrame(notes)
def crossasset(px,rf):
 ix=px.loc["2008-01-01":].index;monthly=ix.to_period("M");weekly=ix.to_period("W")
 used=sorted(set(sum(ASSET.values(),[])).intersection(px.columns))
 price=px[used].reindex(ix)
 rr=price.pct_change(fill_method=None);vol=rr.rolling(63).std()*np.sqrt(252)
 result={};note=[]
 for L in (63,126,252):
  signal=price/price.shift(L)-1
  for side in ("longflat","longshort"):
   for cadence in ("M","W"):
    changing=(monthly!=monthly.shift()) if cadence=="M" else (weekly!=weekly.shift())
    for target in (.10,.15,.20):
     w=np.zeros((len(ix),len(used)),dtype="float32")
     for i in range(254,len(ix)):
      if not changing[i]:w[i]=w[i-1];continue
      z=np.zeros(len(used),dtype="float32")
      for group,groupassets in ASSET.items():
       eligible=[t for t in groupassets if t in used and pd.notna(signal.at[ix[i-1],t]) and pd.notna(vol.at[ix[i-1],t]) and vol.at[ix[i-1],t]>.02]
       for t in eligible:
        direction=float(np.sign(signal.at[ix[i-1],t]))
        if side=="longflat" and direction<0:continue
        z[used.index(t)]=float(target/4/max(len(eligible),1)/max(float(vol.at[ix[i-1],t]),.04)*direction)
      if np.sum(abs(z))>2:z*=2/np.sum(abs(z))
      w[i]=z
     wt=pd.DataFrame(w,index=ix,columns=used)
     if ((wt!=0)&rr.isna()).any(axis=1).any():continue
     turnover=wt.diff().abs().sum(axis=1).fillna(0)
     # Effective financing: collateral yield + spot ETF PnL - funding on gross notional.
     # Long ETF positions are cash equities, short ETF shares have 2% borrowing penalty.
     pnl=(wt*rr).sum(axis=1)+rf.reindex(ix).fillna(0)*(1-wt.sum(axis=1))-.0005*turnover-.02/252*wt.clip(upper=0).abs().sum(axis=1)
     nm=f"macro_{L}_{side}_{cadence}_{int(target*100)}";result[nm]=pnl
     note.append(dict(strategy=nm,avg_gross_exposure=float(wt.abs().sum(axis=1).loc["2019":].mean()),turnover_yr=float(turnover.loc["2019":].sum()/(len(turnover.loc["2019":])/252))))
 return result,pd.DataFrame(note)
def bs(s,k,t,v,rf,kind):
 if t<=0:return max(k-s,0) if kind=="put" else max(s-k,0)
 v=max(float(v),.05);rf=float(rf)
 d1=(math.log(s/k)+(rf+.5*v*v)*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
 if kind=="put":return k*math.exp(-rf*t)*norm.cdf(-d2)-s*norm.cdf(-d1)
 return s*norm.cdf(d1)-k*math.exp(-rf*t)*norm.cdf(d2)
def optionmodels(px,rf):
 result={}
 for index,vix in [("SPY","^VIX"),("QQQ","^VXN")]:
  if index not in px or vix not in px:continue
  ix=px.loc["2011-01-01":].index;pr=px[index].reindex(ix);v=(px[vix].reindex(ix)/100).ffill().clip(.1,1.5)
  ma=pr.rolling(200).mean()
  for kind in ("cash_put","covered_call","protective_put"):
   for delta in (.15,.30):
    for gated in (False,True):
     for ivmult in (1.0,1.2):
      values=[];cash=1.;hold=0.;short=0.;k=0.;days=0;last=-999;position=None
      for i,date in enumerate(ix):
       s=pr.iloc[i];vv=v.iloc[i];rate=float(rf.reindex(ix).fillna(.025/252).iloc[i]*252)
       if not np.isfinite(s) or not np.isfinite(vv):values.append((date,np.nan));continue
       if i:cash*=1+rate/252
       vol=float(vv)*ivmult
       liability=0
       if position:
        days=max(days-1,0)
        opt=bs(s,k,days/252,vol+(.03 if position=="put" else 0),rate,position)
        liability=short*opt
        if days==0:
         if position=="put":cash-=short*max(k-s,0)
         else:cash-=short*max(s-k,0)
         liability=0;short=0;position=None
       wealth=cash+hold*s-liability
       if i>=250 and position is None and i-last>=21:
        last=i
        eligible=(not gated) or (pr.iloc[i-1]>ma.iloc[i-1])
        if eligible:
         days=21;tt=days/252
         if kind=="cash_put":
          iv=vol+.03;d=-norm.ppf(delta)
          k=float(s*math.exp((rate+.5*iv*iv)*tt-d*iv*math.sqrt(tt)))
          credit=bs(s,k,tt,iv,rate,"put")*.97
          short=max(wealth,0)/s;cash+=short*credit;hold=0;position="put"
         elif kind=="covered_call":
          hold=wealth/s;cash=0
          d=norm.ppf(delta)
          k=float(s*math.exp((rate+.5*vol*vol)*tt-d*vol*math.sqrt(tt)))
          short=hold;cash+=short*bs(s,k,tt,vol,rate,"call")*.97;position="call"
         else:
          # protective long put, represented by negative short liability, marked daily with a long option.
          hold=wealth/s;cash=0
          iv=vol+.03;d=-norm.ppf(delta)
          k=float(s*math.exp((rate+.5*iv*iv)*tt-d*iv*math.sqrt(tt)))
          short=-hold;cash+=short*bs(s,k,tt,iv,rate,"put")*1.03;position="put"
       opt=(bs(s,k,days/252,vol+(.03 if position=="put" else 0),rate,position) if position else 0)
       eq=cash+hold*s-short*opt
       values.append((date,max(eq,1e-9)))
      eq=pd.Series(dict(values)).dropna()
      if len(eq)>=500:result[f"options_MODEL_{index}_{kind}_d{int(delta*100)}_{'gate' if gated else 'always'}_iv{int(ivmult*100)}"]=eq.pct_change(fill_method=None)
 return result
def main():
 px=dl();rf=(px["^IRX"]/100/252 if "^IRX" in px else pd.Series(.025/252,index=px.index)).reindex(px.index).ffill().fillna(.025/252)
 q=px.QQQ;qret=q.pct_change(fill_method=None);vr=(qret-rf).rolling(20).std()*np.sqrt(252)
 e=((q>q.rolling(200).mean()).shift(1).fillna(False)).astype(float)*(.35/vr.clip(lower=.04)).clip(upper=3).shift(1).fillna(0)
 allr={"35_0_QQQ_proxy":rf+e*(qret-rf)-.0002*e.diff().abs().fillna(e.abs()),"SPY":px.SPY.pct_change(fill_method=None),"QQQ":qret}
 if "VOO" in px:allr["VOO"]=px.VOO.pct_change(fill_method=None)
 print("sector run",flush=True);sr,si=sector(px,rf);allr.update(sr);si.to_csv(OUT/"sector_metadata.csv",index=False)
 print("macro run",flush=True);mr,mi=crossasset(px,rf);allr.update(mr);mi.to_csv(OUT/"macro_metadata.csv",index=False)
 print("options run",flush=True);allr.update(optionmodels(px,rf))
 for t in OPTIONFUNDS:
  if t in px:allr["options_FUND_"+t]=px[t].pct_change(fill_method=None)
 series=pd.DataFrame(allr).loc[:"2026-11-05"];series.to_csv(OUT/"daily_returns.csv")
 rows=[]
 for period,start,end in [("2019plus","2019-01-01","2026-11-05"),("2021plus","2021-01-01","2026-11-05"),("2016plus","2016-01-01","2026-11-05")]:
  for name in series:
   s=series.loc[start:end,name].dropna()
   total=len(series.loc[start:end])
   if len(s)<250 or len(s)/total<.95:continue
   m=met(s)
   if m:rows.append({"window":period,"strategy":name,**m})
 z=pd.DataFrame(rows);z.to_csv(OUT/"summary.csv",index=False)
 manifest={"status":"EXPLORATORY only. No fresh untouched holdout; previous sector family KILL remains in force.",
 "input":"Actual Yahoo adjusted closes through 2026-11-05, exclusive end 2026-11-06",
 "sector":"Actual 2x/3x daily-leveraged ETF price series (inception respected), rank unlevered ETF underlyings. Weekly/monthly rebalance, 7bp per traded dollar. Closes used for signal and approximate next-session position; no true open execution.",
 "macro":"Broad ETF proxy asset classes, not continuous futures. Long/short equity short-sale margin and borrow simplified; 2pc year short borrow cost assumed.",
 "options":"MODEL = synthetic Black-Scholes repriced at historical VIX/VXN, and 3pc premium trading friction, NOT real bid/ask and NO options-chain validation. FUND = actual fund distribution-adjusted price history, active management/rules vary.",
 "multipletesting":"Hundreds of exploratory variants scanned; the best observation is heavily selection-biased.",
 "limits":["No verified dated futures settlement PnL","No dated options chains","Historical splits adjusted via Yahoo","Weekly rebalance signals last close and execution approximated","Personal tax omitted"]}
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
 print("COMPLETED",len(allr),"variant result rows",len(z),flush=True)
 for group in ("sector_3x","sector_2x","macro_","options_"):
  a=z[(z.window=="2019plus")&z.strategy.str.startswith(group)].sort_values("cagr",ascending=False).head(8)
  print("TOP",group,a[["strategy","cagr","max_dd","vol"]].to_string(index=False),flush=True)
if __name__=="__main__":main()
