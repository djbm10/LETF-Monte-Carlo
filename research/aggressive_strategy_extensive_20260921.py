from __future__ import annotations
import argparse, hashlib, io, json, math, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

TD=252
FF_URL="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
BLOCK=63; WARM=252
LETF_EXPENSE=.0091; ONE_X_EXPENSE=.0003; FIN_SPREAD=.005
LETF_TC=.0003; FUT_TC=.0002
HORIZONS=(10,20,40)

@dataclass(frozen=True)
class Regime:
    name:str; mode:str; drift_ann:float=0.; vol_scale:float=1.
    rf_floor:float|None=None; rf_cap:float|None=None
    crash_every_years:float|None=None

REGIMES=(
 Regime("baseline_history","baseline"),
 Regime("high_inflation_1966_1982","high_inflation"),
 Regime("crisis_heavy_empirical","crisis"),
 Regime("productivity_bull_empirical","productivity"),
 Regime("lost_decade_empirical","lost_decade"),
 Regime("high_rate_low_return","baseline",drift_ann=-.03,rf_floor=.05),
 Regime("stagflation_stress","high_inflation",drift_ann=-.02,vol_scale=1.15,rf_floor=.05),
 Regime("ai_boom_high_vol","baseline",drift_ann=.02,vol_scale=1.15,rf_floor=.03),
 Regime("soft_landing_disinflation","baseline",drift_ann=.005,vol_scale=.85,rf_cap=.04),
 Regime("combined_hostile","crisis_mixed",drift_ann=-.04,vol_scale=1.20,rf_floor=.05,crash_every_years=4.),
 Regime("sequence_shock_first5y","sequence_shock",drift_ann=-.01,vol_scale=1.10),
)

def load_ff():
 req=urllib.request.Request(FF_URL,headers={"User-Agent":"Mozilla/5.0"})
 with urllib.request.urlopen(req,timeout=60) as r: blob=r.read()
 sha=hashlib.sha256(blob).hexdigest(); rows=[]
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  raw=z.read([n for n in z.namelist() if n.lower().endswith(".csv")][0]).decode("latin-1")
 for line in raw.splitlines():
  q=[x.strip() for x in line.split(",")]
  if len(q)>=5 and len(q[0])==8 and q[0].isdigit():
   try: rows.append((pd.to_datetime(q[0],format="%Y%m%d"),(float(q[1])+float(q[4]))/100,float(q[4])/100))
   except ValueError: pass
 d=pd.DataFrame(rows,columns=["date","mkt","rf"]).set_index("date").sort_index().loc["1926-07-01":]
 if len(d)<20000: raise RuntimeError(f"Fama-French history too short: {len(d)}")
 return d,{"source":FF_URL,"sha256":sha,"start":str(d.index.min().date()),"end":str(d.index.max().date()),"rows":len(d)}

def levret(mkt,rf,L,expense=None,spread=FIN_SPREAD):
 mkt=np.asarray(mkt,float);rf=np.asarray(rf,float)
 if expense is None: expense=ONE_X_EXPENSE if L==1 else LETF_EXPENSE
 return np.maximum(L*mkt-max(L-1,0)*(rf+spread/TD)-expense/TD,-1.)

def rollmean1(x,w): return pd.Series(x).rolling(w,min_periods=w).mean().to_numpy()
def rollvol1(x,w=20): return pd.Series(x).rolling(w,min_periods=w).std().to_numpy()*math.sqrt(TD)

def metrics(r):
 r=np.asarray(r,float);r=r[np.isfinite(r)]
 eq=np.cumprod(np.maximum(1+r,0));yrs=len(r)/TD
 peak=np.maximum.accumulate(eq);dd=eq/peak-1;cur=mx=0;hi=-np.inf
 for v in eq:
  if v>=hi: hi=v;cur=0
  else: cur+=1;mx=max(mx,cur)
 return {"cagr":eq[-1]**(1/yrs)-1 if eq[-1]>0 else -1.,"max_drawdown":float(dd.min()),
         "max_underwater_years":mx/TD,"terminal":float(eq[-1]),"vol":float(np.std(r,ddof=1)*math.sqrt(TD))}

def futures_target(px,mkt,rf,bull=.35,bear=.12,sma=200,cap=3.):
 vol=rollvol1(mkt-rf);ma=rollmean1(px,sma);e=np.zeros(len(px))
 for i in range(1,len(px)):
  if np.isfinite(vol[i-1]) and vol[i-1]>0 and np.isfinite(ma[i-1]):
   e[i]=np.clip((bull if px[i-1]>ma[i-1] else bear)/vol[i-1],0,cap)
 return e

def futures_returns(mkt,rf,e,tc=FUT_TC):
 e=np.asarray(e,float);return rf+e*(mkt-rf)-tc*np.abs(np.diff(e,prepend=0.))

def letf_s9(px,r3,rf,bull=.35,bear=.12,sma=200):
 ma=rollmean1(px,sma);vol=rollvol1(r3);a=np.zeros(len(px))
 for i in range(1,len(px)):
  if np.isfinite(ma[i-1]) and np.isfinite(vol[i-1]) and vol[i-1]>0:
   a[i]=np.clip((bull if px[i-1]>ma[i-1] else bear)/vol[i-1],0,1)
 return a*r3+(1-a)*rf-LETF_TC*np.abs(np.diff(a,prepend=0.))

def trend_exposure(px,above=2.,sma=200):
 ma=rollmean1(px,sma);e=np.zeros(len(px))
 for i in range(1,len(px)):
  if np.isfinite(ma[i-1]):e[i]=above if px[i-1]>ma[i-1] else 0.
 return e

def dip_exposure(px,confirmed=True,max_exp=1.5,hold_days=60):
 s=pd.Series(px);high=s.rolling(252,min_periods=126).max();low20=s.rolling(20,min_periods=20).min()
 ma200=s.rolling(200,min_periods=200).mean();r5=s.pct_change(5);dd=s/high-1;rebound=s/low20-1
 e=np.ones(len(px));timer=0
 for i in range(1,len(px)):
  j=i-1
  if confirmed:
   trig=(np.isfinite(dd.iloc[j]) and dd.iloc[j]<=-.10 and np.isfinite(rebound.iloc[j]) and rebound.iloc[j]>=.05
         and np.isfinite(r5.iloc[j]) and r5.iloc[j]>0 and np.isfinite(ma200.iloc[j]) and s.iloc[j]>ma200.iloc[j])
   if trig:timer=hold_days
   e[i]=max_exp if timer>0 else 1.;timer=max(timer-1,0)
  elif np.isfinite(dd.iloc[j]):
   e[i]=min(max_exp,2.) if dd.iloc[j]<=-.30 else min(max_exp,1.75) if dd.iloc[j]<=-.20 else min(max_exp,1.5) if dd.iloc[j]<=-.10 else 1.
 return e

def historical_strategies(ff):
 m=ff.mkt.to_numpy();rf=ff.rf.to_numpy();px=np.cumprod(np.maximum(1+m,1e-12));r3=levret(m,rf,3)
 out={"MKT_1x":m,"LETF_1p5_BH":levret(m,rf,1.5),"LETF_2x_BH":levret(m,rf,2),"LETF_3x_BH":r3,
      "LETF_S9_35_12":letf_s9(px,r3,rf,.35,.12),"LETF_S9_35_0":letf_s9(px,r3,rf,.35,0.),
      "FUT_VOL25":futures_returns(m,rf,futures_target(px,m,rf,.25,.25)),
      "FUT_35_12":futures_returns(m,rf,futures_target(px,m,rf,.35,.12)),
      "FUT_35_0":futures_returns(m,rf,futures_target(px,m,rf,.35,0.)),
      "FUT_40_0":futures_returns(m,rf,futures_target(px,m,rf,.40,0.)),
      "FUT_2x_200DMA":futures_returns(m,rf,trend_exposure(px,2.)),
      "DIP_CONF_1p5_PROXY":futures_returns(m,rf,dip_exposure(px,True,1.5)),
      "DIP_BLIND_2x_PROXY":futures_returns(m,rf,dip_exposure(px,False,2.))}
 return out

def rolling_metrics(r,years):
 r=np.asarray(r,float);n=int(years*TD);vals=[];dds=[]
 for s in range(0,len(r)-n+1,21):
  x=r[s:s+n];eq=np.cumprod(np.maximum(1+x,0))
  vals.append(eq[-1]**(1/years)-1 if eq[-1]>0 else -1.)
  dds.append(float((eq/np.maximum.accumulate(eq)-1).min()))
 a=np.asarray(vals);d=np.asarray(dds)
 return {"rolling_n":len(a),"median_cagr":float(np.median(a)),"p10_cagr":float(np.quantile(a,.10)),
         "min_cagr":float(a.min()),"median_max_dd":float(np.median(d)),"p10_max_dd":float(np.quantile(d,.10))}

CRISES={"Great_Depression_1929_1939":("1929-08-01","1939-12-31"),"Stagflation_1966_1982":("1966-01-01","1982-12-31"),
 "Crash_1987":("1987-08-01","1989-07-31"),"Dotcom_2000_2007":("2000-03-01","2007-06-30"),
 "GFC_2007_2013":("2007-10-01","2013-03-31"),"COVID_2020":("2020-02-01","2020-08-31"),
 "Inflation_2022_2024":("2022-01-01","2024-07-31")}

def period_mask(index,ranges):
 mask=np.zeros(len(index),dtype=bool)
 for a,b in ranges:mask|=(index>=pd.Timestamp(a))&(index<=pd.Timestamp(b))
 return mask

def valid_block_starts(mask,block=BLOCK):
 conv=np.convolve(np.asarray(mask,dtype=np.int8),np.ones(block,dtype=np.int16),mode="valid")
 return np.flatnonzero(conv==block)

def pool_starts(index,mode):
 if mode=="baseline":mask=np.ones(len(index),dtype=bool)
 elif mode=="high_inflation":mask=period_mask(index,[("1966-01-01","1982-12-31")])
 elif mode=="crisis":mask=period_mask(index,[("1929-08-01","1942-12-31"),("1973-01-01","1975-12-31"),
  ("2000-03-01","2009-06-30"),("2020-02-01","2020-06-30"),("2022-01-01","2023-12-31")])
 elif mode in ("crisis_mixed","sequence_shock"):mask=np.ones(len(index),dtype=bool)
 elif mode=="productivity":mask=period_mask(index,[("1982-08-01","2000-03-01"),("2010-01-01","2021-12-31")])
 elif mode=="lost_decade":mask=period_mask(index,[("1966-01-01","1982-12-31"),("2000-03-01","2009-12-31")])
 else:raise ValueError(mode)
 starts=valid_block_starts(mask)
 if len(starts)<10:raise RuntimeError(f"Too few starts for {mode}: {len(starts)}")
 return starts

def sample_blocks(m,rf,starts,b,n,rng):
 nb=math.ceil(n/BLOCK);s=rng.choice(starts,size=(b,nb),replace=True)
 idx=(s[:,:,None]+np.arange(BLOCK)).reshape(b,-1)[:,:n]
 return m[idx],rf[idx]

def generate_regime(ff,regime,b,n,rng):
 m=ff.mkt.to_numpy();rf=ff.rf.to_numpy()
 if regime.mode=="sequence_shock":
  nbad=min(5*TD,n);bm1,br1=sample_blocks(m,rf,pool_starts(ff.index,"crisis"),b,nbad,rng)
  bm2,br2=sample_blocks(m,rf,pool_starts(ff.index,"baseline"),b,n-nbad,rng);bm=np.concatenate([bm1,bm2],1);br=np.concatenate([br1,br2],1)
 elif regime.mode=="crisis_mixed":
  bs=pool_starts(ff.index,"baseline");cs=pool_starts(ff.index,"crisis");nb=math.ceil(n/BLOCK)
  use=rng.random((b,nb))<.55;sb=rng.choice(bs,size=(b,nb),replace=True);sc=rng.choice(cs,size=(b,nb),replace=True)
  idx=(np.where(use,sc,sb)[:,:,None]+np.arange(BLOCK)).reshape(b,-1)[:,:n];bm,br=m[idx],rf[idx]
 else:bm,br=sample_blocks(m,rf,pool_starts(ff.index,regime.mode),b,n,rng)
 ex=bm-br;mu=np.mean(ex,axis=1,keepdims=True);ex=mu+regime.vol_scale*(ex-mu)+regime.drift_ann/TD
 if regime.rf_floor is not None:br=np.maximum(br,regime.rf_floor/TD)
 if regime.rf_cap is not None:br=np.minimum(br,regime.rf_cap/TD)
 bm=np.maximum(br+ex,-.95)
 if regime.crash_every_years:
  lam=n/TD/regime.crash_every_years
  for i in range(b):
   k=rng.poisson(lam)
   if k:
    days=rng.integers(WARM if n>WARM else 0,n,size=k);shocks=rng.uniform(.08,.18,size=k)
    bm[i,days]=np.maximum(bm[i,days]-shocks,-.95)
 return bm,br

def rm2(x,w):
 out=np.full_like(x,np.nan,dtype=float);cs=np.cumsum(x,axis=1);s=cs[:,w-1:].copy();s[:,1:]-=cs[:,:-w];out[:,w-1:]=s/w;return out

def rs2(x,w=20):
 out=np.full_like(x,np.nan,dtype=float);c=np.cumsum(x,1);c2=np.cumsum(x*x,1)
 s=c[:,w-1:].copy();s2=c2[:,w-1:].copy();s[:,1:]-=c[:,:-w];s2[:,1:]-=c2[:,:-w];mu=s/w
 out[:,w-1:]=np.sqrt(np.maximum((s2-w*mu*mu)/(w-1),0));return out

def futures_target_m(px,mkt,rf,bull=.35,bear=.12,sma=200,cap=3.):
 ma=rm2(px,sma);vol=rs2(mkt-rf)*math.sqrt(TD);e=np.zeros_like(px);pv=vol[:,:-1]
 valid=np.isfinite(ma[:,:-1])&np.isfinite(pv)&(pv>0);target=np.where(px[:,:-1]>ma[:,:-1],bull,bear)
 e[:,1:]=np.where(valid,np.clip(target/np.maximum(pv,1e-12),0,cap),0);return e

def futures_ret_m(mkt,rf,e,tc=FUT_TC):
 return rf+e*(mkt-rf)-tc*np.abs(np.diff(e,axis=1,prepend=np.zeros((len(e),1))))

def letf_s9_m(px,r3,rf,bull=.35,bear=.12,sma=200):
 ma=rm2(px,sma);vol=rs2(r3)*math.sqrt(TD);a=np.zeros_like(px);pv=vol[:,:-1]
 valid=np.isfinite(ma[:,:-1])&np.isfinite(pv)&(pv>0);target=np.where(px[:,:-1]>ma[:,:-1],bull,bear)
 a[:,1:]=np.where(valid,np.clip(target/np.maximum(pv,1e-12),0,1),0)
 return a*r3+(1-a)*rf-LETF_TC*np.abs(np.diff(a,axis=1,prepend=np.zeros((len(a),1))))

def trend_exp_m(px,level=2.,sma=200):
 ma=rm2(px,sma);e=np.zeros_like(px);valid=np.isfinite(ma[:,:-1]);e[:,1:]=np.where(valid,np.where(px[:,:-1]>ma[:,:-1],level,0.),0.);return e

def mc_strategies(m,rf):
 px=np.cumprod(np.maximum(1+m,1e-12),axis=1);r3=levret(m,rf,3.)
 out={"MKT_1x":m,"LETF_1p5_BH":levret(m,rf,1.5),"LETF_2x_BH":levret(m,rf,2.),"LETF_3x_BH":r3,
  "LETF_S9_35_12":letf_s9_m(px,r3,rf,.35,.12),"LETF_S9_35_0":letf_s9_m(px,r3,rf,.35,0.)}
 out["FUT_VOL25"]=futures_ret_m(m,rf,futures_target_m(px,m,rf,.25,.25))
 out["FUT_35_12"]=futures_ret_m(m,rf,futures_target_m(px,m,rf,.35,.12))
 out["FUT_35_0"]=futures_ret_m(m,rf,futures_target_m(px,m,rf,.35,0.))
 out["FUT_40_0"]=futures_ret_m(m,rf,futures_target_m(px,m,rf,.40,0.))
 out["FUT_2x_200DMA"]=futures_ret_m(m,rf,trend_exp_m(px,2.))
 return out

def pathstats(r,hds):
 b,n=r.shape;term=np.empty((b,len(hds)));mdd=np.empty_like(term);wealth=np.ones(b);peak=np.ones(b);ddmin=np.zeros(b);j=0
 for t in range(n):
  wealth*=np.maximum(1+r[:,t],0);peak=np.maximum(peak,wealth);ddmin=np.minimum(ddmin,wealth/peak-1)
  if j<len(hds) and t+1==hds[j]:term[:,j]=wealth;mdd[:,j]=ddmin;j+=1
 return term,mdd

def run_mc(ff,regime,paths,seed):
 rng=np.random.default_rng(seed);maxh=max(HORIZONS)*TD;total=maxh+WARM;hds=np.array(HORIZONS)*TD
 store={};beat={};done=0
 while done<paths:
  b=min(32,paths-done);bm,br=generate_regime(ff,regime,b,total,rng);R=mc_strategies(bm,br)
  if not store:
   store={k:[[],[]] for k in R};beat={k:[[] for _ in HORIZONS] for k in R if k!="MKT_1x"}
  bt={}
  for name,rr in R.items():
   t,d=pathstats(rr[:,WARM:WARM+maxh],hds);bt[name]=t;store[name][0].append(t);store[name][1].append(d)
  for name in beat:
   for j in range(len(HORIZONS)):beat[name][j].append(bt[name][:,j]>bt["MKT_1x"][:,j])
  done+=b;print(regime.name,done,"/",paths,flush=True)
 rows=[]
 for name in store:
  T=np.concatenate(store[name][0]);D=np.concatenate(store[name][1])
  for j,y in enumerate(HORIZONS):
   c=np.where(T[:,j]>0,T[:,j]**(1/y)-1,-1.)
   rows.append({"regime":regime.name,"strategy":name,"horizon_years":y,"paths":paths,
    "median_cagr":float(np.median(c)),"p10_cagr":float(np.quantile(c,.10)),"p01_cagr":float(np.quantile(c,.01)),
    "prob_beat_1x":0. if name=="MKT_1x" else float(np.mean(np.concatenate(beat[name][j]))),
    "prob_dd_gt_50":float(np.mean(D[:,j]<=-.50)),"prob_dd_gt_70":float(np.mean(D[:,j]<=-.70)),
    "prob_dd_gt_90":float(np.mean(D[:,j]<=-.90)),"median_max_dd":float(np.median(D[:,j])),
    "p10_max_dd":float(np.quantile(D[:,j],.10)),"prob_terminal_below_start":float(np.mean(T[:,j]<1.)),
    "prob_terminal_below_0p1":float(np.mean(T[:,j]<.1))})
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--paths",type=int,default=2500);ap.add_argument("--seed",type=int,default=20260921)
 ap.add_argument("--output",type=Path,default=Path("results/aggressive_strategy_extensive_20260921"));a=ap.parse_args()
 if a.paths<1000:raise SystemExit("Use at least 1000 paths per regime.")
 a.output.mkdir(parents=True,exist_ok=True);ff,meta=load_ff();strategies=historical_strategies(ff)
 hist=[];roll=[];crisis=[]
 for name,r in strategies.items():
  hist.append({"strategy":name,**metrics(r)})
  for y in (5,10,20,30):roll.append({"strategy":name,"horizon_years":y,**rolling_metrics(r,y)})
  for cname,(x,y) in CRISES.items():
   mask=(ff.index>=x)&(ff.index<=y)
   if mask.sum()>=TD//2:crisis.append({"strategy":name,"crisis":cname,"start":x,"end":y,**metrics(np.asarray(r)[mask])})
 pd.DataFrame(hist).to_csv(a.output/"historical_summary.csv",index=False)
 pd.DataFrame(roll).to_csv(a.output/"rolling_windows.csv",index=False)
 pd.DataFrame(crisis).to_csv(a.output/"crisis_windows.csv",index=False)
 allmc=[]
 for i,reg in enumerate(REGIMES):allmc.append(run_mc(ff,reg,a.paths,a.seed+1009*i))
 mc=pd.concat(allmc,ignore_index=True);mc.to_csv(a.output/"macro_regime_monte_carlo.csv",index=False)
 score=(mc[mc.horizon_years==20].groupby("strategy").agg(worst_regime_p10=("p10_cagr","min"),
        median_of_regime_medians=("median_cagr","median"),worst_prob_dd90=("prob_dd_gt_90","max"),
        worst_prob_loss=("prob_terminal_below_start","max")).reset_index()
        .sort_values(["worst_regime_p10","worst_prob_dd90"],ascending=[False,True]))
 score.to_csv(a.output/"robustness_20y.csv",index=False)
 manifest={"data":meta,"paths_per_regime":a.paths,"n_regimes":len(REGIMES),"total_regime_paths":a.paths*len(REGIMES),
  "seed":a.seed,"block_days":BLOCK,"warmup_days":WARM,"horizons_years":HORIZONS,
  "leverage_model":{"letf":"daily-reset synthetic leverage; historical FF RF; 50bp financing spread; 91bp levered expense",
                    "futures":"cash RF + exposure*(market total return-RF), exposure 0..3x, 2bp cost per 1.0 exposure turnover"},
  "dip_proxy_warning":"Historical DIP strategies use the broad U.S. market as a blue-chip proxy. This is NOT a point-in-time individual-company backtest.",
  "options_warning":"Deep-ITM LEAPS are intentionally not scored here because decision-grade history needs point-in-time option chains/IV surfaces (e.g. OptionMetrics IvyDB/Cboe). A synthetic reconstruction is a separate sensitivity, not observed option history.",
  "regimes":[r.__dict__ for r in REGIMES],
  "limitations":["Scenario regimes are stress worlds, not forecast probabilities.","Empirical regime pools reuse finite historical episodes.",
  "Synthetic drift/vol/rate transforms are assumptions.","Fama-French market is broad U.S. market, not literal SPY before inception.",
  "Strategy parameters were fixed before this run; no optimization on these results."]}
 (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
 print(pd.DataFrame(hist).sort_values("cagr",ascending=False).to_string(index=False))
 print(score.to_string(index=False))
if __name__=="__main__":main()
