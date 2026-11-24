from __future__ import annotations
"""
Point-in-time Bottleneck Winner Tournament.

This engine is intentionally data-provider agnostic. Decision-grade runs require:
- CRSP survivor-bias-free monthly security returns incl. delistings
- Compustat Point-in-Time / Snapshot fundamentals with an explicit availability date
- Hoberg-Phillips TNIC competition data (free) conservatively lagged unless filing dates supplied
- optional I/B/E/S point-in-time estimate/revision file

Canonical input schemas are documented in REQUIRED_COLUMNS below.
No current-membership universe is ever used.
"""
import argparse, io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd

TD_MONTHS=12
TNIC_HHI_URL="https://hobergphillips.tuck.dartmouth.edu/idata/TNIC3HHIdata.zip"
TNIC_PAIRS_URL="https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip"

REQUIRED_COLUMNS={
 "crsp":["permno","date","ret","me"],
 "ccm":["permno","gvkey","linkdt","linkenddt"],
 "fund":["gvkey","available_date","sale","cogs","oibdp","capx","oancf","at","lt","csho","prcc"],
 "tnic":["gvkey","year","tnic3hhi","tnic3tsimm"],
}
OPTIONAL_FUND=["revt","gp","ni","ib","che","dltt","dlc","ceq","xrd","xsga"]
OPTIONAL_IBES=["permno","available_date","eps_rev_1m","eps_rev_3m","sales_rev_1m","surprise"]

def read_any(path):
 p=Path(path)
 if p.suffix.lower()==".parquet":return pd.read_parquet(p)
 if p.suffix.lower() in (".pkl",".pickle"):return pd.read_pickle(p)
 return pd.read_csv(p)

def first_table_in_zip(blob:bytes):
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  names=[n for n in z.namelist() if n.lower().endswith((".csv",".txt",".dat")) and "readme" not in n.lower()]
  if not names:raise RuntimeError(f"No data table in zip: {z.namelist()[:20]}")
  raw=z.read(names[0])
 for sep in [",","\t","|",r"\s+"]:
  try:
   df=pd.read_csv(io.BytesIO(raw),sep=sep,engine="python")
   if df.shape[1]>=3:return df,names[0]
  except Exception:pass
 raise RuntimeError("Could not parse TNIC zip")

def download_tnic_hhi(cache:Path):
 cache.mkdir(parents=True,exist_ok=True);fn=cache/"TNIC3HHIdata.zip"
 if not fn.exists():
  req=urllib.request.Request(TNIC_HHI_URL,headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=120) as r:fn.write_bytes(r.read())
 df,name=first_table_in_zip(fn.read_bytes())
 cols={c.lower().strip():c for c in df.columns}
 ren={}
 for target,aliases in {"gvkey":["gvkey"],"year":["year"],"tnic3hhi":["tnic3hhi","hhi"],"tnic3tsimm":["tnic3tsimm","tsimm"]}.items():
  for a in aliases:
   if a in cols:ren[cols[a]]=target;break
 df=df.rename(columns=ren)
 need={"gvkey","year","tnic3hhi","tnic3tsimm"}
 if not need.issubset(df.columns):raise RuntimeError(f"TNIC schema {list(df.columns)} from {name}")
 df=df[list(need)].copy()
 df["gvkey"]=df.gvkey.astype(str).str.zfill(6)
 df["year"]=pd.to_numeric(df.year).astype(int)
 return df,name

def canonicalize_dates(df,col):
 df=df.copy();df[col]=pd.to_datetime(df[col]);return df

def zscore_cross(df,cols,datecol="date"):
 out=df.copy()
 for c in cols:
  x=pd.to_numeric(out[c],errors="coerce")
  lo=x.groupby(out[datecol]).transform(lambda s:s.quantile(.01))
  hi=x.groupby(out[datecol]).transform(lambda s:s.quantile(.99))
  x=x.clip(lo,hi)
  mu=x.groupby(out[datecol]).transform("mean");sd=x.groupby(out[datecol]).transform("std")
  out[c+"_z"]=(x-mu)/sd.replace(0,np.nan)
 return out

def prepare_crsp(crsp):
 c=crsp.copy();c["date"]=pd.to_datetime(c.date)+pd.offsets.MonthEnd(0)
 c["ret"]=pd.to_numeric(c.ret,errors="coerce")
 if "dlret" in c:
  d=pd.to_numeric(c.dlret,errors="coerce").fillna(0)
  # Only combine if caller has not declared ret already includes delisting.
  if not bool(c.attrs.get("ret_includes_delisting",False)):c["ret"]=(1+c.ret.fillna(0))*(1+d)-1
 c["me"]=pd.to_numeric(c.me,errors="coerce").abs()
 c=c[(c.me>0)&c.ret.notna()].sort_values(["permno","date"])
 g=c.groupby("permno",group_keys=False)
 c["mom12_1"]=g.ret.apply(lambda s:(1+s).rolling(11,min_periods=9).apply(np.prod,raw=True)-1).shift(1)
 c["mom6_1"]=g.ret.apply(lambda s:(1+s).rolling(5,min_periods=4).apply(np.prod,raw=True)-1).shift(1)
 c["price_idx"]=g.ret.transform(lambda s:(1+s).cumprod())
 c["high12"]=g.price_idx.transform(lambda s:s.rolling(12,min_periods=9).max())
 c["high_ratio"]=c.price_idx/c.high12
 c["vol12"]=g.ret.transform(lambda s:s.rolling(12,min_periods=9).std()*math.sqrt(12))
 return c

def prepare_fund(f):
 f=f.copy();f["gvkey"]=f.gvkey.astype(str).str.zfill(6);f["available_date"]=pd.to_datetime(f.available_date)+pd.offsets.MonthEnd(0)
 for c in [x for x in REQUIRED_COLUMNS["fund"][2:] if x in f]:f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values(["gvkey","available_date"])
 g=f.groupby("gvkey",group_keys=False)
 f["sales_growth_yoy"]=g.sale.pct_change(4)
 if "cogs" in f:f["gross_margin"]=(f.sale-f.cogs)/f.sale.replace(0,np.nan)
 elif "gp" in f:f["gross_margin"]=f.gp/f.sale.replace(0,np.nan)
 else:f["gross_margin"]=np.nan
 f["gm_change_yoy"]=g.gross_margin.diff(4)
 f["op_margin"]=f.oibdp/f.sale.replace(0,np.nan)
 f["op_margin_change_yoy"]=g.op_margin.diff(4)
 f["fcf"]=(f.oancf-f.capx);f["fcf_margin"]=f.fcf/f.sale.replace(0,np.nan)
 f["leverage"]=(f.lt/f.at.replace(0,np.nan))
 f["share_growth_yoy"]=g.csho.pct_change(4)
 f["asset_growth_yoy"]=g.at.pct_change(4)
 f["ev_sales_proxy"]=(f.prcc*f.csho+f.lt)/f.sale.replace(0,np.nan)
 return f

def prepare_tnic(t):
 t=t.copy();t["gvkey"]=t.gvkey.astype(str).str.zfill(6);t["year"]=pd.to_numeric(t.year).astype(int)
 # TNIC file is explicitly not lagged. Conservative availability: June 30 of next calendar year.
 t["available_date"]=pd.to_datetime((t.year+1).astype(str)+"-06-30")+pd.offsets.MonthEnd(0)
 t["tnic3hhi"]=pd.to_numeric(t.tnic3hhi,errors="coerce")
 t["tnic3tsimm"]=pd.to_numeric(t.tnic3tsimm,errors="coerce")
 return t.sort_values(["gvkey","available_date"])

def asof_by_key(left,right,key,left_date="date",right_date="available_date"):
 parts=[]
 for k,l in left.groupby(key,sort=False):
  r=right[right[key]==k]
  if r.empty:continue
  parts.append(pd.merge_asof(l.sort_values(left_date),r.sort_values(right_date),left_on=left_date,right_on=right_date,direction="backward",suffixes=("","_r")))
 return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def map_gvkey(crsp,ccm):
 x=crsp.copy();l=ccm.copy()
 l["linkdt"]=pd.to_datetime(l.linkdt);l["linkenddt"]=pd.to_datetime(l.linkenddt,errors="coerce").fillna(pd.Timestamp("2099-12-31"))
 x=x.merge(l[["permno","gvkey","linkdt","linkenddt"]],on="permno",how="left")
 x=x[(x.date>=x.linkdt)&(x.date<=x.linkenddt)].copy()
 x["gvkey"]=x.gvkey.astype(str).str.zfill(6)
 return x

def add_ibes(panel,ibes):
 if ibes is None:return panel
 i=ibes.copy();i["available_date"]=pd.to_datetime(i.available_date)+pd.offsets.MonthEnd(0)
 return asof_by_key(panel,i,"permno")

def build_panel(crsp,ccm,fund,tnic,ibes=None):
 c=map_gvkey(prepare_crsp(crsp),ccm)
 p=asof_by_key(c,prepare_fund(fund),"gvkey")
 p=asof_by_key(p,prepare_tnic(tnic),"gvkey")
 p=add_ibes(p,ibes)
 # Fundamental + market features
 p["competition_scarcity"]=np.log1p(p.tnic3hhi.clip(lower=0))-np.log1p(p.tnic3tsimm.clip(lower=0))
 p["demand_accel"]=p.sales_growth_yoy
 p["pricing_power"]=p.gm_change_yoy.fillna(p.op_margin_change_yoy)
 p["quality"]=p.fcf_margin-p.leverage
 p["dilution_penalty"]=-p.share_growth_yoy
 p["valuation_sanity"]=-np.log1p(p.ev_sales_proxy.clip(lower=0))
 p["momentum_combo"]=.6*p.mom12_1+.4*p.mom6_1
 if "eps_rev_1m" in p:
  p["revision_combo"]=p[[c for c in ["eps_rev_1m","eps_rev_3m","sales_rev_1m","surprise"] if c in p]].mean(axis=1)
 else:p["revision_combo"]=np.nan
 features=["competition_scarcity","demand_accel","pricing_power","quality","dilution_penalty","valuation_sanity","momentum_combo","high_ratio"]
 if p.revision_combo.notna().any():features.append("revision_combo")
 p=zscore_cross(p,features)
 return p

MODELS={
 "momentum":{"momentum_combo_z":.7,"high_ratio_z":.3},
 "quality_momentum":{"momentum_combo_z":.45,"high_ratio_z":.15,"quality_z":.25,"dilution_penalty_z":.15},
 "bottleneck_core":{"competition_scarcity_z":.35,"demand_accel_z":.35,"pricing_power_z":.30},
 "bottleneck_momentum":{"competition_scarcity_z":.20,"demand_accel_z":.20,"pricing_power_z":.15,"momentum_combo_z":.30,"high_ratio_z":.15},
 "bottleneck_full":{"competition_scarcity_z":.15,"demand_accel_z":.15,"pricing_power_z":.10,"quality_z":.15,
                    "dilution_penalty_z":.10,"valuation_sanity_z":.10,"momentum_combo_z":.15,"high_ratio_z":.10},
}
def score_models(p):
 p=p.copy()
 for name,w in MODELS.items():
  cols=[c for c in w if c in p]
  if not cols:continue
  z=pd.DataFrame({c:p[c]*w[c] for c in cols})
  p[name+"_score"]=z.sum(axis=1,min_count=max(2,len(cols)//2))
 return p

def backtest_one(p,score_col,topn=20,cost_bps=15):
 x=p.copy();x["rebalance"]=x.date.dt.month.isin([3,6,9,12])
 x["rank"]=x.groupby("date")[score_col].rank(ascending=False,method="first")
 target=(x.rebalance&(x["rank"]<=topn)).astype(float)
 # Carry last chosen equal weights until next rebalance.
 rows=[];hold={}
 for d,g in x.groupby("date"):
  if g.rebalance.any():
   chosen=g.loc[g["rank"]<=topn,"permno"].tolist();hold={k:1/len(chosen) for k in chosen} if chosen else {}
  r=0.;present=set(g.permno)
  for _,q in g.iterrows():
   if q.permno in hold:r+=hold[q.permno]*q.ret
  # If a held permno disappears without a delisting return it implicitly earns 0 for that month; CRSP delisting data should prevent this.
  turnover=np.nan
  rows.append((d,r,len(hold)))
 out=pd.DataFrame(rows,columns=["date","gross_ret","n_hold"]).set_index("date")
 # Approx quarterly full rebalance cost: use conservative 2*one-way cost on rebalance months.
 rebal=out.index.month.isin([3,6,9,12]);out["net_ret"]=out.gross_ret-(2*cost_bps/10000)*rebal
 return out

def perf(r):
 r=pd.Series(r).dropna();e=(1+r).cumprod();yrs=len(r)/12;dd=e/e.cummax()-1
 return {"years":yrs,"cagr":e.iloc[-1]**(1/yrs)-1 if yrs>0 else np.nan,"vol":r.std()*math.sqrt(12),
         "max_dd":dd.min(),"terminal":e.iloc[-1],"worst_12m":((1+r).rolling(12).apply(np.prod,raw=True)-1).min()}

def block_bootstrap(active,paths=10000,years=20,block=12,seed=20260921):
 a=pd.Series(active).dropna().to_numpy();rng=np.random.default_rng(seed);n=years*12;vals=[]
 if len(a)<36:return {}
 for _ in range(paths):
  k=math.ceil(n/block);starts=rng.integers(0,len(a),size=k);idx=np.concatenate([(s+np.arange(block))%len(a) for s in starts])[:n]
  z=a[idx];vals.append(np.prod(1+z)**(1/years)-1)
 v=np.asarray(vals);return {"mc_paths":paths,"mc_years":years,"active_median_cagr":np.median(v),"active_p10_cagr":np.quantile(v,.1),"active_p01_cagr":np.quantile(v,.01)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--crsp");ap.add_argument("--ccm");ap.add_argument("--fund");ap.add_argument("--tnic");ap.add_argument("--ibes")
 ap.add_argument("--download-tnic-smoke",action="store_true")
 ap.add_argument("--out",default="results/bottleneck_winner_tournament")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if a.download_tnic_smoke:
  t,name=download_tnic_hhi(out/"cache");t.head(1000).to_csv(out/"tnic_hhi_head.csv",index=False)
  (out/"tnic_smoke.json").write_text(json.dumps({"rows":len(t),"source_member":name,"columns":list(t.columns),
      "year_min":int(t.year.min()),"year_max":int(t.year.max())},indent=2))
  print((out/"tnic_smoke.json").read_text())
  if not all([a.crsp,a.ccm,a.fund]):return
 if not all([a.crsp,a.ccm,a.fund]):raise SystemExit("Decision-grade tournament requires --crsp --ccm --fund; TNIC may be --tnic or downloaded separately.")
 crsp=read_any(a.crsp);ccm=read_any(a.ccm);fund=read_any(a.fund)
 tnic=read_any(a.tnic) if a.tnic else download_tnic_hhi(out/"cache")[0]
 ibes=read_any(a.ibes) if a.ibes else None
 p=score_models(build_panel(crsp,ccm,fund,tnic,ibes));p.to_parquet(out/"pit_feature_panel.parquet",index=False)
 # Eligible common-stock-like universe must already be filtered in CRSP extraction; add liquidity screen here.
 p=p[(p.me>=p.groupby("date").me.transform(lambda s:s.quantile(.20))) & p.ret.notna()]
 rows=[];curves=[]
 bench=p.groupby("date").apply(lambda g:np.average(g.ret,weights=g.me),include_groups=False).rename("bench")
 for model in MODELS:
  sc=model+"_score"
  if sc not in p:continue
  for n in [10,20,40]:
   bt=backtest_one(p,sc,n);ix=bt.index.intersection(bench.index);bt=bt.reindex(ix);b=bench.reindex(ix)
   # sub-periods frozen, no tuning on holdout
   for label,s,e in [("train","1989-01-01","2004-12-31"),("validation","2005-01-01","2014-12-31"),("holdout","2015-01-01","2026-12-31"),("full","1989-01-01","2026-12-31")]:
    mask=(ix>=s)&(ix<=e)
    if mask.sum()<24:continue
    pm=perf(bt.net_ret[mask]);bm=perf(b[mask]);mc=block_bootstrap((bt.net_ret-b)[mask],paths=10000,years=min(20,max(5,int(mask.sum()/12))))
    rows.append({"model":model,"topn":n,"period":label,**{f"port_{k}":v for k,v in pm.items()},
                 **{f"bench_{k}":v for k,v in bm.items()},**mc})
   q=bt.copy();q["benchmark"]=b;q["model"]=model;q["topn"]=n;curves.append(q.reset_index())
 pd.DataFrame(rows).to_csv(out/"tournament_results.csv",index=False)
 pd.concat(curves,ignore_index=True).to_csv(out/"portfolio_monthly_returns.csv",index=False)
 manifest={"point_in_time_rules":{
  "CRSP":"survivor-bias-free universe; include delisting returns; never current membership",
  "Compustat":"requires explicit available_date from Point-in-Time/Snapshot or true report/filing date; no datadate-only lookahead",
  "TNIC":"distributed TNIC is explicitly not lagged; default here makes year Y available June 30 Y+1 unless actual filing availability supplied",
  "IBES":"optional and must use historical snapshot/available_date"},
  "models":MODELS,"rebalance":"quarterly","cost_bps_one_way":15,
  "warning":"Do not treat results as decision-grade unless canonical licensed inputs and their availability dates have been independently audited."}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
 print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
