from __future__ import annotations
import argparse, hashlib, io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from letf import config as cfg
from letf.strategy import run_strategy_fixed

TD = 252
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
LETF_EXPENSE = 0.0091
ONE_X_EXPENSE = 0.0003
FIN_SPREAD = 0.005

INFEASIBLE_ROTH = {
    "S7": "original code can allocate 2.0x of TQQQ (up to ~6x equity exposure)",
    "S10": "original code can allocate 1.5x of TQQQ",
    "S16": "original code can allocate 1.2x of TQQQ",
    "S17": "original code can allocate 1.5x of TQQQ",
    "S18": "original code can allocate 1.5x of TQQQ",
    "S19": "original code can allocate 1.5x of TQQQ",
}
LOOKAHEAD_FLAGS = {
    "S7": "realized_vol is not shifted before sizing current-day return",
    "S16": "current-day VIX is used to size current-day return",
}

def load_ff():
    req = urllib.request.Request(FF_URL, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    sha = hashlib.sha256(blob).hexdigest()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(name).decode("latin-1")
    rows=[]
    for line in raw.splitlines():
        q=[x.strip() for x in line.split(",")]
        if len(q)>=5 and len(q[0])==8 and q[0].isdigit():
            try:
                rows.append((pd.to_datetime(q[0],format="%Y%m%d"),
                             (float(q[1])+float(q[4]))/100.0,
                             float(q[4])/100.0))
            except ValueError:
                pass
    df=pd.DataFrame(rows,columns=["date","mkt","rf"]).set_index("date").sort_index()
    df=df.loc["1926-07-01":]
    if len(df)<20000:
        raise RuntimeError(f"Fama-French history too short: {len(df)}")
    return df, {"source":FF_URL,"sha256":sha,"rows":len(df),
                "start":str(df.index.min().date()),"end":str(df.index.max().date())}

def levret(mkt, rf, L, expense=None):
    mkt=np.asarray(mkt,float); rf=np.asarray(rf,float)
    if expense is None: expense = ONE_X_EXPENSE if L == 1 else LETF_EXPENSE
    ret = L*mkt - max(L-1,0)*(rf + FIN_SPREAD/TD) - expense/TD
    return np.maximum(ret,-1.0)

def build_generic(ff):
    d=pd.DataFrame(index=ff.index)
    d["SPY_Ret"]=levret(ff.mkt,ff.rf,1)
    d["SSO_Ret"]=levret(ff.mkt,ff.rf,2)
    d["TQQQ_Ret"]=levret(ff.mkt,ff.rf,3)
    d["UPRO_Ret"]=d["TQQQ_Ret"]
    d["QQQ_Ret"]=d["SPY_Ret"]
    d["Cash_Ret"]=ff.rf.to_numpy()
    d["RF"]=ff.rf.to_numpy()
    d["IRX"]=ff.rf.to_numpy()*TD*100
    d["TNX"]=d["IRX"]+1.5
    d["VIX"]=(pd.Series(ff.mkt,index=ff.index).rolling(20,min_periods=5).std()*math.sqrt(TD)*100).clip(10,80).bfill()
    d["TMF_Ret"]=np.nan
    for a in ["SPY","SSO","TQQQ","UPRO","QQQ"]:
        d[f"{a}_Price"]=(1+d[f"{a}_Ret"].fillna(0)).cumprod()*100
    d["SPY_Price"]=(1+d["SPY_Ret"].fillna(0)).cumprod()*100
    return d

def metrics(eq):
    eq=np.asarray(eq,float)
    eq=eq[np.isfinite(eq)]
    if len(eq)<2 or eq[0]<=0: return {}
    r=eq[1:]/eq[:-1]-1
    years=len(r)/TD
    peak=np.maximum.accumulate(eq)
    dd=eq/peak-1
    cur=mx=0; high=-np.inf
    for v in eq:
        if v>=high:
            high=v;cur=0
        else:
            cur+=1;mx=max(mx,cur)
    return {
        "cagr": (eq[-1]/eq[0])**(1/years)-1 if eq[-1]>0 else -1,
        "max_drawdown": float(dd.min()),
        "vol": float(np.std(r,ddof=1)*math.sqrt(TD)),
        "max_underwater_years": mx/TD,
    }

def rolling_cagrs(eq,index,years,step=21):
    eq=np.asarray(eq,float); n=int(years*TD); out=[]
    for s in range(0,len(eq)-n,step):
        e=s+n
        if eq[s]>0 and eq[e]>0:
            out.append((eq[e]/eq[s])**(1/years)-1)
    if not out:
        return {"rolling_n":0}
    a=np.asarray(out)
    return {"rolling_n":len(a),"rolling_median":float(np.median(a)),
            "rolling_p10":float(np.quantile(a,.1)),"rolling_min":float(a.min()),
            "rolling_prob_beat_1x":np.nan}

def run_all_historical(d):
    rows=[]
    curves={}
    for sid, spec in cfg.STRATEGIES.items():
        if sid=="S6":
            rows.append({"strategy_id":sid,"name":spec["name"],"status":"UNSCORED_NO_REAL_LONG_TREASURY_1926",
                         "roth_feasible":True,"lookahead_flag":False})
            continue
        try:
            eq,_=run_strategy_fixed(d,sid,regime_path=None,correlation_matrices={},apply_costs=True)
            curves[sid]=eq
            m=metrics(eq.values)
            row={"strategy_id":sid,"name":spec["name"],"status":"scored",
                 "roth_feasible":sid not in INFEASIBLE_ROTH,
                 "roth_issue":INFEASIBLE_ROTH.get(sid,""),
                 "lookahead_flag":sid in LOOKAHEAD_FLAGS,
                 "lookahead_issue":LOOKAHEAD_FLAGS.get(sid,""),**m}
            for h in (5,10,20,30):
                rr=rolling_cagrs(eq.values,d.index,h)
                for k,v in rr.items(): row[f"{h}y_{k}"]=v
            rows.append(row)
        except Exception as e:
            rows.append({"strategy_id":sid,"name":spec["name"],"status":"ERROR","error":repr(e),
                         "roth_feasible":sid not in INFEASIBLE_ROTH,
                         "lookahead_flag":sid in LOOKAHEAD_FLAGS})
    if "S2" in curves:
        spy=curves["S2"].values
        for row in rows:
            sid=row["strategy_id"]
            if sid not in curves: continue
            eq=curves[sid].values
            for h in (5,10,20,30):
                n=int(h*TD); beats=[]
                for s in range(0,min(len(eq),len(spy))-n,21):
                    e=s+n
                    if eq[s]>0 and eq[e]>0 and spy[s]>0 and spy[e]>0:
                        ca=(eq[e]/eq[s])**(1/h)-1
                        cb=(spy[e]/spy[s])**(1/h)-1
                        beats.append(ca>cb)
                row[f"{h}y_prob_beat_1x"]=float(np.mean(beats)) if beats else np.nan
    return pd.DataFrame(rows)

def corrected_s7(d, cap=1.0):
    r=d.TQQQ_Ret.to_numpy(); cash=d.Cash_Ret.to_numpy()
    vol=pd.Series(r,index=d.index).rolling(20).std().shift(1).to_numpy()*math.sqrt(TD)
    a=np.where(np.isfinite(vol)&(vol>.01), .20/vol, 1.0)
    a=np.clip(a,.2,cap)
    ret=a*r+(1-a)*cash-.0003*np.abs(np.diff(a,prepend=0))
    return 10000*np.cumprod(1+ret)

def corrected_s16(d, cap=1.0):
    r=d.TQQQ_Ret.to_numpy(); cash=d.Cash_Ret.to_numpy()
    v5=pd.Series(r,index=d.index).rolling(5,min_periods=5).std().shift(1).to_numpy()*math.sqrt(TD)
    v60=pd.Series(r,index=d.index).rolling(60,min_periods=20).std().shift(1).to_numpy()*math.sqrt(TD)
    vix=d.VIX.shift(1).to_numpy()
    ratio=v5/np.maximum(v60,1e-12)
    crisis=(vix>25)|(ratio>1.5)
    target=np.where(crisis,.08,.30)
    a=np.where(np.isfinite(v5)&(v5>.001),target/v5,.5)
    a=np.clip(a,0,cap)
    ret=a*r+(1-a)*cash-.0003*np.abs(np.diff(a,prepend=0))
    return 10000*np.cumprod(1+ret)

def add_corrected(rows,d):
    extras=[
        ("S7_CORRECTED_ROTH","Vol Target 20% lagged, cap 100%",corrected_s7(d,1.0)),
        ("S16_CORRECTED_ROTH","Crisis Alpha lagged VIX, cap 100%",corrected_s16(d,1.0)),
    ]
    for sid,name,eq in extras:
        m=metrics(eq)
        row={"strategy_id":sid,"name":name,"status":"scored","roth_feasible":True,"lookahead_flag":False,**m}
        for h in (5,10,20,30):
            rr=rolling_cagrs(eq,d.index,h)
            for k,v in rr.items(): row[f"{h}y_{k}"]=v
        rows.loc[len(rows)]=row
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("results/final_strategy_screen"))
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    ff,meta=load_ff(); d=build_generic(ff)
    res=run_all_historical(d); res=add_corrected(res,d)
    res.to_csv(a.output/"all_repo_strategies_historical.csv",index=False)
    valid=res[(res.status=="scored")&(~res.lookahead_flag.fillna(False))]
    valid=valid.sort_values(["20y_rolling_p10","cagr"],ascending=False)
    (a.output/"screen_manifest.json").write_text(json.dumps({
        "data":meta,
        "note":"1926 generic 3x U.S.-market proxy tests strategy logic; it is not pre-1985 TQQQ history.",
        "pre_1950_qqq_tlt_scored":False,
        "pre_1985_tqqq_scored":False,
        "S6_status":"separate modern-only real ETF test required",
        "lookahead_flags":LOOKAHEAD_FLAGS,
        "roth_infeasible_original":INFEASIBLE_ROTH,
        "ranking_rule":"20-year rolling p10 CAGR, then full-sample CAGR; lookahead-flagged originals excluded",
        "top_valid":valid[["strategy_id","name","cagr","max_drawdown","20y_rolling_p10","20y_prob_beat_1x"]].head(10).to_dict("records")
    },indent=2))
    print(res[["strategy_id","name","status","roth_feasible","lookahead_flag","cagr","max_drawdown","20y_rolling_p10","20y_prob_beat_1x"]].to_string(index=False))
    print("\nTOP VALID\n",valid[["strategy_id","name","cagr","max_drawdown","20y_rolling_p10","20y_prob_beat_1x"]].head(10).to_string(index=False))
if __name__=="__main__": main()
