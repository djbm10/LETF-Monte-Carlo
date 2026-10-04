from __future__ import annotations
"""
Exchange-grade NQ/MNQ settlement backtest adapter for CME DataMine exports.

This file does not download entitled CME data or store credentials. It ingests
contract-level settlement files legitimately obtained from CME DataMine and
constructs continuous returns under explicit, auditable roll rules.

IMPORTANT: official futures settlements are the P&L source only. Dynamic
strategy exposure must come from the already-frozen external signal series
(QQQ adjusted close for the 35/0 research program). This prevents a future
settlement-grade replication from silently changing the scientific rule by
recomputing trend/volatility from futures prices.

Expected canonical CME columns after normalization:
trade_date, contract, expiry, settle, volume, open_interest
At minimum trade_date, contract, settle are required. expiry may be inferred
from the contract code. volume/open_interest are required for liquidity rolls.

Expected frozen-signal file:
date, adj_close
The price-column name is configurable with --signal-price-column.
"""
import argparse, json, math, re
from pathlib import Path
import numpy as np, pandas as pd

TD=252
MONTH_CODE={"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,"N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}

ALIASES={
 "trade_date":["trade_date","date","business_date","bizdt","settle_date","trading_date"],
 "contract":["contract","symbol","instrument","security_desc","product_code","contract_code"],
 "expiry":["expiry","expiration","expiration_date","last_trade_date","maturity_date"],
 "settle":["settle","settlement","settlement_price","settle_price","final_settle"],
 "volume":["volume","vol","total_volume"],
 "open_interest":["open_interest","oi","openinterest"],
}

def normalize_cols(x):
    x=x.copy();lower={c.lower().strip():c for c in x.columns};ren={}
    for k,als in ALIASES.items():
        for a in als:
            if a in lower:ren[lower[a]]=k;break
    x=x.rename(columns=ren)
    req={"trade_date","contract","settle"}
    if not req.issubset(x.columns):raise ValueError(f"Missing {req-set(x.columns)}; columns={list(x.columns)}")
    x["trade_date"]=pd.to_datetime(x.trade_date)
    if "expiry" in x:x["expiry"]=pd.to_datetime(x.expiry,errors="coerce")
    x["settle"]=pd.to_numeric(x.settle,errors="coerce")
    for c in ["volume","open_interest"]:
        if c in x:x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna(subset=["trade_date","contract","settle"])

def infer_expiry(contract):
    s=str(contract).upper().replace(" ","")
    # Accept NQZ24, MNQH2025, etc. Final settlement is third Friday; last trade approx same date.
    m=re.search(r"(MNQ|NQ)([FGHJKMNQUVXZ])(\d{2}|\d{4})$",s)
    if not m:return pd.NaT
    mon=MONTH_CODE[m.group(2)];y=int(m.group(3));y=y+2000 if y<100 else y
    first=pd.Timestamp(y,mon,1);fridays=pd.date_range(first,first+pd.offsets.MonthEnd(0),freq="W-FRI")
    return fridays[2] if len(fridays)>=3 else pd.NaT

def canonicalize(files):
    parts=[]
    for f in files:
        p=Path(f)
        if p.suffix.lower()==".parquet":x=pd.read_parquet(p)
        else:x=pd.read_csv(p,compression="infer")
        x=normalize_cols(x);parts.append(x)
    z=pd.concat(parts,ignore_index=True)
    if "expiry" not in z:z["expiry"]=z.contract.map(infer_expiry)
    else:z["expiry"]=z.expiry.fillna(z.contract.map(infer_expiry))
    z=z.sort_values(["trade_date","expiry","contract"]).drop_duplicates(["trade_date","contract"],keep="last")
    return z

def choose_active(day,rule="volume_5bd",current=None):
    d=day[day.expiry>=day.trade_date.iloc[0]].copy()
    if d.empty:return None
    today=d.trade_date.iloc[0]
    if current is not None and current in set(d.contract):
        cur=d[d.contract==current].iloc[0]
        bdays=np.busday_count(today.date(),cur.expiry.date()) if pd.notna(cur.expiry) else 999
        later=d[d.expiry>cur.expiry].sort_values("expiry")
        nxt=later.iloc[0] if len(later) else None
        if bdays<=5 and nxt is not None:return nxt.contract
        if rule=="volume_5bd" and nxt is not None and "volume" in d:
            cv=float(cur.volume) if pd.notna(cur.volume) else 0
            nv=float(nxt.volume) if pd.notna(nxt.volume) else 0
            if nv>cv and nv>0:return nxt.contract
        if rule=="oi_5bd" and nxt is not None and "open_interest" in d:
            co=float(cur.open_interest) if pd.notna(cur.open_interest) else 0
            no=float(nxt.open_interest) if pd.notna(nxt.open_interest) else 0
            if no>co and no>0:return nxt.contract
        return current
    # Initialize nearest live contract. Subsequent dates apply the selected roll rule.
    return d.sort_values("expiry").iloc[0].contract

def make_continuous(z,rule="volume_5bd"):
    rows=[];current=None
    for dt,day in z.groupby("trade_date",sort=True):
        nxt=choose_active(day,rule,current)
        if nxt is None:continue
        row=day[day.contract==nxt].iloc[0]
        rolled=current is not None and nxt!=current
        rows.append((dt,nxt,row.expiry,row.settle,row.get("volume",np.nan),row.get("open_interest",np.nan),rolled))
        current=nxt
    c=pd.DataFrame(rows,columns=["date","contract","expiry","settle","volume","open_interest","rolled"]).set_index("date")
    # On a roll date, P&L is earned on the old contract through that close, then
    # state switches to the new contract after close. The new contract's level
    # is never spliced into that day's P&L, so the roll price gap cannot create
    # artificial return.
    rets=[];prev_contract=None;prev_settle=None
    lookup=z.set_index(["trade_date","contract"])["settle"]
    for dt,row in c.iterrows():
        if prev_contract is None:rets.append(np.nan)
        else:
            key=(dt,prev_contract)
            if key in lookup.index:
                today_old=float(lookup.loc[key]);rets.append(today_old/prev_settle-1)
            elif not row.rolled:
                rets.append(float(row.settle)/prev_settle-1)
            else:
                rets.append(np.nan)
        prev_contract=row.contract;prev_settle=float(row.settle)
    c["fut_ret"]=rets
    return c

def rf_series(dates,rf_file=None):
    if rf_file is None:return pd.Series(0.,index=dates)
    r=pd.read_csv(rf_file);r["date"]=pd.to_datetime(r.date)
    if "rf_daily" in r:return r.set_index("date").rf_daily.reindex(dates).ffill().fillna(0)
    if "annual_rate" in r:return (r.set_index("date").annual_rate/TD).reindex(dates).ffill().fillna(0)
    raise ValueError("RF file needs date + rf_daily or annual_rate")

def load_signal_price(signal_file,dates,price_column="adj_close"):
    p=Path(signal_file)
    if p.suffix.lower()==".parquet":x=pd.read_parquet(p)
    else:x=pd.read_csv(p)
    if "date" not in x.columns or price_column not in x.columns:
        raise ValueError(f"signal file requires date + {price_column}; columns={list(x.columns)}")
    x=x[["date",price_column]].copy()
    x["date"]=pd.to_datetime(x.date)
    x[price_column]=pd.to_numeric(x[price_column],errors="coerce")
    x=x.dropna().sort_values("date").drop_duplicates("date",keep="last").set_index("date")[price_column]
    # Forward-fill only from already-observed signal dates; never back-fill.
    return x.reindex(dates).ffill()

def target_exposure(signal_px,rf,bull=.35,bear=0.,cap=3.):
    """Frozen prior-day signal: QQQ price/200DMA and 20d QQQ excess-return vol."""
    signal_px=pd.Series(signal_px).astype(float)
    qret=signal_px.pct_change()
    ex=qret-rf.reindex(signal_px.index).ffill().fillna(0)
    vol=ex.rolling(20).std()*math.sqrt(TD)
    ma=signal_px.rolling(200).mean()
    e=pd.Series(0.,index=signal_px.index)
    for i in range(1,len(signal_px)):
        j=i-1
        if np.isfinite(vol.iloc[j]) and vol.iloc[j]>0 and np.isfinite(ma.iloc[j]) and np.isfinite(signal_px.iloc[j]):
            t=bull if signal_px.iloc[j]>ma.iloc[j] else bear
            e.iloc[i]=np.clip(t/vol.iloc[j],0,cap)
    return e

def backtest(c,rf,signal_px,mode="35_0",cost_per_1x_turnover=.0002):
    if mode=="35_0":e=target_exposure(signal_px,rf,.35,0)
    elif mode=="35_12":e=target_exposure(signal_px,rf,.35,.12)
    elif mode=="25":e=target_exposure(signal_px,rf,.25,.25)
    elif mode=="2x_200":
        ma=signal_px.rolling(200).mean();e=pd.Series(0.,index=c.index)
        prior=(signal_px.iloc[:-1].values>ma.iloc[:-1].values)&np.isfinite(ma.iloc[:-1].values)
        e.iloc[1:]=np.where(prior,2.,0.)
    elif mode=="1x":e=pd.Series(1.,index=c.index)
    else:raise ValueError(mode)
    e=e.reindex(c.index).fillna(0)
    # Correct collateralized futures accounting: collateral yield + notional futures P/L.
    r=rf.reindex(c.index).ffill().fillna(0)+e*c.fut_ret.fillna(0)-cost_per_1x_turnover*e.diff().abs().fillna(e.abs())
    return r,e

def metrics(r):
    r=pd.Series(r).dropna();eq=(1+r).cumprod();yrs=len(r)/TD;dd=eq/eq.cummax()-1
    return {"years":yrs,"cagr":eq.iloc[-1]**(1/yrs)-1,"max_dd":dd.min(),"vol":r.std()*math.sqrt(TD),
            "terminal":eq.iloc[-1],"worst_day":r.min()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("files",nargs="+")
    ap.add_argument("--rule",choices=["volume_5bd","oi_5bd","fixed_5bd"],default="volume_5bd")
    ap.add_argument("--rf-file")
    ap.add_argument("--signal-file",required=True,
                    help="Frozen external strategy signal history; for this research program use QQQ adjusted close.")
    ap.add_argument("--signal-price-column",default="adj_close")
    ap.add_argument("--out",default="results/cme_datamine_futures")
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    z=canonicalize(a.files);c=make_continuous(z,a.rule);rf=rf_series(c.index,a.rf_file)
    signal_px=load_signal_price(a.signal_file,c.index,a.signal_price_column)
    rows=[]
    for mode in ["1x","25","35_12","35_0","2x_200"]:
        r,e=backtest(c,rf,signal_px,mode)
        rows.append({"mode":mode,**metrics(r),"avg_exposure":e.mean(),"max_exposure":e.max(),
                     "turnover_per_year":e.diff().abs().sum()/(len(e)/TD)})
    pd.DataFrame(rows).to_csv(out/"strategy_results.csv",index=False);c.to_csv(out/"continuous_audit.csv")
    pd.DataFrame({"date":c.index,"signal_price":signal_px.values}).to_csv(out/"frozen_signal_audit.csv",index=False)
    (out/"manifest.json").write_text(json.dumps({
      "source_requirement":"Official CME DataMine contract-level historical settlement exports legitimately acquired by user/institution.",
      "pnl_source":"CME contract-level settlements only.",
      "signal_source":"External frozen signal file. For the frozen 35/0 program this must be QQQ adjusted close; futures prices must not redefine the strategy signal.",
      "signal_rule":"prior-day QQQ adjusted close vs 200DMA; 20-session QQQ excess-return volatility; 35% target above trend / 0 below; cap 3x.",
      "roll_rule":a.rule,
      "roll_accounting":"Roll-day return is computed using old contract settlement through roll close, then state switches to new contract; no artificial level jump is counted as P/L.",
      "return_model":"portfolio RF collateral yield + frozen external exposure * futures settlement return - explicit turnover cost",
      "lookahead":"All dynamic exposures use prior-day external signal data. Signal alignment forward-fills past observations only; it never back-fills.",
      "warning":"Do not label output exchange-grade unless input provenance is verified as CME settlement data and contract/expiry fields are audited."
    },indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
