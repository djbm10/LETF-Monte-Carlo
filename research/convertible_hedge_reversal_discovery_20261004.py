from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from bs4 import BeautifulSoup
from scipy import stats
import yfinance as yf

OUT=Path("results/convertible_hedge_reversal_discovery_20261004")
OUT.mkdir(parents=True,exist_ok=True)
PREREG=Path("research/CONVERTIBLE_HEDGE_REVERSAL_PREREG.md")

TRAIN_START=pd.Timestamp("2009-01-01")
TRAIN_END=pd.Timestamp("2014-12-31")
VALID_START=pd.Timestamp("2015-01-01")
VALID_END=pd.Timestamp("2019-12-31")
HOLDOUT_START=pd.Timestamp("2020-01-01")
HORIZONS=(5,10,20)
ROUND_TRIP_COST=.0015
BOOT_DRAWS=5000
SEED=20261004

SEC_SEARCH="https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES="https://www.sec.gov/Archives/edgar/data"
HEADERS={
    "User-Agent":"DouglasOSResearch/1.0 https://github.com/djbm10/LETF-Monte-Carlo",
    "Accept-Encoding":"gzip, deflate",
}
FORMS="8-K,424B5"
CONVERT_QUERIES=['"convertible senior notes" AND priced','"convertible notes" AND priced']
STRAIGHT_QUERY='"senior notes" AND priced NOT convertible NOT exchangeable'

def req(url,params=None,tries=5,timeout=30):
    last=None
    for k in range(tries):
        try:
            r=requests.get(url,params=params,headers=HEADERS,timeout=timeout)
            if r.status_code in (429,403,500,502,503,504):
                time.sleep(1.5*(k+1)); last=RuntimeError(f"{r.status_code} {r.url}"); continue
            r.raise_for_status()
            time.sleep(.16)
            return r
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise last

def search_efts(query,start,end,max_hits=1500):
    hits=[]
    offset=0
    while offset<max_hits:
        params={"q":query,"forms":FORMS,"dateRange":"custom","startdt":start,"enddt":end,"from":offset,"size":100}
        j=req(SEC_SEARCH,params=params).json()
        hh=j.get("hits",{}).get("hits",[])
        if not hh: break
        hits.extend(hh)
        total=j.get("hits",{}).get("total",{})
        total=int(total.get("value",len(hits))) if isinstance(total,dict) else int(total or len(hits))
        offset+=len(hh)
        if offset>=total or len(hh)<10: break
    return hits[:max_hits]

def source_cik(src):
    c=src.get("ciks") or src.get("cik") or []
    if not isinstance(c,list): c=[c]
    if c:
        s=re.sub(r"\D","",str(c[0]))
        if s:return s.zfill(10)
    dn=" ".join(src.get("display_names") or [])
    m=re.search(r"CIK\s*0*([0-9]+)",dn,re.I)
    return m.group(1).zfill(10) if m else None

def hit_record(hit):
    src=hit.get("_source",{})
    ident=str(hit.get("_id",""))
    if ":" not in ident:return None
    acc,doc=ident.split(":",1)
    cik=source_cik(src)
    if not cik:return None
    fd=src.get("file_date") or src.get("filedAt") or src.get("filed")
    if not fd:return None
    url=f"{SEC_ARCHIVES}/{int(cik)}/{acc.replace('-','')}/{doc}"
    return {"cik":cik,"filing_date":str(fd)[:10],"accession":acc,"document":doc,"url":url,
            "form":src.get("form_type") or src.get("root_form") or ""}

def text_from_url(url):
    r=req(url,timeout=45)
    raw=r.text
    if "<" in raw and ">" in raw:
        txt=BeautifulSoup(raw,"html.parser").get_text(" ",strip=True)
    else:txt=raw
    return re.sub(r"\s+"," ",html.unescape(txt)).strip()

def parse_money(s):
    return float(s.replace(",",""))

def scale_money(v,unit):
    u=(unit or "").lower()
    return v*(1e9 if "billion" in u or u=="bn" else 1e6 if "million" in u or u=="mm" else 1)

def parse_principal(text):
    pats=[
      r"aggregate principal amount of\s*\$\s*([0-9][0-9,.]*)\s*(billion|million|bn|mm)?",
      r"\$\s*([0-9][0-9,.]*)\s*(billion|million|bn|mm)?\s*(?:aggregate\s+)?principal amount",
      r"offering of\s*\$\s*([0-9][0-9,.]*)\s*(billion|million|bn|mm)?\s+(?:aggregate principal amount of\s+)?(?:its\s+)?(?:[0-9.]+%\s+)?(?:convertible\s+)?senior notes",
    ]
    for p in pats:
        m=re.search(p,text,re.I)
        if m:
            v=scale_money(parse_money(m.group(1)),m.group(2))
            if 5e6<=v<=50e9:return v
    return None

def parse_conversion(text):
    rate=None;price=None
    pats_rate=[
      r"initial conversion rate(?:\s+is|\s+of)?\s*([0-9]+(?:\.[0-9]+)?)\s+shares[^.]{0,160}?\$\s*1,?000",
      r"convertible at an initial conversion rate(?:\s+of)?\s*([0-9]+(?:\.[0-9]+)?)\s+shares[^.]{0,160}?\$\s*1,?000",
    ]
    for p in pats_rate:
        m=re.search(p,text,re.I)
        if m:
            rate=float(m.group(1));break
    pats_price=[
      r"initial conversion price(?:\s+is|\s+of)?\s*(?:approximately\s*)?\$\s*([0-9]+(?:\.[0-9]+)?)",
      r"equivalent to an initial conversion price(?:\s+of)?\s*(?:approximately\s*)?\$\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for p in pats_price:
        m=re.search(p,text,re.I)
        if m:
            price=float(m.group(1));break
    if price is None and rate and rate>0:price=1000.0/rate
    if rate is None and price and price>0:rate=1000.0/price
    if price is not None and not(0.1<=price<=100000):return None,None
    if rate is not None and not(0.001<=rate<=10000):return None,None
    return rate,price

def repurchase_flag(text):
    pats=[
      r"concurrent(?:ly)?[^.]{0,220}repurchas",
      r"repurchas[^.]{0,220}(?:common stock|common shares|shares of its common)",
      r"enter(?:ed|ing)? into[^.]{0,160}repurchase",
      r"use[^.]{0,120}proceeds[^.]{0,180}repurchas",
    ]
    return any(re.search(p,text,re.I) for p in pats)

def looks_convert(text):
    return bool(re.search(r"convertible\s+(?:senior\s+)?notes|convertible notes",text,re.I))

def looks_priced(text):
    return bool(re.search(r"priced|pricing|offering",text,re.I))

def parse_hits(hits,kind,cache_dir):
    cache_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for idx,h in enumerate(hits):
        rec=hit_record(h)
        if not rec:continue
        cache=cache_dir/(rec["accession"].replace("-","")+"_"+re.sub(r"[^A-Za-z0-9_.-]","_",rec["document"])+".txt")
        try:
            if cache.exists():txt=cache.read_text(errors="ignore")
            else:
                txt=text_from_url(rec["url"]);cache.write_text(txt)
        except Exception as e:
            rows.append({**rec,"kind":kind,"parse_status":"fetch_error","error":repr(e)});continue
        principal=parse_principal(txt)
        isconv=looks_convert(txt)
        if kind=="convert":
            rate,price=parse_conversion(txt)
            ok=isconv and looks_priced(txt) and principal is not None and price is not None
            rows.append({**rec,"kind":kind,"parse_status":"ok" if ok else "term_parse_fail",
                         "principal":principal,"conversion_rate":rate,"conversion_price":price,
                         "repurchase_offset":bool(repurchase_flag(txt)),"text_chars":len(txt)})
        else:
            ok=(not isconv) and ("exchangeable" not in txt.lower()) and looks_priced(txt) and principal is not None
            rows.append({**rec,"kind":kind,"parse_status":"ok" if ok else "term_parse_fail",
                         "principal":principal,"conversion_rate":None,"conversion_price":None,
                         "repurchase_offset":False,"text_chars":len(txt)})
        if idx and idx%100==0:print("PARSED",kind,idx,"/",len(hits),flush=True)
    return pd.DataFrame(rows)

def clean_membership(path):
    m=pd.read_csv(path,dtype={"cik":str})
    def cd(x):return pd.to_datetime(x.astype(str).str.replace("*","",regex=False).str.strip(),errors="coerce").dt.normalize()
    m["date_added"]=cd(m["date_added"]);m["date_removed"]=cd(m["date_removed"])
    m["cik"]=m["cik"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(10)
    m["symbol"]=m["symbol"].astype(str).str.upper().str.replace("-",".",regex=False)
    return m

def attach_membership(events,m):
    rows=[]
    for r in events.itertuples(index=False):
        d=pd.Timestamp(r.filing_date)
        g=m[(m.cik==str(r.cik).zfill(10))&(m.date_added<=d)&(m.date_removed.isna()|(m.date_removed>d))]
        if g.empty:continue
        for sym in sorted(g.symbol.unique()):
            rows.append({**r._asdict(),"symbol":sym})
    return pd.DataFrame(rows)

def load_discovery_prices(path):
    dataset=ds.dataset(path,format="parquet")
    filt=ds.field("Date") < np.datetime64("2020-01-01")
    cols=["Date","symbol","Adj Close","Volume"]
    table=dataset.to_table(columns=cols,filter=filt)
    p=table.to_pandas()
    p["Date"]=pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize()
    p["symbol"]=p.symbol.astype(str).str.upper().str.replace("-",".",regex=False)
    p=p[p.Date<=VALID_END].copy()
    if p.Date.max()>VALID_END:raise AssertionError("HOLDOUT PRICE BREACH")
    return p.sort_values(["symbol","Date"]).drop_duplicates(["symbol","Date"],keep="last")

def dl_spy():
    x=yf.download("SPY",start="2008-10-01",end="2020-01-01",auto_adjust=True,progress=False,threads=False)
    if x.empty:raise RuntimeError("SPY unavailable")
    s=x["Close"];s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s
    s=pd.to_numeric(s,errors="coerce").dropna();s.index=pd.to_datetime(s.index).tz_localize(None).normalize()
    if s.index.max()>VALID_END:raise AssertionError("HOLDOUT SPY BREACH")
    return s

def cluster_events(df):
    if df.empty:return df
    out=[]
    for (cik,kind),g in df.sort_values(["cik","filing_date","text_chars"],ascending=[True,True,False]).groupby(["cik","kind"]):
        last=None;cluster=[]
        for _,r in g.iterrows():
            d=pd.Timestamp(r.filing_date)
            if last is None or (d-last).days>7:
                if cluster:
                    out.append(sorted(cluster,key=lambda z:(z["filing_date"],-z.get("text_chars",0)))[0])
                cluster=[]
            cluster.append(r.to_dict());last=d
        if cluster:out.append(sorted(cluster,key=lambda z:(z["filing_date"],-z.get("text_chars",0)))[0])
    return pd.DataFrame(out)

def event_price_metrics(events,prices,spy,needs_conversion):
    grouped={s:g.set_index("Date").sort_index() for s,g in prices.groupby("symbol")}
    rows=[]
    denom=0
    for r in events.itertuples(index=False):
        denom+=1
        g=grouped.get(r.symbol)
        if g is None:continue
        fd=pd.Timestamp(r.filing_date)
        pre=g[g.index<fd].tail(20)
        postdates=g[g.index>fd].index
        if len(pre)<20 or len(postdates)<21:continue
        if pre["Adj Close"].isna().any() or pre["Volume"].isna().any() or (pre.Volume<=0).any():continue
        entry=postdates[0]
        # Common SPY dates must exist for all measured points.
        pre_event_dates=g[g.index<fd].index
        if len(pre_event_dates)==0:continue
        reaction_start=pre_event_dates[-1]
        if reaction_start not in spy.index or entry not in spy.index:continue
        s0=float(g.loc[reaction_start,"Adj Close"]);s1=float(g.loc[entry,"Adj Close"])
        if s0<=0 or s1<=0:continue
        reaction=(s1/s0-1)-(float(spy.loc[entry])/float(spy.loc[reaction_start])-1)
        adv20=float(pre.Volume.mean())
        if not np.isfinite(adv20) or adv20<=0:continue
        base={**r._asdict(),"entry_date":entry,"reaction_start":reaction_start,
              "reaction_abnormal":reaction,"adv20_shares":adv20}
        if needs_conversion:
            eq=float(r.principal)/float(r.conversion_price)
            base["conversion_equiv_shares"]=eq
            base["hedge_pressure_score"]=eq/adv20
        else:
            base["conversion_equiv_shares"]=np.nan;base["hedge_pressure_score"]=np.nan
        for h in HORIZONS:
            if len(postdates)<=h:continue
            ex=postdates[h]
            if ex not in spy.index:continue
            raw=float(g.loc[ex,"Adj Close"])/s1-1
            spyret=float(spy.loc[ex])/float(spy.loc[entry])-1
            base[f"ret_{h}d_raw"]=raw
            base[f"ret_{h}d_abnormal_net"]=raw-spyret-ROUND_TRIP_COST
        if all(f"ret_{h}d_abnormal_net" in base for h in HORIZONS):rows.append(base)
    return pd.DataFrame(rows),denom

def part(d):
    d=pd.Timestamp(d)
    if TRAIN_START<=d<=TRAIN_END:return "train"
    if pd.Timestamp("2015-01-01")<=d<=VALID_END:return "validation"
    return None

def match_straight(convert,straight):
    if convert.empty or straight.empty:return pd.DataFrame()
    rows=[]
    straight=straight.copy();straight["year"]=pd.to_datetime(straight.filing_date).dt.year
    straight["logadv"]=np.log(straight.adv20_shares)
    for r in convert.itertuples(index=False):
        y=pd.Timestamp(r.filing_date).year;la=math.log(float(r.adv20_shares))
        cand=straight[straight.year==y]
        if cand.empty:cand=straight[(straight.year>=y-1)&(straight.year<=y+1)]
        if cand.empty:continue
        c=cand.assign(dist=(cand.logadv-la).abs()).sort_values(["dist","filing_date"]).iloc[0]
        rows.append({"convert_cik":r.cik,"convert_symbol":r.symbol,"convert_filing_date":r.filing_date,
                     "straight_cik":c.cik,"straight_symbol":c.symbol,"straight_filing_date":c.filing_date,
                     "convert_reaction":r.reaction_abnormal,"straight_reaction":float(c.reaction_abnormal),
                     "reaction_diff":float(r.reaction_abnormal-c.reaction_abnormal)})
    return pd.DataFrame(rows)

def summary_stats(x):
    z=np.asarray(pd.Series(x).dropna(),dtype=float)
    if len(z)==0:return {"n":0}
    mean=float(z.mean());med=float(np.median(z));pos=float((z>0).mean())
    if len(z)>1 and z.std(ddof=1)>0:
        t=float(mean/(z.std(ddof=1)/math.sqrt(len(z))))
        p=float(2*stats.t.sf(abs(t),df=len(z)-1))
    else:t=p=None
    rng=np.random.default_rng(SEED+len(z))
    if len(z)>1:
        boot=z[rng.integers(0,len(z),(BOOT_DRAWS,len(z)))].mean(axis=1)
        lo,hi=float(np.quantile(boot,.025)),float(np.quantile(boot,.975))
    else:lo=hi=mean
    return {"n":int(len(z)),"mean":mean,"median":med,"positive_pct":pos,"tstat":t,"pvalue_2s":p,
            "bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"worst":float(z.min()),"best":float(z.max())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--prices");ap.add_argument("--membership");ap.add_argument("--probe",action="store_true")
    a=ap.parse_args()
    prereg_sha=hashlib.sha256(PREREG.read_bytes()).hexdigest()

    if a.probe:
        h=search_efts(CONVERT_QUERIES[0],"2019-01-01","2019-12-31",max_hits=10)
        print("SEC_PROBE_HITS",len(h),flush=True)
        if h:
            rec=hit_record(h[0]);print("SEC_PROBE_SAMPLE",json.dumps(rec,indent=2),flush=True)
        if not h:raise SystemExit("SEC probe returned zero hits")
        return

    if not a.prices or not a.membership:raise SystemExit("--prices and --membership required")
    all_conv=[];all_straight=[]
    cache=OUT/"filing_cache"
    for year in range(2009,2020):
        ys=f"{year}-01-01";ye=f"{year}-12-31"
        ch=[]
        for q in CONVERT_QUERIES:ch.extend(search_efts(q,ys,ye,max_hits=500))
        # de-duplicate raw search hits
        uniq={str(x.get("_id")):x for x in ch if x.get("_id")}
        print("SEC_SEARCH",year,"convert_hits",len(uniq),flush=True)
        all_conv.append(parse_hits(list(uniq.values()),"convert",cache/"convert"))
        sh=search_efts(STRAIGHT_QUERY,ys,ye,max_hits=800)
        print("SEC_SEARCH",year,"straight_hits",len(sh),flush=True)
        all_straight.append(parse_hits(sh,"straight",cache/"straight"))

    parsed=pd.concat(all_conv+all_straight,ignore_index=True,sort=False)
    parsed.to_csv(OUT/"sec_parsed_candidates.csv",index=False)
    ok=parsed[parsed.parse_status=="ok"].copy()
    conv=cluster_events(ok[ok.kind=="convert"])
    straight=cluster_events(ok[ok.kind=="straight"])

    m=clean_membership(a.membership)
    conv_m=attach_membership(conv,m);straight_m=attach_membership(straight,m)
    conv_member_n=len(conv_m);straight_member_n=len(straight_m)
    if conv_m.empty:raise RuntimeError("No parsed convertible events matched PIT S&P membership")

    prices=load_discovery_prices(a.prices)
    spy=dl_spy()
    conv_px,conv_denom=event_price_metrics(conv_m,prices,spy,True)
    straight_px,straight_denom=event_price_metrics(straight_m,prices,spy,False)
    if not conv_px.empty:conv_px["partition"]=conv_px.filing_date.map(part)
    if not straight_px.empty:straight_px["partition"]=straight_px.filing_date.map(part)
    conv_px=conv_px[conv_px.partition.notna()].copy();straight_px=straight_px[straight_px.partition.notna()].copy()

    no=conv_px[~conv_px.repurchase_offset.astype(bool)].copy()
    off=conv_px[conv_px.repurchase_offset.astype(bool)].copy()

    # TRAIN-only pressure terciles; applied unchanged to validation.
    train_scores=no[no.partition=="train"].hedge_pressure_score.dropna()
    q1,q2=(float(train_scores.quantile(1/3)),float(train_scores.quantile(2/3))) if len(train_scores)>=3 else (np.nan,np.nan)
    def tier(x):
        if not np.isfinite(q1) or not np.isfinite(x):return "unknown"
        return "low" if x<=q1 else "mid" if x<=q2 else "high"
    no["pressure_tier"]=no.hedge_pressure_score.map(tier)

    matches=match_straight(no,straight_px)
    if not matches.empty:matches["partition"]=matches.convert_filing_date.map(part)

    summaries=[]
    for p in ("train","validation"):
        g=no[no.partition==p]
        summaries.append({"partition":p,"metric":"pricing_reaction","horizon":0,**summary_stats(g.reaction_abnormal)})
        for h in HORIZONS:
            summaries.append({"partition":p,"metric":"reversal","horizon":h,**summary_stats(g[f"ret_{h}d_abnormal_net"])})
            for tiername in ("low","mid","high"):
                z=g[g.pressure_tier==tiername]
                summaries.append({"partition":p,"metric":f"reversal_{tiername}","horizon":h,**summary_stats(z[f"ret_{h}d_abnormal_net"])})
        mm=matches[matches.partition==p] if not matches.empty else pd.DataFrame()
        summaries.append({"partition":p,"metric":"reaction_minus_matched_straight","horizon":0,
                          **summary_stats(mm.reaction_diff if not mm.empty else [])})
        if len(off[off.partition==p])>=1:
            diff=float(g.reaction_abnormal.mean()-off[off.partition==p].reaction_abnormal.mean()) if len(g) else np.nan
            summaries.append({"partition":p,"metric":"reaction_minus_offset_convert_mean","horizon":0,
                              "n":int(min(len(g),len(off[off.partition==p]))),"mean":diff})
    summary=pd.DataFrame(summaries)

    def mean_metric(p,metric,h):
        z=summary[(summary.partition==p)&(summary.metric==metric)&(summary.horizon==h)]
        return float(z.iloc[0]["mean"]) if len(z) and pd.notna(z.iloc[0]["mean"]) else np.nan

    mechanism={
      "train_reaction_minus_straight":mean_metric("train","reaction_minus_matched_straight",0),
      "validation_reaction_minus_straight":mean_metric("validation","reaction_minus_matched_straight",0),
      "train_high_minus_low_pressure_reaction":(
          float(no[(no.partition=="train")&(no.pressure_tier=="high")].reaction_abnormal.mean()
                -no[(no.partition=="train")&(no.pressure_tier=="low")].reaction_abnormal.mean())
          if len(no[(no.partition=="train")&(no.pressure_tier=="high")]) and len(no[(no.partition=="train")&(no.pressure_tier=="low")]) else np.nan),
      "validation_high_minus_low_pressure_reaction":(
          float(no[(no.partition=="validation")&(no.pressure_tier=="high")].reaction_abnormal.mean()
                -no[(no.partition=="validation")&(no.pressure_tier=="low")].reaction_abnormal.mean())
          if len(no[(no.partition=="validation")&(no.pressure_tier=="high")]) and len(no[(no.partition=="validation")&(no.pressure_tier=="low")]) else np.nan),
      "train_pressure_terciles":{"q1":q1,"q2":q2},
      "offset_convert_counts":{"train":int(len(off[off.partition=="train"])),"validation":int(len(off[off.partition=="validation"]))},
    }
    mechanism_pass=all(np.isfinite([mechanism["train_reaction_minus_straight"],mechanism["validation_reaction_minus_straight"],
                                    mechanism["train_high_minus_low_pressure_reaction"],mechanism["validation_high_minus_low_pressure_reaction"]])) and \
                   mechanism["train_reaction_minus_straight"]<0 and mechanism["validation_reaction_minus_straight"]<0 and \
                   mechanism["train_high_minus_low_pressure_reaction"]<0 and mechanism["validation_high_minus_low_pressure_reaction"]<0
    mechanism["passed"]=bool(mechanism_pass)

    overall_parsed_member=max(conv_member_n,1)
    price_resolution=float(len(conv_px)/overall_parsed_member)
    train_tiers=no[no.partition=="train"].pressure_tier.value_counts()
    dq={
      "no_offset_convert_total":int(len(no)),
      "no_offset_convert_validation":int(len(no[no.partition=="validation"])),
      "straight_control_total":int(len(straight_px)),
      "convert_member_events_before_price_gate":int(conv_member_n),
      "convert_price_resolution":price_resolution,
      "train_low_n":int(train_tiers.get("low",0)),"train_mid_n":int(train_tiers.get("mid",0)),"train_high_n":int(train_tiers.get("high",0)),
      "max_price_date":str(prices.Date.max().date()),"max_spy_date":str(spy.index.max().date()),
    }
    dq_pass=bool(len(no)>=40 and len(no[no.partition=="validation"])>=15 and len(straight_px)>=30 and price_resolution>=.80 and
                 min(dq["train_low_n"],dq["train_mid_n"],dq["train_high_n"])>=10 and prices.Date.max()<=VALID_END and spy.index.max()<=VALID_END)
    dq["passed"]=dq_pass

    val_means=[mean_metric("validation","reversal",h) for h in HORIZONS]
    train_means=[mean_metric("train","reversal",h) for h in HORIZONS]
    val_positive=sum(np.isfinite(x) and x>0 for x in val_means)
    high_gt_low=0
    for h in HORIZONS:
        hi=mean_metric("validation","reversal_high",h);lo=mean_metric("validation","reversal_low",h)
        high_gt_low+=int(np.isfinite(hi) and np.isfinite(lo) and hi>lo)
    median_val=float(np.nanmedian(val_means));median_train=float(np.nanmedian(train_means))
    advance=bool(dq_pass and mechanism_pass and val_positive>=2 and median_val>0 and median_train>0 and high_gt_low>=2)
    disposition=("MONITOR_PENDING_HOLDOUT" if advance else ("KILL" if dq_pass else "MONITOR_DATA_INSUFFICIENT"))
    advancement={
      "preregistration":str(PREREG),"prereg_sha256":hashlib.sha256(PREREG.read_bytes()).hexdigest(),
      "holdout_status":"SEALED_NOT_DOWNLOADED","discovery_event_end":"2019-12-31",
      "data_quality":dq,"mechanism":mechanism,
      "validation_positive_horizons":int(val_positive),"median_validation_reversal":median_val,
      "median_train_reversal":median_train,"validation_high_pressure_gt_low_horizons":int(high_gt_low),
      "decision":"ADVANCE_TO_SINGLE_FROZEN_HOLDOUT" if advance else "DO_NOT_OPEN_HOLDOUT",
      "disposition":disposition,
    }

    no.to_csv(OUT/"convertible_events_scored.csv",index=False)
    off.to_csv(OUT/"repurchase_offset_convertibles.csv",index=False)
    straight_px.to_csv(OUT/"straight_debt_controls_scored.csv",index=False)
    matches.to_csv(OUT/"matched_straight_controls.csv",index=False)
    summary.to_csv(OUT/"discovery_summary.csv",index=False)
    (OUT/"mechanism_diagnostics.json").write_text(json.dumps(mechanism,indent=2,default=float))
    (OUT/"advancement.json").write_text(json.dumps(advancement,indent=2,default=float))
    (OUT/"source_manifest.json").write_text(json.dumps({
      "sec":"SEC EDGAR EFTS + original filing/exhibit documents, 2009-2019 only",
      "price_panel":"existing survivor-aware historical S&P panel, read with pre-2020 parquet predicate",
      "membership":"lawcal components_history with CIK/date intervals",
      "spy":"Yahoo SPY download ending 2020-01-01 exclusive",
      "holdout":"not fetched",
    },indent=2))

    if advance:
        (OUT/"frozen_holdout_spec.json").write_text(json.dumps({
          "status":"READY_FOR_SINGLE_HOLDOUT_RUN","prereg_sha256":advancement["prereg_sha256"],
          "train_pressure_terciles":{"q1":q1,"q2":q2},
          "rule":"Use identical SEC parsing, PIT S&P universe, event clustering, entry timing, 15bp pair cost, straight-debt matching, and 5/10/20-day horizons on 2020+ exactly once."
        },indent=2))

    print("DISCOVERY_HOLDOUT_FIREWALL",dq["max_price_date"],dq["max_spy_date"],flush=True)
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(advancement,indent=2,default=float),flush=True)

if __name__=="__main__":main()
