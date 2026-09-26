from __future__ import annotations

"""Complete an authoritative direct-SEC Item 1 network from a preserved cache.

The preserved cache contains complete direct-SEC 10-K histories for a subset of
curated S&P issuer CIKs. This script fetches only the remaining issuer CIKs
using the repo's authoritative SEC submission parser, combines the two direct-
SEC sources, and rebuilds the 2009-2019 PIT network.

No Hugging Face text enters the final network.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import sec_item1_network as net
import sec_item1_sp500_network as sp


def normalize_filings(x: pd.DataFrame, source: str) -> pd.DataFrame:
    if x is None or x.empty:
        return pd.DataFrame(
            columns=["cik", "filing_date", "filename", "item1", "word_count", "error", "source"]
        )
    y=x.copy()
    y["cik"]=y["cik"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(10)
    y["filing_date"]=pd.to_datetime(y["filing_date"])
    y["source"]=source
    return y


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--cached-sec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2007)
    ap.add_argument("--end-year", type=int, default=2019)
    ap.add_argument("--formation-start", default="2009-03-31")
    ap.add_argument("--formation-end", default="2019-12-31")
    ap.add_argument("--max-item1-age-days", type=int, default=550)
    ap.add_argument("--pair-density", type=float, default=0.0205)
    ap.add_argument("--rps", type=float, default=4.0)
    args=ap.parse_args()

    args.out.mkdir(parents=True,exist_ok=True)
    ua=os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT with contact email is required")

    membership=sp.load_membership(args.membership)
    dates=list(pd.date_range(args.formation_start,args.formation_end,freq="QE"))
    union_start=min(dates)-pd.Timedelta(days=args.max_item1_age_days)
    union_end=max(dates)
    relevant=membership[
        (membership.date_added<=union_end)
        & (membership.date_removed.isna() | (membership.date_removed>=union_start))
    ]
    union_ciks=set(relevant.cik.astype(str))
    cached=normalize_filings(pd.read_parquet(args.cached_sec),"SEC_DIRECT_CACHE")
    cached=cached[cached.cik.isin(union_ciks)].copy()
    cached_ciks=set(cached.cik.astype(str))

    missing=sorted(union_ciks-cached_ciks)
    (args.out/"missing_ciks_before_fetch.txt").write_text("\n".join(missing)+"\n")
    print(
        "DIRECT_SEC_COMPLETION_START",
        "union_ciks",len(union_ciks),
        "cached_ciks",len(cached_ciks),
        "missing_ciks",len(missing),
        "cached_rows",len(cached),
        flush=True,
    )

    client=net.SecClient(ua,requests_per_second=args.rps)
    idx=net.filing_index(
        client,args.start_year,args.end_year,forms=("10-K",),
        cache=args.out/"index_cache"
    )
    idx=idx[idx.cik.isin(set(missing))].copy()
    idx.to_csv(args.out/"new_sec_index.csv",index=False)
    print("DIRECT_SEC_NEW_INDEX","filings",len(idx),"ciks",idx.cik.nunique(),flush=True)

    fresh=net.fetch_item1_rows(
        client,idx,args.out/"new_item1_cache",ciks=set(missing)
    )
    fresh=normalize_filings(fresh,"SEC_DIRECT_NEW")
    fresh.to_parquet(args.out/"new_direct_sec_filings.parquet",index=False)

    combined=pd.concat([cached,fresh],ignore_index=True,sort=False)
    combined=(
        combined.sort_values(["cik","filing_date","filename"])
        .drop_duplicates(["cik","filing_date","filename"],keep="last")
        .reset_index(drop=True)
    )
    combined.to_parquet(args.out/"direct_sec_filings_all.parquet",index=False)

    metrics_all=[]
    cov=[]
    manifests=[]
    for fd in dates:
        active=sp.active_ciks(membership,fd)
        sample=net.latest_asof(
            combined[combined.cik.isin(active)],fd,
            max_age_days=args.max_item1_age_days
        )
        cov.append({
            "formation_date":str(fd.date()),
            "active_ciks":len(active),
            "valid_item1_ciks":int(sample.cik.nunique()) if len(sample) else 0,
        })
        if len(sample)<20:
            raise RuntimeError(f"{fd.date()}: only {len(sample)} direct-SEC Item1 issuers")
        metrics,edges,manifest=net.build_network(
            sample,fd,pair_density=args.pair_density
        )
        manifest["source"]="direct SEC EDGAR original 10-K submissions only"
        manifest["universe"]="CURATED_PIT_S_AND_P_500_ISSUER_PROXY"
        manifest["max_item1_age_days"]=args.max_item1_age_days
        metrics_all.append(metrics)
        manifests.append(manifest)
        print(
            "DIRECT_SEC_NET",fd.date(),
            "ACTIVE",len(active),"VALID",len(sample),"EDGES",len(edges),
            flush=True
        )

    metrics=pd.concat(metrics_all,ignore_index=True).sort_values(["formation_date","cik"])
    coverage=pd.DataFrame(cov)
    metrics.to_csv(args.out/"network_metrics_all.csv",index=False)
    coverage.to_csv(args.out/"membership_network_coverage.csv",index=False)

    minv=int(coverage.valid_item1_ciks.min())
    med=float(coverage.valid_item1_ciks.median())
    manifest={
        "source":"direct SEC EDGAR original 10-K submissions only",
        "universe":"CURATED_PIT_S_AND_P_500_ISSUER_PROXY",
        "evidence_label":"FREE_DISCOVERY_DIRECT_SEC_NETWORK_REPLICATION",
        "formation_start":str(min(dates).date()),
        "formation_end":str(max(dates).date()),
        "forms":["10-K"],
        "max_item1_age_days":args.max_item1_age_days,
        "pair_density":args.pair_density,
        "union_ciks":len(union_ciks),
        "cached_direct_sec_rows":int(len(cached)),
        "cached_direct_sec_ciks":int(cached.cik.nunique()),
        "new_fetch_target_ciks":len(missing),
        "new_index_rows":int(len(idx)),
        "new_direct_sec_rows":int(len(fresh)),
        "new_direct_sec_success":int(fresh.item1.notna().sum()) if len(fresh) else 0,
        "combined_direct_sec_rows":int(len(combined)),
        "combined_direct_sec_ciks":int(combined.cik.nunique()),
        "valid_item1_min":minv,
        "valid_item1_median":med,
        "network_rows":int(len(metrics)),
        "network_runs":manifests,
        "quality_gate":{
            "min_valid_item1_ciks_required":200,
            "median_valid_item1_ciks_required":350,
            "passed":bool(minv>=200 and med>=350),
        },
        "warning":"Direct-SEC large-cap proxy; still not the frozen full-US universe.",
    }
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    print("DIRECT_SEC_COMPLETION_MANIFEST",json.dumps({
        k:manifest[k] for k in [
            "union_ciks","cached_direct_sec_ciks","new_fetch_target_ciks",
            "new_index_rows","new_direct_sec_success","combined_direct_sec_ciks",
            "valid_item1_min","valid_item1_median","network_rows"
        ]
    }),flush=True)

    if minv<200:
        raise SystemExit(f"DIRECT SEC NETWORK FAIL min {minv} < 200")
    if med<350:
        raise SystemExit(f"DIRECT SEC NETWORK FAIL median {med} < 350")
    print("DIRECT SEC NETWORK REPLICATION PASS",minv,med,flush=True)


if __name__=="__main__":
    main()
