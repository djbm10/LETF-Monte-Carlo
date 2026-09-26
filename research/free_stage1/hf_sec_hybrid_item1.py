from __future__ import annotations

"""Hybrid free Item-1 collector for Bottleneck discovery.

Use the complete public Hugging Face SEC corpus as a cache/accelerator for
historical 10-K Item 1 text, then fetch directly from SEC only for issuer CIKs
that lack a valid <=550-day Item 1 at one or more point-in-time S&P 500
formation dates.

The network parameters remain frozen, but HF section_1 text is produced by an
independent parser and therefore this path is a discovery/replication proxy,
not the authoritative direct-SEC text extraction:
- 10-K only
- actual filing_date information timestamp
- max Item-1 age 550 days
- PIT S&P issuer universe
- TF-IDF cosine network
- target directed pair density 2.05%

HF is never trusted as a substitute for a missing issuer. Direct SEC text fills
that gap. SEC wins duplicate same-CIK/same-filing-date observations.
"""

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path

import orjson
import pandas as pd
import requests

import sec_item1_network as net
import sec_item1_sp500_network as sp


HF_BASE = (
    "https://huggingface.co/datasets/JanosAudran/financial-reports-sec/"
    "resolve/main/data/large"
)


def norm_cik(value) -> str:
    s = str(value).strip()
    return s.zfill(10) if s.isdigit() else s


def hf_urls() -> list[tuple[str, int, str]]:
    return [
        (split, i, f"{HF_BASE}/{split}/shard_{i}.jsonl?download=true")
        for split in ("train", "test", "validate")
        for i in range(10)
    ]


def extract_hf(
    wanted: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int = 6,
) -> tuple[pd.DataFrame, list[dict]]:
    cik_re = re.compile(rb'"cik"\s*:\s*"([0-9]+)"')

    def process(spec):
        split, i, url = spec
        session = requests.Session()
        rows = []
        firms = matched = bytes_seen = 0
        with session.get(url, stream=True, timeout=(30, 600)) as r:
            r.raise_for_status()
            for raw in r.iter_lines(chunk_size=1024 * 1024):
                if not raw:
                    continue
                bytes_seen += len(raw)
                firms += 1
                mm = cik_re.search(raw[:4096]) or cik_re.search(raw)
                if mm is None:
                    continue
                cik = norm_cik(mm.group(1).decode())
                if cik not in wanted:
                    continue
                matched += 1
                data = orjson.loads(raw)
                cik = norm_cik(data.get("cik", ""))
                for filing in data.get("filings") or []:
                    # Frozen authoritative network uses 10-K only.
                    if str(filing.get("form") or "").upper() != "10-K":
                        continue
                    fd = pd.to_datetime(filing.get("filingDate"), errors="coerce")
                    if pd.isna(fd) or fd < start or fd > end:
                        continue
                    report = filing.get("report") or {}
                    sentences = report.get("section_1")
                    if not sentences:
                        continue
                    text = " ".join(str(x).strip() for x in sentences if x)
                    wc = len(text.split())
                    if wc < 250:
                        continue
                    report_date = str(filing.get("reportDate") or "")
                    rows.append(
                        {
                            "cik": cik,
                            "filing_date": fd,
                            "filename": f"HF_RAW:{cik}:10-K:{report_date}",
                            "item1": text,
                            "word_count": wc,
                            "error": None,
                            "source": "HF_RAW_10K",
                            "source_split": split,
                            "source_shard": i,
                        }
                    )
        stat = {
            "split": split,
            "shard": i,
            "bytes": bytes_seen,
            "firms": firms,
            "matched_firms": matched,
            "valid_10k_item1": len(rows),
        }
        print("HF_SHARD", json.dumps(stat), flush=True)
        return rows, stat

    all_rows: list[dict] = []
    stats: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for rows, stat in ex.map(process, hf_urls()):
            all_rows.extend(rows)
            stats.append(stat)

    f = pd.DataFrame(all_rows)
    if f.empty:
        raise RuntimeError("No matching 10-K Item 1 text extracted from HF raw corpus")
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    f = (
        f.sort_values(["cik", "filing_date", "filename"])
        .drop_duplicates(["cik", "filing_date", "filename"], keep="last")
        .reset_index(drop=True)
    )
    return f, stats


def missing_issuers(
    membership: pd.DataFrame,
    filings: pd.DataFrame,
    formation_dates: list[pd.Timestamp],
    max_age_days: int,
) -> tuple[set[str], pd.DataFrame]:
    rows = []
    missing_union: set[str] = set()
    for fd in formation_dates:
        active = sp.active_ciks(membership, fd)
        sample = net.latest_asof(
            filings[filings["cik"].isin(active)],
            fd,
            max_age_days=max_age_days,
        )
        have = set(sample["cik"].astype(str))
        missing = sorted(active - have)
        missing_union.update(missing)
        rows.append(
            {
                "formation_date": str(fd.date()),
                "active_ciks": len(active),
                "hf_valid_ciks": len(have),
                "hf_missing_ciks": len(missing),
            }
        )
    return missing_union, pd.DataFrame(rows)


def fetch_sec_supplement(
    missing: set[str],
    start_year: int,
    end_year: int,
    out: Path,
    rps: float,
) -> pd.DataFrame:
    if not missing:
        return pd.DataFrame(
            columns=[
                "cik",
                "filing_date",
                "filename",
                "item1",
                "word_count",
                "error",
                "source",
            ]
        )
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT with contact email is required")
    client = net.SecClient(ua, requests_per_second=rps)
    idx = net.filing_index(
        client,
        start_year,
        end_year,
        forms=("10-K",),
        cache=out / "sec_index_cache",
    )
    idx = idx[idx["cik"].isin(missing)].copy()
    idx.to_csv(out / "sec_supplement_index.csv", index=False)
    print(
        "SEC_SUPPLEMENT_INDEX",
        "missing_ciks",
        len(missing),
        "filings",
        len(idx),
        flush=True,
    )
    rows = net.fetch_item1_rows(
        client,
        idx,
        out / "sec_item1_cache",
        ciks=missing,
    )
    if rows.empty:
        rows["source"] = []
        return rows
    rows["filing_date"] = pd.to_datetime(rows["filing_date"])
    rows["source"] = "SEC_DIRECT_10K"
    return rows


def combine_filings(hf: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    frames = [hf]
    if sec is not None and len(sec):
        frames.append(sec)
    x = pd.concat(frames, ignore_index=True, sort=False)
    x["cik"] = x["cik"].astype(str).str.zfill(10)
    x["filing_date"] = pd.to_datetime(x["filing_date"])
    x["_priority"] = x["source"].map({"HF_RAW_10K": 1, "SEC_DIRECT_10K": 2}).fillna(0)
    # Same issuer + filing date should represent the same public 10-K. SEC wins.
    x = (
        x.sort_values(["cik", "filing_date", "_priority", "filename"])
        .drop_duplicates(["cik", "filing_date"], keep="last")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )
    return x


def build_networks(
    membership: pd.DataFrame,
    filings: pd.DataFrame,
    formation_dates: list[pd.Timestamp],
    max_age_days: int,
    pair_density: float,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    metrics_all = []
    coverage = []
    manifests = []
    for fd in formation_dates:
        active = sp.active_ciks(membership, fd)
        sample = net.latest_asof(
            filings[filings["cik"].isin(active)],
            fd,
            max_age_days=max_age_days,
        )
        hf_n = int((sample["source"] == "HF_RAW_10K").sum()) if len(sample) else 0
        sec_n = int((sample["source"] == "SEC_DIRECT_10K").sum()) if len(sample) else 0
        coverage.append(
            {
                "formation_date": str(fd.date()),
                "active_ciks": len(active),
                "valid_item1_ciks": int(sample["cik"].nunique()) if len(sample) else 0,
                "hf_source_ciks": hf_n,
                "sec_direct_source_ciks": sec_n,
            }
        )
        if len(sample) < 20:
            raise RuntimeError(f"{fd.date()}: only {len(sample)} valid Item 1 issuers")
        metrics, edges, manifest = net.build_network(
            sample,
            fd,
            pair_density=pair_density,
        )
        manifest["source"] = "HF raw 10-K cache + direct SEC missing-issuer supplement"
        manifest["universe"] = "PIT_S_AND_P_500_ISSUER_PROXY"
        manifest["max_item1_age_days"] = max_age_days
        metrics_all.append(metrics)
        manifests.append(manifest)
        print(
            "HYBRID_NET",
            fd.date(),
            "ACTIVE",
            len(active),
            "VALID",
            len(sample),
            "HF",
            hf_n,
            "SEC",
            sec_n,
            "EDGES",
            len(edges),
            flush=True,
        )

    metrics = pd.concat(metrics_all, ignore_index=True).sort_values(
        ["formation_date", "cik"]
    )
    cov = pd.DataFrame(coverage)
    metrics.to_csv(out / "network_metrics_all.csv", index=False)
    cov.to_csv(out / "membership_network_coverage.csv", index=False)
    return metrics, cov, manifests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--formation-start", default="2009-03-31")
    ap.add_argument("--formation-end", default="2019-12-31")
    ap.add_argument("--hf-start", default="2007-09-01")
    ap.add_argument("--sec-start-year", type=int, default=2007)
    ap.add_argument("--sec-end-year", type=int, default=2019)
    ap.add_argument("--max-item1-age-days", type=int, default=550)
    ap.add_argument("--pair-density", type=float, default=0.0205)
    ap.add_argument("--sec-rps", type=float, default=3.0)
    ap.add_argument("--hf-workers", type=int, default=6)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    membership = sp.load_membership(args.membership)
    dates = list(
        pd.date_range(args.formation_start, args.formation_end, freq="QE")
    )
    union_start = min(dates) - pd.Timedelta(days=args.max_item1_age_days)
    union_end = max(dates)
    relevant = membership[
        (membership["date_added"] <= union_end)
        & (
            membership["date_removed"].isna()
            | (membership["date_removed"] >= union_start)
        )
    ]
    wanted = set(relevant["cik"].astype(str))
    print("PIT_UNION_CIKS", len(wanted), flush=True)

    hf, hf_stats = extract_hf(
        wanted,
        pd.Timestamp(args.hf_start),
        pd.Timestamp(args.formation_end),
        workers=args.hf_workers,
    )
    hf.to_parquet(args.out / "hf_item1_filings.parquet", index=False)

    missing, hf_cov = missing_issuers(
        membership,
        hf,
        dates,
        max_age_days=args.max_item1_age_days,
    )
    hf_cov.to_csv(args.out / "hf_presupplement_coverage.csv", index=False)
    (args.out / "missing_ciks.txt").write_text("\n".join(sorted(missing)) + "\n")
    print(
        "HF_PRE_SUPPLEMENT",
        "valid_filings",
        len(hf),
        "unique_ciks",
        hf["cik"].nunique(),
        "missing_union",
        len(missing),
        "min_valid",
        int(hf_cov["hf_valid_ciks"].min()),
        "median_valid",
        float(hf_cov["hf_valid_ciks"].median()),
        flush=True,
    )

    sec = fetch_sec_supplement(
        missing,
        args.sec_start_year,
        args.sec_end_year,
        args.out,
        args.sec_rps,
    )
    sec.to_parquet(args.out / "sec_supplement_filings.parquet", index=False)

    combined = combine_filings(hf, sec)
    combined.to_parquet(args.out / "combined_item1_filings.parquet", index=False)

    metrics, cov, manifests = build_networks(
        membership,
        combined,
        dates,
        max_age_days=args.max_item1_age_days,
        pair_density=args.pair_density,
        out=args.out,
    )

    min_valid = int(cov["valid_item1_ciks"].min())
    median_valid = float(cov["valid_item1_ciks"].median())
    manifest = {
        "source": "HF raw SEC 10-K cache + direct SEC missing-issuer supplement",
        "universe": "PIT_S_AND_P_500_ISSUER_PROXY",
        "evidence_label": "FREE_DISCOVERY_HF_SEC_PARSER_PROXY",
        "formation_start": str(min(dates).date()),
        "formation_end": str(max(dates).date()),
        "forms": ["10-K"],
        "max_item1_age_days": args.max_item1_age_days,
        "pair_density": args.pair_density,
        "pit_union_ciks": len(wanted),
        "hf_valid_filings": int(len(hf)),
        "hf_unique_ciks": int(hf["cik"].nunique()),
        "sec_missing_union_ciks": len(missing),
        "sec_supplement_rows": int(len(sec)),
        "sec_supplement_success": int(sec["item1"].notna().sum()) if len(sec) else 0,
        "combined_filings": int(len(combined)),
        "combined_unique_ciks": int(combined["cik"].nunique()),
        "valid_item1_min": min_valid,
        "valid_item1_median": median_valid,
        "network_rows": int(len(metrics)),
        "network_runs": manifests,
        "hf_shard_stats": hf_stats,
        "quality_gate": {
            "min_valid_item1_ciks_required": 200,
            "median_valid_item1_ciks_required": 350,
            "passed": bool(min_valid >= 200 and median_valid >= 350),
        },
        "warning": (
            "Large-cap PIT S&P proxy, not the frozen full-US universe. "
            "HF section_1 text uses an independent parser; even at matching "
            "filing dates this is not byte-identical to our direct-SEC Item 1 "
            "extractor. A positive result requires direct-SEC replication "
            "before any holdout is opened."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print("HYBRID_ITEM1_MANIFEST", json.dumps({
        k: manifest[k]
        for k in [
            "hf_valid_filings",
            "hf_unique_ciks",
            "sec_missing_union_ciks",
            "sec_supplement_rows",
            "sec_supplement_success",
            "combined_unique_ciks",
            "valid_item1_min",
            "valid_item1_median",
            "network_rows",
        ]
    }), flush=True)

    if min_valid < 200:
        raise SystemExit(f"HYBRID ITEM1 FAIL min {min_valid} < 200")
    if median_valid < 350:
        raise SystemExit(f"HYBRID ITEM1 FAIL median {median_valid} < 350")
    print("HYBRID HF+SEC ITEM1 NETWORK PASS", min_valid, median_valid, flush=True)


if __name__ == "__main__":
    main()
