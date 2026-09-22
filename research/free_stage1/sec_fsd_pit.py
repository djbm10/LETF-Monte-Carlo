from __future__ import annotations

"""
Build a free point-in-time fundamental panel from SEC Financial Statement Data Sets.

The SEC states these datasets contain numeric information from primary financial
statements and preserve the information "as filed". We therefore use the filing
date in sub.txt as the information timestamp and never join on period-end alone.

Output is intentionally canonical and compact. It is not a Compustat clone.
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{qtr}.zip"

ALIASES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForProceedsFromOtherPropertyPlantAndEquipment",
    ],
    "shares": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
}

FLOW_METRICS = {"revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex"}
STOCK_METRICS = {"assets", "liabilities", "cash", "shares"}


def _request(url: str, user_agent: str) -> bytes:
    if "@" not in user_agent:
        raise ValueError("SEC_USER_AGENT must include a contact email")
    r = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def download_quarter(year: int, qtr: int, cache: Path, user_agent: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{year}q{qtr}.zip"
    if not p.exists():
        p.write_bytes(_request(BASE.format(year=year, qtr=qtr), user_agent))
    return p


def _read_member(z: zipfile.ZipFile, stem: str) -> pd.DataFrame:
    names = {Path(n).name.lower(): n for n in z.namelist()}
    key = next((names[k] for k in names if k == stem.lower()), None)
    if key is None:
        raise KeyError(f"{stem} not found; members={z.namelist()[:20]}")
    return pd.read_csv(z.open(key), sep="\t", low_memory=False)


def load_quarter(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as z:
        sub = _read_member(z, "sub.txt")
        num = _read_member(z, "num.txt")
    sub.columns = [c.lower() for c in sub.columns]
    num.columns = [c.lower() for c in num.columns]

    sub["cik"] = (
        pd.to_numeric(sub["cik"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.zfill(10)
    )
    sub["filed"] = pd.to_datetime(sub["filed"].astype(str), errors="coerce", format="%Y%m%d")
    sub["period"] = pd.to_datetime(sub["period"].astype(str), errors="coerce", format="%Y%m%d")

    num["ddate"] = pd.to_datetime(num["ddate"].astype(str), errors="coerce", format="%Y%m%d")
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    return sub, num


def tag_to_metric() -> dict[str, str]:
    out: dict[str, str] = {}
    for metric, tags in ALIASES.items():
        for tag in tags:
            out[tag] = metric
    return out


def choose_value(g: pd.DataFrame, metric: str) -> float:
    x = g.copy()
    # Prefer consolidated rows.
    if "coreg" in x:
        empty = x["coreg"].isna() | (x["coreg"].astype(str).str.strip() == "")
        if empty.any():
            x = x[empty]
    if x.empty:
        return np.nan

    if metric in FLOW_METRICS:
        # For flows, prefer annual YTD (qtrs=4) for 10-K and single-quarter
        # (qtrs=1) where available for interim periods. The caller keeps qtrs.
        pass

    # SEC can contain multiple taxonomy tags mapping to the same concept.
    # Prefer first alias order, then latest ddate.
    rank = {t: i for i, t in enumerate(ALIASES[metric])}
    x = x.assign(_rank=x.tag.map(rank).fillna(999))
    x = x.sort_values(["_rank", "ddate"], ascending=[True, False])
    return float(x.iloc[0].value) if pd.notna(x.iloc[0].value) else np.nan


def canonicalize_quarter(sub: pd.DataFrame, num: pd.DataFrame) -> pd.DataFrame:
    keep_forms = {"10-K", "10-Q", "20-F", "40-F"}
    s = sub[sub.form.isin(keep_forms)].copy()
    cols = ["adsh", "cik", "name", "form", "filed", "period", "fy", "fp", "sic"]
    cols = [c for c in cols if c in s]
    s = s[cols]

    map_tag = tag_to_metric()
    n = num[num.tag.isin(map_tag)].copy()
    n["metric"] = n.tag.map(map_tag)
    if "segments" in n:
        # Primary-statement scalar facts only.
        n = n[(n.segments.isna()) | (pd.to_numeric(n.segments, errors="coerce").fillna(0) == 0)]

    merged = n.merge(s, on="adsh", how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Keep facts applying to the filing's principal reporting period as closely as possible.
    # Instantaneous facts should usually match period exactly. Flow facts may start earlier.
    merged["period_gap_days"] = (merged.ddate - merged.period).dt.days.abs()
    merged = merged[merged.period_gap_days <= 10]

    rows = []
    group_cols = ["adsh", "cik", "name", "form", "filed", "period"]
    extra = [c for c in ["fy", "fp", "sic"] if c in merged]
    for keys, g in merged.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for c in extra:
            vals = g[c].dropna()
            row[c] = vals.iloc[0] if len(vals) else np.nan
        for metric in ALIASES:
            z = g[g.metric == metric]
            if z.empty:
                row[metric] = np.nan
                continue
            if metric in FLOW_METRICS and "qtrs" in z:
                # Annual filings: prefer qtrs=4; quarterly filings: prefer qtrs=1,
                # but preserve the chosen qtrs so transformations can audit it.
                desired = 4 if row["form"] in ("10-K", "20-F", "40-F") else 1
                zz = z[pd.to_numeric(z.qtrs, errors="coerce") == desired]
                if len(zz):
                    z = zz
                row[metric + "_qtrs"] = desired if len(z) else np.nan
            row[metric] = choose_value(z, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_features(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy().sort_values(["cik", "filed", "period"])
    # Use filing sequence, not period sequence alone.
    g = x.groupby("cik", group_keys=False)
    x["gross_margin"] = x.gross_profit / x.revenue.replace(0, np.nan)
    x["operating_margin"] = x.operating_income / x.revenue.replace(0, np.nan)
    x["fcf"] = x.operating_cash_flow - x.capex.abs()
    x["fcf_margin"] = x.fcf / x.revenue.replace(0, np.nan)
    x["leverage"] = x.liabilities / x.assets.replace(0, np.nan)

    # Approximate YoY by matching filings ~1 year earlier for same form/fp.
    def add_yoy(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("filed").copy()
        for metric in ["revenue", "gross_margin", "operating_margin", "shares", "assets"]:
            vals = []
            for r in group.itertuples():
                prior = group[
                    (group.filed <= r.filed - pd.Timedelta(days=300))
                    & (group.filed >= r.filed - pd.Timedelta(days=450))
                    & (group.form == r.form)
                ]
                if hasattr(r, "fp") and "fp" in group and pd.notna(r.fp):
                    same = prior[prior.fp == r.fp]
                    if len(same):
                        prior = same
                if prior.empty:
                    vals.append(np.nan)
                    continue
                p = prior.iloc[-1][metric]
                cur = getattr(r, metric)
                if metric in ("gross_margin", "operating_margin"):
                    vals.append(cur - p if pd.notna(cur) and pd.notna(p) else np.nan)
                else:
                    vals.append(cur / p - 1 if pd.notna(cur) and pd.notna(p) and p != 0 else np.nan)
            name = {
                "revenue": "revenue_growth_yoy",
                "gross_margin": "gross_margin_change_yoy",
                "operating_margin": "operating_margin_change_yoy",
                "shares": "share_growth_yoy",
                "assets": "asset_growth_yoy",
            }[metric]
            group[name] = vals
        return group

    x = g.apply(add_yoy).reset_index(drop=True)
    x["evidence_label"] = "FREE_DISCOVERY"
    x["information_date"] = x["filed"]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2009)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--quarters", default="1,2,3,4")
    ap.add_argument("--out", type=Path, default=Path("results/free_stage1/sec_fundamentals"))
    args = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT='Research Name your@email.com'")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache"
    parts = []
    failures = []
    quarters = [int(q) for q in args.quarters.split(",") if q.strip()]
    for year in range(args.start_year, args.end_year + 1):
        for qtr in quarters:
            try:
                z = download_quarter(year, qtr, cache, ua)
                sub, num = load_quarter(z)
                c = canonicalize_quarter(sub, num)
                c["source_quarter"] = f"{year}Q{qtr}"
                parts.append(c)
                print(year, qtr, len(c), flush=True)
            except Exception as exc:
                failures.append({"year": year, "qtr": qtr, "error": repr(exc)})

    if not parts:
        raise RuntimeError(f"No SEC quarters parsed: {failures[:5]}")
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["cik", "filed", "period", "adsh"]).drop_duplicates(
        ["adsh", "cik"], keep="last"
    )
    derived = derive_features(panel)
    derived.to_parquet(args.out / "sec_pit_fundamentals.parquet", index=False)

    manifest = {
        "source": "SEC Financial Statement Data Sets",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "rows": int(len(derived)),
        "ciks": int(derived.cik.nunique()),
        "min_information_date": str(derived.information_date.min()),
        "max_information_date": str(derived.information_date.max()),
        "failed_quarters": failures,
        "tag_aliases": ALIASES,
        "point_in_time_rule": "information_date = SEC filed date; never use before filed date",
        "evidence_label": "FREE_DISCOVERY",
        "limitations": [
            "Primary-statement numeric facts only; this is not a full Compustat replacement.",
            "Tag aliases are intentionally narrow and must be audited before performance conclusions.",
            "Restatements enter only when subsequently filed; prior portfolio dates retain prior filings.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
