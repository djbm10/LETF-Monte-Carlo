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
    # SEC integer-like date fields can be inferred as floats when a quarter
    # contains blanks (for example "20230930.0"). Converting those directly
    # to strings makes every such date NaT. Normalize through nullable Int64
    # first so live files remain parseable across pandas versions.
    def parse_yyyymmdd(series: pd.Series) -> pd.Series:
        normalized = (
            pd.to_numeric(series, errors="coerce")
            .astype("Int64")
            .astype(str)
            .replace("<NA>", np.nan)
        )
        return pd.to_datetime(normalized, errors="coerce", format="%Y%m%d")

    sub["filed"] = parse_yyyymmdd(sub["filed"])
    sub["period"] = parse_yyyymmdd(sub["period"])

    num["ddate"] = parse_yyyymmdd(num["ddate"])
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
    keep_forms = {
        "10-K", "10-K/A", "10-Q", "10-Q/A",
        "20-F", "20-F/A", "40-F", "40-F/A",
    }
    s = sub[sub.form.isin(keep_forms)].copy()
    s["form_original"] = s["form"]
    s["amended"] = s["form"].str.endswith("/A")
    # Treat amendments as the same reporting-form family for period semantics
    # and YoY matching, but keep the later amendment filing date as the PIT
    # information timestamp.
    s["form"] = s["form"].str.replace("/A", "", regex=False)
    cols = [
        "adsh", "cik", "name", "form", "form_original", "amended",
        "filed", "period", "fy", "fp", "sic",
    ]
    cols = [col for col in cols if col in s]
    s = s[cols]

    map_tag = tag_to_metric()
    n = num[num.tag.isin(map_tag)].copy()
    n["metric"] = n.tag.map(map_tag)
    if "segments" in n:
        # Primary-statement scalar facts only.
        n = n[(n.segments.isna()) | (pd.to_numeric(n.segments, errors="coerce").fillna(0) == 0)]

    merged = n.merge(s, on="adsh", how="inner")
    if merged.empty:
        raise ValueError(
            "SEC canonicalization produced no joined alias facts: "
            f"sub_rows={len(sub)} kept_forms={len(s)} num_rows={len(num)} "
            f"alias_rows={len(n)}"
        )

    # Keep facts applying to the filing's principal reporting period as closely as possible.
    # Instantaneous facts should usually match period exactly. Flow facts may start earlier.
    merged["period_gap_days"] = (merged.ddate - merged.period).dt.days.abs()
    before_period_filter = len(merged)
    merged = merged[merged.period_gap_days <= 10]
    if merged.empty:
        raise ValueError(
            "SEC canonicalization lost every row at principal-period filter: "
            f"joined_rows={before_period_filter} valid_ddate={int(n.ddate.notna().sum())} "
            f"valid_period={int(s.period.notna().sum())}"
        )

    rows = []
    group_cols = ["adsh", "cik", "name", "form", "filed", "period"]
    extra = [c for c in ["form_original", "amended", "fy", "fp", "sic"] if c in merged]
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
                # Never silently substitute a YTD flow for a single-quarter
                # 10-Q value (or vice versa). That would make YoY growth and
                # valuation denominators incomparable across filings.
                desired = 4 if row["form"] in ("10-K", "20-F", "40-F") else 1
                zz = z[pd.to_numeric(z.qtrs, errors="coerce") == desired]
                if zz.empty:
                    row[metric + "_qtrs"] = np.nan
                    row[metric] = np.nan
                    continue
                z = zz
                row[metric + "_qtrs"] = desired
            row[metric] = choose_value(z, metric)
        rows.append(row)
    return pd.DataFrame(rows)



def apply_amendment_carryforward(panel: pd.DataFrame) -> pd.DataFrame:
    """
    SEC amendments may report only changed facts. For an amended filing, carry
    forward missing raw facts from the latest earlier filing for the same CIK,
    normalized form family, and reporting period. Supplied amendment facts
    always win, and the amendment keeps its own later filing timestamp.
    """
    x = panel.copy().sort_values(["cik", "form", "period", "filed", "adsh"]).reset_index(drop=True)
    if "amended" not in x:
        return x

    raw_cols = [m for m in ALIASES if m in x.columns]
    raw_cols += [f"{m}_qtrs" for m in FLOW_METRICS if f"{m}_qtrs" in x.columns]
    keys = ["cik", "form", "period"]

    for col in raw_cols:
        carried = x.groupby(keys, dropna=False)[col].ffill()
        mask = x["amended"].fillna(False).astype(bool) & x[col].isna()
        x.loc[mask, col] = carried.loc[mask]

    return x

def derive_features(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy().sort_values(["cik", "filed", "period"]).reset_index(drop=True)
    x["gross_margin"] = x.gross_profit / x.revenue.replace(0, np.nan)
    x["operating_margin"] = x.operating_income / x.revenue.replace(0, np.nan)
    x["fcf"] = x.operating_cash_flow - x.capex.abs()
    x["fcf_margin"] = x.fcf / x.revenue.replace(0, np.nan)
    x["leverage"] = x.liabilities / x.assets.replace(0, np.nan)

    # Point-in-time YoY matching, vectorized:
    # for each current filing, choose the latest same-form filing that was
    # filed 300-450 days earlier; if an FP value is present and a same-FP
    # candidate exists, prefer that candidate. This preserves the original
    # matching rule without an O(rows-per-CIK^2) Python loop.
    metrics = ["revenue", "gross_margin", "operating_margin", "shares", "assets"]
    x["_row_id"] = np.arange(len(x))
    x["_target_prior_date"] = x["filed"] - pd.Timedelta(days=300)

    def lookup(keys: list[str], left: pd.DataFrame) -> pd.DataFrame:
        left_cols = ["_row_id", "_target_prior_date", *keys]
        l = left[left_cols].dropna(subset=["_target_prior_date", *keys]).copy()
        right_cols = [*keys, "filed", *metrics]
        r = x[right_cols].dropna(subset=["filed", *keys]).copy()
        rename = {"filed": "prior_filed", **{m: f"prior_{m}" for m in metrics}}
        r = r.rename(columns=rename)
        if l.empty or r.empty:
            return pd.DataFrame(index=pd.Index([], name="_row_id"))
        # merge_asof requires global sorting by the time key even when BY keys
        # are also supplied.
        l = l.sort_values("_target_prior_date")
        r = r.sort_values("prior_filed")
        out = pd.merge_asof(
            l,
            r,
            left_on="_target_prior_date",
            right_on="prior_filed",
            by=keys,
            direction="backward",
            tolerance=pd.Timedelta(days=150),
            allow_exact_matches=True,
        )
        return out.set_index("_row_id")

    fallback = lookup(["cik", "form"], x)
    if "fp" in x.columns:
        same_fp = lookup(["cik", "form", "fp"], x[x["fp"].notna()])
    else:
        same_fp = pd.DataFrame(index=pd.Index([], name="_row_id"))

    row_ids = pd.Index(x["_row_id"], name="_row_id")
    fallback = fallback.reindex(row_ids)
    same_fp = same_fp.reindex(row_ids)
    prefer_same = (
        same_fp["prior_filed"].notna()
        if "prior_filed" in same_fp.columns
        else pd.Series(False, index=row_ids)
    )

    names = {
        "revenue": "revenue_growth_yoy",
        "gross_margin": "gross_margin_change_yoy",
        "operating_margin": "operating_margin_change_yoy",
        "shares": "share_growth_yoy",
        "assets": "asset_growth_yoy",
    }
    for metric in metrics:
        fallback_prior = (
            fallback[f"prior_{metric}"]
            if f"prior_{metric}" in fallback.columns
            else pd.Series(np.nan, index=row_ids)
        )
        same_prior = (
            same_fp[f"prior_{metric}"]
            if f"prior_{metric}" in same_fp.columns
            else pd.Series(np.nan, index=row_ids)
        )
        prior = fallback_prior.copy()
        prior.loc[prefer_same] = same_prior.loc[prefer_same]
        cur = x.set_index("_row_id")[metric]
        if metric in ("gross_margin", "operating_margin"):
            out = cur - prior
        else:
            out = cur / prior.replace(0, np.nan) - 1
        x[names[metric]] = out.reindex(row_ids).to_numpy()

    x = x.drop(columns=["_row_id", "_target_prior_date"])
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
    panel = apply_amendment_carryforward(panel)
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
            "Amended 10-K/10-Q/20-F/40-F filings enter only from their amendment filing date; prior portfolio dates retain the earlier filing.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
