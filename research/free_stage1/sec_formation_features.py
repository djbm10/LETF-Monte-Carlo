from __future__ import annotations

"""
Build a compact formation-date fundamental feature panel from the full SEC
as-filed PIT panel produced by sec_fsd_pit.py.

The output is designed for QuantConnect ingestion: one row per CIK and requested
formation date, using only the latest filing whose information_date <= the
formation date. No future filing can enter an earlier portfolio formation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEEP = [
    "cik",
    "information_date",
    "form",
    "period",
    "revenue",
    "revenue_qtrs",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "leverage",
    "shares",
    "revenue_growth_yoy",
    "gross_margin_change_yoy",
    "operating_margin_change_yoy",
    "share_growth_yoy",
    "asset_growth_yoy",
]


def parse_dates(s: str) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(x.strip()) for x in s.split(",") if x.strip()]
    if not dates:
        raise ValueError("No formation dates supplied")
    return dates


def build_formation_panel(panel: pd.DataFrame, formation_dates: list[pd.Timestamp]) -> pd.DataFrame:
    x = panel.copy()
    if "information_date" not in x:
        raise ValueError("input panel missing information_date")
    if "cik" not in x:
        raise ValueError("input panel missing cik")

    x["information_date"] = pd.to_datetime(x["information_date"], errors="coerce")
    x["period"] = pd.to_datetime(x.get("period"), errors="coerce")
    x["cik"] = x["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    x = x.dropna(subset=["information_date"])
    x = x.sort_values(["information_date", "period", "cik"]).reset_index(drop=True)

    cols = [c for c in KEEP if c in x.columns]
    out = []
    for formation_date in sorted(formation_dates):
        available = x[x["information_date"] <= formation_date]
        if available.empty:
            continue

        # Latest public filing per CIK as of the formation date. If a CIK has
        # multiple records on the same filing date, prefer the later period.
        latest = (
            available.sort_values(["cik", "information_date", "period"])
            .groupby("cik", as_index=False)
            .tail(1)
            .copy()
        )
        latest["formation_date"] = formation_date
        latest["filing_age_days"] = (
            formation_date - latest["information_date"]
        ).dt.days
        out.append(latest[["formation_date", *cols, "filing_age_days"]])

    if not out:
        return pd.DataFrame(columns=["formation_date", *cols, "filing_age_days"])

    result = pd.concat(out, ignore_index=True)
    result = result.sort_values(["formation_date", "cik"]).reset_index(drop=True)

    # Hard PIT invariant.
    if (result["information_date"] > result["formation_date"]).any():
        raise AssertionError("future SEC filing leaked into formation panel")
    if result.duplicated(["formation_date", "cik"]).any():
        raise AssertionError("duplicate formation_date/cik rows")

    result["evidence_label"] = "FREE_DISCOVERY"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/free_stage1/sec_fundamentals/sec_pit_fundamentals.parquet"),
    )
    ap.add_argument("--formation-dates", required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/free_stage1/sec_fundamentals/sec_formation_features.csv"),
    )
    args = ap.parse_args()

    panel = pd.read_parquet(args.input)
    result = build_formation_panel(panel, parse_dates(args.formation_dates))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    coverage = {
        "rows": int(len(result)),
        "ciks": int(result["cik"].nunique()) if len(result) else 0,
        "formation_dates": int(result["formation_date"].nunique()) if len(result) else 0,
        "min_information_date": str(result["information_date"].min()) if len(result) else None,
        "max_information_date": str(result["information_date"].max()) if len(result) else None,
        "future_rows": int((result["information_date"] > result["formation_date"]).sum()) if len(result) else 0,
    }
    print(coverage)


if __name__ == "__main__":
    main()
