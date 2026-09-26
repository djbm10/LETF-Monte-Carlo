from __future__ import annotations

"""Audit historical price coverage against the curated PIT S&P universe.

Denominator: every curated constituent x US trading session in the requested
sample. An unresolved CIK is uncovered. For a resolved issuer, a session is
covered only if at least one ticker alias that belongs to that CIK on that date
has a valid price observation.

This avoids letting a high price-coverage percentage hide membership-identity
failures, or vice versa.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import curated_sp500_membership as cm


def clean_lawcal(path: Path) -> pd.DataFrame:
    return cm.clean_lawcal(path)


def alias_rows_for_cik(
    lawcal: pd.DataFrame,
    cik: str,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp | None,
) -> list[dict]:
    out = []
    hi = interval_end if pd.notna(interval_end) else pd.Timestamp("2100-01-01")
    rows = lawcal[lawcal["cik"] == str(cik).zfill(10)]
    for r in rows.itertuples(index=False):
        lo2 = r.date_added if pd.notna(r.date_added) else pd.Timestamp("1900-01-01")
        hi2 = r.date_removed if pd.notna(r.date_removed) else pd.Timestamp("2100-01-01")
        lo = max(pd.Timestamp(interval_start), pd.Timestamp(lo2))
        end = min(pd.Timestamp(hi), pd.Timestamp(hi2))
        if lo < end:
            out.append(
                {
                    "symbol": cm.norm_symbol(r.symbol),
                    "cik": str(cik).zfill(10),
                    "date_added": lo,
                    "date_removed": end if end < pd.Timestamp("2100-01-01") else pd.NaT,
                    "alias_source": "LAWCAL_SAME_CIK_IDENTITY_WINDOW",
                }
            )

    # Explicit identity eras repair cases where the revision-derived lawcal
    # membership interval is wrong but corporate/ticker identity is known.
    for sym, eras in cm.IDENTITY_OVERRIDES.items():
        for lo2, hi2, c in eras:
            if c != str(cik).zfill(10):
                continue
            lo = max(pd.Timestamp(interval_start), pd.Timestamp(lo2))
            end = min(pd.Timestamp(hi), pd.Timestamp(hi2))
            if lo < end:
                out.append(
                    {
                        "symbol": sym,
                        "cik": str(cik).zfill(10),
                        "date_added": lo,
                        "date_removed": end if end < pd.Timestamp("2100-01-01") else pd.NaT,
                        "alias_source": "CURATED_IDENTITY_OVERRIDE",
                    }
                )
    return out


def build_candidate_membership(
    mapped_intervals: pd.DataFrame,
    lawcal: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for r in mapped_intervals.itertuples(index=False):
        start = pd.Timestamp(r.date_added)
        end = pd.Timestamp(r.date_removed) if pd.notna(r.date_removed) else pd.NaT
        aliases = alias_rows_for_cik(lawcal, str(r.cik), start, end)
        # The curated alias itself is permitted only if our date-aware identity
        # resolver says it is this CIK at the start of the interval.
        if cm.resolve_symbol_cik(r.symbol, start, lawcal, cm.unique_symbol_cik_map(lawcal)[0]) == str(r.cik).zfill(10):
            aliases.append(
                {
                    "symbol": cm.norm_symbol(r.symbol),
                    "cik": str(r.cik).zfill(10),
                    "date_added": start,
                    "date_removed": end,
                    "alias_source": "CURATED_ALIAS_RESOLVED",
                }
            )
        rows.extend(aliases)
    if not rows:
        return pd.DataFrame(columns=["symbol","cik","date_added","date_removed","alias_source"])
    x = pd.DataFrame(rows)
    x["date_added"] = pd.to_datetime(x["date_added"]).dt.normalize()
    x["date_removed"] = pd.to_datetime(x["date_removed"], errors="coerce").dt.normalize()
    return (
        x.sort_values(["cik","date_added","symbol","alias_source"])
        .drop_duplicates(["symbol","cik","date_added","date_removed"], keep="first")
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=Path, required=True)
    ap.add_argument("--mapped-membership", type=Path, required=True)
    ap.add_argument("--lawcal", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--coverage-gate", type=float, default=0.97)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    curated = cm.load_curated(args.curated)
    lawcal = clean_lawcal(args.lawcal)
    unique_map, _ = cm.unique_symbol_cik_map(lawcal)
    mapped = pd.read_csv(args.mapped_membership, dtype={"cik": str})
    mapped["cik"] = mapped["cik"].astype(str).str.zfill(10)
    mapped["date_added"] = pd.to_datetime(mapped["date_added"]).dt.normalize()
    mapped["date_removed"] = pd.to_datetime(mapped["date_removed"], errors="coerce").dt.normalize()

    candidate_membership = build_candidate_membership(mapped, lawcal)
    candidate_membership.to_csv(
        args.out / "curated_sp500_price_candidate_membership.csv",
        index=False,
    )

    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()
    prices["symbol"] = prices["symbol"].map(cm.norm_symbol)
    prices = prices[
        (prices["Date"] >= pd.Timestamp(args.start))
        & (prices["Date"] <= pd.Timestamp(args.end))
        & prices["Close"].notna()
        & (pd.to_numeric(prices["Close"], errors="coerce") > 0)
    ].copy()
    price_keys = set(zip(prices["Date"], prices["symbol"]))

    # Use SPY observations from the panel as the actual US trading-session grid.
    spy_dates = sorted(
        set(prices.loc[prices.symbol == "SPY", "Date"])
    )
    if not spy_dates:
        # SPY may not be a member-panel symbol; fall back to all dates because
        # the source panel is daily US equity data.
        spy_dates = sorted(set(prices["Date"]))

    law_by_cik = {
        cik: g.copy()
        for cik, g in lawcal.groupby("cik")
    }

    total = mapped_identity = price_covered = 0
    yearly = []
    detail = []
    for d in spy_dates:
        syms = cm.snapshot_asof(curated, d)
        day_total = len(syms)
        day_mapped = 0
        day_covered = 0
        missing_identity = []
        missing_price = []
        for sym in syms:
            total += 1
            cik = cm.resolve_symbol_cik(sym, d, lawcal, unique_map)
            if not cik:
                missing_identity.append(sym)
                continue
            mapped_identity += 1
            day_mapped += 1

            cand = set()
            g = law_by_cik.get(cik)
            if g is not None:
                for rr in g.itertuples(index=False):
                    lo = rr.date_added if pd.notna(rr.date_added) else pd.Timestamp("1900-01-01")
                    hi = rr.date_removed if pd.notna(rr.date_removed) else pd.Timestamp("2100-01-01")
                    if lo <= d < hi:
                        cand.add(cm.norm_symbol(rr.symbol))
            # identity overrides can extend an alias beyond noisy lawcal dates.
            for s2, eras in cm.IDENTITY_OVERRIDES.items():
                for lo, hi, c in eras:
                    if c == cik and lo <= d < hi:
                        cand.add(s2)
            if cm.resolve_symbol_cik(sym, d, lawcal, unique_map) == cik:
                cand.add(cm.norm_symbol(sym))

            if any((d, s) in price_keys for s in cand):
                price_covered += 1
                day_covered += 1
            else:
                missing_price.append(
                    {
                        "curated_symbol": sym,
                        "cik": cik,
                        "candidate_symbols": sorted(cand),
                    }
                )
        detail.append(
            {
                "date": str(d.date()),
                "curated_members": day_total,
                "mapped_members": day_mapped,
                "price_covered_members": day_covered,
                "effective_coverage": day_covered / day_total if day_total else 0.0,
                "missing_identity": missing_identity,
                "missing_price": missing_price,
            }
        )

    dd = pd.DataFrame(
        [
            {
                "date": x["date"],
                "curated_members": x["curated_members"],
                "mapped_members": x["mapped_members"],
                "price_covered_members": x["price_covered_members"],
                "effective_coverage": x["effective_coverage"],
            }
            for x in detail
        ]
    )
    dd["year"] = pd.to_datetime(dd["date"]).dt.year
    for y, g in dd.groupby("year"):
        denom = int(g.curated_members.sum())
        cov = int(g.price_covered_members.sum())
        yearly.append(
            {
                "year": int(y),
                "curated_member_sessions": denom,
                "covered_member_sessions": cov,
                "coverage": cov / denom if denom else 0.0,
            }
        )

    effective = price_covered / total if total else 0.0
    identity_cov = mapped_identity / total if total else 0.0
    summary = {
        "sample_start": args.start,
        "sample_end": args.end,
        "denominator": "all curated constituent x observed US equity sessions",
        "total_curated_member_sessions": int(total),
        "identity_mapped_sessions": int(mapped_identity),
        "price_covered_sessions": int(price_covered),
        "identity_mapping_session_coverage": float(identity_cov),
        "effective_identity_plus_price_coverage": float(effective),
        "coverage_gate": float(args.coverage_gate),
        "gate_passed": bool(effective >= args.coverage_gate),
        "candidate_membership_rows": int(len(candidate_membership)),
        "candidate_price_symbols": int(candidate_membership.symbol.nunique()) if len(candidate_membership) else 0,
        "yearly": yearly,
        "worst_days": dd.nsmallest(12, "effective_coverage").to_dict("records"),
    }

    dd.to_csv(args.out / "curated_sp500_daily_price_coverage.csv", index=False)
    (args.out / "curated_sp500_price_coverage_detail.json").write_text(
        json.dumps(detail, indent=2, default=str) + "\n"
    )
    (args.out / "curated_sp500_price_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)

    if effective < args.coverage_gate:
        raise SystemExit(
            f"CURATED PRICE COVERAGE FAIL {effective:.4%} < {args.coverage_gate:.2%}"
        )
    print("CURATED PRICE COVERAGE PASS", effective, flush=True)


if __name__ == "__main__":
    main()
