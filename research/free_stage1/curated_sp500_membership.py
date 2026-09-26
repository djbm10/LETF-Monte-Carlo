from __future__ import annotations

"""Build a curated PIT S&P 500 membership panel with stable SEC CIKs.

Membership timing comes exclusively from thuningxu/sp500nq100 daily historical
snapshots. lawcal/sp500-components-history is used only as an identity crosswalk
for symbols that map to exactly one CIK across the entire file; its membership
dates are deliberately ignored.

This separation prevents revision-history dating errors from contaminating the
point-in-time universe while retaining lawcal's useful manually backfilled CIKs.
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def norm_symbol(x) -> str:
    return str(x).upper().strip().replace("-", ".")


def parse_tickers(v) -> set[str]:
    if isinstance(v, (list, tuple, set)):
        vals = v
    else:
        s = str(v).strip()
        try:
            q = ast.literal_eval(s)
            vals = q if isinstance(q, (list, tuple, set)) else s.split(",")
        except Exception:
            vals = s.split(",")
    return {norm_symbol(x) for x in vals if str(x).strip()}


def clean_lawcal(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, dtype={"cik": str})
    x["symbol"] = x["symbol"].map(norm_symbol)
    x["cik"] = (
        x["cik"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    x = x[x["cik"].str.fullmatch(r"\d{10}")].copy()
    return x


def unique_symbol_cik_map(lawcal: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    g = lawcal.groupby("symbol")["cik"].agg(lambda s: sorted(set(s)))
    ambiguous = set(g[g.map(len) != 1].index)
    mapping = {sym: vals[0] for sym, vals in g.items() if len(vals) == 1}
    return mapping, ambiguous


def load_curated(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    lower = {str(c).lower(): c for c in x.columns}
    dc = lower.get("date")
    tc = next((lower[k] for k in ("tickers", "symbols", "components") if k in lower), None)
    if dc is None or tc is None:
        raise ValueError(f"unexpected curated schema: {list(x.columns)}")
    x["date"] = pd.to_datetime(x[dc], errors="coerce").dt.normalize()
    x["tickers_set"] = x[tc].map(parse_tickers)
    return x.dropna(subset=["date"]).sort_values("date")[["date", "tickers_set"]]


def snapshot_asof(curated: pd.DataFrame, d: pd.Timestamp) -> set[str]:
    z = curated[curated["date"] <= pd.Timestamp(d).normalize()]
    if z.empty:
        raise ValueError(f"no curated snapshot <= {d}")
    return set(z.iloc[-1]["tickers_set"])


def build_intervals(curated: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Collapse to changes only, then turn symbol presence into [added, removed)
    # intervals. The source is daily, so a disappearance date is a valid
    # effective removal boundary for backtest membership.
    x = curated[(curated.date >= start) & (curated.date <= end)].copy()
    if x.empty:
        raise ValueError("curated history empty in requested window")

    change_rows = []
    prev = None
    for row in x.itertuples(index=False):
        cur = set(row.tickers_set)
        if prev is None or cur != prev:
            change_rows.append((pd.Timestamp(row.date), cur))
            prev = cur

    active_since: dict[str, pd.Timestamp] = {}
    intervals = []
    prev_set: set[str] = set()
    for d, cur in change_rows:
        for sym in sorted(cur - prev_set):
            active_since[sym] = d
        for sym in sorted(prev_set - cur):
            intervals.append(
                {"symbol": sym, "date_added": active_since.pop(sym), "date_removed": d}
            )
        prev_set = cur
    for sym in sorted(prev_set):
        intervals.append(
            {"symbol": sym, "date_added": active_since[sym], "date_removed": pd.NaT}
        )
    return pd.DataFrame(intervals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=Path, required=True)
    ap.add_argument("--lawcal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2007-12-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--audit-start", default="2009-03-31")
    ap.add_argument("--audit-end", default="2019-12-31")
    ap.add_argument("--mapping-gate", type=float, default=0.97)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    curated = load_curated(args.curated)
    lawcal = clean_lawcal(args.lawcal)
    mapping, ambiguous = unique_symbol_cik_map(lawcal)

    intervals = build_intervals(
        curated,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end),
    )
    intervals["cik"] = intervals["symbol"].map(mapping)
    intervals["mapping_status"] = intervals["symbol"].map(
        lambda s: "AMBIGUOUS_MULTI_CIK" if s in ambiguous
        else "UNMAPPED" if s not in mapping
        else "UNIQUE_LAWCAL_CIK"
    )

    audit = []
    dates = pd.date_range(args.audit_start, args.audit_end, freq="QE")
    for d in dates:
        syms = snapshot_asof(curated, d)
        mapped = {s: mapping[s] for s in syms if s in mapping}
        amb = sorted(syms & ambiguous)
        unmapped = sorted(syms - set(mapping) - ambiguous)
        audit.append(
            {
                "formation_date": str(d.date()),
                "curated_tickers": len(syms),
                "mapped_unique_tickers": len(mapped),
                "mapped_unique_ciks": len(set(mapped.values())),
                "mapping_coverage": len(mapped) / len(syms) if syms else 0.0,
                "ambiguous_tickers": amb,
                "unmapped_tickers": unmapped,
            }
        )

    a = pd.DataFrame(audit)
    min_cov = float(a["mapping_coverage"].min())
    median_cov = float(a["mapping_coverage"].median())
    summary = {
        "membership_timing_source": "thuningxu/sp500nq100 daily curated snapshots",
        "identity_crosswalk_source": "lawcal/sp500-components-history unique symbol-CIK pairs only",
        "lawcal_dates_used": False,
        "interval_rows": int(len(intervals)),
        "interval_symbols": int(intervals.symbol.nunique()),
        "mapped_interval_rows": int(intervals.cik.notna().sum()),
        "ambiguous_crosswalk_symbols": sorted(ambiguous),
        "quarter_audit_start": args.audit_start,
        "quarter_audit_end": args.audit_end,
        "quarters": int(len(a)),
        "mapping_gate": float(args.mapping_gate),
        "min_quarter_mapping_coverage": min_cov,
        "median_quarter_mapping_coverage": median_cov,
        "gate_passed": bool(min_cov >= args.mapping_gate),
        "worst_quarters": a.nsmallest(8, "mapping_coverage")[
            [
                "formation_date",
                "curated_tickers",
                "mapped_unique_tickers",
                "mapped_unique_ciks",
                "mapping_coverage",
            ]
        ].to_dict("records"),
    }

    intervals.to_csv(args.out / "curated_sp500_membership_intervals.csv", index=False)
    a.to_json(args.out / "curated_sp500_cik_mapping_audit.json", orient="records", indent=2)
    (args.out / "curated_sp500_membership_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)

    if min_cov < args.mapping_gate:
        raise SystemExit(
            f"CURATED CIK MAPPING FAIL min={min_cov:.4%} < {args.mapping_gate:.2%}"
        )
    print("CURATED CIK MAPPING PASS", min_cov, median_cov, flush=True)


if __name__ == "__main__":
    main()
