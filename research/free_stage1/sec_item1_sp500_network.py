from __future__ import annotations

"""Build a point-in-time S&P 500 issuer-level SEC Item 1 network.

This is the free large-cap proxy for Bottleneck Winner. It reuses the frozen
SEC Item 1 extraction/network machinery but restricts each formation-date
network to issuers that were S&P 500 members at that date according to
lawcal/sp500-components-history. The result must not be labeled full-US.

Membership rows use stable SEC CIKs, so ticker changes and future constituent
knowledge are not used to join accounting/text signals.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import sec_item1_network as net


def _clean_date(series: pd.Series) -> pd.Series:
    x = series.astype(str).str.replace("*", "", regex=False).str.strip()
    x = x.replace({"": None, "nan": None, "NaN": None, "None": None})
    return pd.to_datetime(x, errors="coerce")


def load_membership(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"cik": str})
    required = {"symbol", "cik", "date_added", "date_removed"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"membership missing columns: {sorted(missing)}")
    m["cik"] = (
        m["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    m["date_added"] = _clean_date(m["date_added"])
    m["date_removed"] = _clean_date(m["date_removed"])
    m = m.dropna(subset=["cik", "date_added"]).copy()
    return m


def active_ciks(membership: pd.DataFrame, d: pd.Timestamp) -> set[str]:
    x = membership[
        (membership["date_added"] <= d)
        & (membership["date_removed"].isna() | (membership["date_removed"] > d))
    ]
    return set(x["cik"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2007)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--formation-dates", required=True)
    ap.add_argument("--pair-density", type=float, default=net.DEFAULT_PAIR_DENSITY)
    ap.add_argument("--max-item1-age-days", type=int, default=550)
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/free_stage1/sp500_item1_network"),
    )
    args = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT='Research Name your@email.com'")

    membership = load_membership(args.membership)
    formation_dates = net.parse_formation_dates(args.formation_dates)
    union_start = min(formation_dates) - pd.Timedelta(days=args.max_item1_age_days)
    union_end = max(formation_dates)
    relevant = membership[
        (membership["date_added"] <= union_end)
        & (membership["date_removed"].isna() | (membership["date_removed"] >= union_start))
    ]
    union_ciks = set(relevant["cik"].astype(str))

    args.out.mkdir(parents=True, exist_ok=True)
    client = net.SecClient(ua, requests_per_second=args.rps)

    idx = net.filing_index(
        client,
        args.start_year,
        args.end_year,
        forms=("10-K",),
        cache=args.out / "index_cache",
    )
    idx = idx[idx["cik"].isin(union_ciks)].copy()
    idx.to_csv(args.out / "filing_index.csv", index=False)

    filings = net.fetch_item1_rows(
        client,
        idx,
        args.out / "item1_cache",
        ciks=union_ciks,
    )
    filings.to_parquet(args.out / "item1_filings.parquet", index=False)

    manifests = []
    all_metrics = []
    membership_audit = []
    for fd in formation_dates:
        active = active_ciks(membership, fd)
        sample = net.latest_asof(
            filings[filings["cik"].isin(active)],
            fd,
            max_age_days=args.max_item1_age_days,
        )
        membership_audit.append(
            {
                "formation_date": str(fd.date()),
                "active_security_rows": int(
                    len(
                        membership[
                            (membership["date_added"] <= fd)
                            & (
                                membership["date_removed"].isna()
                                | (membership["date_removed"] > fd)
                            )
                        ]
                    )
                ),
                "active_ciks": int(len(active)),
                "valid_item1_ciks": int(sample["cik"].nunique()) if len(sample) else 0,
            }
        )
        if len(sample) < 20:
            raise RuntimeError(
                f"{fd.date()}: only {len(sample)} valid S&P Item 1 issuers"
            )
        metrics, edges, manifest = net.build_network(
            sample,
            fd,
            pair_density=args.pair_density,
        )
        manifest["universe"] = "PIT_S_AND_P_500_ISSUER_PROXY"
        manifest["max_item1_age_days"] = int(args.max_item1_age_days)
        stamp = fd.strftime("%Y%m%d")
        metrics.to_parquet(
            args.out / f"network_metrics_{stamp}.parquet",
            index=False,
        )
        edges.to_parquet(
            args.out / f"network_edges_{stamp}.parquet",
            index=False,
        )
        all_metrics.append(metrics)
        manifests.append(manifest)
        print(
            fd.date(),
            "active_ciks",
            len(active),
            "valid_item1",
            len(sample),
            "edges",
            len(edges),
            flush=True,
        )

    combined = (
        pd.concat(all_metrics, ignore_index=True)
        .sort_values(["formation_date", "cik"])
        .reset_index(drop=True)
    )
    combined.to_csv(args.out / "network_metrics_all.csv", index=False)
    pd.DataFrame(membership_audit).to_csv(
        args.out / "membership_network_coverage.csv",
        index=False,
    )

    manifest = {
        "source": "SEC EDGAR 10-K Item 1 + lawcal/sp500-components-history",
        "universe": "PIT_S_AND_P_500_ISSUER_PROXY",
        "evidence_label": "FREE_DISCOVERY_LARGE_CAP_PROXY",
        "membership_file": str(args.membership),
        "formation_start": str(min(formation_dates).date()),
        "formation_end": str(max(formation_dates).date()),
        "filing_acquisition_start_year": args.start_year,
        "filing_acquisition_end_year": args.end_year,
        "union_ciks": int(len(union_ciks)),
        "item1_rows": int(len(filings)),
        "item1_success": int(filings["item1"].notna().sum()),
        "max_item1_age_days": int(args.max_item1_age_days),
        "pair_density": float(args.pair_density),
        "network_runs": manifests,
        "membership_coverage": membership_audit,
        "warning": (
            "Competition scarcity is measured among point-in-time S&P 500 issuers, "
            "not the frozen full-US common-share universe. This is a large-cap "
            "proxy and must not be presented as the full Stage-1 replication."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
