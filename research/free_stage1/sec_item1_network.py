from __future__ import annotations

"""
Build a CIK-native, point-in-time competition network from SEC 10-K Item 1 text.

Why CIK-native?
---------------
Public Hoberg-Phillips TNIC/ETNIC releases are keyed by GVKEY. A trustworthy
free discovery test should not silently manufacture a historical CIK↔GVKEY
crosswalk from current tickers. SEC filings are natively keyed by CIK, and
QuantConnect/LEAN exposes Symbol.CIK, so the free discovery layer can join
directly on CIK.

This module:
1) reads SEC quarterly master indexes,
2) downloads original 10-K submissions,
3) extracts the 10-K primary DOCUMENT and Item 1 business section,
4) timestamps every observation by actual filing_date,
5) for each requested formation date, uses only each CIK's latest filing that
   was public on or before that date,
6) builds a TF-IDF cosine-similarity network,
7) calibrates the related-pair threshold to a pre-specified pair density.

The baseline density 2.05% follows the Hoberg-Phillips ETNIC-3 documentation's
three-digit-SIC granularity calibration. This is a methodological clone, not
the published TNIC/ETNIC dataset.

SEC automated access:
- Set SEC_USER_AGENT to a descriptive string including a contact email.
- Rate is intentionally conservative by default.
"""

import argparse
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SEC_ROOT = "https://www.sec.gov/"
MASTER = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/master.idx"
DEFAULT_PAIR_DENSITY = 0.0205


@dataclass
class SecClient:
    user_agent: str
    requests_per_second: float = 4.0
    timeout: int = 60

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT must identify the requester and include a contact email."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )
        self._last = 0.0

    def get(self, url: str) -> bytes:
        wait = 1.0 / max(self.requests_per_second, 0.1)
        elapsed = time.monotonic() - self._last
        if elapsed < wait:
            time.sleep(wait - elapsed)
        r = self.session.get(url, timeout=self.timeout)
        self._last = time.monotonic()
        r.raise_for_status()
        return r.content


def parse_master_idx(raw: bytes) -> pd.DataFrame:
    text = raw.decode("latin-1", errors="replace")
    marker = "CIK|Company Name|Form Type|Date Filed|Filename"
    pos = text.find(marker)
    if pos < 0:
        raise ValueError("SEC master index header not found")
    data = text[pos:]
    df = pd.read_csv(io.StringIO(data), sep="|", dtype={"CIK": str})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    return df


def filing_index(
    client: SecClient,
    start_year: int,
    end_year: int,
    forms: tuple[str, ...] = ("10-K",),
    cache: Path | None = None,
) -> pd.DataFrame:
    parts = []
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        for qtr in range(1, 5):
            local = cache / f"{year}Q{qtr}_master.idx" if cache else None
            if local and local.exists():
                raw = local.read_bytes()
            else:
                raw = client.get(MASTER.format(year=year, qtr=qtr))
                if local:
                    local.write_bytes(raw)
            x = parse_master_idx(raw)
            x = x[x.form_type.isin(forms)].copy()
            parts.append(x)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["date_filed", "cik", "filename"])


def _document_blocks(submission: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for block in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", submission, flags=re.I | re.S):
        mtype = re.search(r"<TYPE>\s*([^\r\n<]+)", block, flags=re.I)
        mtext = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.I | re.S)
        if mtype and mtext:
            out.append((mtype.group(1).strip().upper(), mtext.group(1)))
    return out


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Modern inline-XBRL 10-Ks can render real section headings inside table
    # cells (for example "Item 1. | Business"). Preserve only rows that look
    # like Item 1/1A/1B/2 boundaries, then discard the rest of each table so
    # layout/financial-table text does not contaminate the TF-IDF corpus.
    boundary = re.compile(r"(?i)\bitem\s+(?:1(?:a|b)?|2)\b")
    for table in list(soup.find_all("table")):
        heading_rows = []
        for tr in table.find_all("tr"):
            row_text = " ".join(tr.stripped_strings)
            if boundary.search(row_text):
                heading_rows.append(row_text)
        for row_text in heading_rows:
            p = soup.new_tag("p")
            p.string = row_text
            table.insert_before(p)
        table.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


_ITEM1_START = [
    re.compile(r"(?im)^\s*item\s+1[\.\:\-\s]+business\b"),
    re.compile(r"(?im)^\s*item\s+1\b"),
]
_ITEM1_END = [
    re.compile(r"(?im)^\s*item\s+1a[\.\:\-\s]+risk\s+factors\b"),
    re.compile(r"(?im)^\s*item\s+1b\b"),
    re.compile(r"(?im)^\s*item\s+2[\.\:\-\s]+properties\b"),
]


def extract_item1(submission: str) -> str | None:
    blocks = _document_blocks(submission)
    candidates = [txt for typ, txt in blocks if typ == "10-K"]
    if not candidates:
        candidates = [submission]

    best: str | None = None
    for raw in candidates:
        text = _html_to_text(raw)
        starts = []
        for pat in _ITEM1_START:
            starts.extend(m.start() for m in pat.finditer(text))
        if not starts:
            continue

        # Table of contents often contains an early "Item 1". Try each start and
        # keep the longest plausible section before the next Item 1A/1B/2 marker.
        for st in sorted(set(starts)):
            ends = []
            for pat in _ITEM1_END:
                ends.extend(m.start() for m in pat.finditer(text, st + 100))
            en = min(ends) if ends else min(len(text), st + 150_000)
            section = text[st:en].strip()
            words = len(section.split())
            if words < 250:
                continue
            if best is None or len(section) > len(best):
                best = section

    if best is None:
        return None
    # Cap pathological filings while preserving a long Item 1 section.
    return best[:250_000]


def fetch_item1_rows(
    client: SecClient,
    index: pd.DataFrame,
    out_dir: Path,
    ciks: set[str] | None = None,
    max_filings: int | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    work = index.copy()
    if ciks:
        ciks = {str(x).zfill(10) for x in ciks}
        work = work[work.cik.isin(ciks)]
    if max_filings:
        work = work.head(max_filings)

    for n, row in enumerate(work.itertuples(index=False), 1):
        accession = Path(row.filename).stem
        cached = out_dir / f"{row.cik}_{accession}.txt"
        if cached.exists():
            item1 = cached.read_text(errors="replace")
        else:
            url = urljoin(SEC_ROOT, "Archives/" + row.filename)
            try:
                raw = client.get(url).decode("latin-1", errors="replace")
                item1 = extract_item1(raw)
            except Exception as exc:
                rows.append(
                    {
                        "cik": row.cik,
                        "filing_date": row.date_filed,
                        "filename": row.filename,
                        "item1": None,
                        "word_count": 0,
                        "error": repr(exc),
                    }
                )
                continue
            if item1:
                cached.write_text(item1)
        rows.append(
            {
                "cik": row.cik,
                "filing_date": row.date_filed,
                "filename": row.filename,
                "item1": item1,
                "word_count": len(item1.split()) if item1 else 0,
                "error": None if item1 else "ITEM1_NOT_FOUND",
            }
        )
        if n % 100 == 0:
            print(f"processed {n}/{len(work)} filings", flush=True)
    return pd.DataFrame(rows)


def latest_asof(
    filings: pd.DataFrame,
    formation_date: pd.Timestamp,
    max_age_days: int = 550,
) -> pd.DataFrame:
    f = filings.copy()
    f["filing_date"] = pd.to_datetime(f.filing_date)
    min_filing_date = formation_date - pd.Timedelta(days=max_age_days)
    f = f[
        (f.filing_date <= formation_date)
        & (f.filing_date >= min_filing_date)
        & f.item1.notna()
        & (f.word_count >= 250)
    ]
    if f.empty:
        return f
    return (
        f.sort_values(["cik", "filing_date"])
        .groupby("cik", as_index=False)
        .tail(1)
        .sort_values("cik")
        .reset_index(drop=True)
    )


def _global_threshold(similarities: np.ndarray, n: int, pair_density: float) -> float:
    # Desired number of directed non-self edges.
    target = max(1, int(round(pair_density * n * max(n - 1, 1))))
    vals = similarities[np.isfinite(similarities)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    if len(vals) <= target:
        return float(vals.min())
    kth = len(vals) - target
    return float(np.partition(vals, kth)[kth])


def build_network(
    filings_asof: pd.DataFrame,
    formation_date: pd.Timestamp,
    pair_density: float = DEFAULT_PAIR_DENSITY,
    max_features: int = 60_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n = len(filings_asof)
    if n < 10:
        raise ValueError(f"Need >=10 valid filings, got {n}")

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vec.fit_transform(filings_asof.item1.astype(str))
    # Related density ~2.05% => average ~2.05%*(N-1) peers. Query 1.75x
    # that many neighbors so threshold calibration is not artificially capped.
    k = min(n, max(25, int(math.ceil(pair_density * max(n - 1, 1) * 1.75)) + 1))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - distances

    # Exclude self (first neighbor should be self, but don't assume ordering).
    flat = []
    for i in range(n):
        for j, sim in zip(indices[i], sims[i]):
            if i == j:
                continue
            flat.append(float(sim))
    threshold = _global_threshold(np.asarray(flat, dtype=float), n, pair_density)

    ciks = filings_asof.cik.astype(str).tolist()
    edges = []
    metrics = []
    for i in range(n):
        peers = []
        for j, sim in zip(indices[i], sims[i]):
            if i == j or sim < threshold:
                continue
            peers.append((j, float(sim)))
            edges.append(
                {
                    "formation_date": formation_date,
                    "cik": ciks[i],
                    "peer_cik": ciks[j],
                    "similarity": float(sim),
                }
            )
        weights = np.asarray([s for _, s in peers], dtype=float)
        if len(weights):
            w = weights / weights.sum()
            sim_hhi = float(np.sum(w * w))
            total_sim = float(weights.sum())
            mean_sim = float(weights.mean())
            nearest = float(weights.max())
            top5 = float(np.sort(weights)[-5:].mean())
        else:
            sim_hhi = np.nan
            total_sim = 0.0
            mean_sim = 0.0
            nearest = 0.0
            top5 = 0.0
        metrics.append(
            {
                "formation_date": formation_date,
                "cik": ciks[i],
                "source_filing_date": filings_asof.iloc[i].filing_date,
                "peer_count": len(peers),
                "total_similarity": total_sim,
                "mean_similarity": mean_sim,
                "nearest_similarity": nearest,
                "top5_similarity": top5,
                "similarity_hhi": sim_hhi,
                # Higher = fewer/weaker substitutes. This is intentionally
                # transparent and will be z-scored cross-sectionally later.
                "text_scarcity_raw": -math.log1p(len(peers)) - math.log1p(total_sim),
            }
        )
    edge_df = pd.DataFrame(edges)
    metric_df = pd.DataFrame(metrics)
    manifest = {
        "formation_date": str(formation_date.date()),
        "n_firms": n,
        "tfidf_features": int(X.shape[1]),
        "neighbors_queried_per_firm": k,
        "pair_density_target": pair_density,
        "similarity_threshold": threshold,
        "directed_edges": int(len(edge_df)),
        "realized_directed_density": float(len(edge_df) / (n * max(n - 1, 1))),
        "information_rule": "latest 10-K filing_date <= formation_date",
    }
    return metric_df, edge_df, manifest


def parse_formation_dates(s: str) -> list[pd.Timestamp]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(pd.Timestamp(x))
    if not out:
        raise ValueError("No formation dates")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2009)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--forms", default="10-K")
    ap.add_argument("--ciks", default="")
    ap.add_argument("--max-filings", type=int)
    ap.add_argument("--formation-dates")
    ap.add_argument("--pair-density", type=float, default=DEFAULT_PAIR_DENSITY)
    ap.add_argument(
        "--max-item1-age-days",
        type=int,
        default=550,
        help="Exclude stale 10-K business descriptions older than this at formation.",
    )
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=Path("results/free_stage1/sec_item1_network"))
    args = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT='Research Name your@email.com'")
    client = SecClient(ua, requests_per_second=args.rps)
    args.out.mkdir(parents=True, exist_ok=True)

    idx = filing_index(
        client,
        args.start_year,
        args.end_year,
        tuple(x.strip().upper() for x in args.forms.split(",") if x.strip()),
        args.out / "index_cache",
    )
    idx.to_csv(args.out / "filing_index.csv", index=False)

    ciks = {x.strip().zfill(10) for x in args.ciks.split(",") if x.strip()} or None
    filings = fetch_item1_rows(
        client, idx, args.out / "item1_cache", ciks=ciks, max_filings=args.max_filings
    )
    filings.to_parquet(args.out / "item1_filings.parquet", index=False)

    manifests = []
    all_metrics = []
    if args.formation_dates:
        for fd in parse_formation_dates(args.formation_dates):
            sample = latest_asof(
                filings,
                fd,
                max_age_days=args.max_item1_age_days,
            )
            if len(sample) < 10:
                print(f"{fd.date()}: only {len(sample)} firms; skipping network")
                continue
            metrics, edges, manifest = build_network(
                sample, fd, pair_density=args.pair_density
            )
            stamp = fd.strftime("%Y%m%d")
            metrics.to_parquet(args.out / f"network_metrics_{stamp}.parquet", index=False)
            edges.to_parquet(args.out / f"network_edges_{stamp}.parquet", index=False)
            all_metrics.append(metrics)
            manifests.append(manifest)

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True).sort_values(["formation_date", "cik"])
        combined.to_csv(args.out / "network_metrics_all.csv", index=False)

    manifest = {
        "source": "SEC EDGAR original 10-K submissions",
        "forms": args.forms,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "item1_rows": int(len(filings)),
        "item1_success": int(filings.item1.notna().sum()),
        "pair_density_default": DEFAULT_PAIR_DENSITY,
        "max_item1_age_days": int(args.max_item1_age_days),
        "network_runs": manifests,
        "evidence_label": "FREE_DISCOVERY",
        "warnings": [
            "Item 1 extraction is heuristic and must be quality-audited on a stratified sample.",
            "This is a CIK-native TNIC-like free clone, not the published Hoberg-Phillips TNIC/ETNIC dataset.",
            "Do not use a filing before its filing_date.",
            "Exclude Item 1 descriptions older than max_item1_age_days so inactive/dead filers do not persist indefinitely in later networks.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
