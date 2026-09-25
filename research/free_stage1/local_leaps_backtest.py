from __future__ import annotations

"""
Free local LEAPS replication using the preserved Philipp Dubach end-of-day
SPY/QQQ option-chain dataset.

This deliberately does NOT claim to reproduce QuantConnect's next-open QuoteBar
fills because the free dataset is a 4:00 PM ET snapshot. The primary local
execution proxy is therefore:

    close(t) signal -> close(t+1) real quote-side execution

That is lookahead-free and uses actual historical bid/ask quotes. It is
conservative in timing relative to the frozen next-open design. Contract
selection, roll sequencing, parameter grid, minimum OI, and slippage scenarios
match the frozen Stage-1 design.

Data:
- anahatsingh-ui/options-dataset-hist preservation mirror
- SPY/QQQ yearly options_YYYY.parquet
- underlying_prices.parquet
- manisahni/marketdata daily_adjclose_wide.csv for BIL collateral total return

Evidence label:
    FREE_DISCOVERY_LOCAL_EOD_PROXY
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl


TARGET_DELTAS = (0.70, 0.80, 0.90)
TARGET_DTES = (365, 548, 730)
ALLOCATIONS = (0.25, 0.50, 1.00)
SLIPPAGE_BPS = (0, 5, 10)
START_DATE = date(2012, 1, 3)
FROZEN_END_DATE = date(2026, 9, 21)
DATASET_END_DATE = date(2025, 12, 16)
INITIAL_CASH = 1_000_000.0
MIN_OI = 100
OPTION_MULTIPLIER = 100.0
IB_OPTION_FEE_PER_CONTRACT = 0.70
IB_OPTION_MIN_ORDER_FEE = 1.00


@dataclass
class Candidate:
    contract_id: str
    expiration: date
    strike: float
    bid: float
    ask: float
    delta: float
    open_interest: int
    dte: int
    score: float


@dataclass
class PendingEntry:
    candidate: Candidate
    selected_qty: int
    selection_date: date


@dataclass
class StrategyState:
    underlying: str
    target_delta: float
    target_dte: int
    allocation: float
    slippage_bps: int
    cash: float = INITIAL_CASH
    bil_value: float = 0.0
    option_contract: Optional[str] = None
    option_expiration: Optional[date] = None
    option_qty: int = 0
    option_mark: float = 0.0
    entry_date: Optional[date] = None
    pending_entry: Optional[PendingEntry] = None
    pending_exit: bool = False
    entry_count: int = 0
    roll_count: int = 0
    rejected_entries: int = 0
    resized_entries: int = 0
    no_candidate_days: int = 0
    audit: list = field(default_factory=list)

    @property
    def key(self) -> Tuple[float, int, float, int]:
        return (
            self.target_delta,
            self.target_dte,
            self.allocation,
            self.slippage_bps,
        )

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000.0

    def portfolio_value(self) -> float:
        option_value = self.option_qty * self.option_mark * OPTION_MULTIPLIER
        return self.cash + self.bil_value + option_value


def option_fee(qty: int) -> float:
    if qty <= 0:
        return 0.0
    return max(IB_OPTION_MIN_ORDER_FEE, qty * IB_OPTION_FEE_PER_CONTRACT)


def normalize_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def load_underlying(path: Path, start: date, end: date) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "date" not in df:
        raise ValueError(f"{path}: missing date")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if "adjusted_close" not in df.columns:
        if "adj_close" in df.columns:
            df["adjusted_close"] = df["adj_close"]
        else:
            df["adjusted_close"] = df["close"]
    keep = ["date", "open", "close", "adjusted_close"]
    out = df[keep].drop_duplicates("date").sort_values("date").set_index("date")
    if out.empty:
        raise ValueError(f"{path}: empty underlying sample")
    return out


def load_bil_returns(path: Path, dates: Iterable[date]) -> Dict[date, float]:
    df = pd.read_csv(path, usecols=["Date", "BIL"])
    df["date"] = pd.to_datetime(df["Date"]).dt.date
    df["BIL"] = pd.to_numeric(df["BIL"], errors="coerce")
    df = df.dropna(subset=["BIL"]).sort_values("date")
    df["ret"] = df["BIL"].pct_change().fillna(0.0)
    wanted = set(dates)
    out = {d: float(r) for d, r in zip(df["date"], df["ret"]) if d in wanted}
    return out


def valid_chain(day_df: pl.DataFrame, current_date: date) -> pl.DataFrame:
    if day_df.is_empty():
        return day_df
    x = day_df.with_columns(
        [
            pl.col("expiration").cast(pl.Date),
            pl.col("date").cast(pl.Date),
            pl.col("delta").cast(pl.Float64, strict=False),
            pl.col("bid").cast(pl.Float64, strict=False),
            pl.col("ask").cast(pl.Float64, strict=False),
            pl.col("open_interest").cast(pl.Int64, strict=False),
        ]
    )
    x = x.with_columns(
        (pl.col("expiration") - pl.col("date")).dt.total_days().cast(pl.Int32).alias("dte")
    )
    return x.filter(
        (pl.col("type").cast(pl.Utf8).str.to_lowercase() == "call")
        & (pl.col("dte") >= 270)
        & (pl.col("dte") <= 900)
        & (pl.col("open_interest") >= MIN_OI)
        & (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (pl.col("ask") >= pl.col("bid"))
        & (pl.col("delta").abs() >= 0.55)
        & (pl.col("delta").abs() <= 0.99)
    )


def best_candidates(
    valid: pl.DataFrame,
    current_date: date,
) -> Dict[Tuple[float, int], Candidate]:
    out: Dict[Tuple[float, int], Candidate] = {}
    if valid.is_empty():
        return out

    base = valid.with_columns(
        [
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
            pl.col("delta").abs().alias("abs_delta"),
        ]
    ).with_columns(
        (
            (pl.col("ask") - pl.col("bid"))
            / pl.when(pl.col("mid") > 0).then(pl.col("mid")).otherwise(1.0)
        ).alias("spread_pct")
    )

    for td in TARGET_DELTAS:
        for dte_target in TARGET_DTES:
            scored = base.with_columns(
                (
                    (pl.col("abs_delta") - td).abs()
                    + 0.20 * (pl.col("dte") - dte_target).abs() / 365.0
                    + 0.10 * pl.col("spread_pct")
                ).alias("score")
            ).sort(["score", "contract_id"])
            if scored.is_empty():
                continue
            row = scored.row(0, named=True)
            out[(td, dte_target)] = Candidate(
                contract_id=str(row["contract_id"]),
                expiration=normalize_date(row["expiration"]),
                strike=float(row["strike"]),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                delta=abs(float(row["delta"])),
                open_interest=int(row["open_interest"]),
                dte=int(row["dte"]),
                score=float(row["score"]),
            )
    return out


def quote_map(day_df: pl.DataFrame) -> Dict[str, Tuple[float, float]]:
    if day_df.is_empty():
        return {}
    q = {}
    for row in day_df.select(["contract_id", "bid", "ask"]).iter_rows(named=True):
        try:
            bid = float(row["bid"])
            ask = float(row["ask"])
        except Exception:
            continue
        if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid:
            q[str(row["contract_id"])] = (bid, ask)
    return q


def load_year_groups(path: Path, start: date, end: date):
    cols = [
        "contract_id",
        "symbol",
        "expiration",
        "strike",
        "type",
        "bid",
        "ask",
        "open_interest",
        "date",
        "delta",
    ]
    df = pl.read_parquet(path, columns=cols)
    df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.filter(
        (pl.col("date") >= pl.lit(start))
        & (pl.col("date") <= pl.lit(end))
    ).sort(["date", "contract_id"])
    for key, group in df.group_by("date", maintain_order=True):
        d = key[0] if isinstance(key, tuple) else key
        yield normalize_date(d), group


def benchmark_metrics(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return {}
    rets = s.pct_change().dropna()
    years = (pd.Timestamp(s.index[-1]) - pd.Timestamp(s.index[0])).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    dd = s / s.cummax() - 1.0
    ann_vol = rets.std(ddof=1) * math.sqrt(252) if len(rets) > 1 else np.nan
    sharpe = (
        rets.mean() / rets.std(ddof=1) * math.sqrt(252)
        if len(rets) > 1 and rets.std(ddof=1) > 0
        else np.nan
    )
    return {
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
        "days": int(len(s)),
        "terminal_value": float(s.iloc[-1]),
        "total_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe_0rf": float(sharpe),
        "max_drawdown": float(dd.min()),
    }


def run_symbol(
    symbol: str,
    option_dir: Path,
    underlying_path: Path,
    bil_csv: Path,
    start: date,
    end: date,
):
    underlying = load_underlying(underlying_path, start, end)
    trading_dates = [d for d in underlying.index if start <= d <= end]
    if not trading_dates:
        raise ValueError(f"{symbol}: no underlying trading dates")
    bil_rets = load_bil_returns(bil_csv, trading_dates)

    states = []
    for td in TARGET_DELTAS:
        for target_dte in TARGET_DTES:
            for alloc in ALLOCATIONS:
                for slip in SLIPPAGE_BPS:
                    states.append(
                        StrategyState(
                            underlying=symbol,
                            target_delta=td,
                            target_dte=target_dte,
                            allocation=alloc,
                            slippage_bps=slip,
                        )
                    )

    curves: Dict[Tuple[float, int, float, int], list] = {s.key: [] for s in states}
    prev_date: Optional[date] = None

    # Option parquet dates are the authoritative option-chain sessions. Missing
    # chain days are carried through using the last option mark and the BIL
    # return only; in practice the preserved dataset is close to complete.
    option_day_iter = {}
    for year in range(start.year, end.year + 1):
        p = option_dir / f"options_{year}.parquet"
        if p.exists():
            option_day_iter[year] = iter(load_year_groups(p, start, end))

    current_group = {}
    for year, it in option_day_iter.items():
        try:
            current_group[year] = next(it)
        except StopIteration:
            current_group[year] = None

    for current_date in trading_dates:
        # Accrue BIL total-return sleeve from previous trading close to current.
        bil_ret = bil_rets.get(current_date, 0.0)
        for st in states:
            if st.bil_value:
                st.bil_value *= (1.0 + bil_ret)

        day_df = pl.DataFrame()
        item = current_group.get(current_date.year)
        while item is not None and item[0] < current_date:
            try:
                item = next(option_day_iter[current_date.year])
            except StopIteration:
                item = None
                break
        if item is not None and item[0] == current_date:
            day_df = item[1]
            try:
                current_group[current_date.year] = next(option_day_iter[current_date.year])
            except StopIteration:
                current_group[current_date.year] = None

        qmap = quote_map(day_df)
        valid = valid_chain(day_df, current_date) if not day_df.is_empty() else day_df
        candidates = best_candidates(valid, current_date) if not day_df.is_empty() else {}

        # Mark held contracts first using actual close-mid when available.
        for st in states:
            if st.option_contract and st.option_contract in qmap:
                bid, ask = qmap[st.option_contract]
                st.option_mark = (bid + ask) / 2.0

        for st in states:
            # 1) Execute a queued exit at this close using actual bid.
            just_exited = False
            if st.pending_exit and st.option_contract:
                quote = qmap.get(st.option_contract)
                if quote is not None:
                    bid, _ = quote
                    px = bid * (1.0 - st.slippage)
                    qty = st.option_qty
                    proceeds = qty * px * OPTION_MULTIPLIER
                    fee = option_fee(qty)
                    st.cash += proceeds - fee
                    st.audit.append({
                        "event": "EXIT_FILL",
                        "date": str(current_date),
                        "underlying": st.underlying,
                        "target_delta": st.target_delta,
                        "target_dte": st.target_dte,
                        "allocation": st.allocation,
                        "slippage_bps": st.slippage_bps,
                        "contract_id": st.option_contract,
                        "bid": bid,
                        "ask": quote[1],
                        "fill_price": px,
                        "quantity": qty,
                        "fee": fee,
                        "entry_date": str(st.entry_date) if st.entry_date else "",
                        "portfolio_value_before": st.portfolio_value(),
                    })
                    st.option_contract = None
                    st.option_expiration = None
                    st.option_qty = 0
                    st.option_mark = 0.0
                    st.entry_date = None
                    st.pending_exit = False
                    st.roll_count += 1
                    just_exited = True

            # 2) Execute queued entry at this close using actual ask.
            if st.pending_entry is not None and st.option_contract is None and not just_exited:
                pe = st.pending_entry
                quote = qmap.get(pe.candidate.contract_id)
                if quote is not None:
                    _, ask = quote
                    fill_px = ask * (1.0 + st.slippage)
                    pv_before = st.portfolio_value()
                    budget = max(0.0, pv_before * st.allocation)
                    old_cash = st.cash
                    old_bil = st.bil_value

                    # Never increase quantity after selection. Resize only
                    # downward if the next-session quote moved up enough that
                    # the original size would exceed the frozen allocation.
                    affordable = int((budget * 0.995) // (fill_px * OPTION_MULTIPLIER))
                    qty = min(pe.selected_qty, affordable)
                    if qty >= 1:
                        if qty < pe.selected_qty:
                            st.resized_entries += 1
                        option_cost = qty * fill_px * OPTION_MULTIPLIER
                        fee = option_fee(qty)

                        # Re-target the collateral sleeve at this real execution
                        # close. It is represented as a BIL total-return value
                        # sleeve, not raw BIL shares, to retain dividend return.
                        target_bil = max(0.0, pv_before * (1.0 - st.allocation))
                        st.cash += st.bil_value - target_bil
                        st.bil_value = target_bil

                        if st.cash + 1e-9 >= option_cost + fee:
                            st.cash -= option_cost + fee
                            st.option_contract = pe.candidate.contract_id
                            st.option_expiration = pe.candidate.expiration
                            st.option_qty = qty
                            st.option_mark = (quote[0] + quote[1]) / 2.0
                            st.entry_date = current_date
                            st.entry_count += 1
                            st.audit.append({
                                "event": "ENTRY_FILL",
                                "date": str(current_date),
                                "selection_date": str(pe.selection_date),
                                "underlying": st.underlying,
                                "target_delta": st.target_delta,
                                "target_dte": st.target_dte,
                                "allocation": st.allocation,
                                "slippage_bps": st.slippage_bps,
                                "contract_id": pe.candidate.contract_id,
                                "expiration": str(pe.candidate.expiration),
                                "strike": pe.candidate.strike,
                                "selection_delta": pe.candidate.delta,
                                "selection_dte": pe.candidate.dte,
                                "selection_bid": pe.candidate.bid,
                                "selection_ask": pe.candidate.ask,
                                "fill_bid": quote[0],
                                "fill_ask": quote[1],
                                "fill_price": fill_px,
                                "selected_quantity": pe.selected_qty,
                                "fill_quantity": qty,
                                "fee": fee,
                                "portfolio_value_before": pv_before,
                            })
                            st.pending_entry = None
                        else:
                            # Undo collateral retarget if execution cannot be
                            # funded. The next completed chain may select again.
                            st.cash = old_cash
                            st.bil_value = old_bil
                            st.rejected_entries += 1
                            st.pending_entry = None
                    else:
                        st.rejected_entries += 1
                        st.pending_entry = None

            # 3) Close-of-day decision logic.
            if st.pending_exit or st.pending_entry is not None:
                curves[st.key].append((current_date, st.portfolio_value()))
                continue

            if st.option_contract is not None:
                dte = (
                    st.option_expiration - current_date
                ).days if st.option_expiration else 10_000
                held_days = (
                    current_date - st.entry_date
                ).days if st.entry_date else 10_000
                if dte < 180 or held_days >= 182:
                    st.pending_exit = True
            else:
                cand = candidates.get((st.target_delta, st.target_dte))
                if cand is None:
                    st.no_candidate_days += 1
                else:
                    pv = st.portfolio_value()
                    budget = pv * st.allocation
                    qty = int((budget * 0.995) // (cand.ask * OPTION_MULTIPLIER))
                    if qty >= 1:
                        st.audit.append({
                            "event": "ENTRY_SELECTION",
                            "date": str(current_date),
                            "underlying": st.underlying,
                            "target_delta": st.target_delta,
                            "target_dte": st.target_dte,
                            "allocation": st.allocation,
                            "slippage_bps": st.slippage_bps,
                            "contract_id": cand.contract_id,
                            "expiration": str(cand.expiration),
                            "strike": cand.strike,
                            "delta": cand.delta,
                            "dte": cand.dte,
                            "bid": cand.bid,
                            "ask": cand.ask,
                            "open_interest": cand.open_interest,
                            "score": cand.score,
                            "selected_quantity": qty,
                            "portfolio_value": pv,
                        })
                        st.pending_entry = PendingEntry(
                            candidate=cand,
                            selected_qty=qty,
                            selection_date=current_date,
                        )
                    else:
                        st.no_candidate_days += 1

            curves[st.key].append((current_date, st.portfolio_value()))

        prev_date = current_date

    rows = []
    audit_rows = []
    for st in states:
        curve = pd.Series(
            [v for _, v in curves[st.key]],
            index=[d for d, _ in curves[st.key]],
            dtype=float,
        )
        metrics = benchmark_metrics(curve)
        rows.append(
            {
                "underlying": symbol,
                "target_delta": st.target_delta,
                "target_dte": st.target_dte,
                "allocation": st.allocation,
                "slippage_bps": st.slippage_bps,
                "execution_proxy": "signal_close_to_next_close_quote_side",
                "evidence_label": "FREE_DISCOVERY_LOCAL_EOD_PROXY",
                **metrics,
                "entries": st.entry_count,
                "rolls": st.roll_count,
                "resized_entries": st.resized_entries,
                "rejected_entries": st.rejected_entries,
                "no_candidate_days": st.no_candidate_days,
            }
        )
        audit_rows.extend(st.audit)

    # Underlying adjusted-close benchmark normalized to $1m.
    bench = underlying.loc[trading_dates, "adjusted_close"].dropna()
    bench_curve = INITIAL_CASH * bench / float(bench.iloc[0])
    benchmark = benchmark_metrics(bench_curve)
    benchmark["underlying"] = symbol
    benchmark["benchmark"] = f"{symbol}_buy_and_hold_adjusted"
    return pd.DataFrame(rows), benchmark, pd.DataFrame(audit_rows)


def summarize(surface: pd.DataFrame) -> dict:
    out = {
        "cells": int(len(surface)),
        "sample_start": str(surface["start"].min()),
        "sample_end": str(surface["end"].max()),
        "best_cagr_cell": {},
        "median_cagr": float(surface["cagr"].median()),
        "median_max_drawdown": float(surface["max_drawdown"].median()),
    }
    if len(surface):
        best = surface.sort_values("cagr", ascending=False).iloc[0]
        out["best_cagr_cell"] = {
            k: (
                float(best[k])
                if isinstance(best[k], (np.floating, float))
                else int(best[k])
                if isinstance(best[k], (np.integer, int))
                else str(best[k])
            )
            for k in [
                "underlying",
                "target_delta",
                "target_dte",
                "allocation",
                "slippage_bps",
                "cagr",
                "max_drawdown",
                "sharpe_0rf",
            ]
        }
    by_underlying = {}
    for symbol, g in surface.groupby("underlying"):
        by_underlying[symbol] = {
            "cells": int(len(g)),
            "median_cagr": float(g["cagr"].median()),
            "max_cagr": float(g["cagr"].max()),
            "min_cagr": float(g["cagr"].min()),
            "median_max_drawdown": float(g["max_drawdown"].median()),
        }
    out["by_underlying"] = by_underlying
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--bil-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default=str(START_DATE))
    ap.add_argument("--end", default=str(DATASET_END_DATE))
    args = ap.parse_args()

    start = pd.Timestamp(args.start).date()
    requested_end = pd.Timestamp(args.end).date()
    end = min(requested_end, DATASET_END_DATE)

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    benchmarks = []
    all_audits = []

    for symbol in ("SPY", "QQQ"):
        root = args.data_root / symbol.lower()
        surface, benchmark, audit = run_symbol(
            symbol=symbol,
            option_dir=root,
            underlying_path=root / "underlying_prices.parquet",
            bil_csv=args.bil_csv,
            start=start,
            end=end,
        )
        all_rows.append(surface)
        benchmarks.append(benchmark)
        all_audits.append(audit)

    result = pd.concat(all_rows, ignore_index=True)
    # Stable frozen-grid order.
    result = result.sort_values(
        ["underlying", "target_delta", "target_dte", "allocation", "slippage_bps"]
    ).reset_index(drop=True)
    result.insert(0, "local_experiment_id", [f"LEOD{i:03d}" for i in range(1, len(result) + 1)])
    result.to_csv(args.out / "leaps_local_eod_surface.csv", index=False)

    pd.DataFrame(benchmarks).to_csv(args.out / "leaps_local_benchmarks.csv", index=False)
    audits = pd.concat(all_audits, ignore_index=True)
    audits.to_csv(args.out / "leaps_local_trade_audit.csv", index=False)

    summary = summarize(result)
    summary.update(
        {
            "frozen_grid_cells_expected": 162,
            "frozen_grid_cells_completed": int(len(result)),
            "frozen_original_end": str(FROZEN_END_DATE),
            "free_dataset_end": str(DATASET_END_DATE),
            "requested_end": str(requested_end),
            "actual_end": str(end),
            "execution_difference": (
                "Free source is a 4:00 PM ET EOD chain. Contract selection uses "
                "the completed close and execution uses the next trading day's "
                "actual quote-side close, not an unavailable next-open QuoteBar."
            ),
            "sizing_difference": (
                "Quantity is never increased after selection; it may be resized "
                "downward at next-close execution to remain within the frozen "
                "premium allocation after an overnight move."
            ),
            "collateral": (
                "BIL adjusted-close total-return sleeve from manisahni/marketdata; "
                "BIL trading fees are omitted in this local proxy."
            ),
            "option_fee_model": (
                "QuantConnect InteractiveBrokersFeeModel low-volume equity-option "
                "tier: $0.70/contract for premium >= $0.10, $1 minimum."
            ),
            "source_options": "anahatsingh-ui/options-dataset-hist preservation mirror",
            "trade_audit_rows": int(len(audits)),
            "evidence_label": "FREE_DISCOVERY_LOCAL_EOD_PROXY",
        }
    )
    (args.out / "leaps_local_eod_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
