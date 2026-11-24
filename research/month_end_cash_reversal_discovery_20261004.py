from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

OUT = Path("results/month_end_cash_reversal_discovery_20261004")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("1993-02-01")
TRAIN_END = pd.Timestamp("2008-12-31")
VALID_START = pd.Timestamp("2009-01-01")
VALID_END = pd.Timestamp("2019-12-31")
DOWNLOAD_END_EXCLUSIVE = "2020-01-01"  # hard firewall: no holdout prices
T2_EFFECTIVE = pd.Timestamp("2017-09-05")
ROUND_TRIP_COST = 0.0005
HORIZONS = (1, 3, 5)
SEED = 20261004
BOOTSTRAP_DRAWS = 10000
BASELINE_EXCLUSION_SESSIONS = 7
VOL_REFERENCE_SESSIONS = 252

PREREG_PATH = Path("research/MONTH_END_CASH_REVERSAL_PREREG.md")


def download_spy() -> pd.Series:
    x = yf.download(
        "SPY",
        start="1993-01-01",
        end=DOWNLOAD_END_EXCLUSIVE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if x.empty:
        raise RuntimeError("SPY discovery download returned no rows")
    close = x["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    if close.index.max() > VALID_END:
        raise AssertionError(f"HOLDOUT FIREWALL BREACH: max price date {close.index.max()}")
    if close.index.min() > TRAIN_START:
        raise AssertionError("Insufficient SPY discovery history")
    return close


def partition_of(d: pd.Timestamp) -> str | None:
    if TRAIN_START <= d <= TRAIN_END:
        return "train"
    if VALID_START <= d <= VALID_END:
        return "validation"
    return None


def settlement_lag(d: pd.Timestamp) -> int:
    return 3 if d < T2_EFFECTIVE else 2


def build_deadlines(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    s = pd.Series(np.arange(len(index)), index=index)
    months = pd.Series(index, index=index).groupby(index.to_period("M"))
    for _, dates_ser in months:
        dates = pd.DatetimeIndex(dates_ser.values)
        t = dates[-1]
        part = partition_of(t)
        if part is None:
            continue
        lag = settlement_lag(t)
        pos_t = int(s.loc[t])
        pos_d = pos_t - lag
        if pos_d < 0:
            continue
        d = index[pos_d]
        rows.append({
            "month_end": t,
            "deadline": d,
            "settlement_lag": lag,
            "partition": part,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No month-end deadlines constructed")
    if pd.to_datetime(out["deadline"]).max() > VALID_END:
        raise AssertionError("HOLDOUT FIREWALL BREACH in deadline construction")
    return out


def pit_vol_quintile(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ret = close.pct_change()
    rv20 = ret.rolling(20).std(ddof=1) * math.sqrt(252)
    q = pd.Series(np.nan, index=close.index)
    # Date d's matching state uses RV available through d-1, ranked only
    # against RV observations already known by d-1.
    for i in range(21, len(close)):
        signal_rv = rv20.iloc[i - 1]
        if not np.isfinite(signal_rv):
            continue
        lo = max(0, i - 1 - VOL_REFERENCE_SESSIONS + 1)
        hist = rv20.iloc[lo:i].dropna()  # through i-1 only
        if len(hist) < 60:
            continue
        pct = float((hist <= signal_rv).mean())
        q.iloc[i] = min(5, max(1, int(math.ceil(pct * 5))))
    return rv20, q


def forward_return(close: pd.Series, i: int, h: int) -> float:
    if i + h >= len(close):
        return np.nan
    return float(close.iloc[i + h] / close.iloc[i] - 1.0)


def one_day_ending(close: pd.Series, i: int) -> float:
    if i <= 0:
        return np.nan
    return float(close.iloc[i] / close.iloc[i - 1] - 1.0)


def bootstrap_ci(x: np.ndarray, seed_offset: int = 0) -> tuple[float | None, float | None]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None, None
    rng = np.random.default_rng(SEED + seed_offset)
    idx = rng.integers(0, len(x), size=(BOOTSTRAP_DRAWS, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric_row(values: list[float], label: str, h: int) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "partition": label, "horizon": h, "n": 0,
            "mean_excess": np.nan, "median_excess": np.nan,
            "positive_pct": np.nan, "tstat": np.nan,
            "pvalue_2s": np.nan, "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }
    if len(x) > 1 and np.std(x, ddof=1) > 0:
        tstat, pval = stats.ttest_1samp(x, 0.0)
    else:
        tstat, pval = np.nan, np.nan
    lo, hi = bootstrap_ci(x, seed_offset=h + (100 if label == "validation" else 0))
    return {
        "partition": label,
        "horizon": h,
        "n": int(len(x)),
        "mean_excess": float(np.mean(x)),
        "median_excess": float(np.median(x)),
        "positive_pct": float(np.mean(x > 0)),
        "tstat": float(tstat) if np.isfinite(tstat) else np.nan,
        "pvalue_2s": float(pval) if np.isfinite(pval) else np.nan,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
    }


def main() -> None:
    prereg = PREREG_PATH.read_bytes()
    prereg_sha = hashlib.sha256(prereg).hexdigest()

    close = download_spy()
    dates = close.index
    date_to_i = {d: i for i, d in enumerate(dates)}
    rv20, vq = pit_vol_quintile(close)
    deadlines = build_deadlines(dates)

    deadline_positions = sorted(date_to_i[pd.Timestamp(d)] for d in deadlines["deadline"])
    blocked = set()
    for p in deadline_positions:
        for j in range(max(0, p - BASELINE_EXCLUSION_SESSIONS),
                       min(len(dates), p + BASELINE_EXCLUSION_SESSIONS + 1)):
            blocked.add(j)

    # Precompute eligible matched baseline dates, split by partition/weekday/PIT-vol-quintile.
    pools: dict[tuple[str, int, int], list[int]] = {}
    for i, d in enumerate(dates):
        part = partition_of(d)
        if part is None or i in blocked or not np.isfinite(vq.iloc[i]):
            continue
        key = (part, int(d.weekday()), int(vq.iloc[i]))
        pools.setdefault(key, []).append(i)

    event_rows = []
    pressure_by_regime: dict[str, list[float]] = {"T3": [], "T2": []}
    deadline_vs_nearby = []

    for evn, row in deadlines.iterrows():
        d = pd.Timestamp(row.deadline)
        t = pd.Timestamp(row.month_end)
        part = str(row.partition)
        lag = int(row.settlement_lag)
        i = date_to_i[d]
        if not np.isfinite(vq.iloc[i]):
            continue
        key = (part, int(d.weekday()), int(vq.iloc[i]))
        candidates = pools.get(key, [])
        # To prevent mechanically comparing an event to itself or overlapping event zone,
        # candidates already exclude every +/-7-session deadline neighborhood.
        if len(candidates) < 10:
            continue

        event_pressure = one_day_ending(close, i)
        base_pressure = np.asarray([one_day_ending(close, j) for j in candidates], dtype=float)
        base_pressure = base_pressure[np.isfinite(base_pressure)]
        pressure_excess = float(event_pressure - base_pressure.mean()) if len(base_pressure) else np.nan
        regime = "T3" if lag == 3 else "T2"
        if np.isfinite(pressure_excess):
            pressure_by_regime[regime].append(pressure_excess)

        nearby_pressures = []
        for off in (-2, -1, 1, 2):
            k = i + off
            if 0 < k < len(close) and partition_of(dates[k]) == part:
                nearby_pressures.append(one_day_ending(close, k))
        if nearby_pressures and np.isfinite(event_pressure):
            deadline_vs_nearby.append(float(event_pressure - np.nanmean(nearby_pressures)))

        for h in HORIZONS:
            eraw = forward_return(close, i, h)
            braw = np.asarray([forward_return(close, j, h) for j in candidates], dtype=float)
            braw = braw[np.isfinite(braw)]
            if not np.isfinite(eraw) or len(braw) == 0:
                continue
            # Same round-trip cost on event and matched baseline strategies cancels in excess,
            # but net returns are preserved explicitly for auditability.
            event_net = eraw - ROUND_TRIP_COST
            baseline_net_mean = float(np.mean(braw - ROUND_TRIP_COST))
            excess = float(event_net - baseline_net_mean)
            event_rows.append({
                "month_end": str(t.date()),
                "deadline": str(d.date()),
                "partition": part,
                "settlement_regime": regime,
                "settlement_lag_sessions": lag,
                "weekday": int(d.weekday()),
                "pit_vol_quintile": int(vq.iloc[i]),
                "pit_rv20": float(rv20.iloc[i - 1]) if i > 0 and np.isfinite(rv20.iloc[i - 1]) else np.nan,
                "matched_baseline_n": int(len(braw)),
                "horizon": h,
                "event_raw_return": float(eraw),
                "event_net_return": float(event_net),
                "baseline_raw_mean": float(np.mean(braw)),
                "baseline_net_mean": baseline_net_mean,
                "excess_return": excess,
                "deadline_pressure_1d": float(event_pressure),
                "matched_pressure_1d_mean": float(base_pressure.mean()) if len(base_pressure) else np.nan,
                "deadline_pressure_excess": pressure_excess,
            })

    ev = pd.DataFrame(event_rows)
    if ev.empty:
        raise RuntimeError("No discovery events survived frozen matching rules")
    if pd.to_datetime(ev["deadline"]).max() > VALID_END:
        raise AssertionError("HOLDOUT FIREWALL BREACH in event output")

    summaries = []
    for part in ("train", "validation"):
        for h in HORIZONS:
            vals = ev[(ev.partition == part) & (ev.horizon == h)]["excess_return"].to_numpy()
            summaries.append(metric_row(vals.tolist(), part, h))
    summary = pd.DataFrame(summaries)

    # Mechanism diagnostics use one row/event; horizon=1 is just a de-duplication choice,
    # not a mechanism parameter.
    unique_events = ev[ev.horizon == 1].copy()
    reg_diag = {}
    for regime in ("T3", "T2"):
        z = unique_events[unique_events.settlement_regime == regime]["deadline_pressure_excess"].dropna().to_numpy()
        lo, hi = bootstrap_ci(z, seed_offset=300 + (3 if regime == "T3" else 2))
        reg_diag[regime] = {
            "n": int(len(z)),
            "mean_deadline_pressure_excess": float(np.mean(z)) if len(z) else None,
            "median_deadline_pressure_excess": float(np.median(z)) if len(z) else None,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
        }

    nearby_arr = np.asarray(deadline_vs_nearby, dtype=float)
    nearby_arr = nearby_arr[np.isfinite(nearby_arr)]
    mech = {
        "T3": reg_diag["T3"],
        "T2": reg_diag["T2"],
        "deadline_vs_nearby_offsets": {
            "offsets_sessions": [-2, -1, 1, 2],
            "n": int(len(nearby_arr)),
            "mean_deadline_minus_nearby_pressure": float(np.mean(nearby_arr)) if len(nearby_arr) else None,
            "median_deadline_minus_nearby_pressure": float(np.median(nearby_arr)) if len(nearby_arr) else None,
        },
    }

    # Frozen falsifier: require both settlement regimes, with >=12 observations in T2
    # and deadline pressure negative in each; also deadline must be more negative
    # than nearby offsets on average.
    sufficient_t3 = reg_diag["T3"]["n"] >= 12
    sufficient_t2 = reg_diag["T2"]["n"] >= 12
    negative_t3 = sufficient_t3 and reg_diag["T3"]["mean_deadline_pressure_excess"] < 0
    negative_t2 = sufficient_t2 and reg_diag["T2"]["mean_deadline_pressure_excess"] < 0
    deadline_more_negative = len(nearby_arr) >= 24 and float(np.mean(nearby_arr)) < 0
    mechanism_pass = bool(negative_t3 and negative_t2 and deadline_more_negative)
    mech["falsifier"] = {
        "sufficient_T3": bool(sufficient_t3),
        "sufficient_T2": bool(sufficient_t2),
        "negative_pressure_T3": bool(negative_t3),
        "negative_pressure_T2": bool(negative_t2),
        "deadline_more_negative_than_nearby": bool(deadline_more_negative),
        "passed": mechanism_pass,
    }

    tr = summary[summary.partition == "train"].set_index("horizon")
    va = summary[summary.partition == "validation"].set_index("horizon")
    valid_positive = int((va.mean_excess > 0).sum())
    median_valid = float(np.median(va.mean_excess.to_numpy()))
    median_train = float(np.median(tr.mean_excess.to_numpy()))
    data_quality_pass = bool(
        len(unique_events[unique_events.partition == "train"]) >= 120
        and len(unique_events[unique_events.partition == "validation"]) >= 80
        and unique_events.matched_baseline_n.min() >= 10
    )
    advance = bool(
        valid_positive >= 2
        and median_valid > 0
        and median_train > 0
        and mechanism_pass
        and data_quality_pass
    )

    advancement = {
        "preregistration": str(PREREG_PATH),
        "preregistration_sha256": prereg_sha,
        "discovery_price_max_date": str(close.index.max().date()),
        "holdout_status": "SEALED_NOT_DOWNLOADED",
        "frozen_horizons": list(HORIZONS),
        "round_trip_cost": ROUND_TRIP_COST,
        "validation_positive_mean_excess_horizons": valid_positive,
        "median_validation_mean_excess": median_valid,
        "median_train_mean_excess": median_train,
        "mechanism_falsifier_passed": mechanism_pass,
        "data_quality_passed": data_quality_pass,
        "decision": "ADVANCE_TO_SINGLE_FROZEN_HOLDOUT" if advance else "DO_NOT_OPEN_HOLDOUT",
        "disposition": "MONITOR_PENDING_HOLDOUT" if advance else ("KILL" if data_quality_pass and not mechanism_pass else "MONITOR"),
    }

    ev.to_csv(OUT / "discovery_event_level.csv", index=False)
    summary.to_csv(OUT / "discovery_summary.csv", index=False)
    (OUT / "mechanism_diagnostics.json").write_text(json.dumps(mech, indent=2))
    (OUT / "advancement.json").write_text(json.dumps(advancement, indent=2))

    if advance:
        frozen = {
            "source_preregistration": str(PREREG_PATH),
            "source_preregistration_sha256": prereg_sha,
            "discovery_decision": advancement,
            "holdout_sample_start": "2020-01-01",
            "holdout_rule": "Run exactly the preregistered settlement-adjusted deadline strategy at horizons 1,3,5 with identical PIT vol matching, +/-7 session baseline exclusion, and 5bp round trip cost. No parameter changes.",
            "t1_effective": "2024-05-28",
            "status": "READY_FOR_SINGLE_HOLDOUT_RUN",
        }
        (OUT / "frozen_holdout_spec.json").write_text(json.dumps(frozen, indent=2))

    print("HOLDOUT_FIREWALL_MAX_PRICE_DATE", close.index.max().date(), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(mech, indent=2), flush=True)
    print(json.dumps(advancement, indent=2), flush=True)


if __name__ == "__main__":
    main()
