from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("results/mechanism_first/month_end_cash_need_reversal_20261004")
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("1995-06-07")
T2 = pd.Timestamp("2017-09-05")
T1 = pd.Timestamp("2024-05-28")
END_DOWNLOAD = "2026-10-05"
COST_SIDE = 0.0005
ROUND_TRIP_MULT = (1.0 - COST_SIDE) ** 2
BOOT_SEED = 20261004
BOOT_DRAWS = 10_000


def settlement_n(dt: pd.Timestamp) -> int | None:
    if dt < START:
        return None
    if dt < T2:
        return 3
    if dt < T1:
        return 2
    return 1


def regime_name(dt: pd.Timestamp) -> str | None:
    n = settlement_n(dt)
    return None if n is None else f"T+{n}"


def download_spy() -> pd.DataFrame:
    x = yf.download(
        "SPY",
        start="1994-01-01",
        end=END_DOWNLOAD,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if x.empty:
        raise RuntimeError("SPY download returned no rows")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x.sort_index()
    px = pd.to_numeric(x["Adj Close"], errors="coerce").rename("adj_close")
    z = pd.DataFrame({"adj_close": px}).dropna()
    z["ret"] = z.adj_close.pct_change()
    z["rv20"] = z.ret.rolling(20).std(ddof=1) * math.sqrt(252)
    # All date-i matching information must be known before date i begins:
    # use RV observed through i-1, and rank it only against the preceding 252 known RV observations.
    z["rv20_prior"] = z.rv20.shift(1)
    ranks = np.full(len(z), np.nan)
    vals = z.rv20_prior.to_numpy(float)
    for i in range(len(z)):
        if not np.isfinite(vals[i]):
            continue
        lo = max(0, i - 251)
        hist = vals[lo:i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 100:
            continue
        # Percentile rank among already-known observations. Clip 1.0 into Q5.
        ranks[i] = np.mean(hist <= vals[i])
    z["vol_rank"] = ranks
    q = np.floor(np.minimum(z.vol_rank.fillna(np.nan), 0.999999) * 5)
    z["vol_quintile"] = q
    z["weekday"] = z.index.weekday
    z["regime"] = [regime_name(d) for d in z.index]
    return z


def month_end_events(z: pd.DataFrame) -> pd.DataFrame:
    idx = z.index
    positions = {d: i for i, d in enumerate(idx)}
    rows = []
    live = z.loc[z.index >= START]
    for period, group in live.groupby(live.index.to_period("M")):
        T = group.index[-1]
        tpos = positions[T]
        D = None
        dpos = None
        # Latest trade session whose own standard settlement lands no later than month-end session T.
        for p in range(tpos, max(-1, tpos - 10), -1):
            dt = idx[p]
            n = settlement_n(dt)
            if n is None:
                continue
            if p + n <= tpos:
                D = dt
                dpos = p
                break
        if D is None:
            continue
        n = settlement_n(D)
        rows.append(
            {
                "month": str(period),
                "T": T,
                "D": D,
                "regime": f"T+{n}",
                "settlement_lag": n,
                "D_offset_sessions": tpos - dpos,
                "D_weekday": D.weekday(),
            }
        )
    e = pd.DataFrame(rows)
    if e.empty:
        raise RuntimeError("No month-end events generated")
    return e


def add_forward_returns(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    for h in (1, 3):
        z[f"fwd{h}_gross"] = z.adj_close.shift(-h) / z.adj_close - 1.0
    z["fwd3_net"] = (1.0 + z.fwd3_gross) * ROUND_TRIP_MULT - 1.0
    # Price-pressure diagnostic: previous close -> date close.
    z["day_ret"] = z.adj_close / z.adj_close.shift(1) - 1.0
    return z


def build_exclusion_positions(z: pd.DataFrame, events: pd.DataFrame) -> set[int]:
    pos = {d: i for i, d in enumerate(z.index)}
    out: set[int] = set()
    for d in pd.to_datetime(events.D):
        i = pos.get(d)
        if i is None:
            continue
        for j in range(max(0, i - 5), min(len(z), i + 6)):
            out.add(j)
    return out


def matched_pool(
    z: pd.DataFrame,
    date: pd.Timestamp,
    exclusion: set[int],
    require_regime: str | None = None,
) -> pd.DataFrame:
    if date not in z.index:
        return z.iloc[0:0]
    row = z.loc[date]
    if not np.isfinite(row.vol_quintile):
        return z.iloc[0:0]
    reg = require_regime if require_regime is not None else row.regime
    mask = (
        (z.regime == reg)
        & (z.weekday == row.weekday)
        & (z.vol_quintile == row.vol_quintile)
        & z.fwd3_net.notna()
    )
    p = z.loc[mask].copy()
    pos = {d: i for i, d in enumerate(z.index)}
    keep = [d for d in p.index if pos[d] not in exclusion]
    return p.loc[keep]


def event_table(z: pd.DataFrame, events: pd.DataFrame, exclusion: set[int]) -> pd.DataFrame:
    out = []
    for _, ev in events.iterrows():
        d = pd.Timestamp(ev.D)
        if d not in z.index:
            continue
        row = z.loc[d]
        pool = matched_pool(z, d, exclusion, ev.regime)
        if len(pool) == 0 or not np.isfinite(row.fwd3_net):
            continue
        out.append(
            {
                **ev.to_dict(),
                "event_net3": float(row.fwd3_net),
                "event_gross3": float(row.fwd3_gross),
                "event_pressure1": float(row.day_ret),
                "vol_quintile": int(row.vol_quintile) + 1,
                "baseline_n": int(len(pool)),
                "baseline_net3": float(pool.fwd3_net.mean()),
                "baseline_pressure1": float(pool.day_ret.mean()),
                "excess_net3": float(row.fwd3_net - pool.fwd3_net.mean()),
                "pressure_excess1": float(row.day_ret - pool.day_ret.mean()),
            }
        )
    return pd.DataFrame(out)


def stats(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"n": 0}
    v = x.excess_net3.to_numpy(float)
    r = x.event_net3.to_numpy(float)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(BOOT_DRAWS)
    for i in range(BOOT_DRAWS):
        boots[i] = rng.choice(v, len(v), replace=True).mean()
    return {
        "n": int(len(x)),
        "event_net3_mean": float(r.mean()),
        "event_net3_median": float(np.median(r)),
        "event_net3_std": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        "matched_baseline_net3_mean": float(x.baseline_net3.mean()),
        "mean_excess_net3": float(v.mean()),
        "median_excess_net3": float(np.median(v)),
        "positive_event_return_rate": float(np.mean(r > 0)),
        "positive_excess_rate": float(np.mean(v > 0)),
        "worst_event_net3": float(r.min()),
        "best_event_net3": float(r.max()),
        "bootstrap_95_low_mean_excess": float(np.quantile(boots, 0.025)),
        "bootstrap_95_high_mean_excess": float(np.quantile(boots, 0.975)),
        "mean_predeadline_pressure": float(x.event_pressure1.mean()),
        "matched_pressure_mean": float(x.baseline_pressure1.mean()),
        "mean_pressure_excess": float(x.pressure_excess1.mean()),
    }


def subperiod(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp("2006-12-31"):
        return "TRAIN_T3_1995_2006"
    if date < T2:
        return "VALIDATION_T3_2007_2017"
    if date < T1:
        return "OOS_T2_2017_2024"
    return "OOS_T1_2024_2026"


def timing_test(z: pd.DataFrame, events: pd.DataFrame, exclusion: set[int]) -> dict:
    idx = z.index
    pos = {d: i for i, d in enumerate(idx)}
    result = {}
    for reg, expected_k in (("T+3", 3), ("T+2", 2), ("T+1", 1)):
        months = events[events.regime == reg]
        offsets = {}
        detail_rows = []
        for k in (1, 2, 3):
            vals = []
            nobs = 0
            for _, ev in months.iterrows():
                T = pd.Timestamp(ev.T)
                tp = pos.get(T)
                if tp is None or tp - k < 0:
                    continue
                d = idx[tp - k]
                # Keep only observations whose counterfactual date lies in the era being tested.
                if regime_name(d) != reg:
                    continue
                row = z.loc[d]
                pool = matched_pool(z, d, exclusion, reg)
                if len(pool) == 0 or not np.isfinite(row.fwd3_net):
                    continue
                ex = float(row.fwd3_net - pool.fwd3_net.mean())
                vals.append(ex)
                nobs += 1
                detail_rows.append({"regime": reg, "offset": k, "month": ev.month, "date": d, "excess_net3": ex})
            offsets[str(k)] = {
                "n": nobs,
                "mean_excess_net3": None if not vals else float(np.mean(vals)),
            }
        means = {int(k): v["mean_excess_net3"] for k, v in offsets.items() if v["mean_excess_net3"] is not None}
        winner = None
        if len(means) == 3:
            maxv = max(means.values())
            wins = [k for k, v in means.items() if np.isclose(v, maxv, rtol=0, atol=1e-12)]
            winner = wins[0] if len(wins) == 1 else None
        result[reg] = {
            "expected_offset": expected_k,
            "offsets": offsets,
            "winning_offset": winner,
            "pass": bool(winner == expected_k),
        }
    return result


def spy_context(z: pd.DataFrame, event_table_: pd.DataFrame) -> dict:
    if event_table_.empty:
        return {}
    start = pd.Timestamp(event_table_.D.min())
    end = min(pd.Timestamp(event_table_.D.max()), z.index[-1])
    p = z.loc[start:end, "adj_close"].dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    r = p.pct_change().dropna()
    eq = p / p.iloc[0]
    dd = eq / eq.cummax() - 1
    return {
        "start": str(p.index[0].date()),
        "end": str(p.index[-1].date()),
        "cagr": float((p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan,
        "ann_vol": float(r.std(ddof=1) * math.sqrt(252)),
        "max_drawdown": float(dd.min()),
    }


def self_checks(events: pd.DataFrame) -> None:
    # Event construction must mechanically shift with standard settlement.
    assert set(events[events.D.between(pd.Timestamp("2010-01-01"), pd.Timestamp("2016-12-31"))].D_offset_sessions) == {3}
    assert set(events[events.D.between(pd.Timestamp("2018-01-01"), pd.Timestamp("2023-12-31"))].D_offset_sessions) == {2}
    assert set(events[events.D >= pd.Timestamp("2024-06-01")].D_offset_sessions) == {1}
    assert events.D.is_monotonic_increasing
    assert events.D.duplicated().sum() == 0


def main() -> None:
    z = add_forward_returns(download_spy())
    events = month_end_events(z)
    self_checks(events)
    exclusion = build_exclusion_positions(z, events)
    ev = event_table(z, events, exclusion)
    if ev.empty:
        raise RuntimeError("No testable event rows")
    ev["D"] = pd.to_datetime(ev.D)
    ev["T"] = pd.to_datetime(ev.T)
    ev["subperiod"] = ev.D.map(subperiod)

    overall = stats(ev)
    by_regime = {reg: stats(g) for reg, g in ev.groupby("regime")}
    by_subperiod = {name: stats(g) for name, g in ev.groupby("subperiod")}
    timing = timing_test(z, events, exclusion)

    year = (
        ev.assign(year=ev.D.dt.year)
        .groupby("year")
        .agg(
            events=("excess_net3", "size"),
            mean_excess_net3=("excess_net3", "mean"),
            sum_excess_net3=("excess_net3", "sum"),
            positive_excess_rate=("excess_net3", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )

    return_pass = bool(overall["mean_excess_net3"] >= 0.0020)
    timing_pass = bool(all(timing[r]["pass"] for r in ("T+3", "T+2", "T+1")))
    decision = "PROMOTE" if return_pass and timing_pass else "KILL"

    decision_obj = {
        "strategy": "Month-End Institutional Cash-Need Reversal",
        "frozen_spec": "research/mechanism_first/MONTH_END_CASH_NEED_FROZEN_SPEC_2026-10-04.md",
        "mechanism": "settlement-driven institutional cash raising -> temporary pre-deadline price pressure -> post-deadline reversal",
        "primary_falsifier": {
            "return_condition": "full-sample matched-baseline-adjusted 3-session net reversal >= +20 bps",
            "settlement_timing_condition": "actual settlement offset ranks first in each T+3, T+2, and T+1 era",
        },
        "return_condition_pass": return_pass,
        "timing_condition_pass": timing_pass,
        "decision": decision,
        "overall": overall,
        "by_regime": by_regime,
        "by_subperiod": by_subperiod,
        "timing_test": timing,
        "spy_context": spy_context(z, ev),
        "data": {
            "instrument": "SPY",
            "source": "Yahoo Finance adjusted close via yfinance",
            "download_rows": int(len(z)),
            "first_price_date": str(z.index.min().date()),
            "last_price_date": str(z.index.max().date()),
            "first_test_event": str(ev.D.min().date()),
            "last_test_event": str(ev.D.max().date()),
            "settlement_business_day_proxy": "observed US equity trading sessions",
        },
        "costs": {
            "one_way_bps": 5.0,
            "round_trip_bps_nominal": 10.0,
        },
        "notes": [
            "No flow, Friday, funding, ownership, leverage, or cross-sectional filter is used.",
            "No parameter search is performed.",
            "Pre-deadline pressure is a mechanism diagnostic and is not substituted for the preregistered primary falsifier.",
        ],
    }

    ev.to_csv(OUT / "event_audit.csv", index=False)
    year.to_csv(OUT / "year_concentration.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps(decision_obj, indent=2, default=str) + "\n")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "official_settlement_dates": {
                    "T+3_effective": "1995-06-07",
                    "T+2_compliance": "2017-09-05",
                    "T+1_compliance": "2024-05-28",
                },
                "official_sources": [
                    "https://www.sec.gov/rules-regulations/2004/03/securities-transactions-settlement",
                    "https://www.sec.gov/newsroom/press-releases/2017-68-0",
                    "https://www.sec.gov/newsroom/press-releases/2024-62",
                ],
                "freeze_commit_must_predate_results": True,
            },
            indent=2,
        )
        + "\n"
    )

    print(json.dumps(decision_obj, indent=2, default=str))


if __name__ == "__main__":
    main()
