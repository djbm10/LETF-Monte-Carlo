from __future__ import annotations

"""Local free large-cap proxy for the frozen Bottleneck Winner tournament.

This executor is intentionally NOT labeled the full-US Stage-1 replication.
It substitutes:
- point-in-time S&P 500 issuer membership for QuantConnect's full PIT US common-share universe;
- SEC as-filed shares * raw market price for market capitalization;
- prior-session signals and first-quarter-session close execution for the unavailable
  QuantConnect opening fill.

Everything else preserves the frozen research definitions: SEC PIT factors,
CIK-native Item-1 scarcity, model weights, top-N grid, 252-session price signals,
and 15 bps one-way trading drag.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ONE_WAY_COST = 0.0015
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1_000_000.0
MIN_CROSS_SECTION = 20

MODEL_WEIGHTS = {
    "momentum": {"momentum12": 0.70, "high52": 0.30},
    "quality_momentum": {
        "momentum12": 0.35,
        "high52": 0.15,
        "quality": 0.30,
        "dilution": 0.20,
    },
    "bottleneck_core": {
        "demand_accel": 0.35,
        "pricing_power": 0.30,
        "competitive_scarcity": 0.35,
    },
    "bottleneck_momentum": {
        "demand_accel": 0.20,
        "pricing_power": 0.15,
        "competitive_scarcity": 0.20,
        "momentum12": 0.30,
        "high52": 0.15,
    },
    "bottleneck_full": {
        "demand_accel": 0.15,
        "pricing_power": 0.10,
        "competitive_scarcity": 0.15,
        "quality": 0.15,
        "dilution": 0.10,
        "valuation": 0.10,
        "momentum12": 0.15,
        "high52": 0.10,
    },
}


def _clean_date(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace("*", "", regex=False).str.strip()
    x = x.replace({"": None, "nan": None, "NaN": None, "None": None})
    return pd.to_datetime(x, errors="coerce").dt.normalize()


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load_membership(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"cik": str})
    required = {"symbol", "cik", "date_added", "date_removed"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"membership missing columns: {sorted(missing)}")
    m["symbol"] = m["symbol"].astype(str).str.upper().str.strip()
    m["cik"] = (
        m["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    m["date_added"] = _clean_date(m["date_added"])
    m["date_removed"] = _clean_date(m["date_removed"])
    m = m.dropna(subset=["symbol", "cik", "date_added"]).copy()

    # A recycled ticker can splice two unrelated issuers into one vendor price
    # history. Legacy/noisy membership therefore excludes any symbol mapped to
    # multiple CIKs. The curated pipeline may explicitly resolve ticker identity
    # by date. In that case allow reuse only when different-CIK intervals do not
    # overlap; otherwise fail rather than guess.
    reuse = m.groupby("symbol")["cik"].nunique()
    reused_symbols = sorted(reuse[reuse > 1].index.astype(str).tolist())
    resolved = (
        "identity_resolved" in m.columns
        and m["identity_resolved"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ).all()
    )
    m.attrs["reused_symbol_exclusions"] = []
    if reused_symbols and resolved:
        bad = []
        for sym in reused_symbols:
            z = m[m.symbol == sym].sort_values("date_added")
            rows = list(z.itertuples(index=False))
            for i, a in enumerate(rows):
                a_end = (
                    pd.Timestamp(a.date_removed)
                    if pd.notna(a.date_removed)
                    else pd.Timestamp("2100-01-01")
                )
                for b in rows[i + 1:]:
                    if str(a.cik) == str(b.cik):
                        continue
                    b_end = (
                        pd.Timestamp(b.date_removed)
                        if pd.notna(b.date_removed)
                        else pd.Timestamp("2100-01-01")
                    )
                    if max(pd.Timestamp(a.date_added), pd.Timestamp(b.date_added)) < min(a_end, b_end):
                        bad.append(
                            {
                                "symbol": sym,
                                "cik_a": str(a.cik),
                                "start_a": str(pd.Timestamp(a.date_added).date()),
                                "end_a": str(a_end.date()),
                                "cik_b": str(b.cik),
                                "start_b": str(pd.Timestamp(b.date_added).date()),
                                "end_b": str(b_end.date()),
                            }
                        )
        if bad:
            raise ValueError(f"overlapping reused-symbol CIK identities: {bad[:10]}")
        print("ALLOWING_DATE_RESOLVED_REUSED_SYMBOLS", reused_symbols, flush=True)
    elif reused_symbols:
        m.attrs["reused_symbol_exclusions"] = reused_symbols
        print("EXCLUDING_REUSED_SYMBOLS", reused_symbols, flush=True)
        m = m[~m.symbol.isin(reused_symbols)].copy()
    return m


def active_membership(m: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    return m[
        (m.date_added <= d)
        & (m.date_removed.isna() | (m.date_removed > d))
    ].copy()


def load_prices(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    rename = {}
    for c in p.columns:
        k = str(c).strip().lower().replace("_", " ")
        if k == "date":
            rename[c] = "date"
        elif k in ("ticker", "symbol", "yahoo ticker"):
            # Prefer explicit Ticker if both YahooTicker and Ticker exist.
            if c == "Ticker" or "symbol" in k or "ticker" == k:
                rename[c] = "symbol"
        elif k == "open":
            rename[c] = "open"
        elif k == "high":
            rename[c] = "high"
        elif k == "low":
            rename[c] = "low"
        elif k == "close":
            rename[c] = "close"
        elif k in ("adj close", "adjclose", "adjusted close"):
            rename[c] = "adj_close"
        elif k == "volume":
            rename[c] = "volume"
    p = p.rename(columns=rename)
    if "date" not in p or "symbol" not in p or "close" not in p:
        raise ValueError(f"price panel missing date/symbol/close: {list(p.columns)}")
    if "adj_close" not in p:
        p["adj_close"] = p["close"]
    if "volume" not in p:
        p["volume"] = np.nan
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip().str.replace("-", ".", regex=False)
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["date", "symbol", "close", "adj_close"])
    p = p[(p.close > 0) & (p.adj_close > 0)]
    return p.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")


def load_sec(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, dtype={"cik": str})
    s["cik"] = s["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    s["formation_date"] = pd.to_datetime(s["formation_date"]).dt.normalize()
    s["information_date"] = pd.to_datetime(s["information_date"]).dt.normalize()
    if (s.information_date > s.formation_date).any():
        bad = s[s.information_date > s.formation_date].head()
        raise ValueError(f"future SEC rows detected: {bad.to_dict('records')}")
    return s.sort_values(["cik", "formation_date", "information_date"])


def load_network(path: Path) -> pd.DataFrame:
    n = pd.read_csv(path, dtype={"cik": str})
    n["cik"] = n["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    n["formation_date"] = pd.to_datetime(n["formation_date"]).dt.normalize()
    return n.sort_values(["cik", "formation_date"])


def exact_snapshot_by_cik(frame: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Use only the exact frozen quarter-end snapshot.

    Carrying an older network row forward can resurrect a CIK that was
    intentionally excluded by the 550-day Item-1 staleness rule.
    """
    x = frame[frame.formation_date == pd.Timestamp(d).normalize()].copy()
    if x.empty:
        return x
    sort_cols = ["cik", "formation_date"]
    if "information_date" in x.columns:
        sort_cols.append("information_date")
    return x.sort_values(sort_cols).groupby("cik", as_index=False).tail(1)


def prior_quarter_snapshot_date(rebalance_date: pd.Timestamp) -> pd.Timestamp:
    # Rebalances occur in Jan/Apr/Jul/Oct. The frozen accounting/network input
    # is the immediately prior calendar quarter-end, even if that date was a
    # weekend/holiday.
    return (pd.Timestamp(rebalance_date).normalize() - pd.offsets.MonthEnd(1)).normalize()


def winsor_z(values: pd.Series, min_n: int = MIN_CROSS_SECTION) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    good = x.dropna()
    if len(good) < min_n:
        return pd.Series(np.nan, index=values.index, dtype=float)
    a = np.sort(good.to_numpy(dtype=float))
    lo = a[max(0, int(0.01 * (len(a) - 1)))]
    hi = a[min(len(a) - 1, int(0.99 * (len(a) - 1)))]
    clipped = x.clip(lo, hi)
    mu = float(clipped.dropna().mean())
    sd = float(clipped.dropna().std(ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (clipped - mu) / sd


def price_index(prices: pd.DataFrame):
    return {
        symbol: g.set_index("date").sort_index()
        for symbol, g in prices.groupby("symbol", sort=False)
    }


def asof_row(g: pd.DataFrame | None, d: pd.Timestamp):
    if g is None or g.empty:
        return None
    x = g.loc[:d]
    if x.empty:
        return None
    return x.iloc[-1]


def trailing_prices(g: pd.DataFrame | None, d: pd.Timestamp, n: int = 252):
    if g is None or g.empty:
        return None
    x = g.loc[:d, "adj_close"].dropna().tail(n)
    if len(x) < n:
        return None
    return x


def annualized_revenue(row) -> float:
    rev = _num(row.get("revenue"))
    qtrs = _num(row.get("revenue_qtrs"))
    if not math.isfinite(rev) or not math.isfinite(qtrs):
        return np.nan
    if int(qtrs) == 4:
        return rev
    if int(qtrs) == 1:
        return rev * 4.0
    return np.nan


def build_cross_section(
    rebalance_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    formation_date: pd.Timestamp,
    membership: pd.DataFrame,
    sec: pd.DataFrame,
    network: pd.DataFrame,
    prices_by_symbol: dict,
) -> pd.DataFrame:
    active = active_membership(membership, rebalance_date)
    sec_latest = exact_snapshot_by_cik(sec, formation_date)
    net_latest = exact_snapshot_by_cik(network, formation_date)
    sec_map = {r.cik: r for r in sec_latest.itertuples(index=False)}
    net_map = {r.cik: r for r in net_latest.itertuples(index=False)}

    # The S&P can carry multiple listed share classes for one issuer. The
    # frozen full-US universe keeps primary shares only, so in this issuer proxy
    # retain one active symbol per CIK: the eligible class with highest prior-day
    # dollar volume.
    candidates = []
    for r in active.itertuples(index=False):
        sym = str(r.symbol).upper().replace("-", ".")
        g = prices_by_symbol.get(sym)
        prow = asof_row(g, signal_date)
        hist = trailing_prices(g, signal_date, 252)
        if prow is None or hist is None:
            continue
        if pd.Timestamp(prow.name) != signal_date:
            # Do not score a halted/suspended/missing name off a stale quote.
            continue
        close = _num(prow.get("close"))
        volume = _num(prow.get("volume"))
        dollar_volume = close * volume if math.isfinite(close) and math.isfinite(volume) else np.nan
        if not math.isfinite(close) or close < MIN_PRICE:
            continue
        if not math.isfinite(dollar_volume) or dollar_volume < MIN_DOLLAR_VOLUME:
            continue

        p_now = float(hist.iloc[-1])
        p_12m = float(hist.iloc[-252])
        momentum12 = p_now / p_12m - 1.0 if p_12m > 0 else np.nan
        high = float(hist.max())
        high52 = p_now / high if high > 0 else np.nan

        srow = sec_map.get(r.cik)
        nrow = net_map.get(r.cik)

        if srow is not None:
            demand = _num(getattr(srow, "revenue_growth_yoy", np.nan))
            gross_delta = _num(getattr(srow, "gross_margin_change_yoy", np.nan))
            op_delta = _num(getattr(srow, "operating_margin_change_yoy", np.nan))
            pricing = gross_delta if math.isfinite(gross_delta) else op_delta
            fcf = _num(getattr(srow, "fcf_margin", np.nan))
            opm = _num(getattr(srow, "operating_margin", np.nan))
            leverage = _num(getattr(srow, "leverage", np.nan))
            if math.isfinite(fcf) and math.isfinite(leverage):
                quality = fcf - leverage
            elif math.isfinite(opm) and math.isfinite(leverage):
                quality = opm - leverage
            else:
                quality = np.nan
            share_growth = _num(getattr(srow, "share_growth_yoy", np.nan))
            dilution = -share_growth if math.isfinite(share_growth) else np.nan
            shares = _num(getattr(srow, "shares", np.nan))
            mcap = close * shares if math.isfinite(shares) and shares > 0 else np.nan
            ar = annualized_revenue(pd.Series(srow._asdict()))
            multiple = mcap / ar if math.isfinite(mcap) and math.isfinite(ar) and ar > 0 else np.nan
            valuation = -math.log1p(max(multiple, 0.0)) if math.isfinite(multiple) else np.nan
            info_date = getattr(srow, "information_date", pd.NaT)
            fund_formation = getattr(srow, "formation_date", pd.NaT)
        else:
            demand = pricing = quality = dilution = valuation = mcap = np.nan
            info_date = pd.NaT
            fund_formation = pd.NaT

        scarcity = _num(getattr(nrow, "text_scarcity_raw", np.nan)) if nrow is not None else np.nan
        network_formation = getattr(nrow, "formation_date", pd.NaT) if nrow is not None else pd.NaT

        candidates.append({
            "rebalance_date": rebalance_date,
            "signal_date": signal_date,
            "symbol": sym,
            "cik": r.cik,
            "raw_close": close,
            "dollar_volume": dollar_volume,
            "market_cap_proxy": mcap,
            "demand_accel": demand,
            "pricing_power": pricing,
            "competitive_scarcity": scarcity,
            "quality": quality,
            "dilution": dilution,
            "valuation": valuation,
            "momentum12": momentum12,
            "high52": high52,
            "fund_information_date": info_date,
            "fund_formation_date": fund_formation,
            "network_formation_date": network_formation,
            "date_removed": getattr(r, "date_removed", pd.NaT),
        })

    if not candidates:
        return pd.DataFrame()
    x = pd.DataFrame(candidates)
    x = x.sort_values(["cik", "dollar_volume"], ascending=[True, False]).drop_duplicates("cik", keep="first")
    return x.reset_index(drop=True)


def score_cross_section(x: pd.DataFrame, model: str) -> pd.DataFrame:
    if x.empty:
        return x.assign(score=np.nan)
    weights = MODEL_WEIGHTS[model]
    zcols = {}
    for feature in weights:
        zcols[feature] = winsor_z(x[feature])
    score = pd.Series(0.0, index=x.index)
    complete = pd.Series(True, index=x.index)
    for feature, w in weights.items():
        z = zcols[feature]
        complete &= z.notna()
        score = score + w * z.fillna(0.0)
    out = x.copy()
    out["score"] = score.where(complete)
    return out.dropna(subset=["score"]).sort_values("score", ascending=False)


def trading_sessions(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    counts = prices[(prices.date >= start) & (prices.date <= end)].groupby("date").symbol.nunique()
    # S&P proxy panel normally has hundreds of symbols. A low threshold avoids
    # deleting early sessions when historical recovery is incomplete.
    threshold = max(20, int(np.nanmedian(counts.to_numpy()) * 0.20)) if len(counts) else 20
    return pd.DatetimeIndex(counts[counts >= threshold].index).sort_values()


def quarter_rebalances(sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    out = []
    for y in range(int(sessions.min().year), int(sessions.max().year) + 1):
        for m in (1, 4, 7, 10):
            xs = sessions[(sessions.year == y) & (sessions.month == m)]
            if len(xs):
                out.append(pd.Timestamp(xs[0]))
    return out


def position_value_at(
    symbol: str,
    base_value: float,
    entry_price: float,
    current_date: pd.Timestamp,
    removal_date: pd.Timestamp | None,
    prices_by_symbol: dict,
):
    g = prices_by_symbol.get(symbol)
    if g is None or g.empty or not math.isfinite(entry_price) or entry_price <= 0:
        return base_value, current_date, True
    cap_date = current_date
    was_removed = False
    if pd.notna(removal_date) and pd.Timestamp(removal_date) <= current_date:
        cap_date = pd.Timestamp(removal_date)
        was_removed = True
    row = asof_row(g, cap_date)
    if row is None:
        return base_value, cap_date, True
    px = _num(row.get("adj_close"))
    if not math.isfinite(px) or px <= 0:
        return base_value, cap_date, True
    return base_value * px / entry_price, pd.Timestamp(row.name), was_removed


@dataclass
class Position:
    symbol: str
    cik: str
    value_at_entry: float
    entry_price: float
    removal_date: pd.Timestamp | None


def security_identity_key(symbol: str, cik: str) -> str:
    return f"{str(symbol).upper()}::{str(cik).zfill(10)}"


def solve_rebalance(nav_pre: float, current_values: dict[str, float], targets: list[str], cost: float):
    if not targets:
        return nav_pre, {}, 0.0, 0.0
    nav_after = nav_pre
    gross = 0.0
    for _ in range(12):
        target_value = nav_after / len(targets)
        all_syms = set(current_values) | set(targets)
        gross = sum(abs((target_value if s in targets else 0.0) - current_values.get(s, 0.0)) for s in all_syms)
        new_nav = nav_pre - cost * gross
        if abs(new_nav - nav_after) < max(1e-6, nav_pre * 1e-12):
            nav_after = new_nav
            break
        nav_after = new_nav
    target_value = nav_after / len(targets)
    return nav_after, {s: target_value for s in targets}, gross, gross / nav_pre if nav_pre > 0 else np.nan


def metrics_from_curve(
    curve: pd.Series,
    baseline_value: float | None = None,
    baseline_date: pd.Timestamp | None = None,
) -> dict:
    curve = curve.dropna().sort_index()
    if curve.empty:
        return {}
    x = curve.copy()
    if baseline_value is not None and math.isfinite(float(baseline_value)):
        bd = pd.Timestamp(baseline_date if baseline_date is not None else x.index[0])
        bd = bd - pd.Timedelta(nanoseconds=1)
        x = pd.concat([pd.Series([float(baseline_value)], index=[bd]), x]).sort_index()
    if len(x) < 2:
        return {}
    start, end = x.index[0], x.index[-1]
    years = max((end - start).total_seconds() / (365.25 * 86400.0), 1 / 365.25)
    total = float(x.iloc[-1] / x.iloc[0] - 1.0)
    cagr = float((x.iloc[-1] / x.iloc[0]) ** (1.0 / years) - 1.0)
    rets = x.pct_change().dropna()
    ann_vol = float(rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 else np.nan
    sharpe = (
        float(rets.mean() / rets.std(ddof=1) * math.sqrt(252))
        if len(rets) > 1 and rets.std(ddof=1) > 0
        else np.nan
    )
    dd = x / x.cummax() - 1.0
    return {
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "observations": int(len(x)),
        "terminal_value": float(x.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "ann_vol_daily": ann_vol,
        "sharpe_0rf_daily": sharpe,
        "max_drawdown_daily": float(dd.min()),
    }


def period_metrics(
    curve: pd.Series,
    start: str,
    end: str,
    initial_nav: float | None = None,
) -> dict:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    x = curve[(curve.index >= a) & (curve.index <= b)].copy()
    if x.empty:
        return {}
    prior = curve[curve.index < a]
    if len(prior):
        baseline = float(prior.iloc[-1])
    elif initial_nav is not None:
        baseline = float(initial_nav)
    else:
        baseline = float(x.iloc[0])
    return metrics_from_curve(x, baseline_value=baseline, baseline_date=a)


def run_cell(
    model: str,
    top_n: int,
    rebalances: list[pd.Timestamp],
    membership: pd.DataFrame,
    sec: pd.DataFrame,
    network: pd.DataFrame,
    prices_by_symbol: dict,
    sessions: pd.DatetimeIndex,
    all_sessions: pd.DatetimeIndex,
):
    initial_nav = 10_000_000.0
    nav = initial_nav
    positions: dict[str, Position] = {}
    curve: list[tuple[pd.Timestamp, float]] = []
    selection_rows = []
    rebalance_rows = []
    stale_price_events = 0
    removed_seen = set()

    all_session_pos = {pd.Timestamp(d): i for i, d in enumerate(all_sessions)}
    rebalance_set = {pd.Timestamp(d) for d in rebalances}

    for d0 in sessions:
        d = pd.Timestamp(d0)
        current_values = {}
        for position_key, pos in list(positions.items()):
            value, mark_date, removed = position_value_at(
                pos.symbol,
                pos.value_at_entry,
                pos.entry_price,
                d,
                pos.removal_date,
                prices_by_symbol,
            )
            current_values[position_key] = value
            if mark_date < d and not removed:
                stale_price_events += 1
            if removed:
                key = (
                    pos.symbol,
                    pos.cik,
                    str(pos.removal_date),
                    float(pos.entry_price),
                )
                removed_seen.add(key)

        nav_pre = float(sum(current_values.values())) if positions else nav

        if d not in rebalance_set:
            nav = nav_pre
            curve.append((d, nav))
            continue

        pos_idx = all_session_pos.get(d)
        if pos_idx is None or pos_idx == 0:
            nav = nav_pre
            curve.append((d, nav))
            continue

        signal_date = pd.Timestamp(all_sessions[pos_idx - 1])
        formation_date = prior_quarter_snapshot_date(d)

        cross = build_cross_section(
            d, signal_date, formation_date, membership, sec, network, prices_by_symbol
        )
        scored = score_cross_section(cross, model)
        selected = scored.head(top_n).copy()

        if len(selected) < min(top_n, MIN_CROSS_SECTION):
            rebalance_rows.append({
                "rebalance_date": str(d.date()),
                "signal_date": str(signal_date.date()),
                "formation_date": str(formation_date.date()),
                "model": model,
                "top_n": top_n,
                "eligible_rows": int(len(cross)),
                "scored_rows": int(len(scored)),
                "selected_rows": int(len(selected)),
                "executed": False,
                "nav_pre": nav_pre,
            })
            nav = nav_pre
            curve.append((d, nav))
            continue

        selected = selected.copy()
        selected["security_key"] = [
            security_identity_key(sym, cik)
            for sym, cik in zip(selected.symbol, selected.cik)
        ]
        targets = selected.security_key.tolist()
        nav_after, target_values, gross_trade, turnover = solve_rebalance(
            nav_pre, current_values, targets, ONE_WAY_COST
        )

        selected_map = selected.set_index("security_key")
        new_positions = {}
        for position_key, target_value in target_values.items():
            sr = selected_map.loc[position_key]
            sym = str(sr.symbol)
            g = prices_by_symbol.get(sym)
            row = asof_row(g, d)
            if row is None or pd.Timestamp(row.name) != d:
                continue
            px = _num(row.get("adj_close"))
            if not math.isfinite(px) or px <= 0:
                continue
            new_positions[position_key] = Position(
                sym,
                str(sr.cik),
                float(target_value),
                float(px),
                pd.Timestamp(sr.date_removed) if pd.notna(sr.date_removed) else pd.NaT,
            )

        if len(new_positions) != len(targets):
            targets = list(new_positions)
            nav_after, target_values, gross_trade, turnover = solve_rebalance(
                nav_pre, current_values, targets, ONE_WAY_COST
            )
            for position_key in list(new_positions):
                p = new_positions[position_key]
                new_positions[position_key] = Position(
                    p.symbol,
                    p.cik,
                    target_values[position_key],
                    p.entry_price,
                    p.removal_date,
                )

        positions = new_positions
        nav = nav_after
        curve.append((d, nav))

        for rank, r in enumerate(selected.itertuples(index=False), 1):
            position_key = security_identity_key(r.symbol, r.cik)
            if position_key not in positions:
                continue
            row = {
                "rebalance_date": str(d.date()),
                "signal_date": str(signal_date.date()),
                "formation_date": str(formation_date.date()),
                "model": model,
                "top_n": top_n,
                "rank": rank,
                "symbol": r.symbol,
                "cik": r.cik,
                "security_key": position_key,
                "score": r.score,
                "eligible_rows": int(len(cross)),
                "scored_rows": int(len(scored)),
                "market_cap_proxy": r.market_cap_proxy,
                "fund_information_date": r.fund_information_date,
                "fund_formation_date": r.fund_formation_date,
                "network_formation_date": r.network_formation_date,
            }
            for feature in MODEL_WEIGHTS[model]:
                row[feature] = getattr(r, feature)
            selection_rows.append(row)

        rebalance_rows.append({
            "rebalance_date": str(d.date()),
            "signal_date": str(signal_date.date()),
            "formation_date": str(formation_date.date()),
            "model": model,
            "top_n": top_n,
            "eligible_rows": int(len(cross)),
            "scored_rows": int(len(scored)),
            "selected_rows": int(len(positions)),
            "executed": True,
            "nav_pre": nav_pre,
            "nav_post_cost": nav_after,
            "gross_trade": gross_trade,
            "turnover": turnover,
        })

    curve_s = pd.Series(dict(curve), dtype=float).sort_index()
    baseline_date = pd.Timestamp(sessions[0]) if len(sessions) else None
    summary = metrics_from_curve(
        curve_s,
        baseline_value=initial_nav,
        baseline_date=baseline_date,
    )
    summary.update({
        "model": model,
        "top_n": top_n,
        "stale_price_events": int(stale_price_events),
        "removed_position_events": int(len(removed_seen)),
        "mean_turnover": (
            float(pd.DataFrame(rebalance_rows).turnover.dropna().mean())
            if rebalance_rows and "turnover" in pd.DataFrame(rebalance_rows)
            else np.nan
        ),
        "train": period_metrics(curve_s, "2009-01-01", "2014-12-31", initial_nav),
        "validation": period_metrics(curve_s, "2015-01-01", "2019-12-31", initial_nav),
        "holdout": period_metrics(curve_s, "2020-01-01", "2023-12-31", initial_nav),
        "evidence_label": "FREE_DISCOVERY_LARGE_CAP_EOD_PROXY",
    })
    return summary, pd.DataFrame(selection_rows), pd.DataFrame(rebalance_rows), curve_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--fundamentals", type=Path, required=True)
    ap.add_argument("--network", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    membership = load_membership(args.membership)
    reused_symbol_exclusions = membership.attrs.get("reused_symbol_exclusions", [])
    sec = load_sec(args.fundamentals)
    network = load_network(args.network)
    prices = load_prices(args.prices)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    prices = prices[(prices.date >= start - pd.Timedelta(days=550)) & (prices.date <= end)].copy()
    by_symbol = price_index(prices)
    all_sessions = trading_sessions(prices, prices.date.min(), end)
    sessions = all_sessions[(all_sessions >= start) & (all_sessions <= end)]
    if len(sessions) < 1000:
        raise RuntimeError(f"too few trading sessions in price panel: {len(sessions)}")
    rebalances = [d for d in quarter_rebalances(sessions) if start <= d <= end]

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    selections = []
    rebalances_audit = []
    curves = []

    for model in MODEL_WEIGHTS:
        for top_n in (10, 20, 40):
            summary, sel, reb, curve = run_cell(
                model, top_n, rebalances, membership, sec, network,
                by_symbol, sessions, all_sessions
            )
            experiment_id = f"{model}:{top_n}"
            summary["experiment_id"] = experiment_id
            summaries.append(summary)
            if not sel.empty:
                sel.insert(0, "experiment_id", experiment_id)
                selections.append(sel)
            if not reb.empty:
                reb.insert(0, "experiment_id", experiment_id)
                rebalances_audit.append(reb)
            if len(curve):
                c = curve.rename("portfolio_value").reset_index().rename(columns={"index": "date"})
                c.insert(0, "experiment_id", experiment_id)
                curves.append(c)
            print(
                experiment_id,
                "CAGR",
                summary.get("cagr"),
                "train",
                summary.get("train", {}).get("cagr"),
                "validation",
                summary.get("validation", {}).get("cagr"),
                "holdout",
                summary.get("holdout", {}).get("cagr"),
                flush=True,
            )

    surface_rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if k not in ("train", "validation", "holdout")}
        for period in ("train", "validation", "holdout"):
            for k, v in s.get(period, {}).items():
                row[f"{period}_{k}"] = v
        surface_rows.append(row)
    surface = pd.DataFrame(surface_rows)
    surface.to_csv(args.out / "bottleneck_sp500_proxy_surface.csv", index=False)
    if selections:
        pd.concat(selections, ignore_index=True).to_csv(args.out / "bottleneck_sp500_proxy_selection_audit.csv", index=False)
    if rebalances_audit:
        pd.concat(rebalances_audit, ignore_index=True).to_csv(args.out / "bottleneck_sp500_proxy_rebalance_audit.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(args.out / "bottleneck_sp500_proxy_quarterly_curves.csv", index=False)

    result = {
        "cells_expected": 15,
        "cells_completed": int(len(surface)),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "price_rows": int(len(prices)),
        "price_symbols": int(prices.symbol.nunique()),
        "trading_sessions": int(len(sessions)),
        "rebalances": int(len(rebalances)),
        "reused_symbol_exclusions": reused_symbol_exclusions,
        "reused_symbol_exclusion_count": int(len(reused_symbol_exclusions)),
        "universe": "PIT_S_AND_P_500_ISSUER_PROXY",
        "execution_proxy": "prior-session signals; first-quarter-session close rebalance",
        "market_cap_proxy": "SEC as-filed shares * prior-session raw close",
        "transaction_cost": "15 bps one-way gross trade",
        "path_metrics_frequency": "daily marked NAV; trades remain quarterly",
        "warning": (
            "This is a free large-cap EOD proxy, not the frozen full-US QuantConnect "
            "replication. Historical price gaps/delisting coverage must be audited "
            "before interpreting returns."
        ),
        "evidence_label": "FREE_DISCOVERY_LARGE_CAP_EOD_PROXY",
    }
    (args.out / "bottleneck_sp500_proxy_summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    if len(surface) != 15:
        raise RuntimeError(f"expected 15 cells, got {len(surface)}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
