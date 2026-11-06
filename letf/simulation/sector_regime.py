"""Joint conditional-regime Monte Carlo for frozen Nasdaq 35/0 vs 80/20 sectors.

Research infrastructure, not an exchange-settlement-grade performance engine.
All return columns in one historical row are sampled together. All portfolio
signals are recomputed along each synthetic market history. Never sample the
already-computed strategy returns and call it an endogenous simulation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping
import json
import math

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # CI and environments without numba still work
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

SECTOR_PAIRS = {
    "XLK": "ROM", "SOXX": "USD", "XLF": "UYG", "XLV": "RXL",
    "XLE": "DIG", "XLY": "UCC", "XLB": "UYM", "XLI": "UXI",
    "XLP": "UGE", "VNQ": "URE",
}
FROZEN_RULE = {
    "nasdaq": "QQQ above prior 200-close SMA; 35%/20d vol targeting, 3x cap; otherwise 0",
    "sectors": "63-session unlevered momentum top 1; SPY 200SMA gate; 35%/20d actual 2x ETF volatility target, 1x cap; rebalance every 21 sessions",
    "allocation": "80% Nasdaq / 20% sector sleeve; 21-session sleeve rebalancing",
    "accounting": "prior-close signals; next-session open-to-next-open returns; cash at Treasury proxy; modeled commissions, slippage and financing",
}


@dataclass(frozen=True)
class Scenario:
    years: int = 3
    paths: int = 100
    block_days: int = 63
    mode: str = "current"  # current, unconditional, persistent_high, forced_high, forced_low
    transition_mode: str = "semi_markov"  # independently calibrated regime dwell times; or block_markov
    seed: int = 20261106
    high_duration_multiplier: float = 1.0
    funding_spread_bps_annual: float = 100.0
    qqq_turnover_bps: float = 2.0
    sector_turnover_bps: float = 7.0
    sleeve_turnover_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.years < 1 or self.years > 100 or self.paths < 1 or self.paths > 1_000_000:
            raise ValueError("Invalid years or paths")
        if not 5 <= self.block_days <= 504:
            raise ValueError("Block length must be 5..504 sessions")
        if self.mode not in ("current", "unconditional", "persistent_high", "forced_high", "forced_low"):
            raise ValueError("Unknown regime scenario")
        if self.transition_mode not in ("semi_markov", "block_markov"):
            raise ValueError("Unknown transition mode")
        if not 0.25 <= self.high_duration_multiplier <= 5:
            raise ValueError("High-duration multiplier outside supported stress range")
        for key in ("funding_spread_bps_annual", "qqq_turnover_bps", "sector_turnover_bps", "sleeve_turnover_bps"):
            if not 0 <= getattr(self, key) <= 3000:
                raise ValueError(f"Invalid cost: {key}")


@dataclass
class MarketData:
    panel: np.ndarray
    seed_closes: np.ndarray
    names: tuple[str, ...]
    source_start: str
    source_end: str
    asof: str
    last_vix: float
    initial_regime: int
    regime_transition: np.ndarray
    daily_regime_durations: tuple[np.ndarray, np.ndarray]
    high_frequency: float
    rows_dropped: int

    @property
    def sector_count(self) -> int:
        return (len(self.names) - 2) // 2


def _durations(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spells = ([], [])
    if not len(state):
        raise ValueError("Missing regime observations")
    last = int(state[0]); size = 1
    for s in state[1:]:
        s = int(s)
        if s == last:
            size += 1
        else:
            spells[last].append(size);last=s;size=1
    spells[last].append(size)
    return tuple(np.asarray(a, dtype=np.int32) if a else np.asarray([21], dtype=np.int32) for a in spells)


def build_market_data(
    closes: pd.DataFrame, opens: pd.DataFrame, vix: pd.Series,
    asof: str | None = None,
    sector_pairs: Mapping[str, str] = SECTOR_PAIRS,
    min_source_days: int = 252,
) -> MarketData:
    """Build synchronized, labeled daily OPEN and CLOSE return blocks.

    Each source row t is open(t)->open(t+1) for executable P&L and
    close(t-1)->close(t) for next-session signal updates. Missing data are
    not filled with zero. Inputs should be split/dividend adjusted.
    """
    pairs = dict(sector_pairs)
    if not pairs or len(set(pairs.values())) != len(pairs):
        raise ValueError("Sector-to-leveraged-ETF mapping must be nonempty, one-to-one")
    names = ("QQQ", "SPY", *pairs.keys(), *pairs.values())
    require = set(names) | {"^IRX"}
    if not require.issubset(closes.columns) or not set(names).issubset(opens.columns):
        raise ValueError("Missing required QQQ, SPY, sector, leveraged ETF or ^IRX columns")
    if not closes.index.is_monotonic_increasing or not opens.index.is_monotonic_increasing:
        raise ValueError("Input dates must be sorted ascending")
    if closes.index.has_duplicates or opens.index.has_duplicates:
        raise ValueError("Duplicate dates in price histories")
    ix = closes.index.intersection(opens.index).intersection(vix.index)
    if asof is not None:
        ix = ix[ix <= pd.Timestamp(asof)]
    if len(ix) < 271:
        raise ValueError("Need >=271 common trading days to warm up indicators")
    c = closes.reindex(ix)
    o = opens.reindex(ix)
    x = vix.reindex(ix)
    valid_anchor = c.loc[:, list(names)].notna().all(axis=1) & o.loc[:, list(names)].notna().all(axis=1) & x.notna()
    # The anchor must be a real date on which all instruments traded.
    if not valid_anchor.iloc[-1]:
        raise ValueError("As-of date lacks a complete observed market state")
    anchor_date = ix[-1]
    seed = c.loc[:, list(names)].iloc[-270:].to_numpy(dtype=np.float64)
    if not np.isfinite(seed).all() or (seed <= 0).any():
        raise ValueError("Incomplete or nonpositive 270-session as-of anchor")
    raw = pd.DataFrame(index=ix)
    for name in names:
        raw[f"open_{name}"] = o[name].shift(-1) / o[name] - 1
        raw[f"close_{name}"] = c[name] / c[name].shift(1) - 1
    raw["rf"] = c["^IRX"].clip(0, 8) / 100 / 252
    raw["vix"] = x
    # A real sector ETF history is required: never synthesize its pre-inception returns.
    first_valid = raw.dropna().index.min()
    if pd.isna(first_valid):
        raise ValueError("No shared actual ETF history")
    raw = raw.loc[first_valid:].iloc[:-1]  # final day's next-open return unknown
    rows_dropped = int(raw.isna().any(axis=1).sum())
    if rows_dropped > max(5, len(raw) * .01):
        raise ValueError(f"More than 1% missing source days ({rows_dropped}/{len(raw)}): fail closed")
    raw = raw.dropna()
    if len(raw) < min_source_days:
        raise ValueError(f"Insufficient fully synchronized days: {len(raw)}")
    if not np.isfinite(raw.to_numpy()).all() or (raw.filter(like="open_") <= -1).any().any():
        raise ValueError("Invalid historical returns")
    a = len(names)
    panel = raw[[f"open_{n}" for n in names] + [f"close_{n}" for n in names] + ["rf", "vix"]].to_numpy(dtype=np.float64)
    # Daily smoothing creates an independently estimable duration model.
    states = (raw.vix.rolling(10, min_periods=1).median() >= 25).astype(int).to_numpy()
    durations = _durations(states)
    monthly = raw.vix.resample("ME").median().dropna()
    monthly_states = (monthly >= 25).astype(int).to_numpy()
    trans = np.ones((2, 2), dtype=np.float64)
    for prev, after in zip(monthly_states[:-1], monthly_states[1:]):
        trans[prev, after] += 1
    trans /= trans.sum(axis=1, keepdims=True)
    return MarketData(panel, seed, tuple(names), str(raw.index[0].date()), str(raw.index[-1].date()),
                      str(anchor_date.date()), float(x.iloc[-1]), int(x.iloc[-1] >= 25),
                      trans, durations, float(states.mean()), rows_dropped)


def _source_pools(m: MarketData, block_days: int) -> tuple[np.ndarray, np.ndarray]:
    highs = m.panel[:, -1] >= 25
    arr = []
    for state in (0, 1):
        choices = []
        for i in range(len(highs) - block_days + 1):
            if int(np.median(highs[i:i + min(10, block_days)])) == state:
                choices.append(i)
        if len(choices) < 10:
            raise ValueError(f"Too few historical blocks in regime {state} (need >=10)")
        arr.append(np.asarray(choices, dtype=np.int32))
    return arr[0], arr[1]


def sample_joint_indices(m: MarketData, config: Scenario) -> np.ndarray:
    """Deterministic paired sector/index return paths; no independent per-asset shuffle."""
    pools = _source_pools(m, config.block_days)
    rng = np.random.default_rng(config.seed)
    nsteps = config.years * 252
    output = np.empty((config.paths, nsteps), dtype=np.int32)
    for path_idx in range(config.paths):
        if config.mode == "forced_high" or config.mode == "persistent_high":
            state = 1
        elif config.mode == "forced_low":
            state = 0
        elif config.mode == "unconditional":
            state = int(rng.random() < m.high_frequency)
        else:
            state = m.initial_regime
        offset = 0
        remain = 0
        while offset < nsteps:
            if config.transition_mode == "semi_markov":
                if remain <= 0:
                    sampled = int(rng.choice(m.daily_regime_durations[state]))
                    if state == 1:
                        mult = config.high_duration_multiplier * (1.6 if config.mode == "persistent_high" else 1.)
                        sampled = max(1, round(sampled * mult))
                    remain = max(1, sampled)
                length = min(config.block_days, nsteps - offset, remain)
            else:
                length = min(config.block_days, nsteps - offset)
            start = int(rng.choice(pools[state]))
            output[path_idx, offset:offset + length] = np.arange(start, start + length, dtype=np.int32)
            offset += length
            if config.transition_mode == "semi_markov":
                remain -= length
                if remain == 0:
                    state = 1 - state  # two-state alternating spells, empirical dwell lengths
            else:
                p = m.regime_transition[state].copy()
                if config.mode == "persistent_high" and state == 1:
                    p[1] = max(p[1], .9)
                    p[0] = 1 - p[1]
                state = int(rng.choice(2, p=p))
    return output


@njit(cache=True)
def _run_path(panel, seed_closes, indices, k_sector, fund_spread, qqq_turn_cost,
              sector_turn_cost, sleeve_turn_cost):
    """Two strategies use identical market paths and independently updated states."""
    a = 2 + 2 * k_sector
    warm = seed_closes.shape[0]
    nsteps = len(indices)
    closes = np.empty((warm + nsteps, a), dtype=np.float64)
    closes[:warm] = seed_closes
    base_wealth = 1.0
    blend_wealth = 1.0
    peak_base = 1.0
    peak_blend = 1.0
    dd_base = 0.0
    dd_blend = 0.0
    prev_ex = 0.0
    selected = -1
    sector_weight = 0.0
    sleeve_base = .8
    sleeve_sector = .2
    for t in range(nsteps):
        current = warm + t
        prior = current - 1
        obs = indices[t]
        rf = panel[obs, 2 * a]
        smaqq = 0.0
        smasp = 0.0
        for i in range(prior - 199, prior + 1):
            smaqq += closes[i, 0]
            smasp += closes[i, 1]
        smaqq /= 200.
        smasp /= 200.
        meanret = 0.0
        for i in range(prior - 19, prior + 1):
            meanret += closes[i, 0] / closes[i - 1, 0] - 1.
        meanret /= 20.
        variance = 0.0
        for i in range(prior - 19, prior + 1):
            z = closes[i, 0] / closes[i - 1, 0] - 1. - meanret
            variance += z * z
        vol = math.sqrt(variance / 19. * 252.)
        exposure = 0.0
        if closes[prior, 0] > smaqq and vol > 1e-9:
            exposure = min(3., .35 / vol)
        # Borrow/futures spread modeled conservatively, not exchange-level exact.
        rbase = rf + exposure * (panel[obs, 0] - rf) - qqq_turn_cost * abs(exposure - prev_ex)
        rbase -= fund_spread * max(exposure - 1., 0.) / 252.
        prev_ex = exposure
        cost = 0.0
        if t % 21 == 0:
            old_sel = selected
            old_w = sector_weight
            selected = -1
            sector_weight = 0.0
            best_score = -1e100
            if closes[prior, 1] > smasp:
                for sector in range(k_sector):
                    past = closes[prior - 63, 2 + sector]
                    if past <= 0:
                        continue
                    score = closes[prior, 2 + sector] / past - 1.
                    if score > best_score:
                        best_score = score
                        selected = sector
                if selected >= 0:
                    meanl = 0.0
                    for i in range(prior - 19, prior + 1):
                        meanl += closes[i, 2 + k_sector + selected] / closes[i - 1, 2 + k_sector + selected] - 1.
                    meanl /= 20.
                    varl = 0.0
                    for i in range(prior - 19, prior + 1):
                        z = closes[i, 2 + k_sector + selected] / closes[i - 1, 2 + k_sector + selected] - 1. - meanl
                        varl += z * z
                    lv = math.sqrt(varl / 19. * 252.)
                    if lv > 1e-9:
                        sector_weight = min(1., .35 / max(lv, .08))
                    else:
                        selected = -1
            turnover = abs(sector_weight - old_w) if selected == old_sel else sector_weight + old_w
            cost = sector_turn_cost * turnover
        rsector = rf * (1. - sector_weight) - cost
        if selected >= 0:
            rsector += sector_weight * panel[obs, 2 + k_sector + selected]
        # Sector ETF weight naturally drifts with the ETF price between rebalances.
        if selected >= 0 and rsector > -.999:
            sector_weight = min(1., max(0., sector_weight * (1. + panel[obs, 2 + k_sector + selected]) / (1. + rsector)))
        sleeve_cost = 0.0
        if t % 21 == 0:
            sleeve_cost = sleeve_turn_cost * 2. * abs(sleeve_base - .8)
            sleeve_base = .8
            sleeve_sector = .2
        rmix = (1. - sleeve_cost) * (sleeve_base * (1. + rbase) + sleeve_sector * (1. + rsector)) - 1.
        if rbase <= -1. or rmix <= -1.:
            return -1e6, -1e6, -1., -1., 0
        base_wealth *= (1. + rbase)
        blend_wealth *= (1. + rmix)
        sb = sleeve_base * (1. + rbase)
        ss = sleeve_sector * (1. + rsector)
        sleeve_base = sb / (sb + ss)
        sleeve_sector = ss / (sb + ss)
        peak_base = max(peak_base, base_wealth)
        peak_blend = max(peak_blend, blend_wealth)
        dd_base = min(dd_base, base_wealth / peak_base - 1.)
        dd_blend = min(dd_blend, blend_wealth / peak_blend - 1.)
        # Update the simulated close history, AFTER positions/P&L for this period.
        for j in range(a):
            closes[current, j] = closes[prior, j] * (1. + panel[obs, a + j])
    return math.log(base_wealth) * 252 / nsteps, math.log(blend_wealth) * 252 / nsteps, dd_base, dd_blend, 1


@njit(cache=True)
def _run_many(panel, seed_closes, paths, sector_count, spread, qtc, stc, btc):
    out = np.empty((len(paths), 5), dtype=np.float64)
    for i in range(len(paths)):
        a, b, c, d, ok = _run_path(panel, seed_closes, paths[i], sector_count, spread, qtc, stc, btc)
        out[i, 0] = a; out[i, 1] = b; out[i, 2] = c; out[i, 3] = d; out[i, 4] = ok
    return out


def evaluate(m: MarketData, cfg: Scenario) -> dict:
    """Recompute both frozen policies on paired paths; returns summary and metadata."""
    paths = sample_joint_indices(m, cfg)
    r = _run_many(m.panel, m.seed_closes, paths, m.sector_count,
                  cfg.funding_spread_bps_annual / 1e4, cfg.qqq_turnover_bps / 1e4,
                  cfg.sector_turnover_bps / 1e4, cfg.sleeve_turnover_bps / 1e4)
    good = r[r[:, 4] > 0]
    if not len(good):
        raise ValueError("No valid simulated paths (bankruptcy or bad data)")
    diff = good[:, 1] - good[:, 0]
    return {
        **asdict(cfg), "valid": int(len(good)), "bankrupt": int(len(r) - len(good)),
        "p_blend_higher_terminal_wealth": float(np.mean(diff > 0)),
        "p_blend_smaller_drawdown": float(np.mean(good[:, 3] > good[:, 2])),
        "median_annual_log_return_advantage": float(np.median(diff)),
        "p10_annual_log_return_advantage": float(np.quantile(diff, .1)),
        "p90_annual_log_return_advantage": float(np.quantile(diff, .9)),
        "median_baseline_cagr": float(np.median(np.expm1(good[:, 0]))),
        "median_blend_cagr": float(np.median(np.expm1(good[:, 1]))),
        "p50_drawdown_baseline": float(np.median(good[:, 2])),
        "p50_drawdown_blend": float(np.median(good[:, 3])),
        "p_baseline_50pct_drawdown": float(np.mean(good[:, 2] <= -.5)),
        "p_blend_50pct_drawdown": float(np.mean(good[:, 3] <= -.5)),
    }


def manifest(m: MarketData, scenarios: list[Scenario]) -> dict:
    digest = sha256(np.ascontiguousarray(m.panel).tobytes()).hexdigest()
    return {
        "model": "conditional-sector-multivariate-bootstrap-v2", "asof": m.asof,
        "sample_start": m.source_start, "sample_end": m.source_end,
        "sample_rows": len(m.panel), "source_sha256_panel": digest,
        "last_observed_vix": m.last_vix, "starting_high_vol_regime": bool(m.initial_regime),
        "regime_transition_matrix": m.regime_transition.tolist(),
        "empirical_daily_regime_durations": {"low": m.daily_regime_durations[0].tolist(),
                                               "high": m.daily_regime_durations[1].tolist()},
        "missing_rows_excluded": m.rows_dropped,
        "instrument_order": list(m.names), "frozen_rules": FROZEN_RULE,
        "scenarios": [asdict(s) for s in scenarios],
        "important_limits": [
            "Resampling observed joint ETF/index returns cannot create genuinely new structural regimes.",
            "The futures 35/0 leg uses QQQ exposure proxy, not verified CME dated-contract settlement P&L.",
            "The strategy family and evaluation history are already post-selection contaminated.",
            "The sector ETF history is inception-constrained; no pre-inception 2x returns are synthesized.",
            "Open-to-next-open P&L and prior-close signal synchronization are approximations, not live fills.",
            "Rates, financing spreads, margin constraints and taxes are approximations.",
            "Model frequencies are conditional stress outcomes, never calibrated probabilities of future performance.",
        ],
    }
