"""Dated-contract NQ/MNQ settlement replay; explicitly NOT continuous-price P&L.

Required columns: date,contract,expiry,settlement. All contracts are distinct,
settlements unadjusted. Signal file: date,QQQ (split/dividend-adjusted closes).
This daily settlement approximation is not an intraday execution backtest.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FuturesCosts:
    multiplier: float = 2.0                 # MNQ=2 USD per Nasdaq point; NQ=20
    tick_size: float = 0.25
    commission_per_side: float = 2.50
    slippage_ticks_per_side: float = 1.0
    initial_margin_fraction: float = 0.25   # modeled, not actual CME/broker schedule
    maintenance_margin_fraction: float = 0.20
    roll_business_days: int = 5
    initial_cash: float = 100_000.0
    sma_days: int = 200
    volatility_days: int = 20
    annual_target_vol: float = 0.35
    exposure_cap: float = 3.0

    def __post_init__(self):
        if not (self.multiplier > 0 and self.tick_size > 0 and self.initial_cash > 0):
            raise ValueError("Positive multiplier, tick size and capital required")
        if not (0 < self.maintenance_margin_fraction <= self.initial_margin_fraction <= 1):
            raise ValueError("0 < maintenance <= initial margin <= 1")
        if self.sma_days < 2 or self.volatility_days < 2:
            raise ValueError("Insufficient indicator lookback")
        if min(self.commission_per_side, self.slippage_ticks_per_side) < 0:
            raise ValueError("Nonnegative trading costs required")


def validate_contracts(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "contract", "expiry", "settlement"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing {required - set(raw.columns)} in dated contract data")
    df = raw.copy()
    df["date"] = pd.to_datetime(df.date).dt.normalize()
    df["expiry"] = pd.to_datetime(df.expiry).dt.normalize()
    df["settlement"] = pd.to_numeric(df.settlement, errors="coerce")
    df = df.sort_values(["date", "expiry", "contract"]).reset_index(drop=True)
    if df.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate date/contract settlement")
    if df[["date", "contract", "expiry", "settlement"]].isnull().any().any():
        raise ValueError("Invalid dated futures row")
    if (df.settlement <= 0).any() or (df.expiry < df.date).any():
        raise ValueError("Invalid futures prices or expiry")
    if df.groupby("contract").expiry.nunique().max() != 1:
        raise ValueError("Contract expiry must be immutable")
    return df


def target_exposure(qqq: pd.Series, cfg: FuturesCosts) -> float:
    if len(qqq) < max(cfg.sma_days, cfg.volatility_days + 1):
        return 0.0
    last = qqq.iloc[-1]
    if not np.isfinite(last) or last <= qqq.iloc[-cfg.sma_days:].mean():
        return 0.0
    v = qqq.pct_change().iloc[-cfg.volatility_days:].std(ddof=1) * np.sqrt(252)
    if not np.isfinite(v) or v <= 1e-8:
        return 0.0
    return min(cfg.exposure_cap, cfg.annual_target_vol / v)


def _new_contract(day: pd.Timestamp, d: pd.DataFrame, roll_days: int) -> str | None:
    choices = d.loc[d.date.eq(day)].copy()
    choices["business_days_left"] = choices.expiry.map(
        lambda exp: int(np.busday_count(day.date(), exp.date())))
    choices = choices.loc[choices.business_days_left > roll_days]
    if choices.empty:
        return None
    return str(choices.sort_values(["expiry", "contract"]).iloc[0].contract)


def replay_contract_settlements(
    raw: pd.DataFrame,
    qqq_closes: pd.Series,
    costs: FuturesCosts = FuturesCosts(),
    annual_cash_yield: float = 0.0,
) -> pd.DataFrame:
    """Prior close QQQ determines the NEXT futures settlement period exposure.

    Positions are rebalanced at the current daily settlement; old contract P&L
    is marked using its previous and current dated settlements, never a roll
    gap between different contract prices. Liquidations/fees are modeled.
    """
    df = validate_contracts(raw)
    q = qqq_closes.copy().astype(float)
    q.index = pd.to_datetime(q.index).normalize()
    if q.index.has_duplicates or not q.index.is_monotonic_increasing:
        raise ValueError("QQQ dates must be unique, ascending")
    if not np.isfinite(q.to_numpy()).all() or (q <= 0).any():
        raise ValueError("QQQ close values must be positive/finite")
    if not (-.1 <= annual_cash_yield <= .3):
        raise ValueError("Out-of-range cash yield")
    settlement = df.set_index(["date", "contract"]).settlement
    days = df.date.unique()
    if not set(days).issubset(set(q.index)):
        raise ValueError("Missing QQQ closes for settlement dates")
    equity = costs.initial_cash
    contracts = 0
    held = None
    prev_settle = None
    forced_count = 0
    side_cost = costs.commission_per_side + costs.slippage_ticks_per_side * costs.tick_size * costs.multiplier
    records = []
    previous_day = None
    for day in days:
        day = pd.Timestamp(day)
        if previous_day is not None and (day - previous_day).days > 7:
            raise ValueError("Futures settlement calendar gap > 7 days; missing sessions")
        old_contract, old_qty = held, contracts
        pnl = 0.0
        if held is not None and contracts:
            try:
                mark = float(settlement.loc[(day, held)])
            except KeyError as e:
                raise ValueError(f"Held contract {held} missing settlement on {day.date()}") from e
            pnl = contracts * costs.multiplier * (mark - prev_settle)
            prev_settle = mark
        equity = equity * (1 + annual_cash_yield / 252) + pnl
        if equity <= 0:
            records.append(dict(date=day, equity=equity, bankrupt=True, forced_liquidation=True,
                                contract=old_contract, contracts=old_qty, pnl=pnl, costs=0.0))
            break
        maintenance = abs(contracts) * (prev_settle or 0) * costs.multiplier * costs.maintenance_margin_fraction
        liquidated = bool(contracts and equity < maintenance)
        if liquidated:
            forced_count += 1
            equity -= side_cost * abs(contracts)
            contracts = 0
            held = None
            if equity <= 0:
                records.append(dict(date=day, equity=equity, bankrupt=True, forced_liquidation=True,
                                    contract=old_contract, contracts=old_qty, pnl=pnl, costs=side_cost*abs(old_qty)))
                break
        eligible = _new_contract(day, df, costs.roll_business_days)
        exposure = target_exposure(q.loc[:day], costs)
        new_price = float(settlement.loc[(day, eligible)]) if eligible is not None else None
        desired = 0
        if eligible is not None and exposure > 0 and equity > 0:
            desired = int(np.floor(exposure * equity / (new_price * costs.multiplier)))
            capacity = int(np.floor(equity / (new_price * costs.multiplier * costs.initial_margin_fraction)))
            desired = max(0, min(desired, capacity))
        # Trading costs consume margin-eligible cash. Reduce the target if
        # post-fee equity cannot cover the required initial margin.
        while desired > 0:
            prospective_sides = (abs(desired - contracts) if eligible == held
                                 else abs(desired) + abs(contracts))
            prospective_fee = side_cost * prospective_sides
            required = desired * new_price * costs.multiplier * costs.initial_margin_fraction
            if required <= equity - prospective_fee + 1e-9:
                break
            desired -= 1
        sides = abs(desired - contracts) if eligible == held else abs(desired) + abs(contracts)
        fees = side_cost * sides
        equity -= fees
        if equity <= 0:
            records.append(dict(date=day, equity=equity, bankrupt=True, forced_liquidation=liquidated,
                                contract=old_contract, contracts=old_qty, pnl=pnl, costs=fees))
            break
        held = eligible if desired else None
        contracts = desired
        prev_settle = new_price if contracts else None
        records.append(dict(date=day, equity=equity, bankrupt=False, forced_liquidation=liquidated,
                            contract=held, contracts=contracts, desired_exposure=exposure,
                            actual_exposure=contracts * (new_price or 0) * costs.multiplier / equity,
                            pnl=pnl, costs=fees, trading_sides=sides,
                            rolled=bool(old_contract and old_contract != held and old_qty),
                            model_margin=contracts * (new_price or 0) * costs.multiplier * costs.initial_margin_fraction))
        previous_day = day
    out = pd.DataFrame(records).set_index("date")
    out.attrs["limitations"] = "Settlement-close fills estimated, not official execution, no intraday calls; model margin not actual CME schedule."
    out.attrs["forced_margin_liquidations"] = forced_count
    return out
