"""Auditable *synthetic* futures shocks for the dated-contract settlement replay.

Scenarios are counterfactual, never represented as actual market observations.
Unchanged fixed strategy rules rerun on modified contract AND QQQ signals.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Iterable
import pandas as pd
from .futures_reconcile import FuturesCosts, replay_contract_settlements


def shock_joint_series(raw: pd.DataFrame, qqq: pd.Series, shocks: Iterable[tuple[str,float]]):
    contracts = raw.copy()
    prices = qqq.copy()
    dates = pd.to_datetime(contracts.date)
    for date, shock in shocks:
        dt = pd.Timestamp(date)
        if not (-0.95 < shock < 2.0):
            raise ValueError('Shock percent must be greater than -95%, below +200%')
        if dt not in prices.index or dt not in set(dates):
            raise ValueError('Shock date absent from both prices and contract settlements')
        factor = 1 + shock
        contracts.loc[dates >= dt, 'settlement'] *= factor
        prices.loc[dt:] *= factor
    return contracts, prices


def compare_shocks(raw: pd.DataFrame, qqq: pd.Series, costs: FuturesCosts, experiments: dict) -> pd.DataFrame:
    """Each scenario contains joint price shocks and margin/fee sensitivity.

    Use official dated contract data if available. With illustrative/fake inputs,
    the resulting numbers are ONLY unit tests, not investment evidence.
    """
    rows = []
    def collect(name, contracts, prices, cfg):
        out = replay_contract_settlements(contracts, prices, cfg)
        curve = out.equity / cfg.initial_cash
        rows.append({'scenario': name, 'terminal_multiple': curve.iloc[-1],
                     'max_drawdown': float((curve/curve.cummax()-1).min()),
                     'margin_liquidation_days': int(out.forced_liquidation.sum()),
                     'bankrupt': bool(out.bankrupt.any()), 'sessions':len(out),
                     'synthetic_scenario':name != 'unchanged_input'})
    collect('unchanged_input',raw,qqq,costs)
    for name, case in experiments.items():
        contracted, adjusted = shock_joint_series(raw,qqq,case.get('shocks',[]))
        cfg = replace(costs,
                      commission_per_side=case.get('commission_per_side',costs.commission_per_side),
                      slippage_ticks_per_side=case.get('slippage_ticks_per_side',costs.slippage_ticks_per_side),
                      initial_margin_fraction=case.get('initial_margin_fraction',costs.initial_margin_fraction),
                      maintenance_margin_fraction=case.get('maintenance_margin_fraction',costs.maintenance_margin_fraction))
        collect(name,contracted,adjusted,cfg)
    return pd.DataFrame(rows)
