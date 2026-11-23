"""Return/drawdown and cash-flow metrics with explicit conventions."""
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from letf.accuracy import UnsupportedModelError


def xirr(cash_flows):
    """Dated investor-perspective flows, ACT/365.2425; reject ambiguous roots."""
    rows = sorted((pd.Timestamp(d), float(v)) for d, v in cash_flows)
    if not rows or not any(v < 0 for _, v in rows) or not any(v > 0 for _, v in rows):
        raise ValueError('Money-weighted return requires both contributions and receipts')
    t = np.array([(d - rows[0][0]).days / 365.2425 for d, _ in rows])
    amounts = np.array([v for _, v in rows])
    if not np.isfinite(amounts).all() or t[-1] == 0:
        raise ValueError('Invalid cash flows/duration')
    # Search log(1+r) to cover losses and large positive returns stably.
    def objective(log_rate):
        return float(np.dot(amounts, np.exp(-log_rate * t)))
    grid = np.linspace(-15, 15, 1001)
    roots = []
    for a, b in zip(grid[:-1], grid[1:]):
        fa, fb = objective(a), objective(b)
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(brentq(objective, a, b))
    roots = sorted(set(round(r, 10) for r in roots))
    if len(roots) != 1:
        raise UnsupportedModelError('Money-weighted return has no unique root in the documented search domain')
    return float(np.expm1(roots[0]))


def portfolio_metrics(equity, initial_capital, *, start_date, inflation_index=None, external_flows=()):
    values = equity.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or initial_capital <= 0:
        raise ValueError('Invalid wealth curve')
    years = (equity.index[-1] - pd.Timestamp(start_date)).days / 365.2425
    wealth = np.r_[initial_capital, values]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    max_underwater = run = 0
    for value in drawdown:
        run = run + 1 if value < -1e-12 else 0
        max_underwater = max(max_underwater, run)
    result = {'ending_wealth': float(values[-1]), 'max_drawdown': float(-drawdown.min()),
              'max_underwater_sessions': max_underwater, 'cagr': None, 'real_cagr': None,
              'money_weighted_return': None}
    if external_flows:
        # Do not label a contribution-boosted wealth ratio as CAGR.
        flows = [(start_date, -initial_capital), *external_flows, (equity.index[-1], values[-1])]
        result['money_weighted_return'] = xirr(flows)
        result['max_drawdown'] = None
        result['max_underwater_sessions'] = None
    elif years > 0:
        result['cagr'] = (values[-1] / initial_capital) ** (1 / years) - 1
        if inflation_index is not None:
            if not inflation_index.index.equals(equity.index) or not np.isfinite(inflation_index).all() or (inflation_index <= 0).any():
                raise ValueError('Inflation index must be positive, finite and exactly aligned')
            ratio = inflation_index.iloc[-1] / inflation_index.iloc[0]
            result['real_cagr'] = (values[-1] / initial_capital / ratio) ** (1 / years) - 1
    return result
