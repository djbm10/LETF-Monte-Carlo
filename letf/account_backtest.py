"""Account-aware daily replay with prior-close signals and next-open fills.

Audited strategy subset: benchmarks, SMA/bands, fixed-weight portfolios and
unborrowed volatility targets. More elaborate legacy strategies remain gated
until their allocation and execution accounting reconcile. Inputs must be
as-traded OHLC, explicit corporate actions and classified cash distributions.
Vendor adjusted-close levels cannot be used as tax-lot execution prices.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite

import numpy as np
import pandas as pd

from letf import config as cfg
from letf.accounts import PortfolioLedger, calendar_date
from letf.accuracy import Evidence, UnsupportedModelError
from letf.reconstruction import daily_series


@dataclass(frozen=True)
class Distribution:
    security: str
    ex_date: date
    pay_date: date
    amount_per_share: float
    category: str
    source: str
    qualified_verified: bool = False

    def __post_init__(self):
        if self.pay_date < self.ex_date or not self.source or not isfinite(self.amount_per_share) or self.amount_per_share < 0:
            raise ValueError('Invalid sourced distribution schedule')
        if self.category not in ('interest', 'ordinary_dividend', 'qualified_dividend', 'capital_gain_distribution'):
            raise UnsupportedModelError('Return of capital/special distributions require a separate basis action')


def strategy_allocator(strategy_id):
    """Return a stateless allocation rule operating ONLY on supplied history."""
    config = cfg.STRATEGIES[strategy_id]
    kind = config['type']
    if kind not in ('benchmark', 'sma', 'sma_band', 'portfolio', 'vol_targeting', 'adaptive_vol', 'downside_vol'):
        raise UnsupportedModelError(f'{strategy_id}: legacy allocation/execution has not passed the account gate')

    def allocate(history, current_weights):
        if kind == 'portfolio':
            return dict(config['assets'])
        asset = config['asset']
        if kind == 'benchmark':
            return {asset: 1.}
        if kind in ('sma', 'sma_band'):
            period = config['sma_period']
            if len(history) < period:
                return {}
            prices = history['SPY_Close']
            mean = prices.iloc[-period:].mean()
            if kind == 'sma':
                weight = float(prices.iloc[-1] >= mean)
            else:
                band = config['band']
                # True two-sided hysteresis; no same threshold for entry/exit.
                held = current_weights.get(asset, 0.) > .5
                weight = float(prices.iloc[-1] >= mean * (1 - band if held else 1 + band))
            return {asset: weight}
        lookback = config['lookback']
        if len(history) < lookback + 1:
            return {}
        returns = history[f'{asset}_Close'].pct_change(fill_method=None).iloc[-lookback:]
        if kind == 'downside_vol':
            # Semideviation uses all observations, including zero downside on
            # positive days, rather than std of a truncated negative subsample.
            vol = np.sqrt(np.mean(np.minimum(returns, 0.) ** 2) * 252)
            target = config['target_downside_vol']
        else:
            vol = returns.std(ddof=1) * np.sqrt(252)
            target = config.get('target_vol', 0.)
            if kind == 'adaptive_vol':
                if len(history) < config['sma_period']:
                    return {}
                prices = history['SPY_Close']
                bull = prices.iloc[-1] >= prices.iloc[-config['sma_period']:].mean()
                target = config['bull_target'] if bull else config['bear_target']
        return {asset: 1. if vol == 0 else min(1., max(0., target / vol))}
    return allocate


def run_account_backtest(market, securities, profile, *, initial_capital,
                         allocator, price_convention, distributions_complete,
                         distributions=(), rebalance_every=1, account_id='main',
                         tax_inputs=None, tax_payment_dates=None, terminal_liquidation=False,
                         sweep_annual_rates=None, split_events=(), freeze_tax_year=None,
                         inflation_index=None):
    """Replay one cash account, exposing ledger and unpaid terminal tax liability.

    Tax is paid on caller-supplied dates after all ±30-day observations. If cash
    is insufficient, positions are sold at that day's open. Those sales create
    new current-year realizations, included at the next tax close. Terminal
    liability is reported as a payable (not fictitiously reinvested tax cash).
    Tax payment schedules/withholding safe harbors are never inferred.
    """
    if price_convention != 'as_traded':
        raise UnsupportedModelError('Tax/execution replay needs as-traded prices, not adjusted total-return levels')
    if not distributions_complete:
        raise UnsupportedModelError('Complete classified distributions and payable dates are required')
    if not isinstance(rebalance_every, int) or rebalance_every <= 0:
        raise ValueError('Positive integer rebalance schedule required')
    if initial_capital <= 0 or not isfinite(initial_capital):
        raise ValueError('Positive initial capital required')
    symbols = [s.symbol for s in securities]
    for symbol in symbols:
        for suffix in ('Open', 'Close'):
            daily_series(market[f'{symbol}_{suffix}'], f'{symbol}_{suffix}')
            if (market[f'{symbol}_{suffix}'] <= 0).any():
                raise ValueError('Nonpositive security prices require explicit liquidation actions')
    if profile.account_type == 'taxable' and tax_inputs is None:
        raise ValueError('Explicit taxable household inputs are required')
    if profile.account_type == 'taxable':
        for year in range(market.index[0].year, market.index[-1].year):
            if year not in (tax_payment_dates or {}):
                raise UnsupportedModelError(f'Missing annual payment schedule for {year}; taxes cannot be deferred to terminal wealth')
    if sweep_annual_rates is None:
        sweep_annual_rates = pd.Series(0., index=market.index)
    if not sweep_annual_rates.index.equals(market.index) or not np.isfinite(sweep_annual_rates).all():
        raise ValueError('Cash sweep rates must be finite and exactly aligned')
    if terminal_liquidation and not len(market):
        raise ValueError('Empty market')
    book = PortfolioLedger()
    book.add_account(account_id, profile, initial_capital)
    for security in securities:
        book.add_security(security)
    tax_rows = []
    st_cf = lt_cf = 0.
    receivables = []
    paid_years = set()
    equity = []
    last = market.index[0].date()
    withheld = 0.
    distributions = sorted(distributions, key=lambda d: (d.ex_date, d.pay_date))

    def tax_for(year, as_of):
        income = tax_inputs.get('ordinary_income_by_year', {}).get(year, tax_inputs.get('ordinary_income'))
        if income is None:
            raise ValueError(f'Missing ordinary income for {year}')
        return book.annual_tax(account_id, year, income, federal_only=tax_inputs.get('federal_only', False),
                               state_code=tax_inputs.get('state'), filing_status=tax_inputs.get('filing_status', 'single'),
                               st_cf=st_cf, lt_cf=lt_cf, as_of=as_of, freeze_tax_year=freeze_tax_year)

    for i, (timestamp, row) in enumerate(market.iterrows()):
        when = timestamp.date()
        book._advance(when)
        opens = {s: row[f'{s}_Open'] for s in symbols}
        closes = {s: row[f'{s}_Close'] for s in symbols}
        if i:
            days = (when - last).days
            interest = book.accounts[account_id].cash * sweep_annual_rates.iloc[i - 1] * days / 365
            if interest < 0:
                raise UnsupportedModelError('Negative sweep yield requires an explicit fee event')
            if interest:
                book.income(when, account_id, interest, 'interest')
        for action in split_events:
            if calendar_date(action['date']) == when:
                book.split(when, action['security'], action['ratio'])
        for distribution in distributions:
            if distribution.ex_date == when:
                receivables.append((distribution, book.quantity(account_id, distribution.security) * distribution.amount_per_share))
        for distribution, amount in list(receivables):
            if distribution.pay_date <= when:
                book.income(when, account_id, amount, distribution.category,
                            qualified_verified=distribution.qualified_verified)
                receivables.remove((distribution, amount))
        if profile.account_type == 'taxable':
            for year, payment_date in sorted((tax_payment_dates or {}).items()):
                if calendar_date(payment_date) < when and year not in paid_years:
                    raise ValueError('Tax payment date is not a market session; supply an execution/payment schedule')
                if calendar_date(payment_date) != when or year in paid_years:
                    continue
                if year != market.index[0].year + len(paid_years):
                    raise ValueError('Tax years must be closed in order, with carryovers')
                if year >= when.year:
                    raise UnsupportedModelError('Estimated-tax/withholding computations require a dedicated supplied schedule')
                result = tax_for(year, when)
                liability = result['tax']
                if liability > book.accounts[account_id].buying_power:
                    needed = liability - book.accounts[account_id].buying_power
                    slip = (profile.spread_bps / 2 + profile.slippage_bps) / 10000
                    for symbol in symbols:
                        available = book.quantity(account_id, symbol)
                        unit_net = opens[symbol] * (1 - slip)
                        shares = min(available, (needed + profile.commission) / unit_net)
                        if not profile.fractional_shares:
                            shares = min(available, np.ceil(shares))
                        if shares <= 1e-10:
                            continue
                        book.execute(when, account_id, symbol, 'SELL', float(shares), opens[symbol])
                        needed = liability - book.accounts[account_id].buying_power
                        if needed <= 1e-8:
                            break
                book.pay_tax(when, account_id, liability)
                withheld += liability
                st_cf, lt_cf = result['st_cf'], result['lt_cf']
                tax_rows.append(dict(result, year=year, paid_on=when))
                paid_years.add(year)
        if i % rebalance_every == 0:
            wealth = book.wealth(account_id, opens)
            weights = {s: book.quantity(account_id, s) * opens[s] / wealth for s in symbols} if wealth > 0 else {}
            # Only previous observations enter the decision. Today's open is
            # used for sizing/execution, never for computing trend/vol signals.
            target = allocator(market.iloc[:i].copy(), weights)
            if set(target) - set(symbols) or any(not isfinite(w) or w < 0 for w in target.values()) or sum(target.values()) > 1 + 1e-12:
                raise UnsupportedModelError('Target violates cash-account permissions/no-borrowing constraint')
            desired = {s: wealth * target.get(s, 0.) / opens[s] for s in symbols}
            if not profile.fractional_shares:
                desired = {s: np.floor(n) for s, n in desired.items()}
            for symbol in symbols:
                sell = book.quantity(account_id, symbol) - desired[symbol]
                if sell > 1e-9:
                    book.execute(when, account_id, symbol, 'SELL', float(sell), opens[symbol])
            for symbol in symbols:
                buy = desired[symbol] - book.quantity(account_id, symbol)
                slip = (profile.spread_bps / 2 + profile.slippage_bps) / 10000
                affordable = max(0., (book.accounts[account_id].buying_power - profile.commission) / (opens[symbol] * (1 + slip)))
                buy = min(buy, affordable)
                if not profile.fractional_shares:
                    buy = np.floor(buy)
                if buy > 1e-9:
                    book.execute(when, account_id, symbol, 'BUY', float(buy), opens[symbol])
        if terminal_liquidation and i == len(market) - 1:
            for symbol in symbols:
                shares = book.quantity(account_id, symbol)
                if shares > 1e-9:
                    book.execute(when, account_id, symbol, 'SELL', shares, closes[symbol])
        equity.append(book.wealth(account_id, closes) + sum(amount for _, amount in receivables))
        last = when
    terminal_liability = 0.
    if profile.account_type == 'taxable':
        for year in range(market.index[0].year, market.index[-1].year + 1):
            if year in paid_years:
                continue
            # A simulation that stops trading at liquidation can observe the
            # next 30 days without further purchases. Cross-account activity
            # cannot be excluded without an explicit owner-wide assumption.
            if not tax_inputs.get('no_external_wash_replacements_after_end', False):
                raise UnsupportedModelError('Terminal tax requires a completed owner-wide wash observation window')
            result = tax_for(year, date(year + 1, 1, 31) if year < last.year else last + timedelta(days=31))
            st_cf, lt_cf = result['st_cf'], result['lt_cf']
            terminal_liability += result['tax']
            tax_rows.append(dict(result, year=year, paid_on=None))
    curve = pd.Series(equity, index=market.index, name='wealth')
    from letf.performance import portfolio_metrics
    metrics = portfolio_metrics(curve, initial_capital, start_date=market.index[0].date(), inflation_index=inflation_index)
    net_curve = curve.copy()
    net_curve.iloc[-1] -= terminal_liability
    after_tax_metrics = portfolio_metrics(net_curve, initial_capital, start_date=market.index[0].date(), inflation_index=inflation_index) if terminal_liquidation else None
    return {'ledger': book, 'equity': curve, 'metrics': metrics, 'tax_rows': tax_rows,
            'tax_paid': withheld, 'terminal_tax_payable': terminal_liability,
            'ending_wealth_net_payable': curve.iloc[-1] - terminal_liability,
            'after_tax_metrics': after_tax_metrics,
            'trading_costs': sum(e.get('commission', 0.) + e.get('spread_slippage_cost', 0.) for e in book.events),
            'fund_fees': None, 'financing_expenses': None,
            'terminal_liquidation': terminal_liquidation,
            'evidence': Evidence.HYPOTHETICAL.value, 'execution_exact': False,
            'limitations': ['Daily open/close execution convention and configured cost scenario',
                            'Settlement lag must be supplied for execution beyond same-day cash convention',
                            'No estimated-tax penalty, AMT, credits or unsupported state tax model']}
