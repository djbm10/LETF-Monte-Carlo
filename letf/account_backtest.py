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
from functools import lru_cache

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
                         inflation_index=None, cash_flows=(), roth_history_config=None,
                         settlement_policy='immediate', trade_start=None, apply_taxes=True,
                         lot_method='FIFO', source_manifest=None,
                         linked_accounts=(), linked_trades=()):
    """One daily account engine for market replay and generated scenarios.

    Signals see previous closes. Fills use the current open with configured
    costs. US-equity cash proceeds settle T+2 before 2024-05-28 and T+1 after;
    explicit tax/withdrawal funding sales precede the relevant settlement.
    Daily cash, quantity and NAV residuals are independently reconciled.
    """
    from letf.tax.roth import RothHistory
    from letf.performance import portfolio_metrics
    if price_convention != 'as_traded':
        raise UnsupportedModelError('Tax/execution replay needs as-traded prices, not adjusted total-return levels')
    if not distributions_complete:
        raise UnsupportedModelError('Complete classified distributions and payable dates are required')
    if type(rebalance_every) is not int or rebalance_every < 1 or not isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('Positive capital and integer rebalance schedule required')
    if settlement_policy not in ('immediate', 'us_equity'):
        raise UnsupportedModelError('Unknown settlement convention')
    if not len(market):
        raise ValueError('Empty market')
    symbols = [s.symbol for s in securities]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError('Unique instruments required')
    for symbol in symbols:
        for suffix in ('Open', 'Close'):
            daily_series(market[f'{symbol}_{suffix}'], f'{symbol}_{suffix}')
            if (market[f'{symbol}_{suffix}'] <= 0).any():
                raise ValueError('Nonpositive prices require explicit liquidation actions')
    first = pd.Timestamp(trade_start) if trade_start else market.index[0]
    trading = market.loc[first:]
    if not len(trading):
        raise ValueError('No trading sessions after trade_start')
    first, last = trading.index[0].date(), trading.index[-1].date()
    taxable = profile.account_type == 'taxable' and apply_taxes
    if taxable and tax_inputs is None:
        raise ValueError('Explicit taxable household inputs are required')
    payments = {int(y): calendar_date(d) for y, d in (tax_payment_dates or {}).items()} if taxable else {}
    if taxable:
        for year in range(first.year, last.year):
            if year not in payments:
                raise UnsupportedModelError(f'Missing annual payment schedule for {year}; taxes cannot be deferred to terminal wealth')
    if sweep_annual_rates is None:
        sweep_annual_rates = pd.Series(0., index=market.index)
    if not sweep_annual_rates.index.equals(market.index) or not np.isfinite(sweep_annual_rates).all():
        raise ValueError('Cash sweep rates must be finite and exactly aligned')
    calendar = None
    if settlement_policy == 'us_equity':
        import exchange_calendars as xcals
        calendar = xcals.get_calendar('XNYS', start=str(first - timedelta(days=15)), end=str(last + timedelta(days=30)))
        if any(not calendar.is_session(pd.Timestamp(d)) for d in trading.index):
            raise ValueError('Market rows include a non-XNYS session')

    @lru_cache(maxsize=None)
    def settlement(when):
        if calendar is None:
            return when
        lag = 1 if when >= date(2024, 5, 28) else 2
        return calendar.session_offset(pd.Timestamp(when), lag).date()

    @lru_cache(maxsize=None)
    def funding_day(due):
        if calendar is None:
            return due
        if not calendar.is_session(pd.Timestamp(due)):
            raise ValueError('Payment/withdrawal date needs an explicit valid market-session schedule')
        for lag in (1, 2):
            candidate = calendar.session_offset(pd.Timestamp(due), -lag).date()
            if settlement(candidate) == due:
                return candidate
        raise ValueError('Cannot establish settlement-safe funding date')

    flows = []
    for i, flow in enumerate(cash_flows):
        f = dict(flow);f['date'] = calendar_date(f['date']);f['id'] = i
        if f['date'] not in set(trading.index.date):
            raise ValueError('Cash flows must have an explicit market-session date')
        if f['kind'] not in ('contribution', 'deposit', 'withdrawal', 'conversion', 'non_roth_plan_rollover'):
            raise UnsupportedModelError('Unsupported external account flow')
        if not isfinite(f['amount']) or f['amount'] <= 0:
            raise ValueError('External flow amount must be positive')
        flows.append(f)
    book = PortfolioLedger();book.add_account(account_id, profile, initial_capital)
    for linked in linked_accounts:
        book.add_account(linked['id'], linked['profile'], linked['cash'], taxpayer=linked.get('taxpayer','owner'))
    for trade in linked_trades:
        if trade['account'] == account_id or trade['account'] not in book.accounts or not trade.get('source'):
            raise ValueError('Linked trades need a different defined account and observed execution source')
        if calendar_date(trade['date']) not in set(trading.index.date):
            raise ValueError('Linked executions must be inside the replay observation window')
    for security in securities:
        book.add_security(security)
    history = None
    if profile.account_type == 'roth' and flows:
        if roth_history_config is None:
            raise ValueError('Roth flows require aggregate owner birth/basis/clock history')
        history = RothHistory(book, 'owner', **roth_history_config)
    tax_rows, prepared, paid_years = [], {}, set()
    st_cf = lt_cf = state_st_cf = state_lt_cf = 0.
    receivables, equity, unit_values, snapshots, external, decisions = [], [], [], [], [], []
    units = initial_capital
    withheld = 0.
    expected_cash = initial_capital
    expected_quantities = dict.fromkeys(symbols, 0.)
    event_cursor = 0
    previous_date = first
    schedules = sorted(distributions, key=lambda d: (d.ex_date, d.pay_date))

    def tax_for(year, as_of):
        income = tax_inputs.get('ordinary_income_by_year', {}).get(year, tax_inputs.get('ordinary_income'))
        if income is None:
            raise ValueError(f'Missing ordinary income for {year}')
        return book.annual_tax(account_id, year, income, federal_only=tax_inputs.get('federal_only', False),
            state_code=tax_inputs.get('state'), filing_status=tax_inputs.get('filing_status', 'single'),
            st_cf=st_cf, lt_cf=lt_cf, state_st_cf=state_st_cf, state_lt_cf=state_lt_cf,
            as_of=as_of, freeze_tax_year=freeze_tax_year)

    def sell_for_cash(when, required, opens, reason):
        # Reserve exists before sizing; proceeds cannot be reinvested before
        # the actual payment/withdrawal. Include cash already awaiting settlement.
        account = book.accounts[account_id]
        pending = sum(value for due, value in account.unsettled if due <= settlement(when))
        available = account.buying_power + pending
        needed = max(0., required - available)
        for symbol in symbols:
            if needed <= 1e-8:
                break
            owned = book.quantity(account_id, symbol)
            unit_net = opens[symbol] * (1 - (profile.spread_bps / 2 + profile.slippage_bps) / 10000)
            shares = min(owned, (needed + profile.commission) / unit_net)
            if not profile.fractional_shares:
                shares = min(owned, float(np.ceil(shares)))
            if shares <= 1e-10:
                continue
            event = book.execute(when, account_id, symbol, 'SELL', shares, opens[symbol],
                                 method=lot_method, settlement_date=settlement(when))
            event['reason'] = reason
            needed -= shares * event['price'] - profile.commission
        if needed > 1e-6:
            raise ValueError('Insufficient account assets to fund scheduled payment')

    for step, (timestamp, row) in enumerate(trading.iterrows()):
        when = timestamp.date()
        # Reconcile maturation of the previous day's receivables separately.
        prior_pending = list(book.accounts[account_id].unsettled)
        prior_cash = book.accounts[account_id].cash
        book._advance(when)
        expected_cash += sum(value for due, value in prior_pending if due <= when)
        opens = {s: float(row[f'{s}_Open']) for s in symbols}
        closes = {s: float(row[f'{s}_Close']) for s in symbols}
        if step:
            prior_rate = sweep_annual_rates.loc[:timestamp].iloc[-2]
            interest = prior_cash * prior_rate * (when - previous_date).days / 365
            if interest < 0:
                raise UnsupportedModelError('Negative sweep yield requires an explicit fee event')
            if interest:
                book.income(when, account_id, interest, 'interest')
        for action in split_events:
            if calendar_date(action['date']) == when:
                book.split(when, action['security'], action['ratio'])
                expected_quantities[action['security']] *= action['ratio']
        for distribution in schedules:
            if distribution.ex_date == when:
                if taxable and distribution.category == 'qualified_dividend':
                    entitled=[lot for lot in book.lots if lot.account==account_id and
                              lot.security==distribution.security and lot.shares>1e-10]
                    if any((when-lot.acquired).days<=60 for lot in entitled):
                        raise UnsupportedModelError('Qualified dividend needs investor holding-period proof; future 121-day eligibility is not inferred')
                receivables.append((distribution, book.quantity(account_id, distribution.security) * distribution.amount_per_share))
        for distribution, amount in list(receivables):
            if distribution.pay_date <= when:
                book.income(when, account_id, amount, distribution.category,
                            qualified_verified=distribution.qualified_verified)
                receivables.remove((distribution, amount))
        for trade in linked_trades:
            if calendar_date(trade['date']) == when:
                event = book.execute(when, trade['account'], trade['security'], trade['side'],
                    trade['shares'], trade['price'], settlement_date=settlement(when))
                event['source'] = trade['source']
                event['reason'] = 'LINKED_ACCOUNT_EXECUTION'
        # Prepare tax-funded sales early enough for cash-account settlement.
        for year, due in sorted(payments.items()):
            if due > last:
                continue  # accrued/payable at termination, not a future cash event
            if due < when and year not in paid_years:
                raise ValueError('Tax payment date is not a market session or was missed')
            if funding_day(due) == when and year not in prepared:
                if year >= when.year or year != first.year + len(paid_years):
                    raise UnsupportedModelError('Supply ordered prior-year payment dates; estimated taxes require separate instructions')
                result = tax_for(year, when)
                prepared[year] = result
                if result['tax'] > 0:
                    sell_for_cash(when, result['tax'], opens, 'TAX_FUNDING')
                    book.accounts[account_id].collateral_reservations[f'tax:{year}'] = result['tax']
            if due == when:
                if year not in prepared:
                    raise ValueError('Missing settlement-safe tax funding session')
                result = prepared[year]
                book.accounts[account_id].collateral_reservations.pop(f'tax:{year}', None)
                book.pay_tax(when, account_id, result['tax'])
                withheld += result['tax'];paid_years.add(year)
                st_cf, lt_cf = result['st_cf'], result['lt_cf']
                state_st_cf, state_lt_cf = result.get('state_st_cf', 0.), result.get('state_lt_cf', 0.)
                tax_rows.append(dict(result, year=year, paid_on=when))
        for flow in flows:
            if flow['kind'] == 'withdrawal' and funding_day(flow['date']) == when:
                sell_for_cash(when, flow['amount'], opens, 'WITHDRAWAL_FUNDING')
                book.accounts[account_id].collateral_reservations[f"flow:{flow['id']}"] = flow['amount']
        net_flow = 0.
        for flow in flows:
            if flow['date'] != when:
                continue
            before = book.wealth(account_id, opens) + sum(v for _, v in receivables)
            if before <= 0 or units <= 0:
                raise UnsupportedModelError('Cannot unitize new flows into an insolvent or fully closed account')
            amount, kind = flow['amount'], flow['kind']
            if kind == 'withdrawal':
                book.accounts[account_id].collateral_reservations.pop(f"flow:{flow['id']}", None)
                if history:
                    result = history.withdraw(when, account_id, amount, exception=flow.get('exception'))
                    if result['personal_taxable_income'] or result['additional_tax']:
                        raise UnsupportedModelError('Taxable/penalized Roth distribution needs an external personal-tax funding schedule')
                else:
                    if book.accounts[account_id].buying_power < amount - 1e-8:
                        raise ValueError('Withdrawal lacks settled cash')
                    book.accounts[account_id].cash -= amount
                    book.events.append({'date': when, 'account': account_id, 'action': 'WITHDRAWAL', 'amount': amount})
                delta = -amount
            elif history:
                if kind in ('contribution', 'deposit'):
                    history.contribute(when, account_id, amount, tax_year=flow.get('tax_year', when.year),
                        magi=flow['magi'], compensation=flow['compensation'],
                        filing_status=flow.get('filing_status', 'single'),
                        other_ira_contributions=flow.get('other_ira_contributions', 0.))
                else:
                    history.convert(when, account_id, amount, flow['taxable_amount'], kind=kind)
                delta = amount
            else:
                if kind in ('conversion', 'non_roth_plan_rollover'):
                    raise UnsupportedModelError('Roth conversions/rollovers cannot enter a taxable account as internal flows')
                book.accounts[account_id].cash += amount
                book.events.append({'date': when, 'account': account_id, 'action': 'DEPOSIT', 'amount': amount})
                delta = amount
            units *= (before + delta) / before
            net_flow += delta
            external.append((when, -delta))  # investor perspective
        if step % rebalance_every == 0:
            wealth = book.wealth(account_id, opens)
            weights = {s: book.quantity(account_id, s) * opens[s] / wealth for s in symbols} if wealth > 0 else {}
            past = market.loc[market.index < timestamp]
            target = allocator(past.copy(), weights)
            if set(target) - set(symbols) or any(not isfinite(w) or w < 0 for w in target.values()) or sum(target.values()) > 1 + 1e-12:
                raise UnsupportedModelError('Target violates cash-account permissions/no-borrowing constraint')
            decisions.append({'date': when, 'weights': dict(target)})
            desired = {s: wealth * target.get(s, 0.) / opens[s] for s in symbols}
            if not profile.fractional_shares:
                desired = {s: float(np.floor(n)) for s, n in desired.items()}
            for symbol in symbols:
                sell = book.quantity(account_id, symbol) - desired[symbol]
                if sell > 1e-9:
                    event = book.execute(when, account_id, symbol, 'SELL', sell, opens[symbol],
                                         method=lot_method, settlement_date=settlement(when))
                    event['reason'] = 'STRATEGY'
            for symbol in symbols:
                buy = desired[symbol] - book.quantity(account_id, symbol)
                slip = (profile.spread_bps / 2 + profile.slippage_bps) / 10000
                affordable = max(0., (book.accounts[account_id].buying_power - profile.commission) / (opens[symbol] * (1 + slip)))
                buy = min(buy, affordable)
                if not profile.fractional_shares:
                    buy = float(np.floor(buy))
                if buy > 1e-9:
                    event = book.execute(when, account_id, symbol, 'BUY', buy, opens[symbol])
                    event['reason'] = 'STRATEGY'
        if terminal_liquidation and when == last:
            for symbol in symbols:
                shares = book.quantity(account_id, symbol)
                if shares > 1e-9:
                    event = book.execute(when, account_id, symbol, 'SELL', shares, closes[symbol],
                                         method=lot_method, settlement_date=settlement(when))
                    event['reason'] = 'TERMINAL_LIQUIDATION'
        # Independently replay signed cash movements, not changes in NAV.
        for event in book.events[event_cursor:]:
            if event.get('account') != account_id:
                continue
            action = event['action']
            if action == 'BUY':
                expected_cash -= event['shares'] * event['price'] + event['commission']
                expected_quantities[event['security']] += event['shares']
            elif action == 'SELL':
                expected_quantities[event['security']] -= event['shares']
                if event['settlement_date'] <= when:
                    expected_cash += event['shares'] * event['price'] - event['commission']
            elif action in ('INCOME', 'DEPOSIT', 'ROTH_CONTRIBUTION', 'CONVERSION', 'NON_ROTH_PLAN_ROLLOVER'):
                expected_cash += event['amount']
            elif action in ('WITHDRAWAL', 'ROTH_WITHDRAWAL', 'TAX_PAYMENT'):
                expected_cash -= event['amount']
        event_cursor = len(book.events)
        account = book.accounts[account_id]
        receivable = sum(value for _, value in receivables)
        unsettled = sum(value for _, value in account.unsettled)
        position_value = sum(book.quantity(account_id, symbol) * closes[symbol] for symbol in symbols)
        nav = book.wealth(account_id, closes) + receivable
        residual = account.cash - expected_cash
        nav_residual = nav - (account.cash + unsettled + position_value + receivable)
        quantity_residual = max(abs(book.quantity(account_id, s) - expected_quantities[s]) for s in symbols)
        if abs(residual) > 1e-6 or abs(nav_residual) > 1e-6 or quantity_residual > 1e-7 or account.cash < -1e-7:
            raise AssertionError(f'Canonical account reconciliation failed on {when}: cash={residual}, NAV={nav_residual}')
        equity.append(nav)
        unit_values.append(nav / units if units > 0 else np.nan)
        snapshots.append({'date': when, 'cash': account.cash, 'unsettled': unsettled,
                          'reserved': sum(account.collateral_reservations.values()), 'receivable_dividends': receivable,
                          'positions': {s: book.quantity(account_id, s) for s in symbols},
                          'position_value': position_value, 'nav': nav, 'external_flow': net_flow,
                          'cash_residual': residual, 'nav_residual': nav_residual,
                          'quantity_residual': quantity_residual})
        previous_date = when
    terminal_liability = 0.
    if terminal_liquidation and sum(value for _, value in receivables) > 1e-8:
        raise UnsupportedModelError('Terminal distributions still receivable: extend the replay through payable dates')
    if taxable:
        if not tax_inputs.get('no_external_wash_replacements_after_end', False):
            raise UnsupportedModelError('Terminal tax requires a completed owner-wide wash observation window')
        for year in range(first.year, last.year + 1):
            if year in paid_years:
                continue
            result = tax_for(year, date(year + 1, 1, 31) if year < last.year else last + timedelta(days=31))
            st_cf, lt_cf = result['st_cf'], result['lt_cf']
            state_st_cf, state_lt_cf = result.get('state_st_cf', 0.), result.get('state_lt_cf', 0.)
            terminal_liability += result['tax']
            tax_rows.append(dict(result, year=year, paid_on=None))
    curve = pd.Series(equity, index=trading.index, name='wealth')
    cpi = inflation_index.reindex(trading.index) if inflation_index is not None else None
    metrics = portfolio_metrics(curve, initial_capital, start_date=first, inflation_index=cpi, external_flows=external)
    unit_curve = pd.Series(unit_values, index=trading.index) * initial_capital
    twr_metrics = portfolio_metrics(unit_curve, initial_capital, start_date=first, inflation_index=cpi)
    metrics.update(time_weighted_cagr=twr_metrics['cagr'], max_drawdown=twr_metrics['max_drawdown'],
                   max_underwater_sessions=twr_metrics['max_underwater_sessions'])
    net_curve = curve.copy();net_curve.iloc[-1] -= terminal_liability
    after_tax = portfolio_metrics(net_curve, initial_capital, start_date=first, inflation_index=cpi,
                                  external_flows=external) if terminal_liquidation else None
    if after_tax is not None:
        net_units = unit_curve.copy();net_units.iloc[-1] *= net_curve.iloc[-1] / curve.iloc[-1]
        net_twr = portfolio_metrics(net_units, initial_capital, start_date=first, inflation_index=cpi)
        after_tax.update(time_weighted_cagr=net_twr['cagr'], real_time_weighted_cagr=net_twr['real_cagr'],
                         max_drawdown=net_twr['max_drawdown'], max_underwater_sessions=net_twr['max_underwater_sessions'])
    return {'ledger': book, 'equity': curve, 'metrics': metrics, 'tax_rows': tax_rows,
            'decisions': decisions, 'reconciliation': pd.DataFrame(snapshots).set_index('date'),
            'tax_paid': withheld, 'terminal_tax_payable': terminal_liability,
            'ending_wealth_net_payable': curve.iloc[-1] - terminal_liability,
            'terminal_liquidation': terminal_liquidation, 'after_tax_metrics': after_tax,
            'external_flows_investor': external, 'settlement_policy': settlement_policy,
            'trading_costs': sum(e.get('commission', 0.) + e.get('spread_slippage_cost', 0.) for e in book.events),
            'fund_fees': None, 'financing_expenses': None,
            'evidence': Evidence.HYPOTHETICAL.value, 'execution_exact': False,
            'source_manifest': source_manifest,
            'limitations': ['Daily open/close fills with assumed spread/slippage, not observed NBBO fills',
                'Tax payments use the supplied annual schedule; withholding/estimated-tax safe harbors are external inputs',
                'Federal regular tax/NIIT and supported state subset, not a full tax return']}
