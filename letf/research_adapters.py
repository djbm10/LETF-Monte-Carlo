"""Existing research selectors connected to canonical cash/derivative books.

Settlement executions remain scenarios. Explicit dated specifications are
mandatory; continuous charts and inferred expirations are not contract tapes.
"""
from math import floor

import numpy as np
import pandas as pd

from letf.accounts import PortfolioLedger
from letf.derivatives import FuturesBook, OptionContract
from letf.accuracy import Evidence, UnsupportedModelError
from research.free_validation.futures_reconcile import (
    validate_contracts, target_exposure, _new_contract, FuturesCosts)


def replay_futures_account(raw, qqq_closes, contracts, profile, *,
                           costs=FuturesCosts(), cash_rates=None,
                           ira_margin_multiplier=None, profile_source=None,
                           profile_verified_on=None, terminal_liquidation=True):
    """Frozen research 35/0 rule, integer dated contracts, canonical cash.

    Supply a FuturesContract for every contract identifier. Margins are the
    supplied dated house schedule, never the research notional fractions.
    Prior-session QQQ signals avoid assuming the equity close preceded the
    futures settlement. Trades execute at assumed settlement plus costs.
    """
    data = validate_contracts(raw)
    if set(data.contract) - set(contracts):
        raise UnsupportedModelError('Every dated contract requires sourced exchange/house specifications')
    for symbol, spec in contracts.items():
        rows = data.loc[data.contract.eq(symbol)]
        if len(rows) and (rows.expiry.dt.date != spec.expiry).any():
            raise ValueError('Contract expiry conflicts with explicit specification')
        if spec.multiplier != costs.multiplier or spec.tick_size != costs.tick_size:
            raise ValueError('Research sizing and exchange multiplier/tick disagree')
    q = qqq_closes.astype(float).copy()
    q.index = pd.to_datetime(q.index).normalize()
    if not q.index.is_unique or not q.index.is_monotonic_increasing or not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError('Unique increasing finite positive signal history required')
    days = pd.DatetimeIndex(data.date.unique())
    if not set(days).issubset(q.index):
        raise ValueError('Missing signal observation on a settlement date')
    if len(days)>1 and (days.to_series().diff().dropna().dt.days > 7).any():
        raise ValueError('Missing settlement sessions: gap exceeds seven calendar days')
    rates = pd.Series(0.,index=days) if cash_rates is None else cash_rates
    if not rates.index.equals(days) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Nonnegative annual cash rates aligned to settlement dates required')
    ledger=PortfolioLedger();ledger.add_account('main',profile,costs.initial_cash)
    book=FuturesBook(ledger,'main',commission_per_contract=costs.commission_per_side,
        spread_ticks=2*costs.slippage_ticks_per_side,ira_margin_multiplier=ira_margin_multiplier,
        profile_source=profile_source,profile_verified_on=profile_verified_on)
    prices=data.set_index(['date','contract']).settlement
    records=[];previous=None
    for i, day in enumerate(days):
        when=day.date();account=ledger.accounts['main']
        ledger._advance(when)
        if previous is not None:
            interest=account.cash*rates.iloc[i-1]*(day-previous).days/365
            if interest>0:ledger.income(when,'main',interest,'interest')
        marks={symbol:float(prices.loc[day,symbol]) for symbol in book.positions
               if (day,symbol) in prices.index}
        if set(book.positions)-set(marks):
            raise ValueError('Missing held dated contract settlement')
        for symbol in list(book.positions):
            book.settle(when,symbol,marks[symbol])
        liquidated=bool(book.positions and account.cash < book.reserved(maintenance=True))
        if liquidated:
            book.force_liquidate(when,marks)
        candidate=_new_contract(day,data,costs.roll_business_days)
        if terminal_liquidation and day==days[-1]:candidate=None
        # No immediate reopening after a margin call or account insolvency.
        exposure=0. if liquidated or account.cash<=0 else target_exposure(q.loc[q.index<day],costs)
        for symbol, position in list(book.positions.items()):
            if symbol!=candidate:
                book.trade(when,position.contract,-position.quantity,marks[symbol])
        desired=0
        if candidate and exposure>0:
            price=float(prices.loc[day,candidate]);spec=contracts[candidate]
            # Positive NQ-only research universe; negative commodity pricing
            # requires a different exposure-sizing rule (book itself supports it).
            desired=max(0,floor(exposure*account.cash/(price*spec.multiplier)))
            held=book.positions.get(candidate)
            old=held.quantity if held else 0
            cost_per_side=book.commission+book.spread_ticks/2*spec.tick_size*spec.multiplier
            desired=min(desired,max(0,floor(account.cash/(spec.initial_margin*book.multiplier))))
            while desired and desired*spec.initial_margin*book.multiplier > account.cash-abs(desired-old)*cost_per_side+1e-9:
                desired-=1
        old=book.positions[candidate].quantity if candidate in book.positions else 0
        if candidate and desired!=old:
            book.trade(when,contracts[candidate],desired-old,float(prices.loc[day,candidate]))
        pnl=sum(e.get('variation_margin',0.) for e in ledger.events if e.get('date')==when)
        records.append({'date':when,'cash':account.cash,'reserved':book.reserved(),
            'contract':candidate if desired else None,'contracts':desired,'variation_margin':pnl,
            'forced_liquidation':liquidated,'intraday_margin_status':'UNKNOWN',
            'evidence':Evidence.HYPOTHETICAL.value})
        previous=day
        if account.cash<=0:break
    return {'ledger':ledger,'futures_book':book,'daily':pd.DataFrame(records).set_index('date'),
        'evidence':Evidence.HYPOTHETICAL.value,'investment_performance_certification':Evidence.UNSUPPORTED.value,
        'limitations':['Settlement fills are assumed; actual quotes and intraday margin paths unavailable',
            'Taxable Section 1256 accrues in canonical annual ledger; personal tax-payment cash scheduling is not integrated here',
            'One supplied margin vintage per contract; use FuturesBook.change_margin for an event-driven house schedule',
            'Research roll selector counts weekdays, not exchange holidays; not certified as an exchange roll schedule']}


def optionmetrics_contract(row, *, underlying, settlement, multiplier, source):
    """Reuse a parsed real OptionMetrics row as a canonical contract definition.

    A date-only EOD snapshot is NOT converted into a timestamp/depth quote.
    A caller needs those independent observations to use LongOptionBook.trade.
    """
    if not source or row.get('cfadj',1)!=1 or str(row.get('ss_flag',0)) not in ('0','0.0'):
        raise UnsupportedModelError('Adjusted/nonstandard option requires an OCC deliverable memo')
    style={'E':'european','A':'american'}.get(row.get('exercise_style'))
    if style is None:
        raise UnsupportedModelError('Explicit contemporaneous exercise style required')
    if row.cp_flag not in ('C','P'):
        raise ValueError('Invalid option right')
    return OptionContract(str(row.optionid),underlying,float(row.strike),pd.Timestamp(row.exdate).date(),
        'call' if row.cp_flag=='C' else 'put',style,settlement,multiplier)
