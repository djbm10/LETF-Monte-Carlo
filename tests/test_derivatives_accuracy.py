from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from letf.accounts import PortfolioLedger, TAXABLE_CASH, CONSERVATIVE_ROTH, Security
from letf.derivatives import (FuturesBook, FuturesContract, OptionContract, OptionQuote,
                              LongOptionBook, option_collateral, require_derivative_tax_support)
from letf.accuracy import UnsupportedModelError


def future_book(cash=10000):
    book = PortfolioLedger()
    book.add_account('a', replace(TAXABLE_CASH, allowed_kinds=frozenset({'future', 'etf'})), cash)
    futures = FuturesBook(book, 'a', commission_per_contract=1, spread_ticks=1)
    # CME MES multiplier $5 and tick .25; margins are test scenarios, not
    # assertions about actual historical CME/broker requirements.
    contract = FuturesContract('MESZ2025', date(2025, 12, 19), 5., .25, 1500, 1200,
                               date(2024, 1, 1), 'CME contract specs; hypothetical test margin', True, 'cash')
    return book, futures, contract


def test_cme_mes_tick_and_cash_variation_margin():
    book, futures, contract = future_book()
    futures.trade('2024-01-02', contract, 2, 5000)
    assert book.accounts['a'].cash == 9996.75  # 2 * ($1 + half $1.25 tick).
    assert futures.reserved() == 3000
    assert book.accounts['a'].buying_power == 6996.75
    result = futures.settle('2024-01-03', contract.symbol, 5000.25)
    assert result['variation_margin'] == 2.5
    assert book.accounts['a'].cash == 9999.25
    assert result['possible_intraday_breach'] is None
    assert book.wealth('a', {contract.symbol: 5001.25}) == 10009.25


def test_futures_reserves_are_shared_with_equity_orders():
    book, futures, contract = future_book(cash=2000)
    book.add_security(Security('X'))
    futures.trade('2024-01-02', contract, 1, 5000)
    with pytest.raises(ValueError, match='cash'):
        book.execute('2024-01-02', 'a', 'X', 'BUY', 10, 100)


def test_futures_contract_rounding_tick_and_initial_margin():
    _, futures, contract = future_book(cash=2000)
    with pytest.raises(ValueError, match='Integer'):
        futures.trade('2024-01-02', contract, .5, 5000)
    with pytest.raises(ValueError, match='tick'):
        futures.trade('2024-01-02', contract, 1, 5000.1)
    with pytest.raises(ValueError, match='margin'):
        futures.trade('2024-01-02', contract, 2, 5000)


def test_margin_increase_and_overnight_gap_liquidation_can_leave_debt():
    book, futures, contract = future_book(cash=2000)
    futures.trade('2024-01-02', contract, 1, 5000)
    assert futures.change_margin('2024-01-03', contract.symbol, 3000, 2500, source='scenario house notice')
    result = futures.settle('2024-01-04', contract.symbol, 4800, adverse_extreme=4700, extreme_source='scenario OHLC')
    assert result['margin_call'] and result['possible_intraday_breach']
    futures.force_liquidate('2024-01-05', {contract.symbol: 4500})
    assert not futures.positions
    assert book.accounts['a'].cash == pytest.approx(-503.25)


def test_section1256_year_end_sixty_forty_and_no_double_count():
    book, futures, contract = future_book()
    futures.trade('2024-01-02', contract, 1, 5000)
    futures.settle('2024-12-31', contract.symbol, 5100)
    futures.year_end('2024-12-31', {contract.symbol: 5100})
    result = book.annual_tax('a', 2024, 100000, federal_only=True, as_of='2025-01-31')
    assert result['taxable_st'] == pytest.approx(498.375 * .4)
    assert result['taxable_lt'] == pytest.approx(498.375 * .6)
    futures.settle('2025-01-02', contract.symbol, 5101)
    assert book.accounts['a'].income[2025]['section1256'] == 5


def test_explicit_roll_charges_both_contracts():
    book, futures, contract = future_book()
    next_contract = replace(contract, symbol='MESH2026', expiry=date(2026, 3, 20))
    futures.trade('2024-01-02', contract, 1, 5000)
    futures.trade('2024-12-01', contract, -1, 5100)
    futures.trade('2024-12-01', next_contract, 1, 5120)
    assert list(futures.positions) == ['MESH2026']
    assert book.accounts['a'].cash == pytest.approx(10500 - 3 * 1.625)


def test_generic_roth_futures_rejected_and_broker_profile_requires_evidence():
    book = PortfolioLedger()
    book.add_account('r', CONSERVATIVE_ROTH, 10000)
    with pytest.raises(UnsupportedModelError):
        FuturesBook(book, 'r', commission_per_contract=1, spread_ticks=1)
    book.accounts['r'].profile = replace(CONSERVATIVE_ROTH, allowed_kinds=frozenset({'future'}))
    with pytest.raises(UnsupportedModelError):
        FuturesBook(book, 'r', commission_per_contract=1, spread_ticks=1)
    f = FuturesBook(book, 'r', commission_per_contract=1, spread_ticks=1,
                    ira_margin_multiplier=2., profile_source='IBKR FAQ + supplied hypothetical margin schedule',
                    profile_verified_on=date(2026, 11, 23))
    assert f.multiplier == 2


def quote():
    now = datetime(2025, 1, 2, 15, tzinfo=timezone.utc)
    return now, OptionQuote(now, 10., 11., 5, 5, 'synthetic fixture representing an OPRA record',
                             contract_symbol='X270115C00100000')


def test_option_quote_spread_size_timestamps_and_theoretical_rejection():
    now, q = quote()
    assert q.fill(now, 'BUY', 1) == 11
    assert q.fill(now, 'SELL', 1) == 10
    with pytest.raises(ValueError, match='stale'):
        q.fill(now - timedelta(seconds=1), 'BUY', 1)
    with pytest.raises(ValueError, match='stale'):
        q.fill(now + timedelta(minutes=2), 'BUY', 1)
    with pytest.raises(UnsupportedModelError, match='size'):
        q.fill(now, 'BUY', 6)
    with pytest.raises(UnsupportedModelError, match='Theoretical'):
        replace(q, observed=False).fill(now, 'BUY', 1)


def test_option_collateral_and_derivative_tax_gate():
    assert option_collateral('cash_secured_put', 2, strike=100) == 20000
    assert option_collateral('leaps', 2, strike=100, premium_debit=20) == 4000
    with pytest.raises(ValueError):
        option_collateral('covered_call', 2, strike=100, stock_shares=199)
    with pytest.raises(UnsupportedModelError):
        require_derivative_tax_support('covered_call', 'taxable')


def test_futures_delivery_requires_explicit_handler():
    _, futures, contract = future_book()
    physical = replace(contract, symbol='GCH2025', settlement_type='physical', first_notice=date(2025, 2, 28))
    with pytest.raises(UnsupportedModelError, match='delivery'):
        futures.trade('2025-03-01', physical, 1, 5000)


def test_roth_long_leaps_respects_premium_cash_and_actual_quotes():
    book = PortfolioLedger()
    book.add_account('r', replace(CONSERVATIVE_ROTH, allowed_kinds=frozenset({'option'})), 2000)
    options = LongOptionBook(book, 'r', commission_per_contract=1)
    contract = OptionContract('X270115C00100000', 'X', 100., date(2027, 1, 15), 'call', 'american', 'physical')
    now, q = quote()
    with pytest.raises(ValueError, match='premium'):
        options.trade(now, contract, 'BUY', 2, q)
    options.trade(now, contract, 'BUY', 1, q)
    assert book.accounts['r'].cash == 899
    assert book.wealth('r', {contract.symbol: 10}) == 1899
    options.trade(now, contract, 'SELL', 1, q)
    assert book.accounts['r'].cash == 899
    assert book.accounts['r'].unsettled == [(date(2025,1,3),999)]
    assert book.wealth('r',{}) == 1898
    assert book.annual_tax('r', 2025, 100000)['tax'] == 0


def test_option_mark_cannot_use_another_contracts_bid():
    ledger=PortfolioLedger()
    ledger.add_account('r',replace(CONSERVATIVE_ROTH,allowed_kinds=frozenset({'option'})),10000.)
    book=LongOptionBook(ledger,'r')
    contract=OptionContract('X_CALL','X',100.,date(2025,12,19),'call','european','cash')
    ts=datetime(2025,1,2,15,0,tzinfo=timezone.utc)
    quote=OptionQuote(ts,9.,10.,10,10,'Independent quote fixture',contract_symbol='X_CALL')
    book.trade(ts,contract,'BUY',1,quote)
    wrong=replace(quote,contract_symbol='OTHER_CALL',bid=99.,ask=100.)
    with pytest.raises(ValueError,match='Valuation quote'):
        book.mark(ts,{'X_CALL':wrong})
