from dataclasses import replace

import pytest

from letf.accounts import PortfolioLedger, BrokerProfile, Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.accuracy import UnsupportedModelError


def make(profile=TAXABLE_CASH, cash=1000):
    book = PortfolioLedger()
    book.add_account('a', profile, cash)
    book.add_security(Security('X'))
    return book


def test_cash_execution_costs_and_partial_fills():
    book = make(replace(TAXABLE_CASH, commission=1, spread_bps=20, slippage_bps=10))
    event = book.execute('2024-01-01', 'a', 'X', 'BUY', 10, 100, fill_shares=5)
    assert event['price'] == pytest.approx(100.2)
    assert book.accounts['a'].cash == pytest.approx(498)
    assert book.lots[0].basis_per_share == pytest.approx(100.4)
    assert book.wealth('a', {'X': 100}) == pytest.approx(998)


def test_insufficient_cash_and_short_sales_rejected():
    book = make()
    with pytest.raises(ValueError, match='cash'):
        book.execute('2024-01-01', 'a', 'X', 'BUY', 11, 100)
    with pytest.raises(ValueError, match='shares'):
        book.execute('2024-01-01', 'a', 'X', 'SELL', 1, 100)
    assert not book.events and not book.lots


def test_explicit_settlement_cash_cannot_be_spent_early():
    book = make()
    book.execute('2024-01-01', 'a', 'X', 'BUY', 10, 100)
    book.execute('2024-01-02', 'a', 'X', 'SELL', 10, 100, settlement_date='2024-01-03')
    with pytest.raises(ValueError, match='cash'):
        book.execute('2024-01-02', 'a', 'X', 'BUY', 10, 100)
    assert book.wealth('a', {'X': 100}) == 1000
    book.execute('2024-01-03', 'a', 'X', 'BUY', 10, 100)
    assert book.quantity('a', 'X') == 10


def test_hifo_and_specific_identification():
    book = make(cash=10000)
    book.execute('2024-01-01', 'a', 'X', 'BUY', 10, 100)
    book.execute('2024-01-02', 'a', 'X', 'BUY', 10, 200)
    with pytest.raises(ValueError, match='lot ids'):
        book.execute('2024-03-01', 'a', 'X', 'SELL', 2, 300, method='SPEC_ID')
    book.execute('2024-03-01', 'a', 'X', 'SELL', 2, 300, method='HIFO')
    assert book.realizations[0].basis == 400
    book.execute('2024-03-01', 'a', 'X', 'SELL', 2, 300, method='SPEC_ID', lot_ids=['L1'])
    assert book.realizations[1].basis == 200


def test_product_availability_and_unsupported_tax_structure():
    book = make()
    book.add_security(Security('Y', inception=__import__('datetime').date(2025, 1, 1)))
    with pytest.raises(UnsupportedModelError, match='not yet exist'):
        book.execute('2024-01-01', 'a', 'Y', 'BUY', 1, 10)
    with pytest.raises(UnsupportedModelError):
        book.add_security(Security('P', tax_structure='partnership'))
    with pytest.raises(UnsupportedModelError):
        book.add_security(Security('F', kind='future'))


def test_profile_minimum_permissions_and_integer_sizing():
    book = make(replace(CONSERVATIVE_ROTH, fractional_shares=False, minimum_equity=2000))
    with pytest.raises(ValueError, match='integer'):
        book.execute('2024-01-01', 'a', 'X', 'BUY', .5, 100)
    with pytest.raises(ValueError, match='minimum'):
        book.execute('2024-01-01', 'a', 'X', 'BUY', 1, 100)


def test_tax_payment_requires_cash_and_is_not_silently_subtracted():
    book = make()
    book.execute('2024-01-01', 'a', 'X', 'BUY', 10, 100)
    with pytest.raises(ValueError, match='tax cash'):
        book.pay_tax('2024-04-15', 'a', 100)
    book.execute('2024-04-15', 'a', 'X', 'SELL', 1, 200)
    book.pay_tax('2024-04-15', 'a', 100)
    assert book.accounts['a'].cash == 100
    assert book.realizations[0].gain == 100


def test_split_conserves_wealth_and_basis():
    book = make()
    book.execute('2024-01-01', 'a', 'X', 'BUY', 10, 100)
    book.split('2024-06-01', 'X', 2)
    assert book.quantity('a', 'X') == 20
    assert book.wealth('a', {'X': 50}) == 1000
    assert book.lots[0].basis_per_share == 50


def test_calendar_dates_required_and_events_cannot_go_backwards():
    book = make()
    with pytest.raises(ValueError, match='calendar date'):
        book.execute(0, 'a', 'X', 'BUY', 1, 10)
    book.execute('2024-01-03', 'a', 'X', 'BUY', 1, 10)
    with pytest.raises(ValueError, match='chronological'):
        book.execute('2024-01-02', 'a', 'X', 'BUY', 1, 10)


def test_dated_broker_profiles_do_not_grant_historical_or_personal_approval():
    from datetime import date
    from letf.brokers import ibkr_roth_subset, fidelity_roth_cash_subset
    with pytest.raises(UnsupportedModelError, match='historical'):
        ibkr_roth_subset(trading_date=date(2025, 1, 1))
    profile = ibkr_roth_subset(trading_date=date(2026, 11, 23), approved_futures=True)
    assert 'future' in profile.allowed_kinds and profile.source
    profile = fidelity_roth_cash_subset(trading_date=date(2026, 11, 23))
    assert 'future' not in profile.allowed_kinds and 'option' not in profile.allowed_kinds
