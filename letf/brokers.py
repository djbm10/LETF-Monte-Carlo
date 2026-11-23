"""Dated broker capability subsets, never assertions of personal approval.

Sources checked 2026-11-23. These conservative subsets do not enable every
feature the broker offers, and must not be projected backward historically.
"""
from datetime import date
from letf.accounts import BrokerProfile
from letf.accuracy import UnsupportedModelError

VERIFIED_ON = date(2026, 11, 23)
IBKR_IRA_SOURCE = 'https://www.interactivebrokers.com/support/faq/content/1136195127'
FIDELITY_SOURCE = 'https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/limited-margin-agreement.pdf'


def ibkr_roth_subset(*, trading_date, approved_options=False, approved_futures=False):
    if trading_date < VERIFIED_ON:
        raise UnsupportedModelError('The current broker documentation is not historical permission evidence')
    kinds = {'stock', 'etf'}
    if approved_options:
        kinds.add('option')
    if approved_futures:
        kinds.add('future')
    return BrokerProfile('IBKR Roth cash: modeled permission subset', 'roth', frozenset(kinds),
                          source=IBKR_IRA_SOURCE, verified_on=VERIFIED_ON)


def fidelity_roth_cash_subset(*, trading_date, approved_options=False):
    if trading_date < VERIFIED_ON:
        raise UnsupportedModelError('Current documentation does not establish historical broker permissions')
    kinds = {'stock', 'etf', *({'option'} if approved_options else set())}
    # This cash-account subset does not model limited margin's unsettled-fund
    # benefits. No borrowing, naked shorts or futures are enabled.
    return BrokerProfile('Fidelity Roth cash: modeled permission subset', 'roth', frozenset(kinds),
                          source=FIDELITY_SOURCE, verified_on=VERIFIED_ON)
