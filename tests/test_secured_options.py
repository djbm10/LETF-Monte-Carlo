"""Independent cash/share delivery examples; quotes and permissions are fixtures."""
from dataclasses import replace
from datetime import date,datetime,timezone

import pytest

from letf.accounts import PortfolioLedger,CONSERVATIVE_ROTH,TAXABLE_CASH,Security
from letf.accuracy import UnsupportedModelError
from letf.derivatives import OptionContract,OptionQuote
from letf.option_writing import SecuredOptionBook


def setup(right='put',cash=20000.,shares=0):
    ledger=PortfolioLedger();ledger.add_security(Security('X',kind='stock'))
    ledger.add_account('r',replace(CONSERVATIVE_ROTH,allowed_kinds=frozenset({'stock','option'})),cash)
    if shares:ledger.execute('2025-01-02','r','X','BUY',shares,100.)
    book=SecuredOptionBook(ledger,'r',approved_strategies={'covered_call','cash_secured_put'},
        permission_source='SOFTWARE_FIXTURE permission, no actual broker approval',permission_verified_on='2025-01-01',
        commission_per_contract=1.)
    c=OptionContract('X_'+right,'X',100.,date(2025,1,17),right,'american','physical')
    t=datetime(2025,1,3,15,tzinfo=timezone.utc)
    q=OptionQuote(t,2.,2.5,5,5,'SOFTWARE_FIXTURE quote',contract_symbol=c.symbol)
    return ledger,book,c,t,q


def test_put_premium_is_offset_by_liability_and_never_funds_its_own_collateral():
    ledger,book,c,t,q=setup(cash=10001.)
    book.trade(t,c,'SELL',1,q)
    assert ledger.accounts['r'].cash==10000.
    assert ledger.accounts['r'].buying_power==0.
    assert ledger.accounts['r'].unsettled==[(date(2025,1,6),200.)]
    assert ledger.wealth('r',{c.symbol:2.5})==9950.
    with pytest.raises(ValueError,match='collateral'):book.trade(t,c,'SELL',1,q)
    assert book.positions[c.symbol][1]==1


def test_call_locks_shares_and_corporate_actions_fail_before_mutation():
    ledger,book,c,t,q=setup('call',cash=20001.,shares=100)
    book.trade(t,c,'SELL',1,q)
    with pytest.raises(ValueError,match='reserved'):ledger.execute('2025-01-03','r','X','SELL',1,110.)
    with pytest.raises(ValueError,match='unencumbered'):book.trade(t,c,'SELL',1,q)
    for action in (lambda:ledger.split('2025-01-06','X',2.),
                   lambda:ledger.liquidate_security('2025-01-06','X',110.,source='fixture',settlement_date='2025-01-07')):
        with pytest.raises(UnsupportedModelError,match='deliverable|OCC'):action()
    assert ledger.quantity('r','X')==100 and ledger.securities['X'].closure is None
    book.assign('2025-01-06',c.symbol,1,notice_source='SOFTWARE_FIXTURE assignment',assignment_fee=2.)
    assert ledger.quantity('r','X')==0 and not book.positions
    assert ledger.accounts['r'].cash==10200.
    assert ledger.accounts['r'].unsettled==[(date(2025,1,7),9998.)] # fee nets from strike proceeds
    assert ledger.annual_tax('r',2025,100000.)['tax']==0.


def test_put_assignment_uses_full_strike_and_preserves_other_reserved_cash():
    ledger,book,c,t,q=setup(cash=20001.)
    book.trade(t,c,'SELL',1,q)
    ledger.accounts['r'].collateral_reservations['other']=10000.
    book.assign('2025-01-06',c.symbol,1,notice_source='SOFTWARE_FIXTURE assignment')
    assert ledger.quantity('r','X')==100
    assert ledger.accounts['r'].cash==10200.
    assert ledger.accounts['r'].buying_power==200.
    assert not book.positions


def test_failed_assignment_restores_obligation_and_share_locks():
    ledger,book,c,t,q=setup(cash=10001.)
    book.trade(t,c,'SELL',1,q)
    ledger.accounts['r'].collateral_reservations['other']=1000. # adverse external margin call
    with pytest.raises(ValueError,match='cash'):
        book.assign('2025-01-06',c.symbol,1,notice_source='SOFTWARE_FIXTURE assignment')
    assert book.positions[c.symbol][1]==1
    assert ledger.accounts['r'].collateral_reservations['secured_options:'+c.symbol]==10000.
    assert ledger.quantity('r','X')==0


def test_close_or_explicit_nonassignment_releases_only_own_collateral():
    ledger,book,c,t,q=setup(cash=10001.)
    book.trade(t,c,'SELL',1,q)
    ledger.accounts['r'].collateral_reservations['other']=9900.
    with pytest.raises(ValueError,match='close'):book.trade(t,c,'BUY',1,q)
    del ledger.accounts['r'].collateral_reservations['other']
    monday=datetime(2025,1,6,15,tzinfo=timezone.utc)
    book.trade(monday,c,'BUY',1,replace(q,timestamp=monday,ask=1.5,bid=1.))
    assert not book.positions and not ledger.accounts['r'].collateral_reservations
    assert ledger.accounts['r'].cash==10049.
    ledger,book,c,t,q=setup(cash=10001.);book.trade(t,c,'SELL',1,q)
    with pytest.raises(UnsupportedModelError,match='nonassignment'):book.expire(c.expiry,c.symbol,0.)
    book.expire(c.expiry,c.symbol,nonassignment_source='SOFTWARE_FIXTURE final notice')
    assert ledger.accounts['r'].cash==10200. and not book.positions


def test_naked_calls_unapproved_taxable_and_cross_book_netting_remain_blocked():
    ledger,book,c,t,q=setup('call',cash=20000.)
    with pytest.raises(ValueError,match='shares'):book.trade(t,c,'SELL',1,q)
    book.approved=frozenset({'cash_secured_put'})
    with pytest.raises(UnsupportedModelError,match='approval'):book.trade(t,c,'SELL',1,q)
    ledger=PortfolioLedger()
    ledger.add_account('r',replace(TAXABLE_CASH,allowed_kinds=frozenset({'option'})),20000.)
    with pytest.raises(UnsupportedModelError):
        SecuredOptionBook(ledger,'r',approved_strategies={'covered_call'},permission_source='fixture',permission_verified_on='2025-01-01')
