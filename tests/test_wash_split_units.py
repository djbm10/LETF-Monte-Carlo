"""IRS Pub550 wash replacement/basis rules combined with split unit algebra.

Original dollar loss and sale shares never change. Splits change the number
of replacement shares representing that loss, not the amount or tax window.
"""
from letf.accounts import PortfolioLedger,TAXABLE_CASH,CONSERVATIVE_ROTH,Security
from letf.accuracy import UnsupportedModelError
import pytest


def ledger():
    book=PortfolioLedger();book.add_security(Security('X'))
    book.add_account('a',TAXABLE_CASH,3000.);book.add_account('r',CONSERVATIVE_ROTH,3000.)
    book.execute('2024-01-02','a','X','BUY',10.,100.)
    return book


def test_forward_split_partial_taxable_and_roth_replacements_preserve_dollar_loss():
    b=ledger();b.execute('2024-01-10','a','X','SELL',10.,80.)
    loss=b.realizations[0];b.split('2024-01-11','X',2.)
    assert loss.shares==10. and loss.basis==1000. and loss.proceeds==800.
    assert loss.unmatched==20. and loss.matching_share_factor==2.
    b.execute('2024-01-12','r','X','BUY',5.,40.)
    b.execute('2024-01-15','a','X','BUY',10.,40.)
    assert loss.permanent_disallowed==50. and loss.disallowed==150. and loss.gain==-50.
    taxable=[x for x in b.lots if x.account=='a' and x.shares>0]
    roth=[x for x in b.lots if x.account=='r' and x.shares>0]
    assert taxable[0].basis_per_share==50. and roth[0].basis_per_share==40.
    assert b.annual_tax('a',2024,100000.,federal_only=True,as_of='2025-01-31')['capital_loss_deduction']==50.


def test_pre_sale_replacement_split_and_remaining_replacement_do_not_double_match():
    b=ledger();b.execute('2024-01-05','a','X','BUY',4.,90.)
    b.execute('2024-01-10','a','X','SELL',10.,80.)
    loss=b.realizations[0];assert loss.disallowed==80. and loss.unmatched==6.
    b.split('2024-01-11','X',2.)
    b.execute('2024-01-12','a','X','BUY',12.,40.)
    assert loss.disallowed==200. and loss.unmatched==0. and loss.gain==0.
    assert sorted((x.shares,x.basis_per_share) for x in b.lots if x.shares)==[(8.,55.),(12.,50.)]


def test_reverse_split_restates_remaining_loss_without_creating_a_purchase():
    b=ledger();b.execute('2024-01-10','a','X','SELL',10.,80.);b.split('2024-01-11','X',.5)
    b.execute('2024-01-12','a','X','BUY',2.,160.)
    loss=b.realizations[0]
    assert loss.disallowed==80. and loss.unmatched==3. and loss.gain==-120.
    assert [x.basis_per_share for x in b.lots if x.shares]==[200.]
    assert sum(e['action']=='BUY' for e in b.events)==2 # split itself does not purchase replacements


def test_unknown_different_ticker_share_conversion_stays_blocked_before_mutation():
    b=ledger();b.add_security(Security('Y',identity='X'))
    with pytest.raises(UnsupportedModelError,match='conversion'):b.split('2024-01-11','X',2.)
    assert b.quantity('a','X')==10. and not any(e['action']=='SPLIT' for e in b.events)
