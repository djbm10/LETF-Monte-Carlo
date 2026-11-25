from dataclasses import replace
from datetime import date,datetime,timezone

import pytest

from letf.accuracy import UnsupportedModelError
from letf.accounts import PortfolioLedger,CONSERVATIVE_ROTH,Security
from letf.derivatives import LongOptionBook,OptionContract,OptionQuote


def setup(cash=20000.,right='call',style='american'):
    ledger=PortfolioLedger();ledger.add_security(Security('X',kind='stock'))
    ledger.add_account('r',replace(CONSERVATIVE_ROTH,allowed_kinds=frozenset({'option','stock'})),cash)
    options=LongOptionBook(ledger,'r')
    c=OptionContract('X_OPTION','X',100.,date(2027,1,15),right,style,'physical')
    t=datetime(2025,1,2,15,tzinfo=timezone.utc)
    q=OptionQuote(t,10.,11.,5,5,'Independent quote fixture, not historical performance',contract_symbol=c.symbol)
    options.trade(t,c,'BUY',1,q)
    return ledger,options,c,t,q


def test_american_call_delivery_consumes_strike_cash_not_market_spread():
    ledger,options,c,_,_=setup()
    ledger.accounts['r'].profile=replace(ledger.accounts['r'].profile,spread_bps=200.)
    event=options.exercise('2025-01-03',c.symbol,1,notice_source='Test explicit holder exercise notice',exercise_fee=2.)
    assert ledger.accounts['r'].cash==8898.
    assert ledger.quantity('r','X')==100
    assert not options.positions
    assert event['settlement_date']==date(2025,1,6)
    assert ledger.wealth('r',{'X':120.})==20898.
    assert ledger.annual_tax('r',2025,100000.)['tax']==0.


def test_protective_put_delivers_owned_stock_and_proceeds_are_not_free_margin():
    ledger,options,c,_,_=setup(right='put')
    ledger.execute('2025-01-02','r','X','BUY',100,100.)
    options.exercise('2025-01-03',c.symbol,1,notice_source='Test holder put exercise')
    assert ledger.quantity('r','X')==0
    assert ledger.accounts['r'].cash==8900.
    assert ledger.accounts['r'].unsettled==[(date(2025,1,6),10000.)]
    assert ledger.wealth('r',{})==18900.


def test_insufficient_exercise_cash_or_shares_does_not_remove_long_option():
    for right,match in [('call','cash'),('put','shares')]:
        ledger,options,c,_,_=setup(cash=2000.,right=right)
        with pytest.raises(ValueError,match=match):
            options.exercise('2025-01-03',c.symbol,1,notice_source='Test notice')
        assert options.positions[c.symbol][1]==1
        assert ledger.accounts['r'].cash==900.


def test_european_early_exercise_and_underlying_split_are_blocked():
    ledger,options,c,_,_=setup(style='european')
    with pytest.raises(ValueError,match='window'):
        options.exercise('2025-01-03',c.symbol,1,notice_source='Test notice')
    ledger.split('2025-01-03','X',2.)
    with pytest.raises(UnsupportedModelError,match='OCC'):
        options.mark_value({c.symbol:10.})


def test_option_sale_waits_for_settlement_and_new_purchase_settles_prior_cash_first():
    ledger,options,c,t,q=setup(cash=2000.)
    options.trade(t,c,'SELL',1,q)
    with pytest.raises(ValueError,match='cash'):
        options.trade(t,c,'BUY',1,q)
    tomorrow=datetime(2025,1,3,15,tzinfo=timezone.utc)
    options.trade(tomorrow,c,'BUY',1,replace(q,timestamp=tomorrow))
    assert ledger.accounts['r'].cash==800.
    assert not ledger.accounts['r'].unsettled


def test_zero_bid_option_close_cannot_create_unauthorized_commission_debt():
    for capital in (1101.,1102.):
        ledger=PortfolioLedger()
        ledger.add_account('r',replace(CONSERVATIVE_ROTH,allowed_kinds=frozenset({'option'})),capital)
        options=LongOptionBook(ledger,'r',commission_per_contract=1.)
        t=datetime(2025,1,2,15,tzinfo=timezone.utc)
        c=OptionContract('X_CALL','X',100.,date(2027,1,15),'call','american','physical')
        q=OptionQuote(t,0.,11.,5,5,'Independent zero-bid fixture',contract_symbol=c.symbol)
        options.trade(t,c,'BUY',1,q)
        if capital==1101.:
            with pytest.raises(ValueError,match='commission'):options.trade(t,c,'SELL',1,q)
            assert c.symbol in options.positions
        else:
            options.trade(t,c,'SELL',1,q)
            assert not options.positions
        assert ledger.accounts['r'].cash==0.
        assert not ledger.accounts['r'].unsettled


def test_missing_option_depth_is_not_unlimited_liquidity():
    _,_,_,t,q=setup()
    for size in (float('nan'),float('inf'),1.5,True,-1):
        with pytest.raises(ValueError,match='size'):
            replace(q,ask_size=size).fill(t,'BUY',1)
