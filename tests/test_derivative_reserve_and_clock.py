from dataclasses import replace
from datetime import date,datetime,timedelta
from zoneinfo import ZoneInfo

import pytest

from test_derivatives_accuracy import future_book
from letf.accounts import PortfolioLedger,CONSERVATIVE_ROTH
from letf.derivatives import LongOptionBook,OptionQuote,OptionContract


def test_reversing_futures_requires_initial_margin_even_without_larger_absolute_size():
    ledger,futures,contract=future_book(cash=2000.)
    futures.trade('2024-01-02',contract,1,5000.)
    futures.settle('2024-01-03',contract.symbol,4900.)
    before=ledger.accounts['a'].cash;events=len(ledger.events)
    assert before<contract.initial_margin and before>contract.maintenance_margin
    with pytest.raises(ValueError,match='initial/house margin'):
        futures.trade('2024-01-03',contract,-2,4900.)
    assert futures.positions[contract.symbol].quantity==1
    assert ledger.accounts['a'].cash==before and len(ledger.events)==events
    # A risk-reducing close is still allowed and releases collateral.
    futures.trade('2024-01-03',contract,-1,4900.)
    assert not futures.positions


def test_other_book_collateral_is_counted_in_settlement_and_margin_calls():
    ledger,futures,contract=future_book(cash=2000.)
    futures.trade('2024-01-02',contract,1,5000.)
    ledger.accounts['a'].collateral_reservations['other_book']=900.
    result=futures.settle('2024-01-03',contract.symbol,5000.,adverse_extreme=5000.,extreme_source='Independent fixture')
    assert result['margin_call'] and result['possible_intraday_breach']
    assert futures.change_margin('2024-01-03',contract.symbol,1500.,1200.,source='Independent margin fixture')


def test_same_option_instant_in_tokyo_books_correct_us_trade_day_and_expiry():
    ledger=PortfolioLedger();ledger.add_account('r',replace(CONSERVATIVE_ROTH,allowed_kinds=frozenset({'option'})),1000.)
    options=LongOptionBook(ledger,'r')
    contract=OptionContract('X_CALL','X',100.,date(2025,1,2),'call','american','physical')
    tokyo=datetime(2025,1,3,0,tzinfo=ZoneInfo('Asia/Tokyo'))  # Jan2 10:00 NewYork
    quote=OptionQuote(tokyo,1.,2.,2,2,'Independent timezone fixture',contract_symbol=contract.symbol)
    event=options.trade(tokyo,contract,'BUY',1,quote)
    assert event['date']==date(2025,1,2) and event['settlement_date']==date(2025,1,3)
    assert options.mark(tokyo,{contract.symbol:quote})==100.
    with pytest.raises(ValueError,match='chronological'):
        options.trade(tokyo-timedelta(seconds=1),contract,'SELL',1,replace(quote,timestamp=tokyo-timedelta(seconds=1)))
    with pytest.raises(ValueError,match='before'):
        options.mark(tokyo-timedelta(seconds=1),{contract.symbol:quote})
