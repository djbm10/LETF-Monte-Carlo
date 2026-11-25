from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from letf.account_backtest import run_account_backtest
from letf.accounts import Security,CONSERVATIVE_ROTH,TAXABLE_CASH
from letf.accuracy import UnsupportedModelError


@pytest.mark.parametrize('start,expected',[
    ('1998-01-02',date(1998,1,7)),
    ('2017-09-01',date(2017,9,7)), # Labor Day plus transition from T+3.
    ('2017-09-05',date(2017,9,7)),
    ('2024-05-24',date(2024,5,29)), # Memorial Day before T+1 transition.
    ('2024-05-28',date(2024,5,29))])
def test_buy_delivery_is_dated_and_cash_sale_proceeds_follow_effective_law(start,expected):
    end=pd.Timestamp(start)+pd.Timedelta(days=12)
    idx=xcals.get_calendar('XNYS',start=start,end=end+pd.Timedelta(days=7)).sessions_in_range(start,end).tz_localize(None)
    market=pd.DataFrame({'X_Open':100.,'X_Close':100.},index=idx)
    r=run_account_backtest(market,[Security('X')],CONSERVATIVE_ROTH,initial_capital=1000.,
        allocator=lambda h,w:{'X':1.} if len(h)==0 else {},price_convention='as_traded',
        distributions_complete=True,settlement_policy='us_equity')
    buy=next(e for e in r['ledger'].events if e['action']=='BUY')
    sale=next(e for e in r['ledger'].events if e['action']=='SELL')
    assert buy['settlement_date']==expected
    assert sale['settlement_date']>sale['date']
    assert r['reconciliation'].loc[sale['date'],'unsettled']==1000.
    assert r['reconciliation'].loc[sale['settlement_date'],'cash']==1000.
