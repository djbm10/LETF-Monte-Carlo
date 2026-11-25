"""IRS common-share rule, independently counted calendar days and cash."""
from datetime import date
import numpy as np
import pandas as pd
import pytest

from letf.accounts import PortfolioLedger,Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import Distribution,run_account_backtest
from letf.tax.dividend_holding import DividendHolding
from letf.accuracy import UnsupportedModelError

SOURCE='IRS Publication 550 (2025), Qualified dividends, https://www.irs.gov/publications/p550'


def test_partial_sales_count_sale_day_exclude_acquisition_day_and_do_not_duplicate_cash():
    b=PortfolioLedger();b.add_account('a',TAXABLE_CASH,1000.);b.add_security(Security('X'))
    b.execute(date(2024,1,3),'a','X','BUY',10.,100.)
    dist=Distribution('X',date(2024,1,31),date(2024,2,2),1.,'qualified_dividend',SOURCE,True)
    t=DividendHolding(dist,b,'a');t.update(date(2024,1,31),b)
    b.income(date(2024,2,2),'a',10.,'ordinary_dividend')
    b.execute(date(2024,3,1),'a','X','SELL',5.,100.);t.update(date(2024,3,1),b)
    assert (date(2024,3,1)-date(2024,1,3)).days==58
    assert t.unqualified_shares==5. and t.qualified_amount==0.
    b.execute(date(2024,3,4),'a','X','SELL',5.,100.);t.update(date(2024,3,4),b)
    assert (date(2024,3,4)-date(2024,1,3)).days==61
    assert t.qualified_amount==5. and not t.unresolved
    before=b.accounts['a'].cash
    b.reclassify_dividend(date(2024,3,4),'a',2024,5.,source=SOURCE,qualified_verified=True)
    assert b.accounts['a'].cash==before==1010.
    assert b.accounts['a'].income[2024]=={'ordinary_dividend':5.,'qualified_dividend':5.}


@pytest.mark.parametrize('last,expected_tax',[('2024-03-01',2.2),('2024-03-04',1.5)])
def test_canonical_final_sale_resolves_forward_holding_days_without_future_assumption(last,expected_tax):
    idx=pd.bdate_range('2024-01-03',last)
    market=pd.DataFrame({'X_Open':100.,'X_Close':100.},index=idx)
    d=Distribution('X',date(2024,1,31),date(2024,2,2),1.,'qualified_dividend',SOURCE,True)
    kw=dict(initial_capital=1000.,allocator=lambda h,w:{'X':1.},price_convention='as_traded',
        distributions_complete=True,distributions=[d],rebalance_every=999,terminal_liquidation=True,
        tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True},
        qualified_holding_policy='track_unhedged_common')
    taxable=run_account_backtest(market,[Security('X')],TAXABLE_CASH,**kw)
    roth=run_account_backtest(market,[Security('X')],CONSERVATIVE_ROTH,**kw)
    assert taxable['terminal_tax_payable']==pytest.approx(expected_tax)
    assert taxable['ending_wealth_net_payable']==pytest.approx(1010.-expected_tax)
    assert roth['ending_wealth_net_payable']==1010. and roth['terminal_tax_payable']==0.
    assert taxable['reconciliation'].cash_residual.abs().max()<1e-9
    assert [(e['shares'],e['price']) for e in taxable['ledger'].events if e['action'] in ('BUY','SELL')]==[
        (e['shares'],e['price']) for e in roth['ledger'].events if e['action'] in ('BUY','SELL')]


def test_year_end_reclassification_preserves_original_tax_year_and_split_entitlement():
    idx=pd.bdate_range('2024-11-15','2025-01-20');price=pd.Series(100.,index=idx);price.loc['2024-12-20':]=50.
    market=pd.DataFrame({'X_Open':price,'X_Close':price})
    d=Distribution('X',date(2024,12,2),date(2024,12,6),1.,'qualified_dividend',SOURCE,True)
    r=run_account_backtest(market,[Security('X')],TAXABLE_CASH,initial_capital=1000.,allocator=lambda h,w:{'X':1.},
        price_convention='as_traded',distributions_complete=True,distributions=[d],rebalance_every=999,
        split_events=[{'date':'2024-12-20','security':'X','ratio':2.,'source':'Issuer split fixture'}],
        terminal_liquidation=True,tax_payment_dates={2024:'2025-04-15'},qualified_holding_policy='track_unhedged_common',
        tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True})
    assert r['ledger'].accounts['main'].income[2024]['qualified_dividend']==10.
    assert r['ledger'].accounts['main'].income[2024]['ordinary_dividend']==0.
    assert r['ending_wealth_net_payable']==pytest.approx(1008.5)
    reclass=[e for e in r['ledger'].events if e['action']=='DIVIDEND_RECLASSIFICATION']
    assert len(reclass)==1 and reclass[0]['tax_year']==2024 and reclass[0]['date']==date(2025,1,15)


def test_unresolved_forward_holding_and_unverified_issuer_do_not_silently_qualify():
    b=PortfolioLedger();b.add_account('a',TAXABLE_CASH,1000.);b.add_security(Security('X'))
    b.execute(date(2024,1,3),'a','X','BUY',10.,100.)
    d=Distribution('X',date(2024,1,31),date(2024,2,2),1.,'qualified_dividend',SOURCE,False)
    with pytest.raises(UnsupportedModelError,match='issuer'):DividendHolding(d,b,'a')
    with pytest.raises(UnsupportedModelError):
        run_account_backtest(pd.DataFrame({'X_Open':100.,'X_Close':100.},index=pd.bdate_range('2024-01-03','2024-02-05')),
            [Security('X')],TAXABLE_CASH,initial_capital=1000.,allocator=lambda h,w:{'X':1.},price_convention='as_traded',
            distributions_complete=True,distributions=[Distribution('X',d.ex_date,d.pay_date,1.,d.category,SOURCE,True)],
            terminal_liquidation=False,qualified_holding_policy='track_unhedged_common',
            tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True})
