"""Independent calendar counts and22% ordinary-tax/20% deduction examples."""
from datetime import date

import pandas as pd
import pytest

from letf.accounts import PortfolioLedger,Security,TAXABLE_CASH
from letf.account_backtest import Distribution,run_account_backtest
from letf.tax.dividend_holding import DividendHolding
from letf.accuracy import UnsupportedModelError

SOURCE='26CFR1.199A-3,2025 edition; https://www.govinfo.gov/content/pkg/CFR-2025-title26-vol4/xhtml/CFR-2025-title26-vol4-sec1-199A-3.html'


@pytest.mark.parametrize('end,tax',[('2024-02-22',2.2),('2024-02-23',1.76)])
def test_45_vs_46_actual_holding_days_do_not_become_qdi(end,tax):
    index=pd.bdate_range('2024-01-08',end)
    d=Distribution('X',date(2024,1,31),date(2024,2,2),1.,'section199a_dividend',SOURCE,section199a_verified=True)
    r=run_account_backtest(pd.DataFrame({'X_Open':100.,'X_Close':100.},index=index),[Security('X')],TAXABLE_CASH,
        initial_capital=1000.,allocator=lambda h,w:{'X':1.},price_convention='as_traded',distributions_complete=True,
        distributions=[d],rebalance_every=999,terminal_liquidation=True,qualified_holding_policy='track_unhedged_common',
        tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True})
    assert (pd.Timestamp(end).date()-date(2024,1,8)).days in (45,46)
    assert r['terminal_tax_payable']==pytest.approx(tax)
    assert r['ending_wealth_net_payable']==pytest.approx(1010.-tax)
    assert not r['ledger'].accounts['main'].income[2024].get('qualified_dividend',0.)
    assert r['reconciliation'].cash_residual.abs().max()<1e-9


def test_partial_section199a_entitlement_excludes_wash_tacks_and_keeps_cash_once():
    book=PortfolioLedger();book.add_account('a',TAXABLE_CASH,1000.);book.add_security(Security('X'))
    book.execute('2024-01-08','a','X','BUY',10.,100.);book.lots[0].tacked_days=365
    d=Distribution('X',date(2024,1,31),date(2024,2,2),1.,'section199a_dividend',SOURCE,section199a_verified=True)
    tracker=DividendHolding(d,book,'a');book.income('2024-02-02','a',10.,'ordinary_dividend')
    book.execute('2024-02-22','a','X','SELL',5.,100.);tracker.update(date(2024,2,22),book)
    assert tracker.unqualified_shares==5. and tracker.qualified_amount==0.
    book.execute('2024-02-23','a','X','SELL',5.,100.);tracker.update(date(2024,2,23),book)
    before=book.accounts['a'].cash
    book.reclassify_dividend('2024-02-23','a',2024,5.,source=SOURCE,category='section199a_dividend',section199a_verified=True)
    assert book.accounts['a'].cash==before==1010.
    assert book.accounts['a'].income[2024]=={'ordinary_dividend':5.,'section199a_dividend':5.}
    result=book.annual_tax('a',2024,100000.,federal_only=True,as_of='2025-01-31')
    assert result['tax']==pytest.approx(1.98) # (10 -20%*5)*22%


def test_section199a_proof_across_year_and_split_retains_receipt_year():
    idx=pd.bdate_range('2024-11-20','2025-01-10');price=pd.Series(100.,index=idx);price.loc['2024-12-20':]=50.
    d=Distribution('X',date(2024,12,5),date(2024,12,9),1.,'section199a_dividend',SOURCE,section199a_verified=True)
    r=run_account_backtest(pd.DataFrame({'X_Open':price,'X_Close':price}),[Security('X')],TAXABLE_CASH,
        initial_capital=1000.,allocator=lambda h,w:{'X':1.},price_convention='as_traded',distributions_complete=True,
        distributions=[d],rebalance_every=999,terminal_liquidation=True,qualified_holding_policy='track_unhedged_common',
        split_events=[dict(date='2024-12-20',security='X',ratio=2.,source='SOFTWARE_FIXTURE split')],
        tax_payment_dates={2024:'2025-04-15'},tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True})
    inc=r['ledger'].accounts['main'].income
    assert inc[2024]['section199a_dividend']==10. and inc[2024]['ordinary_dividend']==0.
    assert r['ending_wealth_net_payable']==pytest.approx(1008.24)
    events=[e for e in r['ledger'].events if e['action']=='DIVIDEND_RECLASSIFICATION']
    assert len(events)==1 and events[0]['tax_year']==2024 and events[0]['date']==date(2025,1,6)
    unverified=Distribution('X',date(2024,12,5),date(2024,12,9),1.,'section199a_dividend',SOURCE)
    with pytest.raises(UnsupportedModelError,match='issuer'):DividendHolding(unverified,r['ledger'],'main')


def test_section199a_deduction_does_not_reduce_niit_gross_income_or_ma_base():
    book=PortfolioLedger();book.add_account('a',TAXABLE_CASH,0.)
    book.income('2024-06-03','a',10.,'section199a_dividend',section199a_verified=True)
    fed=book.annual_tax('a',2024,300000.,federal_only=True,as_of='2025-01-31')
    assert fed['tax']==pytest.approx(3.18) # 10*80%*35% federal +10*3.8% NIIT
    ma=book.annual_tax('a',2024,300000.,state_code='MA',as_of='2025-01-31')
    assert ma['tax']==pytest.approx(3.68) # MA does not import federal Section199A deduction


def test_truthy_strings_cannot_certify_preferential_income_at_low_level_apis():
    book=PortfolioLedger();book.add_account('a',TAXABLE_CASH,0.)
    for category,flag in [('qualified_dividend','qualified_verified'),('section199a_dividend','section199a_verified')]:
        with pytest.raises(ValueError,match='boolean'):
            Distribution('X',date(2024,6,3),date(2024,6,5),1.,category,SOURCE,**{flag:'False'})
        for value in ('false','True',1):
            with pytest.raises(UnsupportedModelError):book.income('2024-06-03','a',10.,category,**{flag:value})
            with pytest.raises(UnsupportedModelError):book.distribution_tax('2024-06-03','a',10.,category,
                tax_date='2024-06-03',source=SOURCE,**{flag:value})
    assert book.accounts['a'].cash==0. and not book.events
