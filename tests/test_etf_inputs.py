"""Independent distribution/nominal-price conventions, no network fixtures."""
from datetime import date
import pandas as pd
import pytest

from scripts.build_observed_etf_bundles import nominal_bundle
from letf.account_backtest import Distribution, run_account_backtest
from letf.accounts import Security, TAXABLE_CASH
from letf.accuracy import UnsupportedModelError


def test_dated_issuer_values_replace_rounded_dividends_without_fitting_price_returns():
    idx=pd.bdate_range('2023-01-02','2025-11-21')
    frame=pd.DataFrame({'Open':25.,'High':26.,'Low':24.,'Close':25.,'Dividends':0.,'Stock Splits':0.},index=idx)
    frame.loc['2025-11-20','Stock Splits']=2.
    records=[];restated=[]
    for ex,pay,amount,qdi,rounded in [('2024-03-20','2024-03-27',.215144,.04034,.1075),
                                     ('2024-06-26','2024-07-03',.282777,.053021,.1415)]:
        frame.loc[ex,'Dividends']=rounded
        serial=lambda d:str((pd.Timestamp(d)-pd.Timestamp('1899-12-30')).days)
        records.append({'C':'X','H':serial(ex),'I':serial(pay),'J':str(amount),'Q':str(amount),'R':str(qdi)})
        restated.append({'ExDate':ex,'PayableDate':pay,'CashDividendPerShare':amount/2})
    with pytest.raises(UnsupportedModelError,match='not constant'):
        nominal_bundle(frame,records,'X')
    prices,distributions,factor=nominal_bundle(frame,records,'X',restated)
    assert factor==2.
    assert prices.Close.iloc[0]==50.
    assert distributions.loc['2024-03-20','amount_per_share'].sum()==pytest.approx(.215144)
    assert set(distributions.category)=={'ordinary_dividend','qualified_dividend'}
    bad=[dict(r) for r in records];bad[0]['AG']='.001'
    with pytest.raises(UnsupportedModelError,match='special distribution'):
        nominal_bundle(frame,bad,'X',restated)


def test_qualified_etf_payment_requires_investor_holding_period_not_just_fund_qdi():
    idx=pd.bdate_range('2024-01-02',periods=10)
    market=pd.DataFrame({'X_Open':100.,'X_Close':100.},index=idx)
    d=Distribution('X',date(2024,1,4),date(2024,1,5),1.,'qualified_dividend','Dated issuer fixture',True)
    with pytest.raises(UnsupportedModelError,match='holding-period proof'):
        run_account_backtest(market,[Security('X')],TAXABLE_CASH,initial_capital=1000.,
            allocator=lambda h,w:{'X':1.},price_convention='as_traded',distributions_complete=True,
            distributions=[d],tax_inputs={'ordinary_income':100000.,'federal_only':True})
