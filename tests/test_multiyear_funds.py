from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import Distribution,run_account_backtest
from scripts.build_multiyear_fund_bundle import ici_date


def test_issuer_dates_accept_only_explicit_excel_serial_or_documented_string_format():
    assert ici_date('45657')==pd.Timestamp('2024-12-31')
    assert ici_date('03/26/2025')==pd.Timestamp('2025-03-26')
    with pytest.raises(ValueError):ici_date('2025-03-26')


def test_multiyear_cash_distributions_split_and_annual_tax_reconcile_independently():
    idx=xcals.get_calendar('XNYS').sessions_in_range('2024-01-02','2025-12-31').tz_localize(None)
    price=pd.Series(100.,index=idx);price.loc['2025-11-20':]=50.
    market=pd.DataFrame({'X_Open':price,'X_Close':price})
    payouts=[Distribution('X',date(y,6,3),date(y,6,5),1.,'ordinary_dividend','Dated issuer fixture') for y in (2024,2025)]
    common=dict(initial_capital=1000.,allocator=lambda h,w:{'X':1.},price_convention='as_traded',
        distributions_complete=True,distributions=payouts,settlement_policy='us_equity',terminal_liquidation=True,
        split_events=[{'date':'2025-11-20','security':'X','ratio':2.,'source':'Issuer fixture'}],
        tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True},
        tax_payment_dates={2024:'2025-04-15'})
    roth=run_account_backtest(market,[Security('X')],CONSERVATIVE_ROTH,**common)
    taxable=run_account_backtest(market,[Security('X')],TAXABLE_CASH,**common)
    assert roth['ending_wealth_net_payable']==pytest.approx(1020.1)
    assert taxable['tax_paid']==pytest.approx(2.2)
    assert taxable['terminal_tax_payable']==pytest.approx(2.21716)
    assert taxable['ending_wealth_net_payable']==pytest.approx(1015.66084)
    assert len([e for e in taxable['ledger'].events if e['action']=='SPLIT'])==1
    assert taxable['reconciliation'].quantity_residual.abs().max()<1e-9
