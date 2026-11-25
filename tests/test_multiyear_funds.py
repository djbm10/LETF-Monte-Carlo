from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import Distribution,run_account_backtest
from scripts.build_multiyear_fund_bundle import ici_date,check_rounded_cash
from scripts.build_multiyear_fund_bundle import verify_zero_payout


def test_modeled_early_tax_payment_is_a_real_exchange_session():
    from letf.account_cli import modeled_tax_payment_dates
    dates=modeled_tax_payment_dates('2021-01-04','2025-12-31')
    assert dates[2021]==date(2022,4,14) # Good Friday exchange closure, not a filing-deadline assertion
    assert dates[2022]==date(2023,4,14) # Saturday April15
    assert dates[2024]==date(2025,4,15)


def test_legacy_ici_header_validation_rejects_changed_tax_character_columns():
    from scripts.build_observed_etf_bundles import validate_ici_headers
    from letf.accuracy import UnsupportedModelError
    top={'A':'Security','M':'2022','Q':'Ordinary'}
    bottom={'A':'Description','C':'Ticker','H':'Ex-Dividend','I':'Payable','J':'Per Share',
        'M':'(Current Year)','Q':'Dividends','U':'Dividends*','V':'Gain Distr.',
        'Z':'Nondividend','AB':'Liquidation','AC':'Liquidation','AD':'Interest','AJ':'Dividends*'}
    assert validate_ici_headers([top,bottom],2022)==[top,bottom]
    with pytest.raises(UnsupportedModelError):validate_ici_headers([top,{**bottom,'U':'Unknown'}],2022)
    with pytest.raises(UnsupportedModelError):validate_ici_headers([top,bottom],2021)


def test_issuer_dates_accept_only_explicit_excel_serial_or_documented_string_format():
    assert ici_date('45657')==pd.Timestamp('2024-12-31')
    assert ici_date('03/26/2025')==pd.Timestamp('2025-03-26')
    with pytest.raises(ValueError):ici_date('2025-03-26')


def test_issuer_cash_check_verifies_given_split_factor_not_a_fit_to_rounded_payouts():
    check_rounded_cash(.31979,.320,1.)
    check_rounded_cash(.63958,.320,2.)
    with pytest.raises(ValueError):check_rounded_cash(.31979,.160,1.)
    check_rounded_cash(.000119,0.,4.)  # exact issuer amount below vendor's disclosed cash quantum
    with pytest.raises(ValueError):check_rounded_cash(.1,0.,4.)
    with pytest.raises(ValueError):check_rounded_cash(float('nan'),.1,1.)


def test_explicit_zero_issuer_tax_rows_are_not_invented_cash_or_missing_api_payments():
    verify_zero_payout((0.,0.,0.,0.),[],0.)
    verify_zero_payout((0.,0.,0.,0.),[{'CashDividendPerShare':0.}],0.)
    with pytest.raises(ValueError):verify_zero_payout((0.,.01,0.,0.),[],0.)
    with pytest.raises(ValueError):verify_zero_payout((0.,0.,0.,0.),[],.01)
    with pytest.raises(ValueError):verify_zero_payout((0.,0.,0.,0.),[{'CashDividendPerShare':.01}],0.)


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
