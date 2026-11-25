"""IRC 852(b)(7): tax receipt is separate from spendable January cash.

IRS Instructions for Form 1099-DIV (01/2024), Special reporting rules:
https://www.irs.gov/instructions/i1099div
Actual fund examples use issuer final tax-year allocation; no assumed election.
"""
from datetime import date
import exchange_calendars as xcals
import pandas as pd
import pytest
from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import Distribution,run_account_backtest
from letf.account_cli import _issuer_year
from letf.accuracy import UnsupportedModelError

SOURCE='Issuer final 2025 tax-year allocation; https://www.irs.gov/instructions/i1099div'


def distribution(**changes):
    args=dict(security='X',ex_date=date(2025,12,19),pay_date=date(2026,1,30),amount_per_share=1.,
        category='qualified_dividend',source=SOURCE,qualified_verified=True,tax_date=date(2025,12,31),
        tax_date_source=SOURCE,issuer_tax_year=2025,record_date=date(2025,12,19))
    return Distribution(**dict(args,**changes))


def replay(profile,*,start='2025-01-02',end='2025-12-31'):
    idx=xcals.get_calendar('XNYS').sessions_in_range(start,end).tz_localize(None)
    market=pd.DataFrame({'X_Open':100.,'X_Close':100.},index=idx)
    return run_account_backtest(market,[Security('X')],profile,initial_capital=1000.,allocator=lambda h,w:{'X':1.},
        price_convention='as_traded',distributions_complete=True,distributions=[distribution()],
        rebalance_every=9999,terminal_liquidation=True,settlement_policy='us_equity',
        qualified_holding_policy='track_unhedged_common',tax_payment_dates={2025:'2026-04-15'},
        tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True})


def test_deemed_dec31_receipt_taxed_once_without_crediting_spendable_cash():
    taxable=replay(TAXABLE_CASH);roth=replay(CONSERVATIVE_ROTH)
    assert taxable['ending_distribution_receivables']==roth['ending_distribution_receivables']==10.
    assert taxable['terminal_tax_payable']==pytest.approx(1.5)
    assert taxable['ending_wealth_net_payable']==pytest.approx(1008.5)
    assert roth['ending_wealth_net_payable']==1010. and roth['terminal_tax_payable']==0.
    assert taxable['reconciliation'].cash.iloc[-1]==0.
    assert taxable['reconciliation'].unsettled.iloc[-1]==1000.
    assert taxable['ledger'].accounts['main'].income[2025]['qualified_dividend']==10.
    assert not any(e['action']=='DISTRIBUTION_CASH' for e in taxable['ledger'].events)


def test_january_cash_is_not_taxed_again_and_cannot_be_spent_before_pay_date():
    r=replay(TAXABLE_CASH,end='2026-02-02')
    assert r['ending_distribution_receivables']==0.
    assert r['ending_wealth_net_payable']==pytest.approx(1008.5)
    assert r['reconciliation'].loc[date(2026,1,2),'cash']==0.
    assert r['reconciliation'].loc[date(2026,1,30),'cash']==10.
    events=r['ledger'].events
    assert sum(e['amount'] for e in events if e['action']=='DISTRIBUTION_TAX')==10.
    assert sum(e['amount'] for e in events if e['action']=='DISTRIBUTION_CASH')==10.
    assert r['ledger'].accounts['main'].income.get(2026,{}).get('qualified_dividend',0.)==0.
    assert r['reconciliation'].cash_residual.abs().max()<1e-9


def test_terminal_sale_before_qdi_maturity_uses_ordinary_character_even_when_issuer_eligible():
    r=replay(TAXABLE_CASH,start='2025-12-01')
    assert r['terminal_tax_payable']==pytest.approx(2.2)
    assert r['ledger'].accounts['main'].income[2025]['ordinary_dividend']==10.


def test_unknown_year_end_mapping_cannot_be_invented_from_a_january_payment():
    with pytest.raises(UnsupportedModelError):distribution(pay_date=date(2026,2,2))
    with pytest.raises(UnsupportedModelError):distribution(record_date=date(2026,1,2))
    with pytest.raises(ValueError):distribution(tax_date_source='')
    for invalid in (2025.5,True,'2025.2'):
        with pytest.raises(ValueError):_issuer_year(invalid)
    assert _issuer_year(2025.)==2025 and _issuer_year(None) is None
