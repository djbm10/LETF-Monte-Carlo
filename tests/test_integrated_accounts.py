"""Independent cash/settlement examples, not snapshots of engine output."""
from datetime import date
from hashlib import sha256
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from letf.accounts import Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest
from letf.account_cli import compare_accounts, load_bundle
from letf.accuracy import UnsupportedModelError


def frame(start, end, prices=100.):
    sessions = xcals.get_calendar('XNYS').sessions_in_range(start, end).tz_localize(None)
    return pd.DataFrame({'X_Open': prices, 'X_Close': prices}, index=sessions)


def replay(market, allocator, profile=CONSERVATIVE_ROTH, **kwargs):
    return run_account_backtest(market, [Security('X')], profile, initial_capital=1000.,
        allocator=allocator, price_convention='as_traded', distributions_complete=True,
        settlement_policy='us_equity', **kwargs)


@pytest.mark.parametrize('start,end,settles', [('2024-05-23', '2024-05-29', '2024-05-29'),
                                               ('2024-05-28', '2024-05-31', '2024-05-30')])
def test_cash_reinvestment_waits_for_t2_t1_and_memorial_day(start, end, settles):
    market = frame(start, end)
    # Buy first session, sell second; third session requests reinvestment.
    r = replay(market, lambda h,w: {'X': 0. if len(h) == 1 else 1.})
    fills = [e for e in r['ledger'].events if e['action'] in ('BUY', 'SELL')]
    assert fills[1]['settlement_date'] == date.fromisoformat(settles)
    assert fills[2]['date'] == date.fromisoformat(settles)
    assert r['reconciliation'].loc[fills[1]['date'], 'cash'] == 0
    assert r['reconciliation'].loc[fills[1]['date'], 'unsettled'] == 1000


def test_sweep_interest_does_not_accrue_on_not_yet_settled_sale():
    m = frame('2024-06-03', '2024-06-06')
    r = replay(m, lambda h,w: {'X': 1. if len(h) == 0 else 0.},
               sweep_annual_rates=pd.Series(.0365, index=m.index))
    # Mon invested; Tue sale; Wed proceeds settle; Thu first overnight yield.
    income = [e for e in r['ledger'].events if e['action'] == 'INCOME']
    assert len(income) == 1
    assert income[0]['date'] == date(2024, 6, 6)
    assert income[0]['amount'] == pytest.approx(.1)


def test_tax_payment_sells_before_due_date_and_no_reinvestment_of_reserved_cash():
    m = frame('2024-01-02', '2025-04-16')
    m.loc[m.index >= '2024-06-03', :] = 200.
    m.loc[m.index >= '2025-01-02', :] = 250.
    def alloc(h,w):
        return {'X': 0. if len(h) and h.index[-1] == pd.Timestamp('2024-05-31') else 1.}
    r = replay(m, alloc, TAXABLE_CASH, terminal_liquidation=True,
        tax_payment_dates={2024: date(2025,4,15)}, tax_inputs={'ordinary_income':100000.,
        'federal_only':True, 'no_external_wash_replacements_after_end':True})
    funding = [e for e in r['ledger'].events if e.get('reason') == 'TAX_FUNDING']
    # $1000 ST gain in 2024: 22% marginal federal; sell .88 shares in 2025.
    assert r['tax_paid'] == pytest.approx(220.)
    assert funding[0]['date'] == date(2025,4,14)
    assert funding[0]['shares'] == pytest.approx(.88)
    assert funding[0]['settlement_date'] == date(2025,4,15)
    assert r['reconciliation'].loc[date(2025,4,14), 'reserved'] == pytest.approx(220)
    assert r['ending_wealth_net_payable'] == pytest.approx(2170.)
    assert (r['reconciliation'][['cash_residual','nav_residual','quantity_residual']].abs() < 1e-8).all().all()


@pytest.mark.parametrize('profile', [TAXABLE_CASH, CONSERVATIVE_ROTH])
def test_contributions_withdrawals_are_flows_not_investment_return(profile):
    m = frame('2024-06-03','2024-06-10')
    flows = [{'date':'2024-06-04','kind':'contribution','amount':100.,'magi':100000.,'compensation':100000.},
             {'date':'2024-06-07','kind':'withdrawal','amount':50.}]
    r = replay(m, lambda h,w:{'X':1.}, profile, cash_flows=flows,
        roth_history_config={'birth_date':'1980-01-01','first_tax_year':2010,'regular_basis':1000.},
        terminal_liquidation=True, tax_inputs={'ordinary_income':100000.,'federal_only':True,
        'no_external_wash_replacements_after_end':True})
    assert r['ending_wealth_net_payable'] == pytest.approx(1050.)
    assert r['metrics']['cagr'] is None
    assert r['metrics']['time_weighted_cagr'] == pytest.approx(0.)
    assert r['metrics']['money_weighted_return'] == pytest.approx(0., abs=1e-9)
    assert r['tax_paid'] == r['terminal_tax_payable'] == 0.
    funding = [e for e in r['ledger'].events if e.get('reason') == 'WITHDRAWAL_FUNDING']
    assert funding[0]['date'] == date(2024,6,6)
    assert funding[0]['shares'] == pytest.approx(.5)


def test_unsupported_contribution_and_cash_margin_fail():
    m = frame('2024-06-03','2024-06-10')
    with pytest.raises(ValueError, match='owner-wide'):
        replay(m,lambda h,w:{},cash_flows=[{'date':'2024-06-04','kind':'contribution',
            'amount':7100.,'magi':100000.,'compensation':100000.}],
            roth_history_config={'birth_date':'1980-01-01','first_tax_year':2010})
    with pytest.raises(UnsupportedModelError, match='no-borrowing'):
        replay(m,lambda h,w:{'X':1.01})


def test_bundle_and_public_api_compare_both_accounts_without_network(tmp_path):
    m=frame('2024-01-02','2024-01-05')
    data=pd.DataFrame({'Open':[100.,110.,120.,130.], 'Close':[100.,110.,120.,130.],
                       'Dividends':0.,'Stock Splits':0.},index=m.index)
    p=tmp_path/'prices.csv';data.to_csv(p)
    manifest={'data_status':'OBSERVED','price_convention':'as_traded','currency':'USD',
        'frequency':'daily','symbol':'X','distributions_complete':True,
        'files':{'prices':{'path':'prices.csv','sha256':sha256(p.read_bytes()).hexdigest(),
                          'provider':'Independent deterministic fixture','url':'https://example.invalid/fixture'}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    r=compare_accounts(tmp_path,strategy='buy_hold',start='2024-01-02',end='2024-01-05',
                       initial_capital=1000.,state='FEDERAL_ONLY')
    assert r['identical_pre_tax_trades']
    assert r['pre_tax_trade_count']==2
    # Independent buy at 100.02, sell at 129.974, 22% ST marginal tax.
    gross=1000/100.02*129.974
    assert r['accounts']['roth']['ending_wealth_net_payable']==pytest.approx(gross)
    assert r['accounts']['taxable']['ending_wealth_net_payable']==pytest.approx(gross-.22*(gross-1000))
    p.write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        load_bundle(tmp_path)


def test_daily_engine_owner_linked_roth_purchase_permanently_disallows_taxable_loss():
    m=frame('2024-01-02','2024-01-12')
    m.loc[m.index>'2024-01-02',:]=80.
    common=dict(terminal_liquidation=True,tax_inputs={'ordinary_income':100000.,'federal_only':True,
        'no_external_wash_replacements_after_end':True})
    alloc=lambda h,w:{'X':1. if len(h)==0 else 0.}
    isolated=replay(m,alloc,TAXABLE_CASH,**common)
    linked=replay(m,alloc,TAXABLE_CASH,linked_accounts=[{'id':'other_roth','profile':CONSERVATIVE_ROTH,'cash':1000.}],
        linked_trades=[{'date':'2024-01-10','account':'other_roth','security':'X','side':'BUY','shares':5.,
                       'price':80.,'source':'Independent execution fixture'}],**common)
    # Ten shares lost $200; five Roth replacements permanently disallow $100.
    assert isolated['terminal_tax_payable']==pytest.approx(-44.)
    assert linked['terminal_tax_payable']==pytest.approx(-22.)
    assert linked['ledger'].realizations[0].permanent_disallowed==pytest.approx(100.)
    assert linked['reconciliation'].cash_residual.abs().max()==0.
