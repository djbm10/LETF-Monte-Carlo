from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from letf.account_backtest import run_account_backtest, strategy_allocator, Distribution
from letf.accounts import Security, CONSERVATIVE_ROTH, TAXABLE_CASH
from letf.accuracy import UnsupportedModelError
from letf.performance import xirr, portfolio_metrics
from letf.integration import process_trades_with_wired_engine
from letf.tax.engine import TaxpayerElections


def market():
    dates = pd.bdate_range('2024-01-02', periods=10)
    return pd.DataFrame({'SPY_Open': np.arange(100., 110.), 'SPY_Close': np.arange(101., 111.)}, index=dates)


def run(frame, profile=CONSERVATIVE_ROTH, **kwargs):
    return run_account_backtest(frame, [Security('SPY')], profile, initial_capital=1000,
                                allocator=lambda history, weights: {'SPY': 1.}, price_convention='as_traded',
                                distributions_complete=True, **kwargs)


def test_benchmark_replay_cash_lots_and_terminal_wealth_reconcile():
    result = run(market(), terminal_liquidation=True)
    assert result['ending_wealth_net_payable'] == 1100
    assert result['ledger'].quantity('main', 'SPY') == 0
    assert result['tax_paid'] == 0


def test_prior_history_only_and_next_open_execution_prefix_invariance():
    received = []
    frame = market()
    def allocator(history, weights):
        received.append(len(history))
        return {'SPY': 1. if len(history) >= 2 else 0.}
    result = run_account_backtest(frame, [Security('SPY')], CONSERVATIVE_ROTH,
                                  initial_capital=1000, allocator=allocator,
                                  price_convention='as_traded', distributions_complete=True)
    assert received == list(range(10))
    fills = [e for e in result['ledger'].events if e['action'] == 'BUY']
    assert fills[0]['date'] == frame.index[2].date()
    assert fills[0]['price'] == frame['SPY_Open'].iloc[2]
    prefix = run_account_backtest(frame.iloc[:5], [Security('SPY')], CONSERVATIVE_ROTH,
                                  initial_capital=1000, allocator=allocator,
                                  price_convention='as_traded', distributions_complete=True)
    np.testing.assert_array_equal(result['equity'].iloc[:5], prefix['equity'])


def test_distribution_entitlement_precedes_ex_open_purchase_and_cash_waits_for_paydate():
    frame = market()
    distribution = Distribution('SPY', frame.index[2].date(), frame.index[5].date(),
                                 1., 'ordinary_dividend', 'fixture issuer payable notice')
    result = run(frame, distributions=[distribution])
    events = [e for e in result['ledger'].events if e['action'] == 'INCOME']
    assert events[0]['date'] == frame.index[5].date()
    assert events[0]['amount'] == 10
    # Buying on the ex-date itself does not earn this distribution.
    later = run(frame.iloc[2:], distributions=[distribution])
    assert not any(e['amount'] for e in later['ledger'].events if e['action'] == 'INCOME')


def test_taxable_terminal_liquidation_tax_is_reported_and_can_reduce_wealth():
    result = run(market(), TAXABLE_CASH, terminal_liquidation=True,
                 tax_inputs={'ordinary_income': 100000, 'federal_only': True,
                             'no_external_wash_replacements_after_end': True})
    assert result['terminal_tax_payable'] == pytest.approx(22.)
    assert result['ending_wealth_net_payable'] == pytest.approx(1078.)
    assert result['after_tax_metrics']['ending_wealth'] == pytest.approx(1078.)


def test_adjusted_prices_and_unverified_advanced_strategy_gated():
    with pytest.raises(UnsupportedModelError, match='as-traded'):
        run_account_backtest(market(), [Security('SPY')], CONSERVATIVE_ROTH,
                              initial_capital=1000, allocator=strategy_allocator('S2'),
                              price_convention='adjusted', distributions_complete=True)
    with pytest.raises(UnsupportedModelError):
        strategy_allocator('S19')


def test_cash_account_rejects_overleverage():
    with pytest.raises(UnsupportedModelError, match='no-borrowing'):
        run_account_backtest(market(), [Security('SPY')], CONSERVATIVE_ROTH,
                              initial_capital=1000, allocator=lambda h,w: {'SPY': 1.5},
                              price_convention='as_traded', distributions_complete=True)


def test_annual_tax_funded_sales_create_current_year_gains_and_terminal_payable():
    dates = pd.bdate_range('2024-01-02', '2025-04-16')
    prices = np.where(dates >= '2025-01-02', 250., np.where(dates >= '2024-06-03', 200., 100.))
    frame = pd.DataFrame({'SPY_Open': prices, 'SPY_Close': prices}, index=dates)
    def allocate(history, weights):
        if len(history) and history.index[-1] == pd.Timestamp('2024-05-31'):
            return {}
        return {'SPY': 1.}
    result = run_account_backtest(frame, [Security('SPY')], TAXABLE_CASH,
               initial_capital=1000, allocator=allocate, price_convention='as_traded',
               distributions_complete=True, terminal_liquidation=True,
               tax_payment_dates={2024: date(2025, 4, 15)},
               tax_inputs={'ordinary_income': 100000, 'federal_only': True,
                           'no_external_wash_replacements_after_end': True})
    assert result['tax_paid'] == pytest.approx(220.)
    tax_sales = [r for r in result['ledger'].realizations if r.sold == date(2025, 4, 15)]
    assert sum(r.gain for r in tax_sales) == pytest.approx(44.)
    assert result['terminal_tax_payable'] == pytest.approx(110.)
    assert result['ending_wealth_net_payable'] == pytest.approx(2170.)


def test_dated_integration_uses_shared_ledger_without_invented_income_or_interest():
    trades = [{'date': '2024-01-02', 'asset': 'X', 'action': 'BUY', 'shares': 10, 'price': 100},
              {'date': '2024-12-01', 'asset': 'X', 'action': 'SELL', 'shares': 10, 'price': 200}]
    result = process_trades_with_wired_engine(trades, 1, TaxpayerElections(), 1000,
                tax_config={'account_type': 'taxable', 'ordinary_income': 100000,
                            'federal_only': True, 'wash_observation_end': '2025-01-31'})
    assert result['total_tax'] == pytest.approx(220.)
    assert not result['after_tax_wealth_verified']
    assert result['ledger'].accounts['main'].cash == 2000


def test_unmapped_session_ordinals_rejected_by_tax_adapter():
    trades = [{'day_index': 0, 'asset': 'X', 'action': 'BUY', 'shares': 1, 'price': 10}]
    with pytest.raises(UnsupportedModelError, match='Calendar'):
        process_trades_with_wired_engine(trades, 1, TaxpayerElections(), 1000,
                                        tax_config={'account_type': 'roth'})


def test_money_weighted_return_separate_from_lump_sum_cagr():
    flows = [('2024-01-01', -1000), ('2025-01-01', -1000), ('2026-01-01', 2310)]
    assert xirr(flows) == pytest.approx(.10, abs=.0002)
    dates = pd.to_datetime(['2024-01-01', '2025-01-01', '2026-01-01'])
    metrics = portfolio_metrics(pd.Series([1000., 2100., 2310.], index=dates), 1000,
                                 start_date='2024-01-01', external_flows=[('2025-01-01', -1000)])
    assert metrics['cagr'] is None
    assert metrics['money_weighted_return'] == pytest.approx(.1, abs=.0002)


def test_initial_loss_included_in_drawdown_and_real_returns_use_supplied_cpi():
    dates = pd.to_datetime(['2024-01-01', '2025-01-01'])
    metric = portfolio_metrics(pd.Series([900., 1100.], index=dates), 1000,
                                start_date='2024-01-01', inflation_index=pd.Series([100., 110.], index=dates))
    assert metric['max_drawdown'] == pytest.approx(.1)
    assert metric['real_cagr'] == pytest.approx(0.)
