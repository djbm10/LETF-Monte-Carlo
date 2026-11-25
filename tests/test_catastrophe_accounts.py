import numpy as np
import pandas as pd
import pytest
import exchange_calendars as xcals

from letf.accuracy import UnsupportedModelError
from letf.macro.stress import stressed_context,scenario_market
from letf.account_stress import replay_scenario


def context(end='2028-12-29'):
    calendar=xcals.get_calendar('XNYS',start='2027-01-01',end='2029-01-31')
    dates=calendar.sessions_in_range('2027-01-04',end).tz_localize(None)
    # Deliberately reorder columns to catch positional shock writes.
    frame=pd.DataFrame({'short_rate':0.,'inflation_log':0.,'equity_log':0.},index=dates)
    frame.attrs['fixture']='Explicit hypothetical zero-total-return scenario'
    return frame


@pytest.mark.parametrize('case,timing,dead_two',[('daily_extinction_34','close',False),('overnight_extinction_50','open',True)])
def test_extinction_is_a_dated_absorbing_corporate_action_not_a_small_nav_floor(case,timing,dead_two):
    c=stressed_context(context(),case);m=scenario_market(c,3,annual_distribution=0.,fee=0.,spread=0.)
    event=m.liquidation_events[0];day=pd.Timestamp(event['date'])
    assert event['timing']==timing and event['cash_per_share']==0.
    assert (m.market.loc[day:,'CF3_Close']==0.).all()
    assert (m.fund_mechanics.loc[day:].iloc[1:][['fund_fee','financing']]==0.).all().all()
    assert (m.market.loc[day,'CF3_Open']==0.)==(timing=='open')
    two=scenario_market(c,2,annual_distribution=0.,fee=0.,spread=0.)
    assert bool(two.liquidation_events)==dead_two
    for account in ('taxable','roth'):
        summary,result=replay_scenario(m,account,initial=1000.,annual_contribution=0.,allocation=.8,
            rebalance_sessions=100000,spread_bps=0.,slippage_bps=0.)
        # Cash reserve is 200; taxable capital-loss deduction is independently
        # 22% of the 800 basis. It is a household tax receivable, not fund revival.
        assert summary['ending_net_assets']==pytest.approx(376. if account=='taxable' else 200.)
        assert result['ledger'].quantity('main','CF3')==0.
        assert summary['fund_closures']==1 and summary['minimum_cash']>=0.
        assert summary['independent_audit']['passed']


def test_smaller_crash_survives_and_issuer_failure_is_separate_from_benchmark():
    mild=scenario_market(stressed_context(context(),'daily_loss_25'),3,annual_distribution=0.,fee=0.,spread=0.)
    assert not mild.liquidation_events
    assert mild.market.CF3_Close.iloc[-1]==pytest.approx(25.)
    forced=stressed_context(context(),'issuer_failure_zero')
    assert (forced.equity_log==0).all()
    assert scenario_market(forced,1,annual_distribution=0.).liquidation_events
    with pytest.raises(UnsupportedModelError):stressed_context(context(),'probability_weighted_crash')
    forced.attrs['probability']=.01
    with pytest.raises(UnsupportedModelError,match='probabilities'):scenario_market(forced,3)


def test_bear_stagnation_and_daily_volatility_have_explicit_monthly_accounting():
    c=context();c['equity_log']=np.sin(np.arange(len(c)))*.01
    doubled=stressed_context(c,'double_daily_volatility')
    groups=c.index.to_period('M')
    for month in groups.unique():
        original=c.loc[groups==month,'equity_log'];changed=doubled.loc[groups==month,'equity_log']
        assert changed.sum()==pytest.approx(original.sum(),abs=1e-14)
        assert changed.var()==pytest.approx(4*original.var())
    for case,annual in [('bear_7y',-.08),('stagnation_7y_high_rates',0.)]:
        result=stressed_context(c,case)
        for month in groups.unique():
            if month.year!=2028:continue
            selected=result.loc[groups==month]
            assert selected.equity_log.sum()==pytest.approx(np.log1p(annual)/12,abs=1e-14)
            assert selected.inflation_log.sum()==pytest.approx(np.log1p(.06)/12)
            assert (selected.short_rate==.06).all()


@pytest.mark.parametrize('account',['taxable','roth'])
def test_distribution_exprice_and_cash_flows_are_not_fabricated_returns(account):
    m=scenario_market(stressed_context(context(),'reference'),1,fee=0.,spread=0.)
    # Price declines exactly by distributions; no total-return gains exist.
    first=m.distributions[0];day=pd.Timestamp(first.ex_date);i=m.market.index.get_loc(day)
    assert m.market.CF1_Close.iloc[i-1]-m.market.CF1_Close.iloc[i]==pytest.approx(first.amount_per_share)
    summary,result=replay_scenario(m,account,annual_contribution=3000.,allocation=.8,
        rebalance_sessions=63,spread_bps=0.,slippage_bps=0.)
    assert summary['contributions']==6000.
    assert summary['ending_net_assets']==pytest.approx(16000.,abs=1e-7)
    assert summary['metrics']['cagr'] is None
    assert summary['metrics']['money_weighted_return']==pytest.approx(0.,abs=1e-9)
    assert summary['independent_audit']['passed']
    if account=='roth':assert summary['tax_paid']==summary['terminal_tax_payable']==0.


def test_tax_sales_settle_before_payment_and_roth_restrictions_are_not_bypassed():
    m=scenario_market(stressed_context(context(),'reference'),1,fee=0.,spread=0.)
    summary,result=replay_scenario(m,'taxable',annual_contribution=0.,allocation=1.,spread_bps=0.,slippage_bps=0.)
    assert summary['tax_funding_sales']>0 and summary['tax_paid']>0
    events=result['ledger'].events
    for sale in [e for e in events if e.get('reason')=='TAX_FUNDING']:
        payment=next(e for e in events if e['action']=='TAX_PAYMENT' and e['date']>=sale['date'])
        assert sale['settlement_date']<=payment['date']
    with pytest.raises((ValueError,UnsupportedModelError)):
        replay_scenario(m,'roth',annual_contribution=9000.)
    with pytest.raises(UnsupportedModelError,match='buying power'):
        replay_scenario(m,'roth',allocation=1.1)


def test_legal_unbounded_tax_bracket_is_serialized_but_nan_wealth_is_never_published(tmp_path):
    import json
    from letf.account_stress import write_evidence_json
    path=tmp_path/'valid.json';write_evidence_json(path,{'wealth':1000.,'bracket_ceiling':float('inf')})
    assert json.loads(path.read_text())['wealth']==1000.
    assert isinstance(json.loads(path.read_text())['bracket_ceiling'],str)
    with pytest.raises(ValueError,match='NaN'):write_evidence_json(tmp_path/'invalid.json',{'wealth':float('nan')})
    assert not (tmp_path/'invalid.json').exists()
