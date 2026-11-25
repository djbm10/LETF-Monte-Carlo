import numpy as np
import pytest
from letf.account_projection import terminal_replay,CanonicalTerminalMap
from letf.accuracy import UnsupportedModelError


@pytest.mark.parametrize('account',['taxable','roth'])
def test_terminal_map_matches_independent_ledger_at_every_tax_boundary_and_extreme(account):
    m=CanonicalTerminalMap.build(account,.00001,1e5,horizon=30)
    rng=np.random.default_rng(343);nav=np.exp(rng.uniform(-10,11,100))
    net,tax=m.evaluate(nav)
    for i,v in enumerate(nav):
        r=terminal_replay(account,float(v),horizon=30)
        assert net[i]==pytest.approx(r['net'],rel=2e-13,abs=1e-7)
        assert tax[i]==pytest.approx(r['tax'],rel=2e-13,abs=1e-7)
        assert r['positions']==0 and r['gross']==pytest.approx(r['cash']+r['unsettled'])
        if account=='roth':assert tax[i]==0
    with pytest.raises(UnsupportedModelError):m.evaluate([1e6])


def test_lookup_scope_rejects_unsupported_dividend_tax_flow_and_zero_liquidation_scenarios():
    for kwargs in ({'distributions':.01},{'contributions':100.},{'state':'MA'},{'tax_year':2025}):
        with pytest.raises(UnsupportedModelError):terminal_replay('taxable',1.,**kwargs)
    with pytest.raises(UnsupportedModelError):terminal_replay('taxable',0.)
    loss=terminal_replay('taxable',.5)
    assert loss['tax']==pytest.approx(-660.)


def test_terminal_projection_reconciles_to_full_daily_canonical_engine():
    import pandas as pd
    import exchange_calendars as xcals
    from dataclasses import replace
    from letf.account_backtest import run_account_backtest
    from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
    idx=xcals.get_calendar('XNYS',start='2027-01-01',end='2032-02-01').sessions_in_range('2027-01-04','2031-12-31').tz_localize(None)
    calendar=xcals.get_calendar('XNYS',start='2027-01-01',end='2032-02-01')
    for nav in (.1,1.,10.):
        closes=np.geomspace(1.,nav,len(idx));opens=np.r_[1.,closes[:-1]]
        market=pd.DataFrame({'US_CF_Open':opens,'US_CF_Close':closes},index=idx)
        for account,profile in [('taxable',TAXABLE_CASH),('roth',CONSERVATIVE_ROTH)]:
            r=run_account_backtest(market,[Security('US_CF')],replace(profile,spread_bps=2.,slippage_bps=1.),
                initial_capital=10000.,allocator=lambda h,w:{'US_CF':1.},price_convention='scenario_nominal',
                source_manifest={'data_status':'HYPOTHETICAL_STRESS'},distributions_complete=True,terminal_liquidation=True,
                settlement_policy='us_equity',rebalance_every=len(idx)+1,freeze_tax_year=2026,
                tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True},
                tax_payment_dates={y:calendar.date_to_session(pd.Timestamp(f'{y+1}-04-15'),direction='next').date() for y in range(2027,2031)})
            reference=terminal_replay(account,nav,horizon=5)
            assert r['ending_wealth_net_payable']==pytest.approx(reference['net'],abs=1e-8)
            assert r['terminal_tax_payable']==pytest.approx(reference['tax'],abs=1e-8)
            actual=[e['settlement_date'] for e in r['ledger'].events if e['action'] in ('BUY','SELL')]
            assert actual==[e['settlement_date'] for e in reference['events'] if e['action'] in ('BUY','SELL')]
def test_account_interval_family_covers_possible_upstream_stopping_looks(tmp_path):
    """A passed upstream stop is data-selected, not an independent fixed look."""
    import json
    from letf.macro.account_precision import project_accounts
    from letf.macro.precision_run import file_hash
    protocol={'horizons':[5],'leverages':[1],'checkpoints_per_seed':[4,8],
              'family_alpha_quantiles':.025,'family_alpha_probabilities':.025}
    identities=[]
    for seed in (1,2):
        values=np.zeros((8,1,5));values[:,0,0]=np.linspace(-1,1,8)+seed*.01
        values[:,0,1]=.6;values[:,0,3]=.7;values[:,0,4]=values[:,0,0]-.2
        path=tmp_path/f'{seed}.bin';path.write_bytes(values.astype('<f8').tobytes())
        identities.append({'path':path.name,'sha256':file_hash(path),'dtype':'<f8','shape':[8,1,5]})
    # Deliberately small synthetic fixture; no claim this market gate was passed
    # in real data. Only downstream family allocation is under test.
    upstream=tmp_path/'market.json';upstream.write_text(json.dumps({'passed':True,
        'registration':{'protocol':protocol,'source_catalog_sha256':'a'*64},'samples':identities}))
    report=project_accounts(upstream,output=tmp_path/'accounts.json')
    qalphas=[q['alpha'] for row in report['results'] for intervals in row['log_asset_sampling_intervals'].values() for q in intervals]
    palphas=[q['alpha'] for row in report['results'] for q in row['probabilities_conditional'].values()]
    assert sum(qalphas)*2==pytest.approx(.025)
    assert sum(palphas)*2==pytest.approx(.025)
    assert report['family_error_control']['possible_upstream_stopping_looks']==2

