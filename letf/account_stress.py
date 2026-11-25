"""Full daily canonical-ledger scenarios with dividends, flows and catastrophe.

These are unweighted deterministic stress replays, not a Monte Carlo probability
study. Never infer a crash frequency from how many named scenarios extinguish.
"""
from dataclasses import replace
from datetime import date
from hashlib import sha256
import argparse,json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest,equity_calendar
from letf.account_cli import _serialize
from letf.accuracy import UnsupportedModelError
from letf.macro.artifacts import canonical
from letf.macro.precision_run import file_hash,json_safe
from letf.macro.stress import CASES,frozen_anchor,stressed_context,scenario_market


def write_evidence_json(path,value):
    # Unbounded statutory bracket edges are legitimate metadata, not nonfinite
    # wealth. Serialize before publishing a file so invalid numeric results
    # cannot leave an apparently committed partial result.
    blob=json.dumps(json_safe(value),default=_serialize,sort_keys=True,allow_nan=False).encode()
    with Path(path).open('xb') as stream:stream.write(blob)


def independent_account_audit(result,initial):
    """Cash/terminal wealth and flow-neutral returns from exported events/rows.

    No calls to ledger valuation or tax computation. Tax law itself has separate
    authoritative examples; this verifies transaction and performance identities.
    """
    cash=initial;quantities={};terminal_receivable=result['ending_distribution_receivables']
    for event in result['ledger'].events:
        if event.get('account')!='main':continue
        action=event['action']
        if action in ('BUY','SELL'):
            sign=-1 if action=='BUY' else 1
            cash+=sign*event['shares']*event['price']-event['commission']
            symbol=event['security'];quantities[symbol]=quantities.get(symbol,0.)-sign*event['shares']
        elif action in ('INCOME','DISTRIBUTION_CASH','DEPOSIT','ROTH_CONTRIBUTION','CONVERSION','NON_ROTH_PLAN_ROLLOVER'):cash+=event['amount']
        elif action in ('WITHDRAWAL','ROTH_WITHDRAWAL','TAX_PAYMENT'):cash-=event['amount']
    net=cash+terminal_receivable-result['terminal_tax_payable']
    if max(map(abs,quantities.values()),default=0.)>1e-7:raise AssertionError('Terminal positions not liquidated')
    error=abs(net-result['ending_wealth_net_payable'])
    if error>max(1e-7,abs(net)*1e-11):raise AssertionError('Independent terminal cash identity failed')
    units=initial;extinguished=False;values=[]
    for row in result['reconciliation'].itertuples():
        for flow in row.flow_valuations:
            if flow['before_nav']>0:units*=(flow['before_nav']+flow['delta'])/flow['before_nav']
            else:extinguished=True
        if not np.isclose(units,row.units,rtol=1e-12,atol=1e-8):raise AssertionError('External-flow units do not reconcile')
        values.append(0. if extinguished else row.nav/units*initial)
    values[-1]=0. if extinguished else net/units*initial
    curve=np.r_[initial,values];peaks=np.maximum.accumulate(curve);dd=1-curve/peaks
    worst=float(dd.max());length=run=0
    for d in dd:
        run=run+1 if d>1e-12 else 0;length=max(length,run)
    if abs(worst-result['after_tax_metrics']['max_drawdown'])>1e-11:raise AssertionError('Independent flow-neutral drawdown failed')
    if length!=result['after_tax_metrics']['max_underwater_sessions']:raise AssertionError('Independent recovery clock failed')
    return {'terminal_net_identity_error':error,'flow_neutral_max_drawdown':worst,
            'max_underwater_sessions':length,'passed':True,'scope':'Independent exported-event and unitization arithmetic; not independent forecast or full tax-law certification'}


def replay_scenario(scenario,account,*,initial=10000.,annual_contribution=3000.,allocation=.8,
                    rebalance_sessions=63,ordinary_income=100000.,spread_bps=2.,slippage_bps=1.,
                    withdrawals=(),history_config=None):
    if account not in ('taxable','roth') or not 0<=allocation<=1:
        raise UnsupportedModelError('Generic cash-account allocations only; no borrowed buying power')
    market=scenario.market;dates=market.index;symbol=scenario.manifest['symbol']
    calendar=equity_calendar(dates[0].date(),dates[-1].date())
    flows=[]
    for year in sorted(set(dates.year)):
        day=dates[dates.year==year][0]
        if annual_contribution:
            flow={'date':str(day.date()),'kind':'contribution' if account=='roth' else 'deposit','amount':annual_contribution}
            if account=='roth':flow.update(magi=ordinary_income,compensation=ordinary_income)
            flows.append(flow)
    flows.extend(withdrawals)
    payments={y:calendar.date_to_session(pd.Timestamp(date(y+1,4,15)),direction='next').date()
              for y in range(dates[0].year,dates[-1].year)}
    result=run_account_backtest(market,[Security(symbol)],
        replace(TAXABLE_CASH if account=='taxable' else CONSERVATIVE_ROTH,spread_bps=spread_bps,slippage_bps=slippage_bps),
        initial_capital=initial,allocator=lambda past,weights:{symbol:allocation},price_convention='scenario_nominal',
        distributions_complete=True,distributions=scenario.distributions,
        rebalance_every=rebalance_sessions,settlement_policy='us_equity',terminal_liquidation=True,
        tax_inputs={'ordinary_income':ordinary_income,'federal_only':True,'no_external_wash_replacements_after_end':True},
        tax_payment_dates=payments,freeze_tax_year=2026,freeze_roth_contribution_year=2026,
        cash_flows=flows,roth_history_config=history_config or {'birth_date':'1980-01-01','first_tax_year':2010,'regular_basis':initial},
        liquidation_events=scenario.liquidation_events,inflation_index=scenario.inflation,
        source_manifest=scenario.manifest,audit_detail=True)
    audit=independent_account_audit(result,initial)
    r=result['reconciliation'];events=result['ledger'].events
    summary={'ending_net_assets':result['ending_wealth_net_payable'],
        'real_ending_net_assets':result['ending_wealth_net_payable']/scenario.inflation.iloc[-1],
        'metrics':result['after_tax_metrics'],'drawdown_convention':result['drawdown_convention'],
        'tax_paid':result['tax_paid'],'terminal_tax_payable':result['terminal_tax_payable'],
        'trading_costs':result['trading_costs'],'contributions':sum(f['amount'] for f in flows if f['kind']!='withdrawal'),
        'withdrawals':sum(f['amount'] for f in flows if f['kind']=='withdrawal'),
        'fund_closures':sum(e['action']=='FUND_CLOSURE' for e in events),
        'unpaid_distributions_written_off':sum(e['amount'] for e in events if e['action']=='DISTRIBUTION_DEFAULT'),
        'tax_funding_sales':sum(e.get('reason')=='TAX_FUNDING' for e in events),
        'total_incremental_investment_tax_liability':sum(t['tax'] for t in result['tax_rows']),
        'maximum_cash_residual':float(r.cash_residual.abs().max()),'minimum_cash':float(r.cash.min()),
        'maximum_nav_residual':float(r.nav_residual.abs().max()),'maximum_quantity_residual':float(r.quantity_residual.abs().max()),
        'independent_audit':audit,'annual_tax_rows':result['tax_rows'],'investment_certification':'BLOCKED'}
    return summary,result


def run_study(experiment,protocol,output,*,cases=None,anchors=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    selected=cases or protocol['cases'];chosen=protocol['anchor_paths'] if anchors is None else protocol['anchor_paths'][:anchors]
    if set(selected)-set(protocol['cases']) or not chosen:raise ValueError('Only preregistered cases/anchors may be selected')
    expected=json.loads(Path(__file__).parents[1].joinpath('docs/financial-realism-2026-11-25/protocol.json').read_text())
    if protocol!=expected:raise UnsupportedModelError('This version executes its registered scenario definitions only; changes require a new explicit protocol version')
    profile=protocol['account_profile']
    contexts=[frozen_anchor(experiment,a,years=max(protocol['horizons'])) for a in chosen]
    rows=[];inputs=[]
    for ai,anchor in enumerate(contexts):
        for case in selected:
            context=stressed_context(anchor,case)
            for leverage in protocol['leverages']:
                scenario=scenario_market(context,leverage);stem=f'anchor{ai}-{case}-{leverage}x'
                source_dir=output/stem;source_dir.mkdir();scenario.market.to_csv(source_dir/'prices.csv')
                scenario.inflation.to_csv(source_dir/'inflation.csv');scenario.fund_mechanics.to_csv(source_dir/'fund-mechanics.csv')
                (source_dir/'actions.json').write_text(json.dumps({'distributions':[vars(d) for d in scenario.distributions],
                    'liquidations':scenario.liquidation_events},default=_serialize,sort_keys=True))
                inputs.append({'id':stem,'manifest':scenario.manifest,'files':{p.name:file_hash(p) for p in source_dir.iterdir()}})
                for h in protocol['horizons']:
                    end=pd.Timestamp(year=2026+h,month=12,day=31);mask=scenario.market.index<=end
                    from dataclasses import replace as with_fields
                    sample=with_fields(scenario,market=scenario.market.loc[mask],inflation=scenario.inflation.loc[mask],
                        distributions=[d for d in scenario.distributions if d.pay_date<=end.date()],
                        liquidation_events=[e for e in scenario.liquidation_events if pd.Timestamp(e['date'])<=end])
                    for account in ('taxable','roth'):
                        directory=source_dir/f'{h}y-{account}';directory.mkdir()
                        try:
                            summary,result=replay_scenario(sample,account,initial=profile['initial'],
                                annual_contribution=profile['annual_contribution'],allocation=profile['risky_allocation'],
                                rebalance_sessions=profile['rebalance_sessions'],ordinary_income=profile['ordinary_income'],
                                spread_bps=profile['spread_bps'],slippage_bps=profile['slippage_bps'])
                            result['reconciliation'].to_csv(directory/'daily-account.csv')
                            pd.DataFrame({'nav':result['equity'],'unit_nav':result['unit_equity'],
                                'terminal_net_unit_nav':result['terminal_net_unit_equity']}).to_csv(directory/'wealth.csv')
                            (directory/'events.json').write_text(json.dumps(result['ledger'].events,default=_serialize,sort_keys=True))
                            row={'status':'EXECUTED_UNAPPROVED','summary':summary}
                        except (ValueError,UnsupportedModelError,AssertionError) as error:
                            row={'status':'BLOCKED','error_type':type(error).__name__,'error':str(error)}
                        row.update(anchor=protocol['anchor_paths'][ai],case=case,leverage=leverage,horizon=h,account=account,
                            exports={p.name:file_hash(p) for p in directory.iterdir()})
                        write_evidence_json(directory/'result.json',row)
                        rows.append(row);print('canonical stress replay',stem,h,account,row['status'],flush=True)
    report={'evidence':'HYPOTHETICAL_STRESS','model_approved':False,'event_probabilities':'UNSPECIFIED',
        'execution_scope':{'selected_cases':selected,'selected_anchors':chosen,
            'complete_registered_matrix':selected==protocol['cases'] and chosen==protocol['anchor_paths']},
        'purpose':'Unweighted scenario sensitivity; repeated old anchor paths, not new MC evidence',
        'protocol':protocol,'protocol_sha256':sha256(canonical(protocol)).hexdigest(),
        'source_registration_sha256':file_hash(Path(experiment)/'registration.json'),
        'all_account_reconciliations_passed':all(r['status']=='EXECUTED_UNAPPROVED' for r in rows),
        'inputs':inputs,'results':rows,'uncertainty':{
            'sampling':'No new numerical precision or probabilities inferred from these deterministic cases; prior fixed-model study retained',
            'model_and_regime':'Named cases/anchors illustrate dependence; no weights or statistical coverage',
            'costs':'Explicit dated-interval fee and borrowing assumptions, not actual swap invoices',
            'future_tax_policy':'Frozen 2026 federal tax and contribution eligibility; not enacted future law',
            'structure':'Corporate zero-residual failure independent of benchmark; actual issuer closure/relaunch policy unknown'},
        'limitations':['Ordinary distributions are hypothetical; not actual fund tax predictions',
            'Projected XNYS calendar, assumed daily execution, no unexpected closure tape',
            'Drawdowns use taxes paid plus terminal payable; not daily tax-on-liquidation valuations',
            'No certified investment rankings, margin, derivatives or advanced-tax support implied']}
    write_evidence_json(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('experiment','protocol','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=CASES);parser.add_argument('--anchors',type=int,choices=(1,2))
    a=parser.parse_args();r=run_study(a.experiment,json.loads(a.protocol.read_text()),a.output,cases=a.cases,anchors=a.anchors)
    print(json.dumps({'runs':len(r['results']),'all_reconciled':r['all_account_reconciliations_passed'],'model_approved':False}))
