"""Composed real-fund portfolios through the single canonical daily ledger.

This is retrospective research on reviewed daily inputs, not observed trade
fills, a full household return, or forecasting certification. Pairing means
alternative accounts with common policies, not simultaneous owner accounts.
"""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from letf.accounts import Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest, equity_calendar
from letf.account_cli import load_bundle, _serialize, _report_safe
from letf.account_stress import independent_account_audit
from letf.accuracy import UnsupportedModelError
from letf.lab.catalog import file_hash, private_bytes
from letf.lab.evidence import evidence
from letf.lab.strategies import make_allocator

KEYS={'schema','mode','bundles','start','end','strategy','initial_capital','accounts','tax',
      'cash_flows','roth_owner','rebalance_sessions','costs','cash_sweep_annual_rate','signals'}


def validate_request(request):
    if set(request)-KEYS: raise ValueError('Unknown laboratory input fields: '+','.join(sorted(set(request)-KEYS)))
    if request.get('schema')!=1 or request.get('mode')not in ('observed','fixture'): raise ValueError('Expected schema1/observed or explicitly labeled fixture request')
    if not request.get('bundles') or not isinstance(request['bundles'],dict): raise ValueError('Named reviewed input bundles required')
    if request.get('start') is None or request.get('end') is None: raise ValueError('Explicit common start/end required')
    if pd.Timestamp(request['start'])>=pd.Timestamp(request['end']): raise ValueError('Start must precede end')
    accounts=request.get('accounts',['taxable','roth'])
    if not accounts or len(accounts)!=len(set(accounts)) or set(accounts)-{'taxable','roth'}: raise ValueError('Choose unique taxable/roth alternatives')
    capital=request.get('initial_capital',10000.)
    if not np.isfinite(capital) or capital<=0: raise ValueError('Positive finite capital required')
    tax=request.get('tax',{})
    allowed={'state','ordinary_income','ordinary_income_by_year','filing_status','law_mode','freeze_year','external_wash_policy'}
    if set(tax)-allowed: raise ValueError('Unknown household tax inputs')
    if 'taxable' in accounts and (not {'state','ordinary_income','filing_status','law_mode','external_wash_policy'}<=set(tax)):
        raise ValueError('Explicit dated tax, income, state, filing and outside-wash assumptions required')
    if tax.get('law_mode','ACTUAL_DATED') not in ('ACTUAL_DATED','MODERN_LAW_COUNTERFACTUAL'):
        raise UnsupportedModelError('Actual data replay requires dated law or explicit modern-law counterfactual')
    if tax.get('law_mode','ACTUAL_DATED')=='ACTUAL_DATED' and tax.get('freeze_year') is not None:
        raise ValueError('Frozen law cannot be labeled actual historical law')
    if tax.get('law_mode')=='MODERN_LAW_COUNTERFACTUAL' and type(tax.get('freeze_year')) is not int:
        raise ValueError('Explicit enacted frozen law year required')
    if 'taxable' in accounts and tax['external_wash_policy']!='NO_OTHER_OWNER_REPLACEMENTS':
        raise UnsupportedModelError('Linked owner accounts need an explicitly reconciled household run; no silent outside wash assumption')
    if request.get('cash_flows') and 'roth' in accounts and not request.get('roth_owner'):
        raise ValueError('Roth flows require explicit owner birth, clock and basis history')
    for flow in request.get('cash_flows',[]):
        if flow.get('kind') not in ('contribution','deposit','withdrawal'):
            raise UnsupportedModelError('Personal conversion/rollover/earnings taxes require an external funding schedule')
    for key in ('rebalance_sessions',):
        if key in request and (type(request[key])is not int or request[key]<1): raise ValueError('Positive integer rebalance interval required')
    rate=request.get('cash_sweep_annual_rate',0.)
    if not np.isfinite(rate) or rate<0: raise ValueError('Nonnegative explicit hypothetical cash sweep APR required')
    costs=request.get('costs',{})
    if set(costs)-{'spread_bps','slippage_bps','commission','fractional_shares'}: raise ValueError('Unknown cost/profile fields')
    return accounts


def load_portfolio(request):
    rows=[];manifests={};distributions=[];splits=[];cpis=[];securities=[]
    state=request.get('tax',{}).get('state','FEDERAL_ONLY')
    for symbol,directory in sorted(request['bundles'].items()):
        market,manifest,dist,cpi=load_bundle(directory,allow_fixture=request['mode']=='fixture')
        if request['mode']=='fixture' and manifest['data_status']!='SOFTWARE_FIXTURE':
            raise ValueError('Fixture mode requires explicit fixture source labels')
        if symbol!=manifest['symbol']: raise ValueError('Input symbol and manifest disagree')
        if state not in manifest.get('supported_tax_states',[]):
            raise UnsupportedModelError(f'{symbol}: issuer distribution/state adjustments not reviewed for {state}')
        if (request['start']<manifest['account_coverage_start'] or request['end']>manifest['account_coverage_end']):
            raise UnsupportedModelError(f'{symbol}: requested dates exceed reviewed distribution coverage')
        # No incomplete prior-year distribution warmup. A strategy may remain
        # cash until enough genuinely reviewed observations accumulate.
        first=manifest.get('signal_coverage_start',manifest['account_coverage_start'])
        sample=market.loc[first:request['end']]
        if sample.empty: raise ValueError('Empty reviewed source interval')
        rows.append(sample)
        if cpi is not None: cpis.append(cpi)
        securities.append(Security(symbol,kind=manifest.get('instrument_kind','etf'),
                                   tax_structure=manifest.get('tax_structure','corporation')))
        distributions.extend(dist);splits.extend(manifest.get('split_events',[]))
        manifests[symbol]={'manifest_sha256':file_hash(Path(directory)/'manifest.json'),'manifest':manifest}
    left=max(f.index[0] for f in rows);right=min(f.index[-1] for f in rows)
    selected=[f.loc[left:right] for f in rows]
    if any(not f.index.equals(selected[0].index) for f in selected):
        raise UnsupportedModelError('Joint source calendars disagree; missing prices cannot be dropped or filled')
    market=pd.concat(selected,axis=1)
    if pd.Timestamp(request['start'])not in market.index or pd.Timestamp(request['end'])not in market.index:
        raise UnsupportedModelError('Choose actual common source trading-session endpoints')
    cpi=None
    for series in cpis:
        candidate=series.reindex(market.index)
        endpoints=candidate.reindex(pd.to_datetime([request['start'],request['end']]))
        if not np.isfinite(endpoints).all() or (candidate.dropna()<=0).any() or np.isinf(candidate).any():
            raise UnsupportedModelError('Incomplete/invalid observed CPI endpoints; no purchasing-power result can be inferred')
        if cpi is not None and not np.array_equal(candidate.to_numpy(),cpi.to_numpy(),equal_nan=True):
            raise UnsupportedModelError('Conflicting inflation vintages cannot be mixed in a comparison')
        cpi=candidate
    for name,item in request.get('signals',{}).items():
        if name not in ('VIX','RF_DAILY'): raise UnsupportedModelError('Unregistered decision signal')
        if not {'path','sha256','provider','url','column','availability'}<=set(item) or item['availability']!='DAILY_CLOSE':
            raise ValueError('Signal needs reviewed hash/provider/column and daily-close availability')
        if file_hash(item['path'])!=item['sha256']: raise ValueError('Signal checksum mismatch')
        signal=pd.read_csv(item['path'],index_col=0,parse_dates=True)
        if not signal.index.is_unique or not signal.index.is_monotonic_increasing: raise ValueError('Invalid signal chronology')
        values=pd.to_numeric(signal[item['column']],errors='raise').reindex(market.index)
        if not np.isfinite(values).all() or (name=='VIX' and (values<=0).any()):
            raise UnsupportedModelError('Decision signal has missing/nonfinite observations; no fill or backdating')
        market[name]=values
    return market,securities,distributions,splits,cpi,manifests


def fills(result):
    return [{k:e[k] for k in ('date','action','security','shares','price','commission','reason')}
            for e in result['ledger'].events if e['action']in ('BUY','SELL')]


def compare_control_trades(trade_lists):
    """Exact decisions/prices; compensated lot sums permit tiny share roundoff.

    1e-10 shares and $1e-8 are stricter than canonical quantity/cash residual
    limits (1e-7 shares/$1e-6). Never use relative tolerance on large wealth.
    """
    reference=trade_lists[0];share_error=dollar_error=0.
    exact=True
    for other in trade_lists[1:]:
        if len(reference)!=len(other):raise AssertionError('Pre-tax control trade counts differ')
        for a,b in zip(reference,other):
            if any(a[k]!=b[k] for k in ('date','action','security','price','commission','reason')):
                raise AssertionError('Pre-tax control decisions/execution prices differ')
            diff=abs(a['shares']-b['shares']);share_error=max(share_error,diff)
            dollar_error=max(dollar_error,diff*a['price']);exact=exact and a==b
    if share_error>1e-10 or dollar_error>1e-8:
        raise AssertionError('Pre-tax control financial quantities differ beyond canonical-roundoff allowance')
    return {'passed':True,'bitwise_identical':exact,'max_share_difference':share_error,
            'max_notional_difference':dollar_error,'absolute_share_tolerance':1e-10,'absolute_notional_tolerance':1e-8}


def run_observed(request, *, output, progress=None):
    if Path(output).exists():raise FileExistsError('Laboratory runs cannot overwrite existing exports')
    accounts=validate_request(request)
    market,securities,distributions,splits,cpi,manifests=load_portfolio(request)
    _,required,strategy_meta,default_rebalance=make_allocator(request['strategy'])
    if required-set(request['bundles']): raise UnsupportedModelError('Strategy missing reviewed instruments: '+','.join(sorted(required-set(request['bundles']))))
    tax=request.get('tax',{});capital=request.get('initial_capital',10000.)
    tax_year=tax.get('freeze_year');law=tax.get('law_mode','ACTUAL_DATED')
    calendar=equity_calendar(pd.Timestamp(request['start']).date(),pd.Timestamp(request['end']).date())
    payments={year:calendar.date_to_session(pd.Timestamp(date(year+1,4,15)),direction='previous').date()
              for year in range(pd.Timestamp(request['start']).year,pd.Timestamp(request['end']).year)}
    costs=dict(spread_bps=2.,slippage_bps=1.,commission=0.,fractional_shares=True)
    costs.update(request.get('costs',{}))
    results={};controls={};summaries={};completed=0
    for account in accounts:
        for apply_taxes in (False,True):
            def callback(step,total,day):
                if progress:progress({'account':account,'tax_control':not apply_taxes,'session':step,'total_sessions':total,
                                       'date':str(day.date()),'completed_replays':completed,'total_replays':2*len(accounts)})
            flow_records=[]
            for original in request.get('cash_flows',[]):
                flow=dict(original)
                if flow['kind']in ('deposit','contribution'): flow['kind']='contribution' if account=='roth' else 'deposit'
                flow.setdefault('filing_status',tax.get('filing_status','single'))
                flow_records.append(flow)
            allocator,_,_,_=make_allocator(request['strategy'])
            result=run_account_backtest(market,securities,
                replace(TAXABLE_CASH if account=='taxable' else CONSERVATIVE_ROTH,**costs),
                initial_capital=capital,allocator=allocator,price_convention='as_traded',
                distributions_complete=True,distributions=distributions,split_events=splits,
                trade_start=request['start'],rebalance_every=request.get('rebalance_sessions',default_rebalance),
                settlement_policy='us_equity',terminal_liquidation=True,apply_taxes=apply_taxes,
                sweep_annual_rates=pd.Series(request.get('cash_sweep_annual_rate',0.),index=market.index),
                tax_inputs={'ordinary_income':tax.get('ordinary_income',0.),'ordinary_income_by_year':{
                    int(y):v for y,v in tax.get('ordinary_income_by_year',{}).items()},
                    'filing_status':tax.get('filing_status','single'),'state':tax.get('state','FEDERAL_ONLY'),
                    'federal_only':tax.get('state','FEDERAL_ONLY')=='FEDERAL_ONLY',
                    'no_external_wash_replacements_after_end':True},tax_payment_dates=payments,
                freeze_tax_year=tax_year,freeze_roth_contribution_year=tax_year if law!='ACTUAL_DATED' else None,
                cash_flows=flow_records,roth_history_config=request.get('roth_owner'),
                inflation_index=cpi,qualified_holding_policy='track_unhedged_common',audit_detail=True,
                source_manifest={'bundles':manifests},progress_callback=callback if progress else None)
            completed+=1
            independent=independent_account_audit(result,capital)
            if not apply_taxes:controls[account]=result;continue
            results[account]=result
            reconciliation=result['reconciliation']
            summaries[account]={'ending_wealth_net_payable':result['ending_wealth_net_payable'],
                'tax_paid':result['tax_paid'],'terminal_tax_payable':result['terminal_tax_payable'],
                'tax_rows':result['tax_rows'],'metrics':result['after_tax_metrics'],
                'explicit_trading_costs':result['trading_costs'],'internal_fund_costs':'Embedded in observed prices; no duplicate deduction',
                'independent_audit':independent,'trade_count':len(fills(result)),
                'max_cash_residual':float(reconciliation.cash_residual.abs().max()),
                'max_nav_residual':float(reconciliation.nav_residual.abs().max()),
                'max_quantity_residual':float(reconciliation.quantity_residual.abs().max())}
            pi_ratio=(cpi.loc[request['end']]/cpi.loc[request['start']]) if cpi is not None else None
            summaries[account]['real_ending_wealth']=result['ending_wealth_net_payable']/pi_ratio if pi_ratio is not None else None
            summaries[account]['real_money_weighted_return']=None
            if result['external_flows_investor'] and cpi is not None:
                from letf.performance import xirr
                flows=[(pd.Timestamp(request['start']),-capital),*result['external_flows_investor'],
                       (pd.Timestamp(request['end']),result['ending_wealth_net_payable'])]
                levels=cpi.reindex(pd.to_datetime([d for d,v in flows]))
                if np.isfinite(levels).all():
                    try:summaries[account]['real_money_weighted_return']=xirr([
                        (d,v*cpi.loc[request['start']]/level) for (d,v),level in zip(flows,levels)])
                    except UnsupportedModelError as error:summaries[account]['real_IRR_status']='BLOCKED: '+str(error)
                else:summaries[account]['real_IRR_status']='BLOCKED: missing observed CPI at cash-flow dates; no interpolation'
    control_comparison=compare_control_trades([fills(v) for v in controls.values()])
    identical=control_comparison['passed']
    repository=Path(__file__).parents[2]
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repository,text=True).strip())
    except (OSError,subprocess.CalledProcessError):commit='UNKNOWN';dirty=None
    classification=('SOFTWARE_FIXTURE' if request['mode']=='fixture' or any(m['manifest'].get('classification')=='SOFTWARE_FIXTURE' for m in manifests.values())
                    else 'OBSERVED_MARKET' if law=='ACTUAL_DATED' else 'HISTORICAL_COUNTERFACTUAL')
    report={**evidence(classification,limitations=[
        'Retrospective current-vintage market inputs, not independently certified decision-time quotes',
        'Next-open hypothetical fills; personal income, costs, outside-wash and cash-yield assumptions',
        'Regular federal/NIIT and supported state subset; no AMT/credits/aged-blind-dependent worksheet',
        'Annual modeled payment on exchange session on/before April15; sufficient external salary withholding assumed',
        'Alternative accounts, not concurrent owner trades; terminal internal Roth sale is not a personal distribution',
        'Drawdowns use cash taxes paid and final payable, not daily hypothetical liquidation-tax accrual']),
        'schema':1,'mode':request['mode'],'request':request,'source_manifests':manifests,'strategy':strategy_meta,
        'implementation_commit':commit,'python':platform.python_version(),'law_mode':law,'frozen_tax_year':tax_year,
        'implementation_worktree_dirty':dirty,
        'implementation_source_sha256':{str(p.relative_to(repository)):file_hash(p) for p in sorted((repository/'letf').rglob('*.py'))},
        'dependency_lock_sha256':file_hash(repository/'requirements-audit.lock'),
        'same_pre_tax_policy_and_fills':identical,'accounts':summaries,
        'pre_tax_control_comparison':control_comparison,
        'cash_sweep_yield_evidence':'Explicit constant APR assumption, not observed broker/T-bill yields',
        'cash_flow_policy':'Common dated flows; cash-flow IRR separate from lump-sum CAGR',
        'inflation_scope':{'convention':'Observed monthly endpoints for purchasing power, never a uniquely determined daily CPI path',
                           'missing_interior_sessions':int(cpi.isna().sum()) if cpi is not None else None},
        'sampling_uncertainty':'NOT_APPLICABLE: deterministic retrospective replay',
        'comparison_fingerprint_fields':['source_manifests','request','strategy','law_mode']}
    safe=_report_safe(report)
    encoded=json.dumps(safe,default=_serialize,sort_keys=True,allow_nan=False,indent=2).encode()
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    artifacts={}
    for account,result in results.items():
        exports={account+'_daily.csv':result['reconciliation'].to_csv().encode(),
                 account+'_curve.csv':pd.DataFrame({'wealth':result['net_payable_equity'],
                     'flow_neutral_wealth':result['terminal_net_unit_equity']}).to_csv().encode(),
                 account+'_events.json':json.dumps(result['ledger'].events,default=_serialize,allow_nan=False).encode(),
                 account+'_pretax_fills.json':json.dumps(fills(controls[account]),default=_serialize,allow_nan=False).encode()}
        for name,blob in exports.items():
            path=output/name;private_bytes(path,blob);artifacts[name]=file_hash(path)
    private_bytes(output/'report.json',encoded)
    artifacts['report.json']=file_hash(output/'report.json')
    private_bytes(output/'export-manifest.json',json.dumps({'sha256':artifacts,'classification':classification},sort_keys=True).encode())
    return safe
