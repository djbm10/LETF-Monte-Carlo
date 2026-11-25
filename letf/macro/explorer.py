"""Macro-conditional future research routed through the canonical account book.

Monthly forecast distributions do not identify daily paths. A separately seeded
conditional Gaussian bridge supplies explicit hypothetical daily paths. Fund
cash distributions and future tax law are disclosed scenario assumptions.
"""
from dataclasses import replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest,Distribution
from letf.account_cli import _serialize
from letf.reconstruction import SourceRecord,daily_reset_returns
from letf.macro.data import train_slice
from letf.macro.models import MacroModel
from letf.macro.artifacts import ArtifactStore,fingerprint,canonical
from letf.macro.uncertainty import wilson_interval


STRATEGIES=('US1X_CF','US2X_CF','US3X_CF','EQUITY60_CASH40_CF')


def conditional_daily_bridge(monthly,columns,sessions,*,seed,daily_equity_vol,
                             monthly_covariance=None,crash_month=None,crash_return=None,
                             initial_short_rate=None,initial_long_yield=None):
    """Many hypothetical daily paths conditional on given monthly log totals.

    This is the Gaussian conditional bridge, not inferred historical daily data
    and not a post-hoc fit to an observed fund evaluation period.
    """
    rng=np.random.default_rng(seed);n=len(sessions);e=columns.index('equity_log');r=columns.index('short_rate')
    daily=np.zeros(n);rates=np.zeros(n);inflation=np.zeros(n);bond=np.zeros(n)
    pi=columns.index('inflation_log');yield_col=columns.index('long_yield') if 'long_yield' in columns else None
    periods=sessions.to_period('M');unique=periods.unique();previous_yield=(initial_long_yield if initial_long_yield is not None else monthly[0,yield_col]) if yield_col is not None else None
    for t,period in enumerate(unique):
        indices=np.flatnonzero(periods==period);count=len(indices)
        noise=rng.normal(0,daily_equity_vol,count);noise-=noise.mean()
        log_returns=monthly[t,e]/count+noise
        if crash_month==t:
            if crash_return is None or not -1<crash_return<0:raise ValueError('Crash return must be between -100% and zero')
            k=count//2;target=np.log1p(crash_return)
            remainder=np.delete(noise,k);remainder-=remainder.mean()
            log_returns[np.arange(count)!=k]=(monthly[t,e]-target)/(count-1)+remainder
            log_returns[k]=target
        daily[indices]=np.expm1(log_returns)
        # Last month's modeled reference rate applies until next month, not
        # today's unknown future rate. Month 1 is the model's initial state.
        rates[indices]=initial_short_rate if t==0 and initial_short_rate is not None else monthly[max(0,t-1),r]
        inflation[indices]=monthly[t,pi]/count
        if yield_col is not None:
            # Diagnostic only: a par-yield-as-flat-zero-curve model, no claim
            # that this is a Treasury ETF's executable/coupon total return.
            target=monthly[t,yield_col]
            increments=np.full(count,(target-previous_yield)/count)
            if monthly_covariance is not None:
                sig=np.sqrt(monthly_covariance[yield_col,yield_col]/21)
                rho=monthly_covariance[e,yield_col]/np.sqrt(monthly_covariance[e,e]*monthly_covariance[yield_col,yield_col])
                other=rng.normal(0,sig,count);other-=other.mean()
                yield_noise=rho*noise/max(daily_equity_vol,1e-12)*sig+np.sqrt(max(0,1-rho*rho))*other
                yield_noise-=yield_noise.mean();increments+=yield_noise
            for index,change in zip(indices,increments):
                new_yield=previous_yield+change
                dt=1/252 # diagnostic trading-period duration convention
                bond[index]=np.expm1(10*previous_yield-(10-dt)*new_yield)
                previous_yield=new_yield
    if not np.isfinite(daily).all() or not np.isfinite(rates).all():raise ValueError('Nonfinite generated scenario; no paths omitted')
    return pd.DataFrame({'equity_return':daily,'short_rate':rates,'inflation_log':inflation,
        'hypothetical_zero_bond_return':bond},index=sessions)


def scenario_market(daily,*,annual_fee,borrowing_spread,distribution_yield,source):
    if distribution_yield<0 or annual_fee<0 or borrowing_spread<0:raise ValueError('Nonnegative cost/distribution assumptions required')
    dates=daily.index;elapsed=dates.to_series().diff().dt.days.copy();elapsed.iloc[0]=1
    market=pd.DataFrame(index=dates);distributions=[];liquidations=[];costs={}
    # These are reference rates in nominal APR units, not observed invoices.
    rates=daily.short_rate
    quarters=dates.to_period('Q');pay_days={}
    for quarter in quarters.unique():
        ids=np.flatnonzero(quarters==quarter)
        if len(ids)>=3 and dates[ids[-1]].month%3==0:
            pay_days[dates[ids[-3]]]=dates[ids[-1]]
    for leverage in (1.,2.,3.):
        symbol=f'US{leverage:g}X_CF'
        r=daily_reset_returns(daily.equity_return,leverage,annual_fee=annual_fee if leverage>1 else 0.,
            funding_rate_at_previous_close=rates,borrowing_spread=borrowing_spread,
            elapsed_calendar_days=elapsed,source=source)
        opens=[];closes=[];div_cash=[];previous=100.;closed=False
        for t,day in enumerate(dates):
            if closed:opens.append(0.);closes.append(0.);div_cash.append(0.);continue
            # Market-price scenario with no observed overnight alpha. Accrued
            # nontrading-day internal costs enter opening NAV before fills.
            overnight_cost=(r.fund_fee.iloc[t]+r.financing.iloc[t])*max(0,elapsed.iloc[t]-1)/elapsed.iloc[t]
            opening=max(0.,previous*(1-overnight_cost))
            before_distribution=previous*(1+r['return'].iloc[t])
            amount=min(previous*distribution_yield/4,max(0.,before_distribution)*.5) if day in pay_days else 0.
            close=before_distribution-amount
            if close<=0 or opening<=0:
                close=0.;closed=True
                timing='open' if opening<=0 else 'close'
                if timing=='open':opening=0.
                liquidations.append({'date':str(day.date()),'security':symbol,'cash_per_share':0.,
                    'timing':timing,'source':'Explicit daily-reset fund-extinction stress convention, not observed issuer action'})
            if amount:
                distributions.append(Distribution(symbol,day.date(),pay_days[day].date(),amount,
                    'ordinary_dividend','Hypothetical quarterly nonqualified cash distribution policy'))
            opens.append(opening);closes.append(close);div_cash.append(amount);previous=close
        market[f'{symbol}_Open']=opens;market[f'{symbol}_Close']=closes
        costs[symbol]=pd.DataFrame({'fund_fee_rate':r.fund_fee,'financing_rate':r.financing},index=dates)
    return market,distributions,liquidations,costs


def _cost_attribution(result,market,costs):
    """Read held exposure from canonical fills; do not deduct embedded costs twice."""
    held=dict.fromkeys(costs,0.);fee=financing=0.
    events={}
    for e in result['ledger'].events:
        if e.get('account')=='main':events.setdefault(e['date'],[]).append(e)
    for timestamp in market.index:
        day=timestamp.date()
        # Open fills determine interval exposure. Close liquidations happen
        # after the interval; the next row starts from the canonical snapshot.
        for event in events.get(day,[]):
            symbol=event.get('security')
            if symbol not in held or event.get('reason') in ('TERMINAL_LIQUIDATION','FUND_LIQUIDATION'):continue
            if event['action']=='BUY':held[symbol]+=event['shares']
            if event['action']=='SELL':held[symbol]-=event['shares']
        for symbol in held:
            value=held[symbol]*market.loc[timestamp,f'{symbol}_Open']
            fee+=value*costs[symbol].loc[timestamp,'fund_fee_rate']
            financing+=value*costs[symbol].loc[timestamp,'financing_rate']
        for event in events.get(day,[]):
            if event.get('reason') in ('TERMINAL_LIQUIDATION','FUND_LIQUIDATION'):
                held[event['security']]-=event['shares']
    return {'fund_fee_dollars_approx':float(fee),'financing_dollars_approx':float(financing),
        'method':'Modeled internal rate charges times held opening NAV; overnight charges before entry/exit make this an attribution approximation, not a cash debit'}


def explore(sources,*,calibration_end,output,paths=16,horizons=(5,10,20,30,40),seed=20261124,
            model_type='conditional_var_t',initial_capital=10000.,annual_contribution=0.,monthly_withdrawal=0.,
            roth_birth_date=None,roth_first_tax_year=None,roth_regular_basis=None,
            roth_magi=None,annual_compensation=None,
            ordinary_income=100000.,state='FEDERAL_ONLY',filing_status='single',frozen_tax_year=2026,
            rate=None,inflation=None,production_growth=None,equity_drift=None,volatility_scale=1.,
            equity_cagr_constraint=None,
            stock_bond_correlation=None,crash_month=None,crash_return=None,
            annual_fee=.01,borrowing_spread=.015,distribution_yield=.01,strategies=STRATEGIES):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((Path(__file__).resolve().parents[2]/'docs/macro-learning-2026-11-24/protocol.json').read_text())
    horizons=sorted(set(horizons))
    if not horizons or any(type(y)is not int or not 1<=y<=40 for y in horizons):raise ValueError('Integer horizons 1-40 required')
    if set(strategies)-set(STRATEGIES):raise UnsupportedModelError('This explorer does not fabricate futures/options or unsupported portfolio instruments')
    if frozen_tax_year not in (2024,2025,2026):raise UnsupportedModelError('Choose a supported explicit frozen tax-law year')
    if annual_contribution<0 or monthly_withdrawal<0:raise ValueError('Nonnegative external flows required')
    roth_history=None
    if annual_contribution or monthly_withdrawal:
        if any(v is None for v in (roth_birth_date,roth_first_tax_year,roth_regular_basis)):
            raise UnsupportedModelError('Roth cash flows require explicit owner birth date, first Roth tax year and aggregate regular basis')
        roth_history={'birth_date':roth_birth_date,'first_tax_year':roth_first_tax_year,'regular_basis':roth_regular_basis}
    if annual_contribution and (roth_magi is None or annual_compensation is None):
        raise UnsupportedModelError('Roth contributions require separate supplied MAGI and eligible compensation; ordinary income alone is insufficient')
    panel,metadata=sources.panel(calibration_end,include_long_yield=stock_bond_correlation is not None)
    training=train_slice(panel,calibration_end,240,protocol['minimum_conditional_months'] if model_type=='conditional_var_t' else protocol['minimum_bootstrap_months'],allow_missing_transitions=True)
    protocol=dict(protocol,allow_incomplete_training=True)
    model=MacroModel.fit(model_type,training,protocol);columns=model.columns
    constraints={}
    if rate is not None:constraints['short_rate']=rate
    if equity_cagr_constraint is not None:
        if equity_cagr_constraint<=-1:raise ValueError('Constrained equity return must exceed -100%')
        constraints['equity_log']=np.log1p(equity_cagr_constraint)/12
    if inflation is not None:
        if inflation<=-1:raise ValueError('Inflation cannot extinguish the price level')
        constraints['inflation_log']=np.log1p(inflation)/12
    if production_growth is not None:
        if production_growth<=-1:raise ValueError('Production growth below -100% is invalid')
        constraints['production_log']=np.log1p(production_growth)/12
    shifts={}
    if equity_drift is not None:
        if equity_drift<=-1:raise ValueError('Invalid conditional equity drift assumption')
        shifts['equity_log']=np.log1p(equity_drift)/12-training.equity_log.mean()
    overrides={('equity_log','long_yield'):-stock_bond_correlation} if stock_bond_correlation is not None else {}
    schedule={crash_month:{'equity_log':np.log1p(crash_return)}} if crash_month is not None and crash_return is not None else {}
    if crash_month is not None and not 0<=crash_month<12*max(horizons):raise ValueError('Crash month outside simulation')
    gap=(pd.Period('2026-12',freq='M')-training.index[-1].to_period('M')).n
    if gap<0:raise UnsupportedModelError('Calibration includes observations after the scenario origin')
    initial=np.broadcast_to(model.train_values[-1],(paths,len(columns))).copy()
    if gap:initial=model.simulate(gap,paths,seed+98765)[:,-1]
    if rate is not None:initial[:,columns.index('short_rate')]=rate
    monthly=model.simulate(12*max(horizons),paths,seed,constraints=constraints,vol_multiplier=volatility_scale,
        intercept_shifts=shifts,correlation_overrides=overrides,constraint_schedule=schedule,initial_state=initial)
    store=ArtifactStore(output/'artifacts')
    model_id=store.put('models',{'kind':model_type,'parameters':model.parameters,'columns':columns,
        'training_sha256':fingerprint(training),'source_metadata':metadata}, {'training':model.train_values})
    forecast_id=store.put('forecasts',{'model_id':model_id,'seed':seed,'constraints':constraints,
        'intercept_shifts':shifts,'correlation_overrides':{str(k):v for k,v in overrides.items()},
        'volatility_scale':volatility_scale,'data_status':'HYPOTHETICAL_STRESS'}, {'monthly_paths':monthly})
    calendar=xcals.get_calendar('XNYS',start='2026-12-01',end=f'{2028+max(horizons)}-12-31')
    sessions=calendar.sessions[(calendar.sessions>=pd.Timestamp('2027-01-04')) & (calendar.sessions<pd.Timestamp(f'{2027+max(horizons)}-01-01'))].tz_localize(None)
    ff=sources.daily_factors();empirical=(ff['Mkt-RF']+ff.RF).loc[training.index[0]:training.index[-1]]
    daily_vol=float(np.log1p(empirical).std(ddof=1))*volatility_scale
    meta=sources.catalog['sources']['ff_daily']
    source=SourceRecord(meta['provider'],meta['url'],meta['retrieved_at'],str(meta['source_header']),meta['sha256'],
        str(empirical.index[0].date()),'USD','Hypothetical paths calibrated from observed daily factors','daily',False)
    covariance=model.innovation_covariance(vol_multiplier=volatility_scale,correlation_overrides=overrides) if model_type=='conditional_var_t' else None
    rows=[];diagnostics=[]
    for path_id in range(paths):
        daily=conditional_daily_bridge(monthly[path_id],columns,sessions,seed=seed+100000+path_id,
            daily_equity_vol=daily_vol,monthly_covariance=covariance,crash_month=crash_month,crash_return=crash_return,
            initial_short_rate=initial[path_id,columns.index('short_rate')],
            initial_long_yield=initial[path_id,columns.index('long_yield')] if 'long_yield' in columns else None)
        market,divs,liquidations,costs=scenario_market(daily,annual_fee=annual_fee,borrowing_spread=borrowing_spread,
            distribution_yield=distribution_yield,source=source)
        for horizon in horizons:
            sample=market.loc[market.index<pd.Timestamp(f'{2027+horizon}-01-01')]
            final=sample.index[-1];dates=sample.index
            cpi_ratio=float(np.exp(monthly[path_id,:12*horizon,columns.index('inflation_log')].sum()))
            cpi=pd.Series(np.nan,index=dates);cpi.iloc[0]=1.;cpi.iloc[-1]=cpi_ratio
            flow_dates=dates.to_series().groupby(dates.to_period('M')).first()
            flows=[]
            for d in flow_dates:
                if annual_contribution:flows.append({'date':str(d.date()),'kind':'contribution','amount':annual_contribution/12,'magi':roth_magi,'compensation':annual_compensation,'filing_status':filing_status})
                if monthly_withdrawal:flows.append({'date':str(d.date()),'kind':'withdrawal','amount':monthly_withdrawal})
            payments={y:calendar.date_to_session(pd.Timestamp(date(y+1,4,15)),direction='next').date() for y in range(2027,final.year)}
            cash_rates=(daily.short_rate-.005).clip(lower=0.).reindex(dates)
            selected_divs=[d for d in divs if d.pay_date<=final.date()]
            selected_liqs=[a for a in liquidations if pd.Timestamp(a['date'])<=final]
            for strategy in strategies:
                target={'US1X_CF':.6} if strategy=='EQUITY60_CASH40_CF' else {strategy:1.}
                for account,profile in [('taxable',TAXABLE_CASH),('roth',CONSERVATIVE_ROTH)]:
                    try:
                        result=run_account_backtest(sample,[Security(s,kind='etf') for s in costs],
                            replace(profile,spread_bps=2.,slippage_bps=1.),initial_capital=initial_capital,
                            allocator=lambda h,w,t=target:t,price_convention='scenario_nominal',
                            distributions_complete=True,distributions=selected_divs,liquidation_events=selected_liqs,
                            terminal_liquidation=True,settlement_policy='us_equity',rebalance_every=21,
                            tax_inputs={'ordinary_income':ordinary_income,'state':state,'federal_only':state=='FEDERAL_ONLY',
                                    'filing_status':filing_status,
                                'no_external_wash_replacements_after_end':True},tax_payment_dates=payments,
                            freeze_tax_year=frozen_tax_year,freeze_roth_contribution_year=frozen_tax_year,
                                cash_flows=flows,roth_history_config=roth_history,
                            sweep_annual_rates=cash_rates,inflation_index=cpi,
                            source_manifest={'data_status':'HYPOTHETICAL_STRESS','forecast_id':forecast_id,'daily_bridge_seed':seed+100000+path_id})
                    except Exception as exc:
                        (output/'blocked.json').write_text(json.dumps({'path':path_id,'horizon':horizon,'strategy':strategy,
                            'account':account,'reason':str(exc),'no_rankings_released':True},indent=2))
                        raise UnsupportedModelError('Canonical account scenario blocked: '+str(exc)) from exc
                    net=float(result['ending_wealth_net_payable'])
                    tracked=[a for a in selected_liqs if target.get(a['security'],0)>0]
                    rows.append({'path':path_id,'horizon':horizon,'strategy':strategy,'account':account,
                        'net_terminal_wealth':net,'real_terminal_wealth':net/cpi_ratio,'tax_paid':result['tax_paid'],
                        'terminal_tax_payable':result['terminal_tax_payable'],'metrics':result['after_tax_metrics'],
                        'trading_costs':result['trading_costs'],'cost_attribution':_cost_attribution(result,sample,costs),
                        'fund_liquidation':bool(tracked),'cash_residual_max':float(result['reconciliation'].cash_residual.abs().max()),
                        'terminal_wealth_scope':'Cash inside the Roth after internal liquidation; personal distribution not executed' if account=='roth' else 'Account assets net of modeled investment tax payable',
                        'nav_residual_max':float(result['reconciliation'].nav_residual.abs().max())})
                print('account comparisons complete',path_id,horizon,strategy,flush=True)
        diagnostics.append({'path':path_id,'short_rate_min':float(daily.short_rate.min()),'short_rate_max':float(daily.short_rate.max()),
            'monthly_equity_log_mean':float(monthly[path_id,:,columns.index('equity_log')].mean()),
            'diagnostic_bond_only':'Zero-bond model is a scenario diagnostic; taxable bond OID/distribution mechanics not certified'})
        print('account path complete',path_id,flush=True)
    summary=[]
    for horizon in horizons:
        for account in ('taxable','roth'):
            benchmark={r['path']:r for r in rows if r['horizon']==horizon and r['account']==account and r['strategy']=='US1X_CF'}
            for strategy in strategies:
                selected=[r for r in rows if r['horizon']==horizon and r['account']==account and r['strategy']==strategy]
                wealth=np.array([r['net_terminal_wealth'] for r in selected]);real=np.array([r['real_terminal_wealth'] for r in selected])
                counts={'drawdown_over_50':sum(r['metrics']['max_drawdown']>.5 for r in selected),
                    'drawdown_over_90':sum(r['metrics']['max_drawdown']>.9 for r in selected),
                    'fund_liquidation':sum(r['fund_liquidation'] for r in selected)}
                if benchmark:counts['outperform_paired_1x']=sum(r['net_terminal_wealth']>benchmark[r['path']]['net_terminal_wealth'] for r in selected)
                summary.append({'horizon':horizon,'account':account,'strategy':strategy,'paths':len(selected),
                    'wealth_p10_p50_p90':np.quantile(wealth,[.1,.5,.9]).tolist(),
                    'purchasing_power_p10_p50_p90':np.quantile(real,[.1,.5,.9]).tolist(),
                    'probability_drawdown_over_50':float(np.mean([r['metrics']['max_drawdown']>.5 for r in selected])),
                    'probability_drawdown_over_90':float(np.mean([r['metrics']['max_drawdown']>.9 for r in selected])),
                    'probability_fund_liquidation':float(np.mean([r['fund_liquidation'] for r in selected])),
                    'probability_outperform_paired_1x':float(np.mean([r['net_terminal_wealth']>benchmark[r['path']]['net_terminal_wealth'] for r in selected])) if benchmark else None,
                    'median_tax_drag_dollars':float(np.median([r['tax_paid']+r['terminal_tax_payable'] for r in selected])),
                    'conditional_sampling_95_intervals':{k:wilson_interval(int(v),len(selected)) for k,v in counts.items()},
                    'median_max_underwater_sessions':float(np.median([r['metrics']['max_underwater_sessions'] for r in selected]))})
    report={'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED','source_catalog_sha256':sources.catalog_hash,
        'model_id':model_id,'forecast_id':forecast_id,'model_type':model_type,'model_approval':'UNAPPROVED_RESEARCH_MODEL',
        'calibration_end_requested':calibration_end,'training_end_actual':str(training.index[-1].date()),'columns':columns,
        'blind_nowcast_bridge_months':gap,'incomplete_training_months':model.parameters['incomplete_training_months'],
        'seed':seed,'paths':paths,'horizons':horizons,'frozen_tax_and_roth_eligibility_year':frozen_tax_year,
        'initial_capital':initial_capital,'annual_contribution':annual_contribution,'monthly_withdrawal':monthly_withdrawal,
        'filing_status':filing_status,'ordinary_income':ordinary_income,'state':state,
        'supplied_roth_owner_history':roth_history,
        'supplied_annual_roth_magi':roth_magi,'supplied_annual_eligible_compensation':annual_compensation,
        'conditional_assumptions':{'reference_rate':rate,'annual_inflation':inflation,'annual_production_growth':production_growth,
            'constant_monthly_equity_annualized_constraint':equity_cagr_constraint,
            'equity_log_intercept_shift':shifts.get('equity_log'),'joint_volatility_multiplier':volatility_scale,
            'innovation_stock_bond_correlation':stock_bond_correlation,'crash_month_zero_based':crash_month,'crash_return':crash_return,
            'annual_fund_fee':annual_fee,'borrowing_spread':borrowing_spread,'quarterly_nonqualified_distribution_yield':distribution_yield,
            'cash_sweep':'max(reference rate minus 0.5%, 0) nominal APR; generic assumed contract',
            'rebalance':'every 21 trading sessions','tax_payment':'assumed April15 advanced to next projected XNYS session'},
        'results':summary,'path_results':rows,'macro_diagnostics':diagnostics,
        'uncertainty':{'conditional_monte_carlo_paths':paths,'convergence_certified':False,
            'parameter_uncertainty':'Not included in these conditional path percentiles; compare training/model/specification sensitivities separately',
            'source_uncertainty':'Current revised market history and limited macro vintages; research only'},
        'limitations':['Synthetic daily bridge, fund units and distribution policy are hypothetical, not observed TQQQ/UPRO quotes',
            'Interest/inflation/growth interventions are conditional statistical assumptions, not identified causal effects',
            'Fund price floor produces explicit closure; closed units cannot be bought or resurrected',
            'Future tax law and Roth eligibility are explicit frozen-year assumptions; no prediction of enacted future law',
            'Government bonds, futures and options are not ranked as interchangeable ETF sleeve implementations',
            'Stock/bond correlation is an innovation intervention; realized correlation need not equal its input',
            'Ordinary distribution policy is a disclosed tax stress assumption, not predicted future issuer classification',
            'Internal expense attribution uses opening exposure approximately; expenses are deducted only once in modeled NAV',
            'Small demonstration path counts do not establish calibrated real-world probabilities']}
    (output/'report.json').write_text(json.dumps(report,default=_serialize,indent=2,allow_nan=False)+'\n')
    return report
