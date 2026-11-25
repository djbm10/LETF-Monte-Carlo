"""Unweighted catastrophe scenarios; never extensions of forecast probabilities.

Reuse frozen source-row lineages and the existing empirical daily bridge. The
calendar clock is explicit. Shocks are interventions assumed by the researcher,
not estimated frequencies, causal predictions or observed historical quotes.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.account_backtest import Distribution
from letf.macro.artifacts import ArtifactStore,canonical
from letf.macro.daily_tails import daily_bridge
from letf.macro.models import MacroModel
from letf.macro.precision_run import stream_seeds
from letf.macro.resampling import sample_with_lineage
from letf.reconstruction import SourceRecord,daily_reset_returns

CASES=('reference','daily_loss_25','daily_extinction_34','overnight_extinction_50',
       'bear_7y','stagnation_7y_high_rates','inflation_funding_10y',
       'double_daily_volatility','issuer_failure_zero','higher_cost')


def frozen_anchor(experiment,anchor,*,years=40):
    """Recreate a registered old stream; no draws or paths selected by outcomes."""
    r=json.loads((Path(experiment)/'registration.json').read_text());p=r['protocol']
    if anchor['root_seed'] not in p['seeds'] or anchor['stream_block']!=0 or anchor['path_index']!=0:
        raise UnsupportedModelError('Only preregistered first-path old stream anchors are supported')
    meta,a=ArtifactStore(Path(experiment)/'artifacts').load('models',r['model_id'])
    model=MacroModel(meta['kind'],meta['columns'],a['training'],meta['parameters'])
    origin=pd.Timestamp(p['origin']);first=origin+pd.Timedelta(days=1)
    last=pd.Timestamp(year=origin.year+years,month=12,day=31)
    calendar=xcals.get_calendar('XNYS',start=first-pd.Timedelta(days=15),end=last+pd.Timedelta(days=31))
    dates=calendar.sessions_in_range(first,last).tz_localize(None)
    periods=dates.to_period('M');months=periods.unique();counts=np.array([(periods==m).sum() for m in months])
    gap=(origin.to_period('M')-pd.Period(model.parameters['training_end'],'M')).n
    monthly_seed,bridge_seed=stream_seeds(anchor['root_seed'],anchor['stream_block'])
    _,full=sample_with_lineage(model,len(months)+gap,p['fixed_stream_block_paths'],monthly_seed,return_values=False)
    lineage=full[:1,gap:];monthly=model.train_values[lineage]
    logs=daily_bridge(monthly,lineage,model.columns,a['templates'],a['lengths'],bridge_seed,sessions=counts)[0]
    rate_col=model.columns.index('short_rate');inf_col=model.columns.index('inflation_log')
    previous=np.r_[model.train_values[full[0,gap-1],rate_col] if gap else model.train_values[-1,rate_col],monthly[0,:-1,rate_col]]
    frame=pd.DataFrame({'equity_log':logs,'short_rate':np.repeat(previous,counts),
        'inflation_log':np.repeat(monthly[0,:,inf_col]/counts,counts)},index=dates)
    frame.attrs.update(model_id=r['model_id'],source_catalog_sha256=r['source_catalog_sha256'],anchor=anchor,
        clock='Variable projected XNYS sessions/month; preserves sampled monthly equity log totals and variance; not the old fixed-21 clock')
    return frame


def stressed_context(anchor,case):
    if case not in CASES:raise UnsupportedModelError('Unknown or probability-weighted stress case')
    if set(anchor.columns)!={'equity_log','short_rate','inflation_log'} or not np.isfinite(anchor.to_numpy()).all():
        raise ValueError('Complete aligned joint scenario required')
    frame=anchor.copy();frame.attrs=dict(anchor.attrs);frame.attrs['case']=case
    frame.attrs['probability']='UNSPECIFIED_NOT_ESTIMATED'
    frame.attrs['timing']='close';frame.attrs['forced_issuer_failure']=None
    index=frame.index;shock=int(index.searchsorted(pd.Timestamp('2028-06-15')))
    if case in ('daily_loss_25','daily_extinction_34','overnight_extinction_50','issuer_failure_zero'):
        if shock>=len(index):raise ValueError('Scenario does not reach its registered shock date')
        if case=='issuer_failure_zero':frame.attrs['forced_issuer_failure']=str(index[shock].date())
        else:frame.loc[index[shock],'equity_log']=np.log1p({'daily_loss_25':-.25,'daily_extinction_34':-.34,'overnight_extinction_50':-.5}[case])
        if case=='overnight_extinction_50':frame.attrs['timing']='open'
    if case in ('bear_7y','stagnation_7y_high_rates','inflation_funding_10y'):
        years=10 if case=='inflation_funding_10y' else 7
        selected=(index>=pd.Timestamp('2028-01-01'))&(index<pd.Timestamp(f'{2028+years}-01-01'))
        frame.loc[selected,'short_rate']=.10 if years==10 else .06
        for period in index[selected].to_period('M').unique():
            mask=index.to_period('M')==period;n=int(mask.sum())
            frame.loc[mask,'inflation_log']=np.log1p(.08 if years==10 else .06)/12/n
            if years==7:
                daily=frame.loc[mask,'equity_log'];target=np.log1p(-.08 if case=='bear_7y' else 0.)/12
                frame.loc[mask,'equity_log']=daily-daily.mean()+target/n
    if case=='double_daily_volatility':
        for period in index.to_period('M').unique():
            mask=index.to_period('M')==period;daily=frame.loc[mask,'equity_log']
            frame.loc[mask,'equity_log']=daily.mean()+2*(daily-daily.mean())
    return frame


@dataclass
class ScenarioMarket:
    market:pd.DataFrame
    distributions:list
    liquidation_events:list
    inflation:pd.Series
    fund_mechanics:pd.DataFrame
    manifest:dict


def scenario_market(context,leverage,*,annual_distribution=.02,fee=.01,spread=.015):
    if leverage not in (1,2,3) or not 0<=annual_distribution<1:
        raise UnsupportedModelError('Registered long ETF exposure and finite distribution assumption required')
    if context.attrs.get('probability')!='UNSPECIFIED_NOT_ESTIMATED':
        raise UnsupportedModelError('Scenario cannot be promoted or assigned forecast probabilities')
    dates=context.index;case=context.attrs['case'];symbol=f'CF{leverage}'
    if len(dates)<2:raise ValueError('At least two scenario sessions required')
    if case=='higher_cost':fee=.02;spread=.03
    elif case=='inflation_funding_10y':spread=pd.Series(np.where((dates>='2028-01-01')&(dates<'2038-01-01'),.03,spread),index=dates)
    source=SourceRecord('Registered hypothetical scenario','scenario://financial-realism-v1','2026-11-25',case,
        sha256(context.to_csv().encode()+canonical(context.attrs)).hexdigest(),str(dates[0].date()),
        'USD','Assumed benchmark total returns, not observed fund quotes','daily',False)
    # First row is the enrollment price, not a pre-purchase overnight return.
    returns=np.expm1(context.equity_log.iloc[1:])
    result=daily_reset_returns(returns,leverage,annual_fee=fee if leverage>1 else 0.,
        borrowing_spread=spread.iloc[1:] if isinstance(spread,pd.Series) else spread,
        funding_rate_at_previous_close=context.short_rate.iloc[1:],
        elapsed_calendar_days=pd.Series(dates.to_series().diff().dt.days.iloc[1:],index=dates[1:]),source=source)
    first=pd.DataFrame({'return':[0.],'fund_fee':[0.],'financing':[0.],'liquidated':[False],
                        'evidence':['HYPOTHETICAL_STRESS']},index=dates[:1])
    result=pd.concat([first,result]);forced=context.attrs.get('forced_issuer_failure')
    if forced:
        t=dates.get_loc(pd.Timestamp(forced));result.loc[dates[t:],'liquidated']=True
        result.loc[dates[t],'return']=-1.;result.loc[dates[t+1:],'return']=0.
        result.loc[dates[t+1:],['fund_fee','financing']]=0.
    closed=np.flatnonzero(result.liquidated.to_numpy());end=int(closed[0]) if len(closed) else None
    # The distribution is carved out of ex-price, never added to a total-return
    # quote a second time. Per-share distributions are scenario assumptions.
    ex={}
    for period in dates.to_period('Q').unique():
        month=period.end_time.month;year=period.year
        candidate=int(dates.searchsorted(pd.Timestamp(year=year,month=month,day=10)))
        if candidate>0 and candidate+5<len(dates) and (end is None or candidate<end):ex[candidate]=candidate+5
    opening=np.zeros(len(dates));closing=np.zeros(len(dates));opening[0]=closing[0]=100.
    distributions=[];events=[]
    for t in range(1,len(dates)):
        previous=closing[t-1]
        if end is not None and t>=end:
            opening[t]=previous if t==end and context.attrs['timing']=='close' else 0.
            if t==end:events.append({'date':str(dates[t].date()),'security':symbol,'cash_per_share':0.,
                'timing':context.attrs['timing'],'source':f'Stress assumption {case}; {source.sha256}; not actual issuer policy'})
            continue
        opening[t]=previous;cash=previous*annual_distribution/4 if t in ex else 0.
        closing[t]=previous*(1+result['return'].iloc[t])-cash
        if closing[t]<=0 or not np.isfinite(closing[t]):
            raise UnsupportedModelError('Distribution depletion or numerical underflow needs a separate explicit closure; no NAV floor')
        if cash:distributions.append(Distribution(symbol,dates[t].date(),dates[ex[t]].date(),cash,
            'ordinary_dividend',f'Explicit hypothetical ordinary distribution and ex-price reduction; {source.sha256}'))
    market=pd.DataFrame({symbol+'_Open':opening,symbol+'_Close':closing},index=dates)
    inflation=pd.Series(np.exp(np.r_[0.,np.cumsum(context.inflation_log.to_numpy()[1:])]),index=dates)
    manifest={'data_status':'HYPOTHETICAL_STRESS','case':case,'symbol':symbol,'leverage':leverage,
        'source_sha256':source.sha256,'context':context.attrs,'fees_embedded_in_prices':True,
        'ordinary_distribution_yield_assumption':annual_distribution,'fee_apr':float(fee if leverage>1 else 0.),
        'borrowing_convention':'Net borrowed notional (L-1), ACT/360; fees ACT/365; lagged scenario short rates',
        'event_probability':None,'certification':'BLOCKED','enrollment_row_has_no_return':True,
        'execution':'All ordinary daily change assumed intraday; registered overnight extinction is an opening shock; not observed fills'}
    return ScenarioMarket(market,distributions,events,inflation,result,manifest)
