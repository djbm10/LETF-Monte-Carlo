"""Registered Monte Carlo precision study of hypothetical fund UNIT NAV.

This module calculates no portfolio balances, trades, contributions or taxes.
Account wealth requires the canonical account explorer. Thousands of instrument
paths are not a substitute for thousands of reconciled account replays.
"""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from letf.macro.artifacts import ArtifactStore,canonical,fingerprint
from letf.macro.data import train_slice
from letf.macro.daily_tails import realized_monthly_variance,training_templates,daily_bridge
from letf.macro.resampling import sample_with_lineage
from letf.macro.models import MacroModel
from letf.macro.scoring import drawdowns
from letf.macro.uncertainty import wilson_interval
from letf.reconstruction import SourceRecord,daily_reset_returns


def precision_report(values,policy):
    """Nested prefixes and separate-seed replication; zero NAVs are retained."""
    arrays=[np.asarray(x,float) for x in values]
    n=policy['master_paths'];qs=policy['quantiles'];counts=policy['prefix_counts']
    if len(arrays)!=2 or any(x.shape!=(n,) or not np.isfinite(x).all() or (x<0).any() for x in arrays):
        raise ValueError('Two complete, nonnegative path samples required')
    terminal=[]
    for a in arrays:
        terminal.append({str(k):np.quantile(a[:k],qs).tolist() for k in counts})
    reference=np.asarray(terminal[0][str(n)]);differences=[]
    for seed_index,k in [(0,k) for k in counts[:-1]]+[(1,n)]:
        sample=np.asarray(terminal[seed_index][str(k)])
        # A zero quantile is not discarded or clipped to a positive NAV.
        differences.append(float(np.max(abs(np.log(sample)-np.log(reference)))) if (sample>0).all() and (reference>0).all() else None)
    intervals=[wilson_interval(int((a<1.).sum()),n) for a in arrays]
    half=max((hi-lo)/2 for lo,hi in intervals)
    passed=all(v is not None and v<=policy['maximum_abs_log_wealth_quantile_difference'] for v in differences)
    return {'unit_nav_p10_p50_p90_by_seed_and_prefix':terminal,'maximum_abs_log_quantile_differences':differences,
        'probability_unit_nav_below_initial_by_seed':[float(np.mean(a<1.)) for a in arrays],
        'conditional_sampling_95_intervals':intervals,'maximum_wilson_half_width':half,
        'quantile_convergence_passed':passed,'probability_precision_passed':half<=policy['maximum_probability_wilson_half_width'],
        'convergence_passed':bool(passed and half<=policy['maximum_probability_wilson_half_width'])}


def sampling_study(sources,*,output,protocol,calibration_end='2026-09-30',chunk_paths=128):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if type(chunk_paths)is not int or chunk_paths<1:raise ValueError('Positive chunk size required')
    policy=protocol['convergence'];n=policy['master_paths'];horizons=policy['horizons'];months=12*max(horizons)
    panel,meta=sources.panel(calibration_end);ff=sources.daily_factors().loc[:pd.Timestamp(calibration_end)]
    observed=np.log1p(ff['Mkt-RF']+ff.RF)
    panel['equity_log_variance']=realized_monthly_variance(observed).reindex(panel.index)
    frame=train_slice(panel,calibration_end,480,120,allow_missing_transitions=True)
    model=MacroModel.fit('stationary_block',frame,protocol)
    template_logs=observed.loc[frame.index[0].to_period('M').start_time:frame.index[-1]]
    templates,lengths=training_templates(template_logs,frame)
    store=ArtifactStore(output/'artifacts');gap=(pd.Period('2026-12','M')-frame.index[-1].to_period('M')).n
    if gap<0:raise ValueError('Calibration later than research origin')
    vi=model.columns.index('short_rate');pi=model.columns.index('inflation_log')
    code_hash=sha256(Path(__file__).read_bytes()+Path(__file__).with_name('daily_tails.py').read_bytes()+
        Path(__file__).with_name('resampling.py').read_bytes()+Path(__file__).with_name('models.py').read_bytes()+
        Path(__file__).parents[1].joinpath('reconstruction.py').read_bytes()).hexdigest()
    mid=store.put('models',{'kind':model.kind,'parameters':model.parameters,'columns':model.columns,
        'training_sha256':fingerprint(frame),'source_metadata':meta,'implementation_sha256':code_hash}, {'training':model.train_values})
    # Ordinal date labels only to call the existing daily-reset mechanics API.
    # No date is represented as an actual historical/future exchange session.
    ordinal=pd.date_range('2000-01-01',periods=21*months,freq='D')
    source=SourceRecord('Joint stationary research bootstrap','https://statistics.stanford.edu/technical-reports/stationary-bootstrap',
        '2026-11-24','Registered research scenario',sources.catalog_hash,str(ordinal[0].date()),'USD',
        'Hypothetical 21-session months with ordinal date labels, no observed fund quotes','daily',False)
    all_values={};all_real={};all_dd={};all_liquidation={};fids=[]
    for seed in policy['independent_seeds']:
        full,lineage=sample_with_lineage(model,months+gap,n,seed)
        initial=full[:,gap-1,vi] if gap else np.full(n,frame.short_rate.iloc[-1])
        monthly=full[:,gap:];lineage=lineage[:,gap:]
        fid=store.put('forecasts',{'model_id':mid,'seed':seed,'paths':n,'implementation_sha256':code_hash,
            'clock':'21 synthetic sessions/month, 252/year; not a future exchange calendar',
            'annual_fee':.01,'borrowing_spread':.015,'calibration_end':calibration_end,
            'origin':'2026-12-31','evidence':'HYPOTHETICAL_STRESS','account_wealth_calculated':False},
            {'monthly':monthly,'lineage':lineage,'initial_rate':initial})
        fids.append(fid)
        values={(h,l):np.empty(n) for h in horizons for l in (1,2,3)}
        real={key:np.empty(n) for key in values};dd={key:np.empty(n) for key in values};liqs={key:np.empty(n,dtype=bool) for key in values}
        for begin in range(0,n,chunk_paths):
            ids=np.arange(begin,min(begin+chunk_paths,n))
            logs=daily_bridge(monthly[ids],lineage[ids],model.columns,templates,lengths,seed+100000,path_ids=ids)
            rates=np.repeat(np.column_stack([initial[ids],monthly[ids,:-1,vi]]),21,axis=1)
            for k,p in enumerate(ids):
                underlying=pd.Series(np.expm1(logs[k]),index=ordinal)
                for leverage in (1,2,3):
                    result=daily_reset_returns(underlying,leverage,annual_fee=.01 if leverage>1 else 0.,
                        borrowing_spread=.015,funding_rate_at_previous_close=pd.Series(rates[k],index=ordinal),source=source)
                    growth=np.cumprod(1+result['return'].to_numpy());peak=np.maximum.accumulate(np.r_[1.,growth])[1:]
                    for h in horizons:
                        last=h*252;key=h,leverage
                        values[key][p]=growth[last-1]
                        real[key][p]=growth[last-1]/np.exp(monthly[p,:h*12,pi].sum())
                        dd[key][p]=np.max(1-growth[:last]/peak[:last])
                        liqs[key][p]=bool(result.liquidated.iloc[last-1])
            print('sampling chunk complete',seed,int(ids[-1])+1,flush=True)
        all_values[seed]=values;all_real[seed]=real;all_dd[seed]=dd;all_liquidation[seed]=liqs
        store.put('evaluations',{'forecast_id':fid,'scope':'UNIT NAV precision, no historical outcomes or account balances'},
            {f'{h}y_{l}x_{name}':data[(h,l)] for h in horizons for l in (1,2,3)
                for name,data in [('unit_nav',values),('real_unit_nav',real),('drawdown',dd),('liquidated',liqs)]})
    rows=[];seeds=policy['independent_seeds']
    for h in horizons:
        for leverage in (1,2,3):
            key=h,leverage;row={'horizon':h,'leverage':leverage,**precision_report([all_values[s][key] for s in seeds],policy)}
            row['real_unit_nav_p10_p50_p90_by_seed']=[np.quantile(all_real[s][key],[.1,.5,.9]).tolist() for s in seeds]
            row['maximum_drawdown_p10_p50_p90_by_seed']=[np.quantile(all_dd[s][key],[.1,.5,.9]).tolist() for s in seeds]
            row['conditional_liquidation_probability_by_seed']=[float(np.mean(all_liquidation[s][key])) for s in seeds]
            rows.append(row)
    report={'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'model_approval':'UNAPPROVED_RESEARCH_REFERENCE','scope':'Fund UNIT NAV, not account wealth, after-tax returns or retirement income',
        'account_wealth_convergence':'NOT_TESTED; use canonical account replay, no replacement tax/wealth engine',
        'paths_per_seed':n,'seeds':seeds,'forecast_ids':fids,'source_catalog_sha256':sources.catalog_hash,
        'training_start':str(frame.index[0].date()),'training_end':str(frame.index[-1].date()),
        'protocol':policy,'results':rows,'all_quantile_and_probability_gates_passed':all(r['convergence_passed'] for r in rows),
        'limitations':['Sampling error does not measure source, parameter, model or tax-policy uncertainty',
            'No distribution cash flows, tax, commissions, exchange fills or personal withdrawals in unit-NAV paths',
            'Fixed fee/spread scenarios and synthetic session clock; no real TQQQ historical claim',
            'No probability of outperforming investments is certified by these unapproved models']}
    (output/'report.json').write_bytes(canonical(report));return report
