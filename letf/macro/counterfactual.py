"""Authentic daily broad/growth/industry return counterfactuals to 1930.

Not literal S&P 500/Nasdaq-100 before their existence, not an ETF quote tape,
not a historical personal-tax/account simulation. Uses daily_reset_returns.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.reconstruction import SourceRecord,daily_reset_returns
from letf.macro.data import french_daily
from letf.accuracy import UnsupportedModelError

CLOSURES={'1933-03-15':{'previous_close':'1933-03-03','elapsed_days':12,
    'source':'https://www.federalreservehistory.org/essays/emergency-banking-act-of-1933'}}


def path_statistics(returns):
    values=np.asarray(returns,float)
    if not np.isfinite(values).all() or (values < -1).any():raise ValueError('Invalid return path')
    wealth=np.r_[1.,np.cumprod(1+values)]
    dd=1-wealth/np.maximum.accumulate(wealth)
    run=longest=0
    for x in dd:
        run=run+1 if x>1e-12 else 0;longest=max(longest,run)
    return {'terminal_index':float(wealth[-1]),'max_drawdown':float(dd.max()),'max_underwater_observations':longest,
            'observations':len(values),'liquidated':bool((values==-1).any())}


ERAS={'depression_1930s':('1930-01-01','1939-12-31'),'postwar':('1946-01-01','1969-12-31'),
    'stagflation_1970s':('1970-01-01','1979-12-31'),'crash_1987':('1987-01-01','1987-12-31'),
    'lost_decade':('2000-01-01','2012-12-31'),'gfc_recovery':('2008-01-01','2012-12-31'),
    'bull_2012_2025':('2012-01-01','2025-12-31'),'covid_2020':('2020-01-01','2020-12-31'),
    'inflation_2022':('2022-01-01','2022-12-31')}


def dated_statistics(returns,elapsed):
    stats=path_statistics(returns)
    previous=returns.index[0]-pd.Timedelta(days=float(elapsed.iloc[0]))
    years=(returns.index[-1]-previous).days/365.2425
    stats.update(cagr=stats['terminal_index']**(1/years)-1,period_start_close=str(previous.date()),
        first_return_date=str(returns.index[0].date()),end=str(returns.index[-1].date()))
    return stats


def historical_windows(returns,elapsed):
    eras={}
    failed=returns.index[returns.eq(-1)]
    closure=failed[0] if len(failed) else None
    for label,(a,b) in ERAS.items():
        sample=returns.loc[a:b]
        eras[label]=(dated_statistics(sample,elapsed.reindex(sample.index)) if len(sample) and
            (closure is None or sample.index[0]<=closure) else {'status':'BLOCKED','reason':'No source coverage or fund already liquidated'})
    rolling={}
    for horizon in (5,10,20,30,40):
        records=[]
        for year in range(returns.index[0].year,returns.index[-1].year-horizon+2):
            end=pd.Timestamp(year+horizon,1,1)-pd.Timedelta(days=1)
            if end>returns.index[-1]:continue
            sample=returns.loc[f'{year}-01-01':end]
            if closure is not None and sample.index[0]>closure:continue
            records.append(dated_statistics(sample,elapsed.reindex(sample.index)))
        rolling[str(horizon)]={'windows':records,'overlapping_windows_are_not_independent':True,
            'cagr_p10_p50_p90_across_historical_windows':np.quantile([r['cagr'] for r in records],[.1,.5,.9]).tolist() if records else None}
    return {'eras':eras,'rolling_year_start_windows':rolling}


def historical_counterfactual(sources,*,start='1930-01-01',end=None,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ff=sources.daily_factors();rates=french_daily(sources.bytes('ff_monthly')[0],date_digits=6).RF
    # Monthly bill returns infer a rate, NOT a unique daily price path. Use
    # only the previous completed month's rate, piecewise constant, disclosed
    # as a counterfactual financing proxy. Early RF has source rounding error.
    annual=rates.pipe(np.log1p).mul(12).pipe(np.expm1)
    prior_month=(ff.index.to_period('M')-1).to_timestamp('M')
    rate=pd.Series(annual.reindex(prior_month).to_numpy(),index=ff.index)
    elapsed=ff.index.to_series().diff().dt.days
    daily={'US_BROAD_MARKET':ff['Mkt-RF']+ff.RF}
    for name,col in [('LARGE_GROWTH_BOOK_TO_MARKET','BIG LoBM'),('SMALL_GROWTH_BOOK_TO_MARKET','SMALL LoBM')]:
        frame=french_daily(sources.bytes('size_value_daily')[0])
        if col not in frame:raise UnsupportedModelError(f'Missing authentic proxy column: {col}')
        daily[name]=frame[col]
    industry=french_daily(sources.bytes('industry_daily')[0])
    if 'HiTec' in industry:daily['HISTORICAL_HITEC_INDUSTRY']=industry.HiTec
    meta=sources.catalog['sources']['ff_daily']
    provenance=SourceRecord(provider=meta['provider'],url=meta['url'],retrieved_at=meta['retrieved_at'],
        publication_vintage=str(meta['source_header']),sha256=meta['sha256'],first_observation=str(ff.index[0].date()),
        currency='USD',convention='Observed daily total returns of research portfolios',frequency='daily',fees_embedded=False)
    report={'source_catalog_sha256':sources.catalog_hash,'start_requested':start,'end_requested':end,
        'evidence':'HYPOTHETICAL_STRESS','account_history_certification':'BLOCKED',
        'not_literal_modern_products':True,'closure_evidence':CLOSURES,'exposures':{},
        'financing_proxy':'Previous completed month French one-month bill return annualized, held until next month; not observed historical swap borrowing',
        'limitations':['Broad market is French market portfolio, not a literal pre-1957 S&P 500 fund',
            'Growth means dated book-to-market portfolios / HiTec industry, not pre-1985 Nasdaq-100',
            'Current CIZ/revised portfolio vintage, not historical publication vintage',
            'Spreads/fees/product structure are counterfactual; modern funds, swaps and Roth accounts did not exist in 1930',
            'No intraday reset, execution or precise intraday breach is inferred',
            'Daily observations use the historical trading calendar, including Saturday trading',
            'Inflation is observed monthly; only endpoint purchasing power is calculated',
            'Gold source unavailable; no fixed-monetary-regime spot-gold leverage is fabricated']}
    cpi,_=sources.series('CPIAUCNS')
    source_names={'US_BROAD_MARKET':'ff_daily','LARGE_GROWTH_BOOK_TO_MARKET':'size_value_daily',
        'SMALL_GROWTH_BOOK_TO_MARKET':'size_value_daily','HISTORICAL_HITEC_INDUSTRY':'industry_daily'}
    for name,series in daily.items():
        genuine_first=str(series.index[0].date())
        source_meta=sources.catalog['sources'][source_names[name]]
        provenance=SourceRecord(provider=source_meta['provider'],url=source_meta['url'],retrieved_at=source_meta['retrieved_at'],
            publication_vintage=str(source_meta['source_header']),sha256=source_meta['sha256'],first_observation=genuine_first,
            currency='USD',convention='Observed daily total returns of research portfolios',frequency='daily',fees_embedded=False)
        series=series.reindex(ff.index).loc[start:end]
        r=rate.reindex(series.index);days=elapsed.reindex(series.index)
        if series.isna().any() or r.isna().any():raise UnsupportedModelError('Incomplete daily proxy or prior-month financing input')
        records=[]
        for leverage in (1.,2.,3.):
            for spread in (.005,.015,.03):
                for fee in (.005,.01,.02):
                    result=daily_reset_returns(series,leverage,annual_fee=fee,funding_rate_at_previous_close=r,
                        borrowing_spread=spread,elapsed_calendar_days=days,source=provenance,verified_closures=CLOSURES)
                    stats=dated_statistics(result['return'],days)
                    years=(series.index[-1]-pd.Timestamp(stats['period_start_close'])).days/365.2425
                    pi0=cpi.reindex([series.index[0].to_period('M').to_timestamp('M')]).iloc[0]
                    pi1=cpi.reindex([series.index[-1].to_period('M').to_timestamp('M')]).iloc[0]
                    stats['real_cagr']=(stats['terminal_index']/(pi1/pi0))**(1/years)-1
                    stats.update(leverage=leverage,annual_spread=spread,annual_fee=fee,
                        financing_rate_days_sum=float(result.financing.sum()),fund_fee_rate_days_sum=float(result.fund_fee.sum()),
                        start=str(series.index[0].date()),end=str(series.index[-1].date()))
                    records.append(stats)
                    if spread==.015 and fee==.01:
                        result.to_csv(output/f'{name}_{leverage:g}x_daily.csv',index_label='date')
                        stats['historical_windows']=historical_windows(result['return'],days)
        report['exposures'][name]={'first_genuine_observation':genuine_first,'results':records,
            'underlying_source':source_names[name],'underlying_source_sha256':source_meta['sha256'],
            'proxy_identity':'Observed portfolio return inputs; hypothetical leveraged product'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
