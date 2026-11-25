"""Cost/benchmark attribution on ALREADY INSPECTED TQQQ observations.

This is not fresh validation, a fitted invoice, or an approved replacement.
Cost coefficients are fitted ONLY through 2019. Price-versus-TR is reported as
an unresolved convention sensitivity, not chosen to obtain a passing CAGR.
"""
import argparse,json,io,sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from letf.fund_nav import issuer_nav_returns
from letf.reconstruction import SourceRecord,ReconstructionGate,daily_reset_returns,validate_reconstruction


def run(official,prices,output):
    if output.exists():raise ValueError('No evidence overwrite')
    catalog=json.loads((official/'catalog.json').read_text());vendor=json.loads((prices/'manifest.json').read_text())
    def checked(folder,item):
        p=folder/item['path'];blob=p.read_bytes()
        if sha256(blob).hexdigest()!=item['sha256']:raise ValueError('Immutable input changed')
        return blob
    nav=pd.read_csv(io.BytesIO(checked(official,catalog['sources']['TQQQ_nav'])),index_col=0,parse_dates=True).sort_index()
    events=[]
    for year in range(2006,2026):events+=json.loads(checked(official,catalog['sources'][f'TQQQ_distributions_{year}']))
    vendor_fund=pd.read_csv(io.BytesIO(checked(prices,vendor['sources']['TQQQ'])),index_col=0,parse_dates=True)['Adj Close'].pct_change(fill_method=None)
    start=max(nav.index[1],vendor_fund.first_valid_index())
    events=[e for e in events if start<=pd.Timestamp(e['ExDate'])<=pd.Timestamp('2025-12-31')]
    actual,_=issuer_nav_returns(nav,events,start=start,end='2025-12-31')
    rates=pd.read_csv(io.BytesIO(checked(prices,vendor['sources']['DFF'])),index_col=0,parse_dates=True).iloc[:,0].shift(2)/100
    indexes={}
    for name in ('NASDAQXNDX','NASDAQ100'):
        level=pd.read_csv(io.BytesIO(checked(official,catalog['sources'][name])),index_col=0,parse_dates=True).iloc[:,0].dropna()
        indexes[name]=level.pct_change(fill_method=None)
    fee=.0086;rows={};gate=ReconstructionGate()
    for index in indexes:
        frame=pd.DataFrame({'actual':actual,'base':indexes[index].reindex(actual.index),'rate':rates.reindex(actual.index),
            'days':nav.index.to_series().diff().dt.days.reindex(actual.index)})
        if frame.isna().any().any():raise ValueError('Missing authentic daily inputs')
        train=frame.loc[:'2019-12-31'];test=frame.loc['2020-01-01':'2025-12-31']
        meta=catalog['sources'][index]
        src=SourceRecord(meta['provider'],meta['url'],meta['retrieved_at'],meta['publication_vintage'],meta['sha256'],meta['first_observation'],
            'USD','Total return' if index=='NASDAQXNDX' else 'Price return, fund dividends/collateral income not separately reconciled','daily',False)
        target=3*train.base-fee*train.days/365-train.actual
        design=np.column_stack([train.rate*train.days/360,train.days/360])
        effective=lsq_linear(design,target,bounds=([2.,0.],[3.,np.inf]))
        baseline_spread=max(0.,float(np.dot(2*train.days/360,target-2*train.rate*train.days/360)/np.dot(2*train.days/360,2*train.days/360)))
        for kind in ('legacy_net_borrowing','training_only_effective_funding'):
            beta,cost=(2.,2*baseline_spread) if kind=='legacy_net_borrowing' else effective.x
            pred=daily_reset_returns(test.base,3,source=src,annual_fee=fee,funding_rate_at_previous_close=test.rate,
                elapsed_calendar_days=test.days,borrowing_spread=cost/3,financed_notional=3.,
                collateral_income_per_nav=(3-beta)*test.rate,collateral_day_basis=360,
                financing_source='Explicit fully-swapped cost attribution hypothesis; coefficients trained before 2020, not actual historical invoices')['return']
            result=validate_reconstruction(test.actual,pred,training_end='2019-12-31',gate=gate,provenance_verified=False)
            result.update(effective_funding_notional=float(beta),annual_nonmanagement_cost_per_nav=float(cost),training_observations=len(train),
                training_residual_rmse=float(np.sqrt(np.mean((target-design@np.array([beta,cost]))**2))),
                certification='BLOCKED: fitted effective coefficients cannot establish actual notionals/collateral/dividend conventions')
            result['daily_bias_by_rate']={label:{'observations':int(mask.sum()),'daily_bias':float((pred-test.actual)[mask].mean())} for label,mask in
                [('under_1pct',test.rate<.01),('at_least_3pct',test.rate>=.03)]}
            rows[index+'_'+kind]=result
    report={'status':'ATTRIBUTION_DIAGNOSTIC_ALREADY_INSPECTED_NOT_NEW_VALIDATION','official_catalog_sha256':sha256((official/'catalog.json').read_bytes()).hexdigest(),
        'vendor_manifest_sha256':sha256((prices/'manifest.json').read_bytes()).hexdigest(),'training_end':'2019-12-31','evaluation':'2020-2025',
        'gate_unchanged':asdict(gate),'fee_scenario':fee,'cases':rows,'promoted':False,
        'limitations':['Neither a price-return match nor a fitted funding coefficient proves full fund total-return mechanics',
            'NDX is the named daily objective; actual fund NAV total returns include income and all internal expenses',
            'Dated swap notional, actual collateral yield, fees and dividend terms remain necessary',
            'Source fee disclosures are date-specific; no current fee is backfilled as a proven historical schedule']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('official','prices','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.official,a.prices,a.output)
    print(json.dumps({k:{'cagr_error':v['errors']['cagr_error'],'rmse':v['errors']['daily_rmse'],'beta':v['effective_funding_notional'],'cost':v['annual_nonmanagement_cost_per_nav']} for k,v in r['cases'].items()},indent=2))
