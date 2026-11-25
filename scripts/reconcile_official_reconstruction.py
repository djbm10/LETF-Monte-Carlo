"""Source corrections on an ALREADY INSPECTED reconstruction diagnostic.

No new holdout claim, no tuning against 2020-25, no threshold modification.
Official Nasdaq TR replaces net-ETF proxies where available. S&P/Treasury
substitutes remain explicitly blocked rather than called exact target indices.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.fund_nav import issuer_nav_returns
from letf.reconstruction import SourceRecord,ReconstructionGate,daily_reset_returns,validate_reconstruction,reconstruction_errors
from scripts.validate_daily_reconstruction import PRODUCTS


def run(official,inputs,output):
    if output.exists():raise ValueError('Existing evidence cannot be overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((official/'catalog.json').read_text());vendor=json.loads((inputs/'manifest.json').read_text())
    spec=json.loads(Path('docs/accuracy-audit/reconstruction-gate.json').read_text())
    gate=ReconstructionGate(**{k:spec[k] for k in asdict(ReconstructionGate())})
    def checked(folder,item):
        path=folder/item['path'];blob=path.read_bytes()
        if sha256(blob).hexdigest()!=item['sha256']:raise ValueError('Source checksum mismatch')
        return blob
    def vload(name):return pd.read_csv(io.BytesIO(checked(inputs,vendor['sources'][name])),index_col=0,parse_dates=True)
    funding=vload('DFF').iloc[:,0].shift(2)/100
    index=pd.read_csv(io.BytesIO(checked(official,catalog['sources']['NASDAQXNDX'])),index_col=0,parse_dates=True).iloc[:,0]
    # FRED publishes non-session rows as missing. They are not observations and
    # never become synthetic zero-return sessions. Genuine session equality is
    # checked below against issuer NAV before any return comparison.
    index=index.dropna();index_return=index.pct_change(fill_method=None)
    report={'status':'DIAGNOSTIC_ALREADY_INSPECTED','investment_performance_certification':'BLOCKED',
        'original_gate':spec,'gate_sha256':gate.fingerprint,
        'official_catalog_sha256':sha256((official/'catalog.json').read_bytes()).hexdigest(),
        'vendor_manifest_sha256':sha256((inputs/'manifest.json').read_bytes()).hexdigest(),'products':{}}
    for symbol in ('TQQQ','QLD','UPRO','SSO'):
        base,leverage,fee=PRODUCTS[symbol]
        nav=pd.read_csv(io.BytesIO(checked(official,catalog['sources'][symbol+'_nav'])),index_col=0,parse_dates=True).sort_index()
        events=[]
        for year in range(2006,2026):
            events.extend(json.loads(checked(official,catalog['sources'][f'{symbol}_distributions_{year}'])))
        vendor_fund=vload(symbol)['Adj Close'].pct_change(fill_method=None)
        start=max(nav.index[1],vendor_fund.first_valid_index())
        selected=[e for e in events if start<=pd.Timestamp(e['ExDate'])<=pd.Timestamp('2025-12-31')]
        actual,cash=issuer_nav_returns(nav,selected,start=start,end='2025-12-31')
        original=vendor_fund.reindex(actual.index)
        underlying=(index_return if symbol in ('TQQQ','QLD') else vload(base)['Adj Close'].pct_change(fill_method=None))
        rows=pd.DataFrame({'actual':actual,'base':underlying.reindex(actual.index),
            'rate':funding.reindex(actual.index),'days':nav.index.to_series().diff().dt.days.reindex(actual.index)})
        if rows.isna().any().any():raise ValueError('Missing authentic session observations; no internal alignment/drop is permitted')
        train=rows.loc[:spec['training_end']];test=rows.loc[spec['evaluation_start']:spec['evaluation_end']]
        meta=catalog['sources']['NASDAQXNDX'] if symbol in ('TQQQ','QLD') else vendor['sources'][base]
        source=SourceRecord(meta['provider'],meta['url'],meta['retrieved_at'],meta['publication_vintage'],meta['sha256'],
            meta['first_observation'],'USD','Nasdaq-100 total-return index' if symbol in ('TQQQ','QLD') else 'Net ETF total-return PROXY; exact target index BLOCKED', 'daily',symbol not in ('TQQQ','QLD'))
        raw=daily_reset_returns(train.base,leverage,annual_fee=fee,funding_rate_at_previous_close=train.rate,elapsed_calendar_days=train.days,source=source)
        x=(leverage-1)*train.days.to_numpy()/360
        spread=max(0.,float(np.dot(x,raw['return']-train.actual)/np.dot(x,x)))
        predicted=daily_reset_returns(test.base,leverage,annual_fee=fee,funding_rate_at_previous_close=test.rate,
            borrowing_spread=spread,elapsed_calendar_days=test.days,source=source)['return']
        result=validate_reconstruction(test.actual,predicted,training_end=spec['training_end'],gate=gate,provenance_verified=False)
        result.update(fitted_training_spread=spread,annual_fee_scenario=fee,training_observations=len(train),
            index_convention=source.convention,actual_convention='Issuer NAV, same-restatement cash reinvested at ex-date NAV',
            issuer_vs_vendor_total_return_errors=reconstruction_errors(original.loc[test.index],test.actual),
            cash_distributions_reviewed=len(events),
            certification_blockers=['Historical fee/waiver/swap/collateral cash-yield schedules incomplete',
                'Already inspected evaluation period; source correction cannot create new independent evidence',
                'Current revised/downloaded inputs, not decision-time archive']+
                ([] if symbol in ('TQQQ','QLD') else ['Official S&P total-return index not available in this input bundle; SPY remains a proxy']))
        report['products'][symbol]=result
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('official','inputs','output'):p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args();r=run(a.official,a.inputs,a.output)
    print(json.dumps({s:{'cagr_error':v['errors']['cagr_error'],'daily_rmse':v['errors']['daily_rmse'],
        'spread':v['fitted_training_spread'],'numerical_pass':v['numerical_pass']} for s,v in r['products'].items()},indent=2))
