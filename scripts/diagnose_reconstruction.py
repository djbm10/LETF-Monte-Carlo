"""Additional diagnostics of the ALREADY INSPECTED ETF evaluation window.

No refit on evaluation data, changed thresholds or new independent-test claim.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.reconstruction import SourceRecord,daily_reset_returns
from scripts.validate_daily_reconstruction import PRODUCTS


def holding_windows(actual,predicted):
    report={}
    for years in (1,3,5,10):
        length=252*years;errors=[]
        for start in range(0,len(actual)-length+1,21):
            a=np.prod(1+actual.iloc[start:start+length]);p=np.prod(1+predicted.iloc[start:start+length])
            errors.append(p/a-1)
        report[str(years)]={'windows':len(errors),'sessions_per_window':length,
            'relative_error_p10_p50_p90':np.quantile(errors,[.1,.5,.9]).tolist() if errors else None,
            'max_abs_error':float(np.max(np.abs(errors))) if errors else None,
            'status':'DIAGNOSTIC_ALREADY_INSPECTED' if errors else 'BLOCKED_INSUFFICIENT_DAILY_COVERAGE',
            'overlap_is_not_independent_evidence':True}
    return report


def diagnose(inputs,frozen_report,output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Refusing to overwrite reconstruction evidence')
    old=json.loads(Path(frozen_report).read_text());catalog=json.loads((Path(inputs)/'manifest.json').read_text())
    if sha256((Path(inputs)/'manifest.json').read_bytes()).hexdigest()!=old['input_manifest_sha256']:
        raise ValueError('Diagnostics must use the same immutable inputs as their frozen fit')
    def load(s):
        meta=catalog['sources'][s];p=Path(inputs)/meta['path']
        if sha256(p.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Source checksum mismatch')
        return pd.read_csv(p,index_col=0,parse_dates=True),meta
    rate=load('DFF')[0].DFF.shift(2)/100;results={}
    for product,(benchmark,leverage,fee) in PRODUCTS.items():
        fund,_=load(product);base,meta=load(benchmark)
        x=pd.DataFrame({'actual':fund['Adj Close'].pct_change(fill_method=None),
            'underlying':base['Adj Close'].pct_change(fill_method=None),'rate':rate.reindex(base.index),
            'days':base.index.to_series().diff().dt.days})
        e=x.loc[old['split']['evaluation_start']:old['split']['evaluation_end']]
        if e.isna().any().any():raise ValueError('Missing evaluation data; no dropped observations')
        result=old['products'][product]
        predicted=daily_reset_returns(e.underlying,leverage,annual_fee=fee,borrowing_spread=result['fitted_training_spread'],
            funding_rate_at_previous_close=e.rate,elapsed_calendar_days=e.days,
            source=SourceRecord(**{k:meta[k] for k in SourceRecord.__dataclass_fields__}))['return']
        groups={};residual=predicted-e.actual
        masks={'reference_rate_under_1pct':e.rate<.01,'reference_rate_at_least_3pct':e.rate>=.03,
            'covid_crash_recovery':(e.index>='2020-02-01')&(e.index<='2020-06-30'),
            'inflation_shock_2022':e.index.year==2022}
        for label,mask in masks.items():
            sample=residual.loc[mask]
            groups[label]={'observations':len(sample),'daily_bias':float(sample.mean()) if len(sample) else None,
                'daily_rmse':float(np.sqrt(np.mean(sample**2))) if len(sample) else None,
                'no_noncontiguous_compounding':True}
        results[product]={'frozen_gate_unchanged':result['checks'],'fitted_training_spread':result['fitted_training_spread'],
            'cagr_error_before':result['errors']['cagr_error'],'cagr_error_after':result['errors']['cagr_error'],
            'holding_periods':holding_windows(e.actual,predicted),'rate_crisis_diagnostics':groups,
            'certification':'BLOCKED','numerical_pass':result['numerical_pass']}
    report={'frozen_report_sha256':sha256(Path(frozen_report).read_bytes()).hexdigest(),'products':results,
        'evaluation_status':'ALREADY_INSPECTED_2020_2025_DIAGNOSTIC','parameter_changes':False,'threshold_changes':False,
        'evidence_needed':['Official target-index daily total returns and verified index revisions, not net-fee QQQ/SPY/TLT substitutes',
            'Dated fund expense waivers, swap/funding spreads and collateral yields',
            'Issuer NAV total returns versus market-price premium/discount effects',
            'Fresh future evaluation after registering any new candidate; this old window cannot be made untouched']}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','frozen-report','output'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();diagnose(a.inputs,a.frozen_report,a.output)
