"""Fixed-old-parameter diagnostics on newly captured, already elapsed history.

This never refits inspected outcomes or manufactures an untouched holdout. The
original 2020–25 errors/gates stay authoritative. Short2026 history cannot pass
the original504-session minimum and does not establish generalization.
"""
import argparse
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from letf.fund_nav import issuer_nav_returns
from letf.lab.catalog import SourceLake,file_hash,private_json
from letf.reconstruction import SourceRecord,daily_reset_returns,reconstruction_errors


def diagnose(lake,snapshots,old_report,output,*,start='2026-01-01',end='2026-11-24'):
    if start<'2026-01-01' or end>'2026-12-31':raise ValueError('Explicit2026 diagnostic subset only')
    lake=SourceLake(lake);loaded={};hashes={};metadata={}
    for name in ('TQQQ_nav','TQQQ_cash_2026','NASDAQXNDX','DFF'):
        record,path=lake.load(snapshots[name])
        if record['name']!=name:raise ValueError('Snapshot series identity mismatch')
        metadata[name]=record['metadata'];hashes[name]=record['raw_sha256']
        loaded[name]=json.loads(path.read_bytes()) if name.endswith('2026') else pd.read_csv(io.BytesIO(path.read_bytes()))
    nav=loaded['TQQQ_nav'];nav.index=pd.to_datetime(nav.pop('Date'),format='%m/%d/%Y');nav=nav.sort_index()
    if pd.Timestamp(end)>nav.index.max():raise ValueError('Requested end exceeds captured issuer NAV')
    events=[e for e in loaded['TQQQ_cash_2026'] if pd.Timestamp(start)<=pd.Timestamp(e['ExDate'])<=pd.Timestamp(end)]
    actual,_=issuer_nav_returns(nav,events,start=start,end=end)
    idx=loaded['NASDAQXNDX'];idx.index=pd.to_datetime(idx.iloc[:,0]);series=pd.to_numeric(idx.iloc[:,1],errors='coerce').dropna()
    underlying=series.pct_change(fill_method=None)
    dff=loaded['DFF'];dff.index=pd.to_datetime(dff.iloc[:,0]);rates=pd.to_numeric(dff.iloc[:,1],errors='coerce')/100
    # Two calendar-day publication lag, identical to the old diagnostic. Do
    # not fill genuine missing rates or infer exact swap financing from DFF.
    rate=pd.Series(rates.reindex(actual.index-pd.Timedelta(days=2)).to_numpy(),index=actual.index)
    rows=pd.DataFrame({'base':underlying.reindex(actual.index),'rate':rate,
        'days':nav.index.to_series().diff().dt.days.reindex(actual.index)})
    if rows.isna().any().any():raise ValueError('Missing authentic joint observations; no internal alignment drops')
    old=json.loads(Path(old_report).read_text());prior=old['products']['TQQQ']
    meta=metadata['NASDAQXNDX']
    source=SourceRecord(meta['provider'],meta['url'],meta['retrieved_at'],meta['publication_vintage'],hashes['NASDAQXNDX'],
        meta['first_observation'],'USD','Nasdaq-origin current total-return index','daily',False)
    predicted=daily_reset_returns(rows.base,3.,annual_fee=prior['annual_fee_scenario'],
        funding_rate_at_previous_close=rows.rate,borrowing_spread=prior['fitted_training_spread'],
        elapsed_calendar_days=rows.days,source=source)['return']
    errors=reconstruction_errors(actual,predicted)
    result={'classification':'HISTORICAL_COUNTERFACTUAL','status':'NEWLY_EVALUATED_RETROSPECTIVE_FIXED_OLD_PARAMETERS',
        'start':str(actual.index[0].date()),'end':str(actual.index[-1].date()),'source_raw_hashes':hashes,
        'source_snapshot_ids':snapshots,'old_report_sha256':file_hash(old_report),
        'old_gate_sha256':old['gate_sha256'],'old_INSPECTED_2020_2025_errors':prior['errors'],
        'fitted_training_spread_unchanged':prior['fitted_training_spread'],'annual_fee_scenario_unchanged':prior['annual_fee_scenario'],
        'new_errors':errors,'original_minimum_observation_count':old['original_gate']['min_observations'],
        'new_period_meets_original_minimum':len(actual)>=old['original_gate']['min_observations'],
        'provenance_certified':False,'investment_performance_certification':'BLOCKED',
        'limitations':['Elapsed2026 data are retrospective, not frozen-before-outcomes evidence',
            'Current index/NAV/cash restatements; exact dated financing/fee/collateral/target-index vintage incomplete',
            'Cash API does not certify final tax classes or independently prove annual distribution completeness',
            'Partial-year CAGR is annualized, not realized full-year performance; no gate reset or model promotion']}
    private_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('lake','snapshots','old-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--end',default='2026-11-24');a=p.parse_args()
    r=diagnose(a.lake,json.loads(a.snapshots.read_text()),a.old_report,a.output,end=a.end)
    print(json.dumps({'status':r['status'],'errors':r['new_errors'],'certification':'BLOCKED'}))
