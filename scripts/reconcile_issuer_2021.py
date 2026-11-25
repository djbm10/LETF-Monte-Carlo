"""Independent publication/API convention check; no retuning of consumed gates."""
import argparse,io,json,sys
from pathlib import Path
from hashlib import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.fund_nav import issuer_nav_returns
from letf.macro.artifacts import canonical
from letf.macro.precision_run import file_hash


def reconcile(official,facts,pdf,output):
    if output.exists():raise ValueError('No evidence overwrite')
    f=json.loads(facts.read_text())
    if file_hash(pdf)!=f['pdf_sha256']:raise ValueError('Issuer publication byte identity mismatch')
    catalog=json.loads((official/'catalog.json').read_text())
    def read(name):
        s=catalog['sources'][name];p=official/s['path']
        if file_hash(p)!=s['sha256']:raise ValueError('Immutable issuer/index source changed')
        return p.read_bytes()
    nav=pd.read_csv(io.BytesIO(read('TQQQ_nav')),index_col=0,parse_dates=True).sort_index()
    level=pd.read_csv(io.BytesIO(read('NASDAQXNDX')),index_col=0,parse_dates=True).iloc[:,0].dropna()
    events=[]
    for year in range(2006,2026):events+=json.loads(read(f'TQQQ_distributions_{year}'))
    end=nav.loc[:'2021-05-31'].index[-1];rows=[]
    for years,published in f['reported_annualized_returns'].items():
        years=int(years);base=nav.loc[:f'{2021-years}-05-31'].index[-1]
        start=nav.index[nav.index>base][0]
        selected=[e for e in events if start<=pd.Timestamp(e['ExDate'])<=end]
        returns,cash=issuer_nav_returns(nav,selected,start=start,end=end)
        # Two genuinely independent arithmetic implementations for the NAV
        # reinvestment convention: compounded gross returns versus share ledger.
        shares=1.
        for day in returns.index:shares+=shares*cash.loc[day]/nav.loc[day,'NAV']
        direct=shares*nav.loc[end,'NAV']/nav.loc[base,'NAV']
        compounded=float(np.prod(1+returns))
        if abs(direct-compounded)>max(1e-10,direct*1e-12):raise AssertionError('NAV share reinvestment reconciliation failed')
        actual=compounded**(1/years)-1
        index=(level.loc[end]/level.loc[base])**(1/years)-1
        tolerance=f['rounding_tolerance_absolute_return']
        rows.append({'years':years,'baseline_close':str(base.date()),'last_close':str(end.date()),
            'sessions':len(returns),'published_fund_annualized':published['fund'],'nav_api_annualized':actual,
            'fund_error_pp':100*(actual-published['fund']),'fund_publication_rounding_passed':abs(actual-published['fund'])<=tolerance,
            'published_index_annualized':published['index_total_return'],'XNDX_annualized':float(index),
            'index_error_pp':100*(index-published['index_total_return']),
            'index_publication_rounding_passed':bool(abs(index-published['index_total_return'])<=tolerance),
            'independent_share_ledger_terminal_relative_error':float(direct/compounded-1)})
    total=sum(r['notional'] for r in f['swaps'])
    if total!=f['reported_total_swap_notional']:raise AssertionError('Transcribed swap amounts do not reconcile to published total')
    interest=sum(r['notional']*r['floating_rate_paid'] for r in f['swaps'])
    report={'status':'SOURCE_RECONCILIATION_NOT_NEW_OOS_FORECAST_VALIDATION','rows':rows,
        'facts_sha256':file_hash(facts),'pdf_sha256':f['pdf_sha256'],'official_catalog_sha256':file_hash(official/'catalog.json'),
        'snapshot':{'asof':f['statement_date'],'total_swap_notional':total,'NAV':f['net_assets'],
            'swap_notional_per_NAV':total/f['net_assets'],'notional_weighted_swap_floating_rate':interest/total,
            'annualized_gross_swap_cost_per_NAV_at_snapshot':interest/f['net_assets'],
            'actual_stock_exposure':f['common_stock_value']/f['net_assets'],'futures_notional_per_NAV':f['futures_notional']/f['net_assets']},
        'preserved_TQQQ_reconstruction_CAGR_error_pp':2.4270821740895565,
        'reconstruction_certification':'BLOCKED','models_promoted':False,
        'conclusions':['Published-period NAV and index checks report every pass/failure at issuer rounding precision',
            'Actual swap financing applies to gross dated notionals; stock, futures and collateral exposures vary',
            'The reporting-date gross annualized swap cost excludes commissions/collateral income and is not annual realized expense',
            'No current expense, spot exposure or rate is backfilled to an unevidenced daily historical schedule']}
    output.parent.mkdir(parents=True,exist_ok=True)
    blob=canonical(report)
    with output.open('xb') as stream:stream.write(blob)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('official','facts','pdf','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=reconcile(a.official,a.facts,a.pdf,a.output)
    print(json.dumps({'rows':r['rows'],'snapshot':r['snapshot']},indent=2))
