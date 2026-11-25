"""Issuer-reviewed 2024-25 fund inputs, including dated forward splits.

No data download or price fitting occurs here. Original sources are retained
and hashed; unsupported tax components reject instead of being approximated.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.build_observed_etf_bundles import ici_rows
from letf.accuracy import UnsupportedModelError

OTHER_ISSUERS={
    'SPY':{'provider':'State Street','files':{2025:('letf-ssga-ici-primary.xlsx',
        'https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/SPDR_ICI_Tax_Summary_Primary.xlsx')}},
    'TMF':{'provider':'Direxion','files':{y:(f'letf-direxion-ici-{y}.xlsx',f'https://www.direxion.com/uploads/{y}-ICI-Primary-Direxion-ETF.xlsx') for y in (2024,2025)}},
    'QQQ':{'provider':'Invesco','files':{
        2024:('letf-invesco-ici-2024.xlsx','https://www.invesco.com/content/dam/invesco/us/en/documents/tax-document/ce-ici-primary-and-secondary-distribution-file-file-ivz-etf-2024-02-10-2025.xlsx'),
        2025:('letf-invesco-ici-2025.xlsx','https://www.invesco.com/content/dam/invesco/us/en/documents/tax-documents/ICI-Primary-and-Secondary-and-NRA-File-IVZ-ETF-2025.xlsx')}}}


def check_rounded_cash(original,vendor,known_split_factor):
    """Verify a GIVEN split factor against vendor's 0.001/share cash quantum.

    Never estimate a split factor from rounded cash. Exact issuer cash is used
    for all tax/income events, even when vendor cash is compatible by rounding.
    """
    if vendor<=0 or abs(original-vendor*known_split_factor)>.000500001*known_split_factor:
        raise ValueError('Dated issuer cash is inconsistent with split actions and vendor reporting precision')


def ici_date(value):
    if '/' in str(value):
        return pd.to_datetime(value,format='%m/%d/%Y')
    return pd.Timestamp('1899-12-30')+pd.Timedelta(days=float(value))


def build(inputs, issuer_dir, output, symbol,*,macro_sources=None,years=None):
    reviewed_years=tuple(years) if years is not None else ((2025,) if symbol=='SPY' else (2024,2025))
    if (not reviewed_years or any(type(y)is not int for y in reviewed_years) or
        reviewed_years!=tuple(range(min(reviewed_years),max(reviewed_years)+1))):
        raise ValueError('A contiguous increasing range of final issuer tax years is required')
    if symbol in OTHER_ISSUERS and any(y not in OTHER_ISSUERS[symbol]['files'] for y in reviewed_years):
        raise UnsupportedModelError('No captured dated issuer taxonomy for requested years')
    output.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((inputs/'manifest.json').read_text())
    meta=catalog['sources'][symbol];file=inputs/meta['path']
    if sha256(file.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Source checksum mismatch')
    raw=pd.read_csv(file,index_col=0,parse_dates=True)
    splits=raw['Stock Splits'].loc[lambda s:s.ne(0)]
    scale=pd.Series(1.,index=raw.index)
    for day,ratio in splits.items():scale.loc[scale.index<day]*=ratio
    prices=raw.loc[f'{min(reviewed_years)-1}-01-01':f'{max(reviewed_years)}-12-31'].copy()
    for col in ('Open','High','Low','Close'):prices[col]*=scale.reindex(prices.index)
    rows=[];files={};source_checks=[]
    for year in reviewed_years:
        issuer=OTHER_ISSUERS.get(symbol)
        provider=issuer['provider'] if issuer else 'ProShares'
        original=issuer_dir/(issuer['files'][year][0] if issuer else f'letf-ici-primary-{year}.xlsx')
        original_url=issuer['files'][year][1] if issuer else f'https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_{year}.xlsx'
        restated=issuer_dir/f'letf-{symbol}-issuer-{year}.json'
        api=None if issuer else json.loads(restated.read_text())
        ici=[r for r in ici_rows(original,year=year) if r.get('C')==symbol]
        if not ici:
            # A zero payout year needs affirmative issuer evidence; an empty
            # ticker match cannot prove distribution completeness.
            raise UnsupportedModelError(f'Missing issuer {year} tax rows')
        num=lambda r,c:float(r.get(c) or 0.)
        for r in ici:
            # Supported corporate RIC subset. Never ignore 199A, foreign,
            # spillover, exempt, ROC or special-rate capital-gain components.
            excluded=('K','L','P','T','W','X','Y','Z','AA','AB','AC','AD','AE','AI','AK','AL','AM','AN','AO')
            if any(num(r,c) for c in excluded):raise UnsupportedModelError(f'{symbol}/{year}: additional tax components')
            total,ordinary,qdi,cg,reit=num(r,'J'),num(r,'Q'),num(r,'U'),num(r,'V'),num(r,'AJ')
            if not np.isclose(total,ordinary+cg,atol=1e-9) or not 0<=qdi+reit<=ordinary:
                raise ValueError('Issuer tax components do not reconcile')
            ex=ici_date(r['H'])
            pay=ici_date(r['I'])
            timing={}
            if pay.year!=year or ex.year!=year:
                record=ici_date(r['G'])
                if (symbol!='SPY' or pay.year!=year+1 or pay.month!=1 or ex.year!=year or
                    ex.month<10 or record.year!=year or record.month<10 or not np.isclose(num(r,'M'),total,atol=1e-9)):
                    raise UnsupportedModelError('Tax-year spillover lacks final issuer allocation evidence')
                timing={'tax_date':f'{year}-12-31','issuer_tax_year':year,'record_date':str(record.date()),
                    'tax_date_source':original_url+f'; final {year} ICI column M tax-year allocation, IRC 852(b)(7)'}
            restatement=None
            if api is not None:
                matches=[a for a in api if pd.Timestamp(a['ExDate'])==ex]
                if len(matches)!=1 or pd.Timestamp(matches[0]['PayableDate'])!=pay:raise ValueError('Issuer date disagreement')
                api_total=sum(float(matches[0].get(k) or 0) for k in ('CashDividendPerShare','LongTermCapGains','ShortTermCapGains','ReturnOfCapital'))
                restatement=total/api_total
                if not np.isclose(restatement,scale.loc[ex],rtol=2e-5,atol=1e-8):
                    raise ValueError('Dated/current issuer payouts conflict with split scaling')
            else:check_rounded_cash(total,float(raw.loc[ex,'Dividends']),float(scale.loc[ex]))
            source_checks.append({'ex_date':str(ex.date()),'nominal_cash':total,'issuer_restatement':restatement,
                'dated_split_factor':float(scale.loc[ex]),'rounded_vendor_cash':float(raw.loc[ex,'Dividends'])})
            for category,amount in [('ordinary_dividend',ordinary-qdi-reit),('qualified_dividend',qdi),
                                   ('section199a_dividend',reit),('capital_gain_distribution',cg)]:
                if amount:rows.append({'ex_date':ex,'pay_date':str(pay.date()),'amount_per_share':amount,
                    'category':category,'qualified_verified':category=='qualified_dividend',
                    'section199a_verified':category=='section199a_dividend',
                    'source':original_url,**timing})
        issuer_dates={ici_date(r['H']) for r in ici}
        if issuer_dates!=set(raw.loc[str(year)].loc[lambda f:f.Dividends.ne(0)].index):raise ValueError('Incomplete cash payout coverage')
        source_files=[(original,original_url)]
        if api is not None:source_files.append((restated,f'https://www.proshares.com/api/distributionsummary/?fund={symbol}&year={year}'))
        for src,url in source_files:
            shutil.copyfile(src,output/src.name)
            files[src.stem]={'path':src.name,'sha256':sha256(src.read_bytes()).hexdigest(),'provider':provider,
                             'url':url,'format':src.suffix[1:]}
    prices.to_csv(output/'prices.csv',index_label='Date',float_format='%.12g')
    pd.DataFrame(rows).set_index('ex_date').sort_index().to_csv(output/'distributions.csv')
    for name,provider,url in [('prices',meta['provider'],meta['url']),('distributions',provider+' dated ICI records',original_url)]:
        p=output/f'{name}.csv';files[name]={'path':p.name,'sha256':sha256(p.read_bytes()).hexdigest(),
            'provider':provider,'url':url,'format':'csv'}
    if macro_sources is not None:
        from letf.macro.data import HistoricalSources
        sources=HistoricalSources(macro_sources);blob,cpi_meta=sources.bytes('CPIAUCNS')
        (output/'cpi.csv').write_bytes(blob)
        files['cpi']={'path':'cpi.csv','sha256':sha256(blob).hexdigest(),'provider':cpi_meta['provider'],
            'url':cpi_meta['url'],'format':'csv','retrieved_at':cpi_meta['retrieved_at'],
            'scope':'Observed CPI endpoints used ex post only; missing interior observations retained'}
    active_splits=splits.loc[prices.index.min():prices.index.max()]
    manifest={'symbol':symbol,'instrument_kind':'etf','tax_structure':'corporation','currency':'USD',
        'frequency':'daily','data_status':'OBSERVED','price_convention':'as_traded','distributions_complete':True,
        'account_coverage_start':str(prices.loc[f'{min(reviewed_years)}-01-01':].index[0].date()),
        'account_coverage_end':str(prices.index[-1].date()),
        'signal_coverage_start':f'{min(reviewed_years)}-01-01',
        'reviewed_distribution_tax_years':list(reviewed_years),
        'supported_tax_states':['FEDERAL_ONLY','TX','FL','NV'],'corporate_actions_reviewed':True,
        'split_events':[{'date':str(d.date()),'security':symbol,'ratio':float(v),
            'source':'Dated vendor split actions crosschecked against original/restated issuer cash payouts'} for d,v in active_splits.items()],
        'retrieved_at':datetime.now(timezone.utc).isoformat(),'original_vendor_sha256':meta['sha256'],
        'original_vendor_capture':catalog['retrieved_at'],'files':files,'issuer_cross_checks':source_checks,
        'limitations':['Actual daily observations; assumed open/close execution, not NBBO fills',
            'Final fund tax classifications enter tax accounting only, never trading signals',
            'State-specific government-obligation exemptions and extra tax components remain gated']}
    if symbol=='SPY':
        manifest['supported_strategies']=['buy_hold']
        manifest['limitations'].append('SPY 2024 final classification not captured; 2025 buy/hold only, no distribution-complete 2024 signal warmup')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('inputs','issuer-dir','output'):p.add_argument('--'+a,required=True,type=Path)
    p.add_argument('--symbol',required=True,choices=['SPY','TQQQ','QLD','SSO','UPRO','QQQ','TMF'])
    p.add_argument('--macro-sources',type=Path)
    p.add_argument('--years',type=int,nargs='+',help='Contiguous final issuer tax years; no missing year is inferred')
    a=p.parse_args();m=build(a.inputs,a.issuer_dir,a.output,a.symbol,macro_sources=a.macro_sources,years=a.years)
    print(json.dumps({'symbol':a.symbol,'issuer_checks':len(m['issuer_cross_checks']),'splits':m['split_events']},indent=2))
