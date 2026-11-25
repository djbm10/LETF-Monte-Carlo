"""Build 2024 account inputs from frozen vendor observations and issuer ICI data.

No future-price fitting: a constant split-restatement factor is established by
matching ALL dated issuer nominal cash payouts to vendor ex-date payouts.
Only no-in-period-split histories and supported tax components are accepted.
An optional exact issuer restated schedule replaces rounded vendor payouts
when establishing nominal scale; both original observations are retained.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError

TAX_URL='https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_2024.xlsx'


def ici_rows(path, *, year=2024):
    """Explicit issuer primary layouts; unrecognized fields fail closed.

    The 2021/22 two-line header has the same reviewed column meanings. Legacy
    binary XLS uses pinned xlrd; no formula execution or spreadsheet macros.
    """
    if Path(path).suffix.lower()=='.xls':
        try:import xlrd
        except ImportError as error:raise UnsupportedModelError('Binary issuer XLS requires pinned xlrd from requirements-audit.lock') from error
        book=xlrd.open_workbook(path)
        if book.datemode!=0:raise UnsupportedModelError('Unreviewed 1904 Excel date system')
        sheet=book.sheet_by_index(0)
        rows=[{xlrd.formula.colname(j):str(sheet.cell_value(i,j)) for j in range(sheet.ncols)} for i in range(sheet.nrows)]
        return validate_ici_headers(rows,year)
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[''.join(el.itertext()) for el in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        tree=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows=[]
    for node in tree.findall('.//m:sheetData/m:row',ns):
        row={}
        for cell in node:
            value=cell.find('m:v',ns)
            text=value.text if value is not None else ''
            if cell.get('t')=='s' and text:text=strings[int(text)]
            row[re.sub(r'\d','',cell.get('r'))]=text
        rows.append(row)
    return validate_ici_headers(rows,year)


def validate_ici_headers(rows,year):
    headers=[r for r in rows if r.get('A')=='Security Description (Fund Name)']
    if not headers and year in (2021,2022):
        candidates=[i for i,r in enumerate(rows[:-1]) if r.get('A')=='Security' and rows[i+1].get('A')=='Description']
        required={'C':'Ticker','H':'Ex-Dividend','I':'Payable','J':'Per Share','M':'(Current Year)',
                  'Q':'Dividends','U':'Dividends*','V':'Gain Distr.','Z':'Nondividend',
                  'AB':'Liquidation','AC':'Liquidation','AD':'Interest','AJ':'Dividends*'}
        if len(candidates)!=1:raise UnsupportedModelError('Unknown legacy issuer primary header')
        i=candidates[0];top,bottom=rows[i],rows[i+1]
        if (float(top.get('M',0))!=year or top.get('Q','').strip()!='Ordinary' or
            any(bottom.get(c,'').strip()!=v for c,v in required.items())):
            raise UnsupportedModelError('Legacy issuer primary columns/year do not match reviewed layout')
        return rows
    if len(headers)!=1 or headers[0].get('H')!='Ex-Dividend Date' or headers[0].get('M','').split()[0]!=str(year) or headers[0].get('Q')!='Ordinary Dividends (14+15+16)':
        raise UnsupportedModelError('Unrecognized issuer ICI primary 2024 column layout')
    return rows


def nominal_bundle(frame, rows, symbol, restated_issuer_rows=None):
    source=[r for r in rows if r.get('C')==symbol]
    if not source:raise UnsupportedModelError('Ticker absent from the dated issuer tax supplement')
    def number(row,column):return float(row.get(column) or 0.)
    # J total cash, K/L prior/next-year allocation, Q ordinary, R qualified.
    # Reject REIT/foreign/exempt/special-rate/capital-gain components rather
    # than silently folding them into the ordinary corporate tax subset.
    records=[];ratios=[]
    for r in source:
        total=number(r,'J');ordinary=number(r,'Q');qualified=number(r,'R')
        if any(number(r,c) for c in ('K','L','O','P','S','T','V','W','X','Y','Z','AA','AB','AC','AD','AE','AG','AH','AI','AJ','AK','AL','AM','AN','AO')):
            raise UnsupportedModelError(f'{symbol}: special distribution components require additional tax rules')
        if (not np.isclose(ordinary,total,atol=1e-9) or not 0<=qualified<=ordinary or
            ('U' in r and not np.isclose(number(r,'U'),qualified,atol=1e-9))):
            raise UnsupportedModelError('ICI tax character does not reconcile to total cash distribution')
        ex=pd.Timestamp('1899-12-30')+pd.Timedelta(days=number(r,'H'))
        pay=pd.Timestamp('1899-12-30')+pd.Timedelta(days=number(r,'I'))
        if ex.year!=2024 or pay.year!=2024:
            raise UnsupportedModelError('Year-end spillover distributions require separate tax-year mapping')
        vendor=float(frame.loc[ex,'Dividends'])
        if vendor<=0:raise ValueError('Issuer payout missing from vendor ex-date actions')
        if restated_issuer_rows is None:
            ratios.append(total/vendor)
        else:
            matches=[row for row in restated_issuer_rows if pd.Timestamp(row['ExDate'])==ex]
            if len(matches)!=1 or pd.Timestamp(matches[0]['PayableDate'])!=pay:
                raise ValueError('Dated and current issuer payable/ex-date records disagree')
            ratios.append(total/float(matches[0]['CashDividendPerShare']))
        for category,value in [('ordinary_dividend',ordinary-qualified),('qualified_dividend',qualified)]:
            if value:records.append({'ex_date':ex,'pay_date':pay.date(),'amount_per_share':value,
                'category':category,'source':TAX_URL,'qualified_verified':category=='qualified_dividend'})
    factor=float(np.median(ratios))
    if not np.allclose(ratios,factor,rtol=2e-5,atol=1e-7):
        raise UnsupportedModelError('Nominal scaling not constant across issuer distributions')
    known_splits=frame.loc[frame.index>='2025-01-01','Stock Splits']
    known_factor=float(known_splits.loc[known_splits.ne(0)].prod())
    if not np.isclose(factor,known_factor,rtol=2e-5):
        raise UnsupportedModelError('Issuer restatement factor conflicts with dated vendor split actions')
    sample=frame.loc['2023-01-01':'2024-12-31'].copy()
    if sample['Stock Splits'].ne(0).any():
        raise UnsupportedModelError('In-period splits require a reviewed action schedule')
    vendor_dates=set(frame.loc['2024'].loc[lambda d:d.Dividends.ne(0)].index)
    if vendor_dates!=set(pd.Timestamp('1899-12-30')+pd.Timedelta(days=number(r,'H')) for r in source):
        raise UnsupportedModelError('Issuer/vendor distribution coverage mismatch')
    for column in ('Open','High','Low','Close','Dividends'):
        sample[column]*=factor
    return sample,pd.DataFrame(records).set_index('ex_date').sort_index(),factor


def build(inputs,tax_supplement,output,issuer_distributions_dir=None):
    output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((inputs/'manifest.json').read_text());rows=ici_rows(tax_supplement)
    report={}
    for symbol in ('TQQQ','QLD','SSO','UPRO'):
        source=metadata['sources'][symbol];file=inputs/source['path']
        if sha256(file.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Frozen input checksum mismatch')
        frame=pd.read_csv(file,index_col=0,parse_dates=True)
        issuer_file=issuer_distributions_dir/f'letf-{symbol}-issuer-2024.json' if issuer_distributions_dir else None
        issuer_rows=json.loads(issuer_file.read_text()) if issuer_file and issuer_file.is_file() else None
        try:prices,distributions,factor=nominal_bundle(frame,rows,symbol,issuer_rows)
        except UnsupportedModelError as exc:
            report[symbol]={'status':'BLOCKED','reason':str(exc)};continue
        folder=output/symbol;folder.mkdir()
        prices.to_csv(folder/'prices.csv',index_label='Date',float_format='%.12g')
        distributions.to_csv(folder/'distributions.csv',index_label='ex_date')
        shutil.copyfile(tax_supplement,folder/'ici_primary_2024.xlsx')
        if issuer_rows is not None:shutil.copyfile(issuer_file,folder/'issuer-restated-distributions.json')
        files={}
        for key,name,provider,url,fmt in [('prices','prices.csv',source['provider'],source['url'],'csv'),
              ('distributions','distributions.csv','ProShares dated 2024 ICI primary supplement',TAX_URL,'csv'),
              ('issuer_tax_original','ici_primary_2024.xlsx','ProShares',TAX_URL,'xlsx')]:
            files[key]={'path':name,'sha256':sha256((folder/name).read_bytes()).hexdigest(),
                        'provider':provider,'url':url,'format':fmt}
        if issuer_rows is not None:
            path=folder/'issuer-restated-distributions.json'
            files['issuer_restated']={'path':path.name,'sha256':sha256(path.read_bytes()).hexdigest(),
                'provider':'ProShares','url':f'https://www.proshares.com/api/distributionsummary/?fund={symbol}&year=2024','format':'json'}
        manifest={'symbol':symbol,'instrument_kind':'etf','tax_structure':'corporation','currency':'USD',
            'frequency':'daily','data_status':'OBSERVED','price_convention':'as_traded',
            'distributions_complete':True,'account_coverage_start':'2024-01-02','account_coverage_end':'2024-12-31',
            'supported_tax_states':['FEDERAL_ONLY','TX','FL','NV'],
            'retrieved_at':datetime.now(timezone.utc).isoformat(),'files':files,
            'nominal_price_scaling':{'factor':factor,'method':'Dated nominal/current restated issuer payouts crosschecked against dated vendor splits',
                'original_vendor_sha256':source['sha256'],'original_vendor_capture':metadata['retrieved_at'],
                'relative_tolerance_frozen':2e-5,'in_period_splits':False},
            'limitations':['Vendor daily OHLC is not NBBO execution evidence',
                'Rounded vendor dividend amounts are replaced by dated issuer tax-supplement amounts',
                'Qualified classification requires separate investor holding-period proof at each ex-date',
                'No support for state government-obligation exclusions or special distribution components',
                '2024 tax classification is ex-post information; never used in strategy signals']}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report[symbol]={'status':'OBSERVED','nominal_price_factor':factor,'distributions':len(distributions)}
    (output/'build-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--tax-supplement',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--issuer-distributions-dir',type=Path)
    a=p.parse_args();build(a.inputs,a.tax_supplement,a.output,a.issuer_distributions_dir)
