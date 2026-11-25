"""Public credential-free macro/factor capture into an immutable NEW directory.

Raw source files stay local. Current revisions, vintage data and publication
metadata remain distinct; a requested vintage must be confirmed in its header.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import requests

FRENCH='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FRENCH_SOURCES={
 'ff_daily':'F-F_Research_Data_Factors_daily_CSV.zip',
 'ff_monthly':'F-F_Research_Data_Factors_CSV.zip',
 'size_value_daily':'6_Portfolios_2x3_daily_CSV.zip',
 'industry_daily':'10_Industry_Portfolios_daily_CSV.zip'}
FRED={
 'CPIAUCNS':('monthly','BLS','CPI-U NSA level; observation month is not release date'),
 'INDPRO':('monthly','Federal Reserve','Industrial production index, seasonally adjusted, revised'),
 'BAA':('monthly',"Moody's via Federal Reserve",'Seasoned Baa yield, percent; provider copyright restrictions'),
 'AAA':('monthly',"Moody's via Federal Reserve",'Seasoned Aaa yield, percent; provider copyright restrictions'),
 'GS10':('monthly','Federal Reserve/Treasury','10-year constant maturity par yield; not bond total return'),
 'UNRATE':('monthly','BLS','Unemployment percent, seasonally adjusted, revised'),
 'USREC':('monthly','NBER via Federal Reserve','Ex-post business cycle labels; forbidden as contemporaneous predictor'),
 'GDPC1':('quarterly','BEA','Real GDP, seasonally adjusted annual rate, revised; no monthly interpolation'),
 'DFF':('daily','Federal Reserve','Effective federal funds annual percent; not broker swap financing'),
 'DTB3':('daily','Federal Reserve/Treasury','3-month T-bill discount yield; not a total-return series'),
 'GOLDAMGBD228NLBM':('daily','LBMA via FRED','London gold fix; monetary regimes and rights require review')}
VINTAGE_IDS=('CPIAUCNS','INDPRO','BAA','AAA','GS10','UNRATE')


def capture(output,protocol,local_ff=None,reuse=None):
    output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(protocol.read_text())
    requests_list=[]
    previous=json.loads((reuse/'catalog.json').read_text()) if reuse else None
    for name,file in FRENCH_SOURCES.items():
        requests_list.append((name,FRENCH+file,'zip','monthly' if name=='ff_monthly' else 'daily','Kenneth French Data Library',
            'Research returns, current CIZ vintage; dividends reinvested, not executable ETF prices',None))
    for name,(freq,provider,convention) in FRED.items():
        requests_list.append((name,f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}','csv',freq,provider,convention,None))
    for origin in spec['origins']:
        for name in VINTAGE_IDS:
            key=f'{name}_asof_{origin}'
            url=f'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={name}&cosd=1900-01-01&coed={origin}&vintage_date={origin}&revision_date={origin}'
            freq,provider,convention=FRED[name]
            requests_list.append((key,url,'csv',freq,provider,convention,origin))
    def fetch(item):
        name,url,fmt,freq,provider,convention,vintage=item
        timestamp=datetime.now(timezone.utc).isoformat()
        try:
            if previous and name in previous['sources']:
                old=previous['sources'][name];source=reuse/old['path']
                if old['url']==url and sha256(source.read_bytes()).hexdigest()==old['sha256']:
                    shutil.copyfile(source,output/old['path'])
                    return name,dict(old,reused_from_catalog_sha256=sha256((reuse/'catalog.json').read_bytes()).hexdigest()),None
            if name=='ff_daily' and local_ff:
                blob=local_ff.read_bytes()
            else:
                response=requests.get(url,timeout=35);response.raise_for_status();blob=response.content
            filename=f'{name}.{fmt}';path=output/filename
            entry={'path':filename,'url':url,'provider':provider,'retrieved_at':timestamp,
                'publication_vintage':vintage or 'Current revised source at capture; historical release vintage not supplied',
                'frequency':freq,'convention':convention,'currency':'USD','sha256':sha256(blob).hexdigest(),
                'license':'Local research capture only; redistribution not certified; retain original provider terms',
                'license_url':'https://fred.stlouisfed.org/legal/' if fmt=='csv' else 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html',
                'point_in_time_verified':False,'role':'ex_post_diagnostic_only' if name=='USREC' else 'research_input'}
            if fmt=='csv':
                frame=pd.read_csv(io.BytesIO(blob),index_col=0,parse_dates=True)
                if frame.empty or len(frame.columns)!=1:raise ValueError('Unexpected observation schema')
                values=pd.to_numeric(frame.iloc[:,0],errors='coerce')
                valid=values.dropna()
                if valid.empty:raise ValueError('No observations available in requested vintage')
                entry.update(first_observation=str(valid.index.min().date()),last_observation=str(valid.index.max().date()),
                    observations=len(valid),missing_observations=int(values.isna().sum()))
                if vintage:
                    expected=f'{name.split("_asof_")[0]}_{vintage.replace("-","")}'
                    if frame.columns[0]!=expected or valid.index.max()>pd.Timestamp(vintage):
                        raise ValueError('Provider did not confirm the requested vintage/header/date boundary')
                    entry['point_in_time_verified']=True
            else:
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    text=z.read(next(n for n in z.namelist() if n.lower().endswith('.csv'))).decode('latin-1')
                entry['source_header']=text.splitlines()[:6]
            path.write_bytes(blob)
            return name,entry,None
        except Exception as exc:
            return name,None,{'url':url,'reason':str(exc),'requested_vintage':vintage,'status':'BLOCKED'}
    catalog={'captured_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(protocol.read_bytes()).hexdigest(),
        'sources':{},'blocked':{},'raw_data_redistribution':'Not certified; local snapshots only',
        'market_revision_warning':'French current CIZ includes retrospective revisions and RF source changes. Archived macro vintage does not make the equity history point-in-time.'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,entry,error in pool.map(fetch,requests_list):
            if error:catalog['blocked'][name]=error
            else:catalog['sources'][name]=entry
            print(name,'BLOCKED' if error else 'captured',flush=True)
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    return catalog


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--protocol',type=Path,default=Path('docs/macro-learning-2026-11-24/protocol.json'))
    p.add_argument('--local-ff',type=Path)
    p.add_argument('--reuse',type=Path)
    a=p.parse_args();c=capture(a.output,a.protocol,a.local_ff,a.reuse)
    print(json.dumps({'captured':len(c['sources']),'blocked':len(c['blocked'])}))
