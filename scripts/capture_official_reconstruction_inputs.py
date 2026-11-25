"""Archive public issuer NAV/distributions and Nasdaq-origin index observations.

Local research snapshot only. Public download does NOT confer redistribution
rights; copyrighted Nasdaq/index and issuer records remain outside Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import io
import json
from pathlib import Path

import pandas as pd
import requests


def capture(output,*,offline_cache=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    captured=datetime.now(timezone.utc).isoformat();items=[]
    for s in ('TQQQ','QLD','UPRO','SSO'):
        items.append((f'{s}_nav',f'https://accounts.profunds.com/etfdata/ByFund/{s}-historical_nav.csv','ProShares','csv'))
        for year in range(2006,2026):
            items.append((f'{s}_distributions_{year}',f'https://www.proshares.com/api/distributionsummary/?fund={s}&year={year}','ProShares','json'))
    for series in ('NASDAQXNDX','NASDAQ100'):
        items.append((series,f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1985-01-01&coed=2026-11-24','Nasdaq, Inc. via Federal Reserve FRED','csv'))
    def fetch(item):
        name,url,provider,fmt=item
        try:
            previous=None
            if offline_cache is not None:
                cache=json.loads((offline_cache/'catalog.json').read_text());previous=cache['sources'].get(name)
            if previous is not None:
                blob=(offline_cache/previous['path']).read_bytes()
                if previous['url']!=url or sha256(blob).hexdigest()!=previous['sha256']:raise ValueError('Invalid retained snapshot')
            else:
                response=requests.get(url,timeout=45);response.raise_for_status();blob=response.content
            info={}
            if fmt=='json':
                rows=json.loads(blob)
                if not isinstance(rows,list):raise ValueError('Issuer distribution schema changed')
                info={'records':len(rows),'convention':'Issuer historical distribution API; inspect split restatement against dated NAV before use'}
            else:
                frame=pd.read_csv(io.BytesIO(blob));col='Date' if name.endswith('_nav') else frame.columns[0]
                dates=pd.to_datetime(frame[col],format='%m/%d/%Y' if name.endswith('_nav') else '%Y-%m-%d')
                if not dates.is_unique:raise ValueError('Duplicate observations')
                info={'first_observation':str(dates.min().date()),'last_observation':str(dates.max().date()),'records':len(frame),
                    'convention':'Issuer split-restated NAV levels, NOT dividend-adjusted' if name.endswith('_nav') else 'Official Nasdaq index levels; XNDX total return, NASDAQ100 price return'}
            path=output/f'{name}.{fmt}';path.write_bytes(blob)
            return name,{'path':path.name,'url':url,'provider':provider,'format':fmt,'sha256':sha256(blob).hexdigest(),
                'retrieved_at':previous['retrieved_at'] if previous else captured,
                'archived_reuse':previous is not None,'frequency':'daily' if fmt=='csv' else 'dated cash events','currency':'USD',
                'publication_vintage':'Current downloadable revision; not historical decision-time vintage',
                'license':'Copyright retained by provider; local research only, redistribution not certified',**info},None
        except Exception as exc:return name,None,{'url':url,'reason':str(exc),'status':'BLOCKED'}
    sources={};blocked={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,entry,error in pool.map(fetch,items):
            if entry:sources[name]=entry
            else:blocked[name]=error
    catalog={'retrieved_at':captured,'sources':sources,'blocked':blocked,
        'scope':'Local research inputs, no redistribution/real-time-vintage certification'}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');return catalog


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline-cache',type=Path);a=p.parse_args()
    r=capture(a.output,offline_cache=a.offline_cache);print(json.dumps({'sources':len(r['sources']),'blocked':r['blocked']},indent=2))
