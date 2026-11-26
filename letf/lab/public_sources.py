"""Bounded, explicit refresh of authorized public feeds; never auto-download.

Only registered HTTPS sources. New snapshots never replace old vintages. This
adapter archives observations, not final tax classes or executable fill tapes.
"""
from datetime import datetime, timezone
from io import BytesIO
import json
import zipfile
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd
import requests

from letf.lab.catalog import SourceLake
from letf.macro.data import french_daily

FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
REGISTERED = {
    'ff_daily': (FRENCH + 'F-F_Research_Data_Factors_daily_CSV.zip', 'Kenneth French Data Library', 'french_daily'),
    'ff_monthly': (FRENCH + 'F-F_Research_Data_Factors_CSV.zip', 'Kenneth French Data Library', 'french_monthly'),
    'size_value_daily': (FRENCH + '6_Portfolios_2x3_daily_CSV.zip', 'Kenneth French Data Library', 'french_daily'),
    'industry_daily': (FRENCH + '10_Industry_Portfolios_daily_CSV.zip', 'Kenneth French Data Library', 'french_daily'),
}
for _series in ('CPIAUCNS', 'INDPRO', 'UNRATE', 'GS10', 'DFF', 'DTB3', 'BAA', 'AAA', 'SP500', 'NASDAQXNDX', 'NASDAQ100'):
    REGISTERED[_series] = (f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={_series}',
                           'Original provider via Federal Reserve FRED; inspect series terms', 'fred')
for _symbol in ('TQQQ', 'QLD', 'UPRO', 'SSO'):
    REGISTERED[_symbol + '_nav'] = (f'https://accounts.profunds.com/etfdata/ByFund/{_symbol}-historical_nav.csv', 'ProShares', 'nav')
    REGISTERED[_symbol + '_cash_2026'] = (f'https://www.proshares.com/api/distributionsummary/?fund={_symbol}&year=2026','ProShares','cash')
ALLOWED_HOSTS = frozenset(urlsplit(v[0]).hostname for v in REGISTERED.values())


def download(url, *, session=None, max_bytes=64 * 1024 * 1024):
    client = session or requests.Session()
    for _ in range(4):
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or
                parsed.username or parsed.password or parsed.port not in (None, 443)):
            raise ValueError('Public refresh endpoint outside registered HTTPS providers')
        with client.get(url, timeout=(10, 45), allow_redirects=False, stream=True) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, response.headers['Location']); continue
            response.raise_for_status()
            pieces = []; count = 0
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                if count > max_bytes: raise ValueError('Public source exceeds reviewed download size')
                pieces.append(block)
            return b''.join(pieces), {'final_url': url, 'etag': response.headers.get('ETag'),
                'last_modified': response.headers.get('Last-Modified'), 'content_type': response.headers.get('Content-Type')}
    raise ValueError('Public source redirect limit exceeded')


def inspect(name, blob):
    if name not in REGISTERED: raise ValueError('Source is not registered')
    fmt = REGISTERED[name][2]
    if fmt=='cash':
        records=json.loads(blob)
        if not isinstance(records,list) or not records:raise ValueError('Issuer cash schema empty/changed; absence is not proof of zero payouts')
        dates=[]
        for row in records:
            ex=pd.Timestamp(row['ExDate']);pay=pd.Timestamp(row['PayableDate'])
            if ex.year!=2026 or pay<ex or ex.date()>datetime.now(timezone.utc).date():raise ValueError('Invalid issuer cash dates')
            for key in ('CashDividendPerShare','LongTermCapGains','ShortTermCapGains','ReturnOfCapital','OtherPerShare','SpecialPerShare'):
                amount=float(row.get(key) or 0.)
                if not np.isfinite(amount) or amount<0:raise ValueError('Invalid issuer cash amount')
            dates.append(ex)
        if len(dates)!=len(set(dates)):raise ValueError('Duplicate cash ex-date')
        return {'first_observation':str(min(dates).date()),'last_observation':str(max(dates).date()),
            'rows':len(records),'format':'json','currency':'USD','frequency':'dated cash events',
            'convention':'Current split-restated issuer cash API; no final2026 tax classification or completeness certification',
            'interpolated':False,'final_tax_classes_verified':False}
    if fmt.startswith('french'):
        frame = french_daily(blob, date_digits=6 if fmt == 'french_monthly' else 8)
        convention = 'Research portfolio total returns, current revised CIZ vintage; not executable prices'
        frequency = 'monthly' if fmt == 'french_monthly' else 'daily'
    else:
        frame = pd.read_csv(BytesIO(blob))
        if fmt == 'nav':
            if 'Date' not in frame or 'NAV' not in frame: raise ValueError('Issuer NAV schema changed')
            dates = pd.to_datetime(frame.pop('Date'), format='%m/%d/%Y')
            frame = frame[['NAV']].apply(pd.to_numeric, errors='raise')
            convention = 'Current split-restated issuer NAV; NOT dividend-adjusted or executable market prices'
        else:
            if frame.shape[1] != 2 or frame.columns[1] != name: raise ValueError('FRED series schema changed')
            dates = pd.to_datetime(frame.iloc[:, 0], format='%Y-%m-%d')
            # FRED documents '.' as missing. Unexpected text is corruption or
            # a changed schema, never an observation silently turned into NaN.
            frame = frame.iloc[:, 1:].replace('.',np.nan).apply(pd.to_numeric, errors='raise')
            convention = ('Nasdaq-origin total-return index' if name == 'NASDAQXNDX' else
                'Price-return index, excludes dividends' if name in ('SP500', 'NASDAQ100') else
                'Observed macro/rate series in provider units; yield is not a bond total return')
        frame.index = pd.DatetimeIndex(dates)
        frame = frame.sort_index()
        frequency = 'provider observation frequency; gaps retained'
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Source dates empty, duplicate or out of order')
    values = frame.to_numpy(float)
    if np.isinf(values).any() or not np.isfinite(values).any(): raise ValueError('Invalid source values')
    finite_dates = frame.dropna(how='all').index
    if finite_dates.max().date() > datetime.now(timezone.utc).date(): raise ValueError('Future source observation')
    extra={}
    if fmt.startswith('french'):
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            name_in_zip=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
            extra['source_header']=archive.read(name_in_zip).decode('latin-1').splitlines()[:6]
    return {'first_observation': str(finite_dates.min().date()), 'last_observation': str(finite_dates.max().date()),
            'rows': len(frame), 'missing_values': int(np.isnan(values).sum()), 'columns': list(frame),
            'frequency': frequency, 'convention': convention, 'interpolated': False,
            'format':'zip' if fmt.startswith('french') else 'csv',
            'currency':None if name in ('CPIAUCNS','INDPRO','UNRATE') else 'USD',**extra}


def refresh(lake, names, *, downloader=download):
    if not names or len(names) != len(set(names)) or set(names) - set(REGISTERED):
        raise ValueError('Explicit unique registered source names required')
    lake = lake if isinstance(lake, SourceLake) else SourceLake(lake)
    results = {}
    for name in names:
        url, provider, _ = REGISTERED[name]
        try:
            blob, headers = downloader(url)
            details = inspect(name, blob)
            metadata = {'provider': provider, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'publication_vintage': 'Current provider revision at capture; historical release vintage not supplied',
                'point_in_time_verified': False, 'license': 'Provider rights retained; private local research only; redistribution NOT approved',
                'license_url': 'https://fred.stlouisfed.org/legal/' if 'fred.stlouisfed.org' in url else
                               'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html' if 'dartmouth.edu' in url else
                               'https://www.proshares.com/resources/terms-and-conditions', **headers, **details}
            results[name] = {'status': 'CAPTURED', 'snapshot': lake.put(name, blob, metadata), **details}
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            results[name] = {'status': 'BLOCKED', 'reason': str(error), 'url': url}
    return {'classification': 'OBSERVED_MARKET', 'financial_certification': 'BLOCKED', 'sources': results,
            'complete': all(r['status'] == 'CAPTURED' for r in results.values())}
