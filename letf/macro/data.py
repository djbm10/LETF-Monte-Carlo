"""Source-checked historical observations and origin-specific macro panels.

Monthly macro observations are never filled into fabricated daily prices.
Archived releases are kept separate from current revised history. A current
French return vintage is not silently called a historical publication vintage.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError

CORE=('equity_log','inflation_log','short_rate','credit_spread','production_log')


def french_daily(blob, *, date_digits=8):
    """Read the FIRST value-weighted daily block; reject duplicates/sentinels."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text=z.read(next(n for n in z.namelist() if n.lower().endswith('.csv'))).decode('latin-1')
    header=None;rows=[]
    for line in text.splitlines():
        cells=[c.strip() for c in line.split(',')]
        if header is None and len(cells)>2 and cells[0]=='':header=cells[1:];continue
        if re.fullmatch(r'\d{'+str(date_digits)+'}',cells[0] or ''):
            if header is None:raise ValueError('Missing daily field definitions')
            if len(cells)!=len(header)+1:raise ValueError('Daily row width mismatch')
            values=np.asarray([float(v) for v in cells[1:]])
            if not np.isfinite(values).all() or (values<=-99).any():
                raise UnsupportedModelError('Daily missing/sentinel values require an explicit shorter supported interval')
            day=pd.to_datetime(cells[0],format='%Y%m%d' if date_digits==8 else '%Y%m')
            if date_digits==6:day=day+pd.offsets.MonthEnd(0)
            rows.append((day,*list(values/100)))
        elif rows:break  # never append equal-weighted or annual sections
    if not rows:raise ValueError('No daily observations')
    frame=pd.DataFrame(rows,columns=['date',*header]).set_index('date')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:raise ValueError('Invalid daily chronology')
    return frame


class HistoricalSources:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.catalog=json.loads((self.folder/'catalog.json').read_text())
        self.catalog_hash=sha256((self.folder/'catalog.json').read_bytes()).hexdigest()
        self._daily_cache={}

    def bytes(self,name):
        if name not in self.catalog['sources']:
            raise UnsupportedModelError(f'Source unavailable: {name}')
        item=self.catalog['sources'][name];path=self.folder/item['path']
        if not path.resolve().is_relative_to(self.folder.resolve()):raise ValueError('Source escapes bundle')
        blob=path.read_bytes()
        if sha256(blob).hexdigest()!=item['sha256']:raise ValueError(f'Immutable source checksum mismatch: {name}')
        return blob,item

    def series(self,name):
        blob,meta=self.bytes(name)
        frame=pd.read_csv(io.BytesIO(blob),index_col=0,parse_dates=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:raise ValueError('Invalid source chronology')
        values=pd.to_numeric(frame.iloc[:,0],errors='coerce')
        if np.isinf(values).any():raise ValueError('Infinite source observations')
        values.index=values.index.to_period('M').to_timestamp('M')
        return values,meta

    def daily_factors(self):
        blob,meta=self.bytes('ff_daily')
        if meta['sha256'] not in self._daily_cache:self._daily_cache[meta['sha256']]=french_daily(blob)
        return self._daily_cache[meta['sha256']].copy()

    def panel(self,origin=None,*,require_macro_vintage=False,include_long_yield=False):
        daily=self.daily_factors()
        returns=daily['Mkt-RF']+daily.RF
        monthly=pd.DataFrame({'equity_log':np.log1p(returns).resample('ME').sum(),
            'short_rate':np.expm1(np.log1p(daily.RF).resample('ME').sum()*12)})
        used={'ff_daily':self.catalog['sources']['ff_daily']['sha256']};vintages={};missing={}
        if 'ff_monthly' in self.catalog['sources']:
            ff_month=french_daily(self.bytes('ff_monthly')[0],date_digits=6)
            monthly['short_rate']=np.expm1(np.log1p(ff_month.RF)*12).reindex(monthly.index)
            used['ff_monthly']=self.catalog['sources']['ff_monthly']['sha256']
        cutoff=pd.Timestamp(origin) if origin else monthly.index[-1]
        def macro(name):
            key=f'{name}_asof_{cutoff.date()}'
            if origin and key in self.catalog['sources']:
                values,meta=self.series(key)
                if not meta['point_in_time_verified']:raise ValueError('Unconfirmed macro vintage')
                vintages[name]=meta['publication_vintage'];used[key]=meta['sha256']
            else:
                if require_macro_vintage and origin:raise UnsupportedModelError(f'No archived macro vintage at origin: {name}')
                values,meta=self.series(name);used[name]=meta['sha256']
                vintages[name]='CURRENT_REVISED_PSEUDO_REAL_TIME'
                # Conservative publication lag for current-vintage research;
                # not a substitute for unavailable revision vintages.
                if origin:values=values.loc[:cutoff-pd.offsets.MonthEnd(2)]
            missing[name]=int(values.isna().sum())
            return values
        cpi=macro('CPIAUCNS');production=macro('INDPRO')
        monthly['inflation_log']=np.log(cpi).diff().reindex(monthly.index)
        monthly['production_log']=np.log(production).diff().reindex(monthly.index)
        monthly['credit_spread']=(macro('BAA')-macro('AAA')).reindex(monthly.index)/100
        if include_long_yield:monthly['long_yield']=macro('GS10').reindex(monthly.index)/100
        monthly=monthly.loc[:cutoff]
        metadata={'source_hashes':used,'catalog_sha256':self.catalog_hash,'macro_vintages':vintages,
            'missing_source_rows':missing,'requested_origin':str(cutoff.date()),
            'point_in_time_verified':False,
            'macro_vintages_verified':all(v!='CURRENT_REVISED_PSEUDO_REAL_TIME' for v in vintages.values()),
            'market_vintage':'Current CIZ French vintage; retrospective corrections not historical decision-time records',
            'short_rate_convention':'Annualized compounded monthly French RF return; funding proxy, not an observed borrowing invoice',
            'growth_convention':'Industrial production growth, not GDP',
            'frequency':'monthly','variables':list(monthly.columns),'evidence':'OBSERVED',
            'limitations':['Observed factors are research portfolios, not executable ETF quotes',
                'Missing interior monthly observations are retained as NaN; never interpolated']}
        return monthly,metadata


def train_slice(panel,origin,window_months,minimum):
    eligible=panel.loc[:pd.Timestamp(origin)]
    complete=eligible.dropna()
    if complete.empty:raise UnsupportedModelError('No complete monthly information state')
    end=complete.index[-1]
    frame=eligible.loc[:end].tail(window_months)
    if frame.isna().any().any():raise UnsupportedModelError('Missing interior training observations; shorten or source the missing history explicitly')
    if len(frame)<minimum:raise UnsupportedModelError('Insufficient independent training months')
    if not frame.index.equals(pd.date_range(frame.index[0],frame.index[-1],freq='ME')):
        raise ValueError('Monthly training calendar gap')
    return frame
