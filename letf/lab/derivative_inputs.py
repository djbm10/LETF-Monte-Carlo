"""Hash-checked dated derivative inputs for existing canonical books.

This boundary validates records; it does not manufacture fills, assignment,
house margins, missing depth or legal classifications from continuous charts.
Passing it is necessary, never sufficient, for historical certification.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
from math import isfinite
from zoneinfo import ZoneInfo

import pandas as pd

from letf.accuracy import UnsupportedModelError
from letf.accounts import calendar_date
from letf.derivatives import FuturesContract,OptionContract,OptionQuote
from letf.lab.catalog import file_hash,inside_file


def instant(value):
    stamp=pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:raise ValueError('Explicit timezone-aware publication/quote instant required')
    return stamp.tz_convert('UTC').to_pydatetime()


def strict_bool(value):
    if type(value)is bool:return value
    if value in ('true','false'):return value=='true'
    raise ValueError('Explicit boolean classification required')


def positive_int(value):
    # CSV strings are permitted; fractional, missing and boolean sizes are not.
    if isinstance(value,bool):raise ValueError('Integer depth required')
    result=int(value)
    if float(value)!=result or result<0:raise ValueError('Integer nonnegative depth required')
    return result


def load_inputs(folder,*,decision_at=None):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('kind') not in ('future','option'):
        raise UnsupportedModelError('Reviewed dated future/option input schema1 required')
    classification=manifest.get('classification')
    if classification not in ('OBSERVED_MARKET','SOFTWARE_FIXTURE'):
        raise UnsupportedModelError('Raw inputs must be observed records or explicit software fixtures')
    now=datetime.now(timezone.utc);cutoff=instant(decision_at) if decision_at else None
    files=manifest.get('files',{});frames={};hashes={}
    if set(files)-{'contracts','settlements' if manifest['kind']=='future' else 'quotes'}:
        raise UnsupportedModelError('Unreviewed derivative file role; margin/action changes require explicit canonical events')
    for name,meta in files.items():
        if not all(meta.get(k) for k in ('path','sha256','provider','license','retrieved_at','convention')):
            raise ValueError('Source identity, rights, timestamp, units and hash required')
        if instant(meta['retrieved_at'])>now:raise ValueError('Future capture timestamp')
        path=inside_file(folder,meta['path'])
        if file_hash(path)!=meta['sha256']:raise ValueError('Derivative source checksum mismatch')
        frames[name]=pd.read_csv(path,keep_default_na=False);hashes[name]=meta['sha256']
    if not files or 'contracts' not in frames:raise ValueError('Dated contract specifications required')
    contracts={};available={}
    for row in frames['contracts'].to_dict('records'):
        published=instant(row.pop('available_at'))
        if published>now or (cutoff and published>cutoff):raise UnsupportedModelError('Contract definition not available at decision time')
        if manifest['kind']=='future':
            for key in ('expiry','margin_as_of','first_notice'):
                row[key]=calendar_date(row[key]) if row.get(key) else None
            row['section1256']=strict_bool(row['section1256'])
            c=FuturesContract(**row)
        else:
            row['expiry']=calendar_date(row['expiry']);row['multiplier']=positive_int(row.get('multiplier',100))
            row['adjusted']=strict_bool(row.get('adjusted',False));c=OptionContract(**row)
        if c.symbol in contracts:raise ValueError('Duplicate dated contract identity')
        contracts[c.symbol]=c;available[c.symbol]=published
    if not contracts:raise ValueError('Empty contract definitions')
    if manifest['kind']=='future':
        required={'date','contract','expiry','settlement','available_at'}
        tape=frames.get('settlements')
        if tape is None or not required.issubset(tape):raise ValueError('Dated settlements and publication instants required')
        if tape.duplicated(['date','contract']).any():raise ValueError('Duplicate dated settlement')
        for row in tape.to_dict('records'):
            c=contracts[row['contract']];day=calendar_date(row['date']);pub=instant(row['available_at'])
            if calendar_date(row['expiry'])!=c.expiry or day>c.expiry:raise ValueError('Settlement/expiry identity conflict')
            if pub.date()<day or pub<available[c.symbol] or pub>now:raise ValueError('Invalid settlement publication instant')
            if cutoff and pub>cutoff:raise UnsupportedModelError('Future settlement unavailable at decision time')
            c.validate_price(float(row['settlement']))
        if tape.empty:raise ValueError('Empty settlement tape')
        quotes=[];quote_availability=[]
    else:
        tape=frames.get('quotes');quotes=[];quote_availability=[]
        required={'timestamp','contract_symbol','bid','ask','bid_size','ask_size','source','available_at'}
        if tape is None or not required.issubset(tape):raise ValueError('Dated bid/ask/size and publication instants required')
        if tape.duplicated(['timestamp','contract_symbol']).any():raise ValueError('Duplicate option quote timestamp')
        for row in tape.to_dict('records'):
            c=contracts[row['contract_symbol']];stamp=instant(row['timestamp']);pub=instant(row['available_at'])
            if pub<stamp or pub<available[c.symbol] or pub>now:raise ValueError('Invalid quote publication instant')
            if cutoff and pub>cutoff:raise UnsupportedModelError('Future option quote unavailable at decision time')
            if stamp.astimezone(ZoneInfo('America/New_York')).date()>c.expiry:
                raise ValueError('Quote after contract expiry')
            q=OptionQuote(stamp,float(row['bid']),float(row['ask']),positive_int(row['bid_size']),
                positive_int(row['ask_size']),row['source'],observed=True,contract_symbol=c.symbol)
            # Validate both sides without assuming positive displayed depth.
            for side,size in (('BUY',q.ask_size),('SELL',q.bid_size)):
                if size:q.fill(stamp,side,1)
            if not (0<=q.bid<=q.ask and q.ask>0) or not isfinite(q.ask):raise ValueError('Invalid option quote')
            quotes.append(q);quote_availability.append(pub)
        if tape.empty:raise ValueError('Empty quote tape')
    return {'manifest':manifest,'contracts':contracts,'frames':frames,'quotes':quotes,'quote_availability':quote_availability,
        'summary':{'classification':classification,'kind':manifest['kind'],'contracts':len(contracts),
            'observations':len(tape),'source_hashes':hashes,'manifest_sha256':file_hash(folder/'manifest.json'),
            'historical_execution_certification':'BLOCKED','account_certification':'BLOCKED',
            'missing_evidence':['Order-time executable depth/fill and continuous complete session coverage',
                'Effective broker permissions, house-margin changes and exact funding/roll instructions',
                'OCC deliverables/exercise/assignment and account-specific tax evidence where applicable'],
            'note':'Use these existing FuturesContract/OptionContract/OptionQuote objects with canonical books; no new cash ledger'}}


def quote_at(inputs,symbol,decision_at):
    """Time-available quote lookup; canonical fill still checks age/depth."""
    when=instant(decision_at)
    choices=[q for q,pub in zip(inputs['quotes'],inputs['quote_availability'])
             if q.contract_symbol==symbol and pub<=when and q.timestamp<=when]
    if not choices:raise UnsupportedModelError('No time-available quote for this decision')
    return max(choices,key=lambda q:q.timestamp)
