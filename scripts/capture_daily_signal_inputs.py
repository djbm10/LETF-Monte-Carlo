"""Capture the original frozen-rule IRX and VIX public closes, privately.

Uses the repository's existing yfinance feed, not a paid/authenticated service.
No filling missing sessions, substituting DFF for IRX or guessing a daily bill
return from a future monthly return. Current-vintage retrospective signals only.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yfinance as yf
from letf.lab.catalog import SourceLake,private_bytes,private_json,file_hash


def normalize(frame,symbol):
    if symbol not in ('^VIX','^IRX'):raise ValueError('Only explicitly registered original signal identifiers')
    if isinstance(frame.columns,pd.MultiIndex):frame=frame.copy();frame.columns=frame.columns.get_level_values(0)
    if frame.empty or 'Close'not in frame:raise ValueError('Missing genuine signal close observations')
    series=pd.to_numeric(frame.Close,errors='raise').copy();series.index=pd.to_datetime(series.index).tz_localize(None)
    if not series.index.is_unique or not series.index.is_monotonic_increasing or np.isinf(series).any():raise ValueError('Invalid signal chronology/values')
    if symbol=='^VIX' and (series.dropna()<=0).any():raise ValueError('Nonpositive VIX observation')
    return series.rename('VIX') if symbol=='^VIX' else (series/100/252).rename('RF_DAILY')


def capture(output,*,lake,start='2024-01-01',end='2026-01-01',downloader=None):
    if pd.Timestamp(end)>pd.Timestamp(datetime.now(timezone.utc).date()):raise ValueError('Historical signal capture cannot request future sessions')
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700);results={};lake=SourceLake(lake)
    downloader=downloader or yf.download
    for symbol in ('^VIX','^IRX'):
        try:
            frame=downloader(symbol,start=start,end=end,auto_adjust=False,progress=False,threads=False)
            signal=normalize(frame,symbol);stem=symbol[1:]
            raw=frame.to_csv(index_label='Date').encode();path=output/(stem+'-provider-frame.csv');private_bytes(path,raw)
            url=f'https://finance.yahoo.com/quote/{symbol}/history/'
            metadata=dict(provider='Yahoo Finance via pinned yfinance0.2.66',url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),
                license='Provider rights retained; private research, no redistribution approval',
                convention='Provider client DataFrame capture, not original HTTP bytes; index close in annual-percent yield/volatility points, not an investment return',
                publication_vintage='Current download; old decision-time archive not supplied',point_in_time_verified=False,
                first_observation=str(signal.first_valid_index().date()),last_observation=str(signal.last_valid_index().date()),
                currency=None,frequency='daily index close',classification='OBSERVED_MARKET')
            ident=lake.put(stem+'_daily_signal_frame',raw,metadata)
            target=output/(stem+'-signal.csv');private_bytes(target,signal.to_csv(index_label='Date').encode())
            results[signal.name]=dict(path=str(target.resolve()),sha256=file_hash(target),provider=metadata['provider'],url=url,
                column=signal.name,availability='DAILY_CLOSE',raw_frame_snapshot=ident,missing_values=int(signal.isna().sum()),
                transform='Close/100/252, exact original IRX convention' if symbol=='^IRX' else 'Close unchanged')
        except Exception as error:results['VIX' if symbol=='^VIX' else 'RF_DAILY']={'status':'BLOCKED','reason':str(error)}
    result={'classification':'OBSERVED_MARKET','signals':results,'complete':all('sha256'in v for v in results.values()),
        'scope':'Current-vintage close observations for next-open retrospective signals; no historical publication-vintage or trading-fill certification'}
    private_json(output/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--lake',type=Path,required=True)
    a=p.parse_args();r=capture(a.output,lake=a.lake);print(json.dumps(r));raise SystemExit(0 if r['complete'] else 2)
