"""Capture public daily observations into a NEW, immutable audit directory.

Usage: .venv/bin/python scripts/capture_daily_audit_inputs.py --output audit-output/inputs
No credentials, orders, subscriptions or purchases. Source revisions can change
future downloads: the saved CSV checksums identify this exact capture.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {'retrieved_at': timestamp, 'start': '1990-01-01', 'end_exclusive': '2026-01-01',
                'sources': {}, 'errors': {}, 'currency': 'USD'}
    for ticker in ('SPY', 'QQQ', 'TQQQ', 'QLD', 'UPRO', 'SSO', 'TLT', 'TMF'):
        try:
            frame = yf.download(ticker, start=manifest['start'], end=manifest['end_exclusive'],
                                auto_adjust=False, actions=True, progress=False, threads=False)
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            if frame.empty or 'Adj Close' not in frame or 'Close' not in frame:
                raise ValueError('Missing genuine daily raw/adjusted observations')
            frame = frame.dropna(subset=['Close', 'Adj Close'])
            path = output / f'{ticker}.csv'
            frame.to_csv(path, index_label='Date', float_format='%.12g')
            manifest['sources'][ticker] = {
                'path': path.name, 'provider': 'Yahoo Finance via yfinance 0.2.66',
                'url': f'https://finance.yahoo.com/quote/{ticker}/history/',
                'retrieved_at': timestamp, 'publication_vintage': timestamp,
                'sha256': sha256(path.read_bytes()).hexdigest(),
                'first_observation': str(frame.index[0].date()),
                'last_observation': str(frame.index[-1].date()), 'rows': len(frame),
                'currency': 'USD', 'frequency': 'daily', 'fees_embedded': True,
                'convention': 'vendor OHLC (may be split-restated), plus adjusted close (splits/distributions)',
                'execution_price_convention_verified': False,
                'independent_issuer_reconciliation': False,
            }
            print(f'{ticker}: {len(frame)} daily observations', flush=True)
        except Exception as exc:
            manifest['errors'][ticker] = str(exc)
            print(f'{ticker}: unavailable: {exc}', flush=True)
    try:
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF&cosd=1990-01-01&coed=2025-12-31'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
        frame.columns = ['DFF']
        frame['DFF'] = pd.to_numeric(frame['DFF'], errors='coerce')
        if frame['DFF'].isna().any():
            raise ValueError('FRED DFF has missing observations; no interpolation is authorized')
        path = output / 'DFF.csv'
        frame.to_csv(path, index_label='Date')
        manifest['sources']['DFF'] = {
            'path': path.name, 'provider': 'Federal Reserve via FRED', 'url': url,
            'retrieved_at': timestamp, 'publication_vintage': timestamp,
            'sha256': sha256(path.read_bytes()).hexdigest(),
            'first_observation': str(frame.index[0].date()), 'last_observation': str(frame.index[-1].date()),
            'rows': len(frame), 'frequency': 'daily', 'currency': 'USD',
            'convention': 'annual effective federal funds rate, percentage points; proxy funding, not swap invoices',
            'fees_embedded': False,
        }
        print(f'DFF: {len(frame)} daily observations', flush=True)
    except Exception as exc:
        manifest['errors']['DFF'] = str(exc)
        print(f'DFF: unavailable: {exc}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if manifest['errors']:
        raise SystemExit('Capture incomplete; inspect manifest. Missing data has not been fabricated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    capture(parser.parse_args().output)
