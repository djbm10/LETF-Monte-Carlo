"""Free, credential-free capture for the canonical account demonstration.

Vendor quotes are not historical NBBO fills. BRK-B is used because verified
absence of distributions/splits avoids inventing fund tax distributions.
Raw snapshots remain local; do not redistribute vendor data without rights.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    # Include all corporate actions up to retrieval to detect post-window
    # split restatement of vendor OHLC. A split would require another loader.
    history = yf.Ticker('BRK-B').history(start='2023-01-01',
        end=str(date.today() + timedelta(days=1)), auto_adjust=False, actions=True, repair=False)
    if history.empty or any(c not in history for c in ('Open', 'Close', 'Adj Close', 'Dividends', 'Stock Splits')):
        raise ValueError('Incomplete vendor history')
    if history[['Dividends', 'Stock Splits']].to_numpy().any():
        raise ValueError('Corporate action found; this no-distribution/as-traded demonstration must be reviewed')
    if not np.allclose(history.Close, history['Adj Close'], rtol=1e-7, atol=1e-7):
        raise ValueError('Price/adjusted-close convention mismatch')
    history.index = history.index.tz_localize(None)
    frozen = history.loc['2023-01-01':'2025-12-31'].copy()
    frozen.index.name = 'Date'
    price_path = output / 'BRK-B.csv'
    frozen.to_csv(price_path, float_format='%.12g')
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS&cosd=2023-01-01&coed=2025-12-31'
    response = requests.get(url, timeout=60);response.raise_for_status()
    cpi = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
    cpi_path = output / 'CPIAUCNS.csv';cpi.to_csv(cpi_path, index_label='Date')
    manifest = {'retrieved_at': timestamp, 'symbol': 'BRK-B', 'instrument_kind': 'stock',
        'currency': 'USD', 'frequency': 'daily', 'price_convention': 'as_traded',
        'distributions_complete': True, 'corporate_actions_checked_through': str(history.index[-1].date()),
        'data_status': 'OBSERVED', 'execution_status': 'HYPOTHETICAL_STRESS',
        'scope': 'One corporate stock; no vendor or issuer distributions/splits during the captured period',
        'issuer_distribution_sources': ['https://www.berkshirehathaway.com/2024ar/2024ar.pdf',
                                        'https://www.berkshirehathaway.com/2025ar/2025ar.pdf'],
        'files': {
            'prices': {'path': price_path.name, 'sha256': sha256(price_path.read_bytes()).hexdigest(),
                       'provider': 'Yahoo Finance via yfinance 0.2.66', 'url': 'https://finance.yahoo.com/quote/BRK-B/history/',
                       'first_observation': str(frozen.index[0].date()), 'last_observation': str(frozen.index[-1].date()),
                       'rows': len(frozen), 'license': 'Vendor personal-use terms; local research capture; no redistribution certified'},
            'cpi': {'path': cpi_path.name, 'sha256': sha256(cpi_path.read_bytes()).hexdigest(),
                    'provider': 'BLS via FRED', 'url': url, 'convention': 'CPI-U NSA monthly levels, ex-post performance only'}},
        'limitations': ['Observed daily OHLC does not establish executable NBBO fills',
                        'Current vendor/revised CPI vintage, not a preserved historical release vintage']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, required=True)
    capture(parser.parse_args().output)
