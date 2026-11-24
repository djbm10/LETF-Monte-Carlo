"""Runnable canonical account comparison: python -m letf.account_cli compare.

Observed daily quotes and hypothetical fill assumptions remain separate. This
CLI does not certify investments or fetch data implicitly during a backtest.
"""
import argparse
from dataclasses import replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.accounts import Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest, Distribution
from letf.accuracy import UnsupportedModelError


def load_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('data_status') != 'OBSERVED' or manifest.get('price_convention') != 'as_traded':
        raise UnsupportedModelError('Account replay needs observed as-traded quotes and reviewed corporate actions')
    if manifest.get('currency') != 'USD' or manifest.get('frequency') != 'daily':
        raise UnsupportedModelError('Only daily USD account inputs are integrated')
    files = {}
    for name, item in manifest['files'].items():
        path = folder / item['path']
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Source file must be inside its input bundle')
        if sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        if not item.get('provider') or not item.get('url'):
            raise ValueError('Provider and source URL required')
        files[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    symbol = manifest['symbol']
    prices = files['prices']
    if not manifest.get('distributions_complete'):
        raise UnsupportedModelError('Distribution classification/payable dates are incomplete')
    distributions = []
    if 'distributions' in files:
        for ex_date, row in files['distributions'].iterrows():
            distributions.append(Distribution(symbol, ex_date.date(), pd.Timestamp(row.pay_date).date(),
                float(row.amount_per_share), row.category, row.source,
                qualified_verified=bool(row.get('qualified_verified', False))))
    elif ('Dividends' not in prices or 'Stock Splits' not in prices or
          prices[['Dividends', 'Stock Splits']].to_numpy().any()):
        raise UnsupportedModelError('Nonzero/missing corporate actions need a complete explicit action schedule')
    if 'Stock Splits' in prices and prices['Stock Splits'].to_numpy().any():
        raise UnsupportedModelError('Use a reviewed split/deliverable bundle; vendor split-restated OHLC is not sufficient')
    market = prices[['Open', 'Close']].rename(columns={'Open': f'{symbol}_Open', 'Close': f'{symbol}_Close'})
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Daily observations must be unique and increasing')
    # CPI is used only in ex-post metrics; it is never attached to strategy
    # histories or supplied as a contemporaneous trading signal.
    cpi = None
    if 'cpi' in files:
        monthly = files['cpi'].iloc[:, 0]
        mapping = {d.to_period('M'): float(v) for d, v in monthly.items()}
        cpi = pd.Series([mapping.get(d.to_period('M'), np.nan) for d in market.index], index=market.index)
    return market, manifest, distributions, cpi


def allocator_for(symbol, strategy, lookback):
    if strategy not in ('buy_hold', 'sma', 'vol_target'):
        raise UnsupportedModelError('Unreviewed strategy; frozen legacy rules are not silently redefined')
    if lookback < 2:
        raise ValueError('Lookback must be at least two sessions')
    def allocate(history, weights):
        if strategy == 'buy_hold':
            return {symbol: 1.}
        if len(history) < lookback + (strategy == 'vol_target'):
            return {}
        close = history[f'{symbol}_Close']
        if strategy == 'sma':
            return {symbol: float(close.iloc[-1] >= close.iloc[-lookback:].mean())}
        vol = close.pct_change(fill_method=None).iloc[-lookback:].std(ddof=1) * np.sqrt(252)
        return {symbol: min(1., .20 / vol) if vol > 0 else 1.}
    return allocate


def _fills(result):
    return [{k: e[k] for k in ('date', 'action', 'security', 'shares', 'price', 'commission', 'reason')}
             for e in result['ledger'].events if e['action'] in ('BUY', 'SELL')]


def _serialize(value):
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def compare_accounts(bundle, *, strategy='sma', lookback=50, start='2024-01-02', end='2025-12-31',
                     initial_capital=100000., state='MA', ordinary_income=100000.,
                     cash_flows=(), output=None):
    market, manifest, distributions, cpi = load_bundle(bundle)
    market = market.loc[:end]
    cpi = cpi.reindex(market.index) if cpi is not None else None
    symbol = manifest['symbol']
    rules = allocator_for(symbol, strategy, lookback)
    security = Security(symbol, kind=manifest.get('instrument_kind', 'stock'),
                        tax_structure=manifest.get('tax_structure', 'corporation'))
    # Equal ex ante costs for equal instruments in the compared cash accounts.
    profiles = {'taxable': replace(TAXABLE_CASH, spread_bps=2., slippage_bps=1.),
                'roth': replace(CONSERVATIVE_ROTH, spread_bps=2., slippage_bps=1.)}
    payments = {year: date(year + 1, 4, 15) for year in range(pd.Timestamp(start).year, market.index[-1].year)}
    common = dict(initial_capital=initial_capital, allocator=rules, price_convention='as_traded',
        distributions_complete=True, distributions=distributions, trade_start=start,
        terminal_liquidation=True, inflation_index=cpi, settlement_policy='us_equity',
        source_manifest=manifest, cash_flows=cash_flows)
    # Treat the initial Roth balance as supplied historical regular basis, not
    # a new above-limit contribution. No new Roth balance is created for free.
    roth_history = dict(birth_date='1980-01-01', first_tax_year=2010, regular_basis=initial_capital)
    gross = {}
    net = {}
    for account, profile in profiles.items():
        kwargs = dict(common)
        if account == 'roth' and cash_flows:
            kwargs['roth_history_config'] = roth_history
        gross[account] = run_account_backtest(market, [security], profile, apply_taxes=False, **kwargs)
        net[account] = run_account_backtest(market, [security], profile,
            tax_inputs={'ordinary_income': ordinary_income, 'state': state,
                        'federal_only': state == 'FEDERAL_ONLY',
                        'no_external_wash_replacements_after_end': True},
            tax_payment_dates=payments, **kwargs)
    identical = json.dumps(_fills(gross['taxable']), default=_serialize, sort_keys=True) == json.dumps(
                           _fills(gross['roth']), default=_serialize, sort_keys=True)
    if not identical:
        raise AssertionError('Account-neutral pre-tax trade decisions diverged')
    summary = {'data_status': 'OBSERVED', 'execution_status': 'HYPOTHETICAL_STRESS',
        'investment_performance_certification': 'BLOCKED',
        'reason': 'Observed daily OHLC + disclosed fill/annual-tax schedule; no NBBO or full household tax return',
        'strategy': {'rule': strategy, 'lookback_sessions': lookback, 'symbol': symbol},
        'start': start, 'end': end, 'source_manifest_sha256': sha256((Path(bundle)/'manifest.json').read_bytes()).hexdigest(),
        'identical_pre_tax_trades': identical, 'pre_tax_trade_count': len(_fills(gross['taxable'])),
        'pre_tax_wealth': gross['taxable']['ending_wealth_net_payable'], 'accounts': {},
        'assumptions': {'state': state, 'ordinary_income': ordinary_income, 'settlement': 'US T+2/T+1 effective dates',
            'spread_bps': 2., 'slippage_bps': 1., 'commission': 0., 'cash_sweep_yield': 0.,
            'initial_roth_basis': 'Supplied existing regular basis; no new above-limit contribution',
            'tax_timing': 'Annual settlement funded; sufficient external salary withholding assumed for safe harbor',
            'cpi': 'Monthly CPI-U NSA mapped to its observation month for ex-post metrics only'}}
    for account, result in net.items():
        reconciled = result['reconciliation']
        summary['accounts'][account] = {
            'ending_wealth_net_payable': result['ending_wealth_net_payable'],
            'tax_paid': result['tax_paid'], 'terminal_tax_payable': result['terminal_tax_payable'],
            'metrics': result['after_tax_metrics'], 'trading_costs': result['trading_costs'],
            'tax_rows': result['tax_rows'], 'trade_count': len(_fills(result)),
            'max_cash_residual': float(reconciled.cash_residual.abs().max()),
            'max_nav_residual': float(reconciled.nav_residual.abs().max()),
            'ending_positions': reconciled.positions.iloc[-1],
            'final_cash': float(reconciled.cash.iloc[-1]),
            'unsettled_liquidation_proceeds': float(reconciled.unsettled.iloc[-1]),
            'days': len(reconciled), 'external_flows_investor': result['external_flows_investor']}
    if output is not None:
        output = Path(output);output.mkdir(parents=True, exist_ok=False)
        for account in net:
            net[account]['reconciliation'].to_csv(output/f'{account}_daily_ledger.csv')
            pd.DataFrame(_fills(net[account])).to_csv(output/f'{account}_fills.csv', index=False)
            pd.DataFrame(_fills(gross[account])).to_csv(output/f'{account}_pretax_fills.csv', index=False)
            (output/f'{account}_events.json').write_text(json.dumps(net[account]['ledger'].events, default=_serialize, indent=2))
        (output/'summary.json').write_text(json.dumps(summary, default=_serialize, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    compare = sub.add_parser('compare')
    compare.add_argument('--bundle', required=True, type=Path)
    compare.add_argument('--output', required=True, type=Path)
    compare.add_argument('--strategy', default='sma', choices=['buy_hold', 'sma', 'vol_target'])
    compare.add_argument('--lookback', default=50, type=int)
    compare.add_argument('--start', default='2024-01-02');compare.add_argument('--end', default='2025-12-31')
    compare.add_argument('--initial-capital', default=100000., type=float)
    compare.add_argument('--ordinary-income', default=100000., type=float)
    compare.add_argument('--state', required=True, choices=['MA', 'TX', 'FL', 'NV', 'FEDERAL_ONLY'])
    compare.add_argument('--flows', type=Path)
    args = parser.parse_args()
    flows = json.loads(args.flows.read_text()) if args.flows else ()
    result = compare_accounts(args.bundle, strategy=args.strategy, lookback=args.lookback,
        start=args.start, end=args.end, initial_capital=args.initial_capital, state=args.state,
        ordinary_income=args.ordinary_income, cash_flows=flows, output=args.output)
    print(json.dumps(result, default=_serialize, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
