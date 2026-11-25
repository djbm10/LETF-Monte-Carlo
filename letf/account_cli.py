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


def _verified_flag(value):
    if value is None or pd.isna(value):return False
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,str) and value.lower() in ('true','false'):return value.lower()=='true'
    raise ValueError('Tax verification flags require explicit true/false values')


def _issuer_year(value):
    if value is None or pd.isna(value):return None
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(float(value)) or float(value)!=int(float(value)):
        raise ValueError('Issuer tax year must be an explicit integer')
    return int(float(value))


def load_bundle(folder, *, allow_fixture=False):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    status=manifest.get('data_status')
    if (status != 'OBSERVED' and not (allow_fixture and status=='SOFTWARE_FIXTURE')) or manifest.get('price_convention') != 'as_traded':
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
        if item.get('format','csv') == 'csv':
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
                qualified_verified=_verified_flag(row.get('qualified_verified', False)),
                section199a_verified=_verified_flag(row.get('section199a_verified',False)),
                tax_date=pd.Timestamp(row.tax_date).date() if pd.notna(row.get('tax_date')) else None,
                tax_date_source=str(row.tax_date_source) if pd.notna(row.get('tax_date_source')) else '',
                issuer_tax_year=_issuer_year(row.get('issuer_tax_year')),
                record_date=pd.Timestamp(row.record_date).date() if pd.notna(row.get('record_date')) else None))
    elif ('Dividends' not in prices or 'Stock Splits' not in prices or
          prices[['Dividends', 'Stock Splits']].to_numpy().any()):
        raise UnsupportedModelError('Nonzero/missing corporate actions need a complete explicit action schedule')
    if 'Stock Splits' in prices and prices['Stock Splits'].to_numpy().any():
        actions=manifest.get('split_events',[])
        observed=prices.loc[prices['Stock Splits'].ne(0),'Stock Splits']
        supplied={pd.Timestamp(a['date']):a['ratio'] for a in actions}
        if (not manifest.get('corporate_actions_reviewed') or len(supplied)!=len(actions) or
            set(supplied)!=set(observed.index) or any(not a.get('source') or a.get('security')!=symbol for a in actions) or
            any(not np.isfinite(v) or v<=0 or v!=float(observed.loc[d]) for d,v in supplied.items())):
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


def _report_safe(value, key=None):
    """Encode unbounded legal bracket metadata; reject nonfinite money/results."""
    if isinstance(value, dict):
        return {k: _report_safe(v, k) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_report_safe(v) for v in value]
    if isinstance(value, np.generic): value=value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if key == 'bracket_ceiling' and np.isposinf(value): return 'UNBOUNDED'
        raise ValueError('Nonfinite financial result cannot be exported')
    return value


def compare_accounts(bundle, *, strategy='sma', lookback=50, start='2024-01-02', end='2025-12-31',
                     initial_capital=100000., state='MA', ordinary_income=100000.,
                     filing_status='single',cash_flows=(),roth_history_config=None,output=None,
                     buy_hold_initial_only=False,qualified_holding_policy='verified_at_ex'):
    market, manifest, distributions, cpi = load_bundle(bundle)
    if start < manifest.get('account_coverage_start',start) or end > manifest.get('account_coverage_end',end):
        raise UnsupportedModelError('Replay dates exceed reviewed corporate-action/distribution coverage')
    if state not in manifest.get('supported_tax_states',[state]):
        raise UnsupportedModelError('Distribution state-tax adjustments have not been reviewed for this input bundle')
    if strategy not in manifest.get('supported_strategies',[strategy]):
        raise UnsupportedModelError('Strategy needs distribution-complete signal warmup unavailable in this bundle')
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
        source_manifest=manifest, cash_flows=cash_flows, split_events=manifest.get('split_events',[]),
        qualified_holding_policy=qualified_holding_policy)
    if buy_hold_initial_only:
        if strategy!='buy_hold' or cash_flows:raise UnsupportedModelError('Initial-only buy/hold requires a lump sum and explicit buy_hold rule')
        common['rebalance_every']=len(market)+1
    if cash_flows and roth_history_config is None:
        raise UnsupportedModelError('Roth flows require supplied owner birth, contribution clock and aggregate basis history')
    common['cash_flows']=[dict(f,filing_status=f.get('filing_status',filing_status)) for f in cash_flows]
    gross = {}
    net = {}
    for account, profile in profiles.items():
        kwargs = dict(common)
        if account == 'roth' and cash_flows:
            kwargs['roth_history_config'] = roth_history_config
        gross[account] = run_account_backtest(market, [security], profile, apply_taxes=False, **kwargs)
        net[account] = run_account_backtest(market, [security], profile,
            tax_inputs={'ordinary_income': ordinary_income, 'state': state,
                        'filing_status':filing_status,
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
        'cash_distribution_policy':'Accumulate in cash; initial purchase only' if buy_hold_initial_only else 'Reinvest through ordinary strategy rebalancing',
        'qualified_holding_policy':qualified_holding_policy,
        'start': start, 'end': end, 'source_manifest_sha256': sha256((Path(bundle)/'manifest.json').read_bytes()).hexdigest(),
        'identical_pre_tax_trades': identical, 'pre_tax_trade_count': len(_fills(gross['taxable'])),
        'pre_tax_wealth': gross['taxable']['ending_wealth_net_payable'], 'accounts': {},
        'assumptions': {'state': state, 'filing_status':filing_status,'ordinary_income': ordinary_income, 'settlement': 'US T+2/T+1 effective dates',
            'spread_bps': 2., 'slippage_bps': 1., 'commission': 0., 'cash_sweep_yield': 0.,
            'roth_owner_history':roth_history_config,
            'initial_roth_balance': 'Existing balance, not a new contribution; internal liquidation does not execute a personal distribution',
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
            'ending_distribution_receivables':result['ending_distribution_receivables'],
            'days': len(reconciled), 'external_flows_investor': result['external_flows_investor']}
    summary=_report_safe(summary)
    serialized=json.dumps(summary,default=_serialize,indent=2,allow_nan=False)+'\n'
    if output is not None:
        output = Path(output);output.mkdir(parents=True, exist_ok=False)
        for account in net:
            net[account]['reconciliation'].to_csv(output/f'{account}_daily_ledger.csv')
            pd.DataFrame(_fills(net[account])).to_csv(output/f'{account}_fills.csv', index=False)
            pd.DataFrame(_fills(gross[account])).to_csv(output/f'{account}_pretax_fills.csv', index=False)
            (output/f'{account}_events.json').write_text(json.dumps(net[account]['ledger'].events, default=_serialize, indent=2))
        (output/'summary.json').write_text(serialized)
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
    compare.add_argument('--filing-status',default='single',choices=['single','married','head_of_household','married_filing_separately'])
    compare.add_argument('--state', required=True, choices=['MA', 'TX', 'FL', 'NV', 'FEDERAL_ONLY'])
    compare.add_argument('--flows', type=Path)
    compare.add_argument('--roth-history',type=Path)
    compare.add_argument('--buy-hold-initial-only',action='store_true')
    compare.add_argument('--qualified-holding-policy',choices=['verified_at_ex','track_unhedged_common'],default='verified_at_ex')
    args = parser.parse_args()
    flows = json.loads(args.flows.read_text()) if args.flows else ()
    result = compare_accounts(args.bundle, strategy=args.strategy, lookback=args.lookback,
        start=args.start, end=args.end, initial_capital=args.initial_capital, state=args.state,
        ordinary_income=args.ordinary_income, filing_status=args.filing_status,cash_flows=flows,
        roth_history_config=json.loads(args.roth_history.read_text()) if args.roth_history else None,output=args.output,
        buy_hold_initial_only=args.buy_hold_initial_only,qualified_holding_policy=args.qualified_holding_policy)
    print(json.dumps(result, default=_serialize, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
