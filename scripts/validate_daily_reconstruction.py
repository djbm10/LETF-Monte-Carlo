"""Frozen, one-pass reconstruction diagnostic; never tunes on evaluation data.

Usage: .venv/bin/python scripts/validate_daily_reconstruction.py --inputs audit-output/inputs --output audit-output/reconstruction.json
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from letf.reconstruction import SourceRecord, ReconstructionGate, daily_reset_returns, validate_reconstruction, reconstruction_errors

# Fixed ex ante. These are fee scenarios, NOT certified historical fee schedules.
PRODUCTS = {'TQQQ': ('QQQ', 3., .0086), 'QLD': ('QQQ', 2., .0095),
            'UPRO': ('SPY', 3., .0091), 'SSO': ('SPY', 2., .0089),
            'TMF': ('TLT', 3., .0108)}


def run(inputs, output):
    if output.exists():
        raise ValueError('Refusing to overwrite an existing evaluation result')
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path('docs/accuracy-audit/reconstruction-gate.json').read_text())
    gate = ReconstructionGate(**{k: spec[k] for k in asdict(ReconstructionGate())})
    manifest = json.loads((inputs / 'manifest.json').read_text())

    def load(symbol):
        metadata = manifest['sources'][symbol]
        path = inputs / metadata['path']
        if sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError(f'Immutable input checksum mismatch: {symbol}')
        return pd.read_csv(path, index_col=0, parse_dates=True), metadata

    report = {'gate': asdict(gate), 'gate_sha256': gate.fingerprint, 'split': spec,
              'input_manifest_sha256': sha256((inputs / 'manifest.json').read_bytes()).hexdigest(),
              'products': {}, 'strategy_comparisons_authorized': False,
              'limitation': 'ETF benchmarks embed their own fees; constant product fees and DFF proxy funding are assumptions. No point-in-time historical expense/swap/closure schedule.'}
    try:
        rates, _ = load('DFF')
        # DFF is published with a lag. Lag by TWO calendar days as a conservative
        # observable funding proxy; no same-day revised rate is used in signals.
        funding = rates['DFF'].shift(2) / 100
    except Exception as exc:
        report['blocked'] = f'No usable contemporaneous funding proxy: {exc}'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        return report
    for product, (benchmark, leverage, fee) in PRODUCTS.items():
        try:
            fund, fund_meta = load(product)
            base, base_meta = load(benchmark)
            # Vendor adjusted closes are total-return proxies, never raw fill prices.
            actual = fund['Adj Close'].pct_change(fill_method=None)
            underlying = base['Adj Close'].pct_change(fill_method=None)
            elapsed = base.index.to_series().diff().dt.days
            joined = pd.DataFrame({'actual': actual, 'underlying': underlying,
                                   'days': elapsed, 'rate': funding.reindex(base.index)})
            # Only the common inception/warmup prefix may be removed; no internal
            # NaNs may disappear from the training or evaluation intervals.
            start = max(actual.first_valid_index(), underlying.first_valid_index(),
                        funding.first_valid_index())
            joined = joined.loc[start:'2025-12-31']
            if joined.isna().any().any():
                raise ValueError('Missing aligned daily inputs inside available history')
            training = joined.loc[:spec['training_end']]
            evaluation = joined.loc[spec['evaluation_start']:spec['evaluation_end']]
            if len(training) < 504:
                raise ValueError('Insufficient independent training observations')
            source = SourceRecord(**{k: base_meta[k] for k in SourceRecord.__dataclass_fields__})
            uncalibrated = daily_reset_returns(training['underlying'], leverage, annual_fee=fee,
                                               funding_rate_at_previous_close=training['rate'],
                                               elapsed_calendar_days=training['days'], source=source)
            # Least-squares constant annual spread with nonnegative constraint;
            # intercept-free, all fit on training only. No drift optimization.
            x = (leverage - 1) * training['days'].to_numpy() / 360
            residual = uncalibrated['return'].to_numpy() - training['actual'].to_numpy()
            fitted_spread = max(0., float(np.dot(x, residual) / np.dot(x, x)))
            predicted = daily_reset_returns(evaluation['underlying'], leverage, annual_fee=fee,
                                            funding_rate_at_previous_close=evaluation['rate'],
                                            borrowing_spread=fitted_spread,
                                            elapsed_calendar_days=evaluation['days'], source=source)
            result = validate_reconstruction(evaluation['actual'], predicted['return'],
                                             training_end=spec['training_end'], gate=gate,
                                             provenance_verified=True)
            result['fitted_training_spread'] = fitted_spread
            result['training_observations'] = len(training)
            result['fee_assumption_annual'] = fee
            result['benchmark'] = benchmark
            # Validation confirms post-inception proxy errors only. Extension
            # before inception remains hypothetical without dated fund terms.
            result['pre_inception_evidence'] = 'HYPOTHETICAL STRESS MODEL'
            result['pre_inception_start_available'] = str(underlying.first_valid_index().date())
            result['by_regime'] = {}
            vol = evaluation['underlying'].rolling(21, min_periods=21).std().shift(1) * np.sqrt(252)
            for label, mask in [('calm', vol < .2), ('high_volatility', vol >= .2)]:
                # Regime samples are NONCONTIGUOUS. Report residual statistics
                # only; compounding them would fabricate a holding-period path.
                err = (predicted['return'] - evaluation['actual']).loc[mask]
                result['by_regime'][label] = {'n': len(err), 'bias': float(err.mean()),
                                             'rmse': float(np.sqrt(np.mean(err ** 2)))}
            # Add pre-inception proxy values without altering observed fund CSVs.
            pre = pd.DataFrame({'underlying': underlying, 'days': elapsed,
                                'rate': funding.reindex(base.index)}).loc[:fund.index[0] - pd.Timedelta(days=1)]
            pre = pre.loc[pre.dropna().index.min():]
            if len(pre):
                if pre.isna().any().any():
                    raise ValueError('Missing pre-inception daily inputs')
                proxy = daily_reset_returns(pre['underlying'], leverage, annual_fee=fee,
                                            funding_rate_at_previous_close=pre['rate'], borrowing_spread=fitted_spread,
                                            elapsed_calendar_days=pre['days'], source=source)
                proxy_path = output.parent / f'{product}_pre_inception_proxy.csv'
                if proxy_path.exists():
                    raise ValueError('Refusing to overwrite prior proxy artifact')
                proxy.to_csv(proxy_path, index_label='Date')
                result['pre_inception_proxy'] = {'path': proxy_path.name, 'rows': len(proxy),
                                                'sha256': sha256(proxy_path.read_bytes()).hexdigest(),
                                                'evidence': 'HYPOTHETICAL STRESS MODEL',
                                                'parameter_vintage': spec['training_end'],
                                                'tradable_historically': False}
            report['products'][product] = result
        except Exception as exc:
            report['products'][product] = {'passed': False, 'evidence': 'UNSUPPORTED', 'error': str(exc)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.inputs, args.output)
    print(json.dumps({symbol: {'passed': r['passed'], 'evidence': r['evidence'],
                               'errors': r.get('errors'), 'error': r.get('error')}
                       for symbol, r in report['products'].items()}, indent=2))
