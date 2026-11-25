"""Daily-reset proxies with explicit provenance and frozen evaluation gates.

No daily paths are inferred from monthly data. Tracking residuals are measured
against observations, not randomly added to make historical prices plausible.
Passing a mathematical check does not establish source authenticity.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from letf.accuracy import Evidence, UnsupportedModelError


@dataclass(frozen=True)
class SourceRecord:
    provider: str
    url: str
    retrieved_at: str
    publication_vintage: str
    sha256: str
    first_observation: str
    currency: str
    convention: str
    frequency: str
    fees_embedded: bool

    def validate(self):
        for key, value in asdict(self).items():
            if key != 'fees_embedded' and not value:
                raise UnsupportedModelError(f'Missing provenance: {key}')
        if len(self.sha256) != 64 or any(c not in '0123456789abcdef' for c in self.sha256):
            raise ValueError('Source checksum must be a SHA-256 hex digest')
        if self.frequency != 'daily' or self.currency != 'USD':
            raise UnsupportedModelError('Daily USD series required; currency hedging and monthly paths are unsupported')


def daily_series(series, name, *, allow_total_loss=True):
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f'{name} must be a dated pandas Series')
    if not len(series) or series.index.hasnans or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f'{name} must have unique increasing nonmissing dates')
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f'{name} contains missing/nonfinite observations')
    if len(series) > 1 and np.median(np.diff(series.index.values).astype('timedelta64[D]').astype(int)) > 4:
        raise UnsupportedModelError(f'{name}: monthly/sparse observations cannot determine a daily path')
    if not allow_total_loss and (series <= -1).any():
        raise ValueError(f'{name} includes an impossible underlying return')


def daily_reset_returns(benchmark_returns, leverage, *, annual_fee=0.,
                        funding_rate_at_previous_close=0., borrowing_spread=0.,
                        elapsed_calendar_days=None, source=None,
                        observed_fund_returns=False, funding_day_basis=360,
                        verified_closures=None,financed_notional=None,
                        collateral_income_per_nav=None,financing_source=None,
                        collateral_day_basis=365):
    """Long daily-reset model, or direct passthrough of observed fund returns.

    Fees accrue ACT/365 and borrowing ACT/360 with real dated intervals. If
    intervals are omitted, use explicit 1/252-year simulation intervals. A
    funding rate passed here must already be aligned to the interval start.
    Loss of all NAV closes the hypothetical fund permanently. This is a stress
    liquidation convention, not proof of a real issuer's intraday behavior.

    Default financing is net cash borrowing (L-1)*(rate+spread). An explicit
    gross-notional mode instead charges financed_notional*(rate+spread) and
    credits collateral_income_per_nav, an exposure-weighted APR per unit NAV.
    These are distinct contracts: a swap spread can apply to the full swap
    notional. Dated notionals/yields must be supplied, never inferred from NAV
    curve fit. ProShares Trust financial statements May31,2026, swap schedules:
    https://www.proshares.com/globalassets/proshares/documents/annual-reports/annual_proshares_geared_financial_statement.pdf
    """
    daily_series(benchmark_returns, 'benchmark', allow_total_loss=False)
    if source is None:
        raise UnsupportedModelError('Source provenance is required')
    source.validate()
    if not np.isfinite(leverage) or leverage < 1:
        raise UnsupportedModelError('This engine supports long leverage >= 1 only')

    def aligned(value, name):
        if isinstance(value, pd.Series):
            if not value.index.equals(benchmark_returns.index):
                raise ValueError(f'{name} index does not exactly match the benchmark')
            result = value.to_numpy(dtype=float)
        else:
            result = np.full(len(benchmark_returns), value, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError(f'Nonfinite {name}')
        return result

    fee = aligned(annual_fee, 'fee')
    rate = aligned(funding_rate_at_previous_close, 'funding rate')
    spread = aligned(borrowing_spread, 'spread')
    if (fee < 0).any() or (spread < 0).any() or not np.isfinite(funding_day_basis) or funding_day_basis <= 0:
        raise ValueError('Fee/spread must be nonnegative and day basis positive')
    if observed_fund_returns:
        if leverage != 1 or np.any(fee) or np.any(rate) or np.any(spread) or financed_notional is not None or collateral_income_per_nav is not None:
            raise ValueError('Observed adjusted fund returns already include internal costs; no extra leverage/fees/funding')
        return pd.DataFrame({'return': benchmark_returns, 'fund_fee': 0., 'financing': 0.,
                             'liquidated': False, 'evidence': Evidence.OBSERVED.value})
    if elapsed_calendar_days is None:
        fee_fraction = np.full(len(fee), 1 / 252)
        funding_fraction = fee_fraction
    else:
        days = aligned(elapsed_calendar_days, 'elapsed days')
        if (days <= 0).any():
            raise UnsupportedModelError('Historical intervals must represent consecutive daily observations, not missing months')
        for i in np.flatnonzero(days>7):
            key=str(benchmark_returns.index[i].date())
            notice=(verified_closures or {}).get(key,{})
            if (not notice.get('source') or notice.get('elapsed_days')!=days[i] or i==0 or
                notice.get('previous_close')!=str(benchmark_returns.index[i-1].date())):
                raise UnsupportedModelError('Extended gap requires dated independent market-closure evidence')
        fee_fraction = days / 365
        funding_fraction = days / funding_day_basis
    fees = fee * fee_fraction
    if financed_notional is None:
        if collateral_income_per_nav is not None:raise ValueError('Collateral credit requires explicit financed notional; no double credit')
        financing = (leverage - 1) * (rate + spread) * funding_fraction
    else:
        if not financing_source or collateral_income_per_nav is None or not np.isfinite(collateral_day_basis) or collateral_day_basis<=0:
            raise UnsupportedModelError('Gross financing requires notional, collateral income/day basis and dated source assumptions')
        notional=aligned(financed_notional,'financed notional')
        collateral=aligned(collateral_income_per_nav,'collateral income per NAV')
        if (notional<0).any():raise ValueError('Nonnegative gross financed notional required')
        credit_fraction=np.full(len(fee),1/252) if elapsed_calendar_days is None else days/collateral_day_basis
        financing=notional*(rate+spread)*funding_fraction-collateral*credit_fraction
    returns = leverage * benchmark_returns.to_numpy() - fees - financing
    liquidated = np.zeros(len(returns), dtype=bool)
    failed = np.flatnonzero(returns <= -1)
    if len(failed):
        first = failed[0]
        returns[first] = -1.
        returns[first + 1:] = 0.
        fees[first + 1:] = 0.
        financing[first + 1:] = 0.
        liquidated[first:] = True
    return pd.DataFrame({'return': returns, 'fund_fee': fees, 'financing': financing,
                         'liquidated': liquidated, 'evidence': Evidence.HYPOTHETICAL.value},
                        index=benchmark_returns.index)


@dataclass(frozen=True)
class ReconstructionGate:
    """Fixed engineering thresholds; not an investment-performance target."""
    min_observations: int = 504
    daily_rmse: float = .003
    daily_bias: float = .0002
    annual_tracking_error: float = .05
    cagr_error: float = .02
    drawdown_error: float = .05
    terminal_relative_error: float = .15

    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def return_metrics(returns):
    values = np.asarray(returns, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < -1).any():
        raise ValueError('Invalid returns for metrics')
    wealth = np.r_[1., np.cumprod(1 + values)]
    return {'cagr': wealth[-1] ** (252 / len(values)) - 1,
            'max_drawdown': float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            'annual_volatility': float(np.std(values, ddof=1) * np.sqrt(252)) if len(values) > 1 else 0.,
            'terminal': float(wealth[-1])}


def reconstruction_errors(actual, predicted):
    daily_series(actual, 'actual'); daily_series(predicted, 'predicted')
    if not actual.index.equals(predicted.index):
        raise ValueError('Validation indices must match exactly; gaps cannot be dropped silently')
    a = return_metrics(actual); p = return_metrics(predicted)
    residual = predicted.to_numpy() - actual.to_numpy()
    if a['terminal'] <= 0:
        raise UnsupportedModelError('Relative validation against an extinguished observed product is undefined')
    return {'n': len(actual), 'daily_rmse': float(np.sqrt(np.mean(residual ** 2))),
            'daily_bias': float(np.mean(residual)),
            'annual_tracking_error': float(np.std(residual, ddof=1) * np.sqrt(252)),
            'cagr_error': p['cagr'] - a['cagr'],
            'drawdown_error': p['max_drawdown'] - a['max_drawdown'],
            'terminal_relative_error': p['terminal'] / a['terminal'] - 1,
            'actual': a, 'proxy': p}


def validate_reconstruction(actual, predicted, *, training_end, gate, provenance_verified=False):
    if len(actual) == 0 or pd.Timestamp(training_end) >= actual.index[0]:
        raise ValueError('Evaluation must start strictly after all training observations')
    errors = reconstruction_errors(actual, predicted)
    checks = {'min_observations': errors['n'] >= gate.min_observations}
    for key in ('daily_rmse', 'daily_bias', 'annual_tracking_error', 'cagr_error',
                'drawdown_error', 'terminal_relative_error'):
        checks[key] = bool(abs(errors[key]) <= getattr(gate, key))
    numerical_pass = all(checks.values())
    years = {}
    for year in actual.index.year.unique():
        mask = actual.index.year == year
        years[str(year)] = reconstruction_errors(actual.loc[mask], predicted.loc[mask])
    holding_errors = {}
    for horizon in (21, 63, 252, 1260):
        if len(actual) < horizon:
            continue
        ratios = []
        for start in range(0, len(actual) - horizon + 1, 21):
            a = np.prod(1 + actual.iloc[start:start + horizon])
            p = np.prod(1 + predicted.iloc[start:start + horizon])
            ratios.append(p / a - 1)
        holding_errors[str(horizon)] = {'n': len(ratios), 'median_relative_error': float(np.median(ratios)),
                                        'max_abs_relative_error': float(np.max(np.abs(ratios)))}
    return {'errors': errors, 'checks': checks, 'numerical_pass': numerical_pass,
            'provenance_verified': provenance_verified,
            'passed': numerical_pass and provenance_verified,
            'evidence': Evidence.VALIDATED_RECONSTRUCTION.value if numerical_pass and provenance_verified
                        else Evidence.HYPOTHETICAL.value,
            'gate': asdict(gate), 'gate_sha256': gate.fingerprint,
            'by_year': years, 'by_holding_period_sessions': holding_errors}
