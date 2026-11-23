"""Unbiased moving-block resampling of joint daily observations.

One draw supplies every strategy on a path. No crisis downweighting,
directional momentum selection, missing-data filling or future holdout rows.
Resampled paths are hypothetical, even when every input row was observed.
"""
import numpy as np
import pandas as pd

from letf.accuracy import Evidence, UnsupportedModelError
from letf.reconstruction import daily_series


def paired_block_paths(observations, *, paths, years, block_sessions, seed,
                       training_end, evaluation_start, provenance_verified):
    if not provenance_verified:
        raise UnsupportedModelError('Observed joint daily input provenance must be verified')
    if years not in (5, 10, 20, 30, 40):
        raise ValueError('Supported audit horizons: 5, 10, 20, 30, 40 years')
    if type(paths) is not int or paths < 1 or type(block_sessions) is not int or block_sessions < 1:
        raise ValueError('Positive integer path and block sizes required')
    if pd.Timestamp(training_end) >= pd.Timestamp(evaluation_start):
        raise ValueError('Training must end before the frozen evaluation window')
    if observations.index.max() > pd.Timestamp(training_end):
        raise ValueError('Input includes observations after the training cutoff')
    if len(observations) < block_sessions:
        raise ValueError('Insufficient daily block history')
    for column in observations:
        daily_series(observations[column], column)
    children = np.random.SeedSequence(seed).spawn(paths)
    n = years * 252
    for path_id, child in enumerate(children):
        rng = np.random.default_rng(child)
        starts = rng.integers(0, len(observations) - block_sessions + 1,
                              size=(n + block_sessions - 1) // block_sessions)
        indices = np.concatenate([np.arange(s, s + block_sessions) for s in starts])[:n]
        result = observations.iloc[indices].reset_index(drop=True)
        result.attrs.update({'path_id': path_id, 'seed': seed, 'spawn_key': child.spawn_key,
                             'source_indices': indices.tolist(), 'evidence': Evidence.HYPOTHETICAL.value,
                             'training_end': str(training_end), 'evaluation_start': str(evaluation_start)})
        yield result


def paired_mean_difference_ci(left, right, *, seed, confidence=.95, bootstrap_samples=2000):
    """Paired bootstrap CI on independent path-level outcomes (not percentiles).

    If antithetic paths are used, aggregate each pair first. Overlapping
    historical holding periods are not independent Monte Carlo paths.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError('At least two matched path outcomes are required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite or failed paths must fail the comparison, not disappear')
    if not 0 < confidence < 1 or bootstrap_samples < 1:
        raise ValueError('Invalid CI settings')
    difference = left - right
    rng = np.random.default_rng(seed)
    sample_means = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        sample_means[i] = rng.choice(difference, len(difference), replace=True).mean()
    tail = (1 - confidence) / 2
    low, high = np.quantile(sample_means, [tail, 1 - tail])
    return {'mean_difference': float(difference.mean()), 'low': float(low), 'high': float(high),
            'confidence': confidence, 'independent_path_count': len(difference), 'seed': seed}
