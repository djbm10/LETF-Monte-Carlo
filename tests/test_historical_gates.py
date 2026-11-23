import numpy as np
import pandas as pd
import pytest

from letf.accuracy import UnsupportedModelError
from letf.historical import calculate_historical_rolling_cagrs
from letf.simulation.paired import paired_block_paths, paired_mean_difference_ci
from letf.data import fetch_historical_data, label_legacy_data


def test_legacy_loader_requires_explicit_hypothetical_optin():
    with pytest.raises(UnsupportedModelError, match='unverified proxies'):
        fetch_historical_data()


def test_missing_real_history_does_not_become_observed_at_inception():
    frame = pd.DataFrame({'TMF_Ret': [.01, .01], 'TMF_IsSynthetic': [False, False]},
                          index=pd.bdate_range('2020-01-01', periods=2))
    labelled = label_legacy_data(frame)
    assert labelled['TMF_IsSynthetic'].all()
    assert not labelled.attrs['historical_comparison_authorized']


def test_historical_cagr_uses_full_endpoint_interval_and_rejects_internal_gaps():
    dates = pd.bdate_range('2020-01-01', periods=253)
    frame = pd.DataFrame({'X_Price': np.exp(np.arange(253) * .001), 'X_IsSynthetic': False}, index=dates)
    result = calculate_historical_rolling_cagrs(frame, 'X', 1)
    years = (dates[-1] - dates[0]).days / 365.2425
    assert result['cagrs'][0] == pytest.approx(np.exp(.252 / years) - 1)
    frame.loc[dates[50], 'X_IsSynthetic'] = True
    assert calculate_historical_rolling_cagrs(frame, 'X', 1) is None


def test_paired_block_paths_keep_joint_rows_and_seed_without_touching_global_rng():
    frame = pd.DataFrame({'X': np.arange(100) / 1000, 'Y': np.arange(100) / 500},
                          index=pd.bdate_range('2010-01-01', periods=100))
    kwargs = dict(paths=2, years=5, block_sessions=20, seed=42, training_end='2010-12-31',
                   evaluation_start='2011-01-01', provenance_verified=True)
    a = list(paired_block_paths(frame, **kwargs))
    b = list(paired_block_paths(frame, **kwargs))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[0]['Y'], a[0]['X'] * 2)
    assert not a[0].equals(a[1])
    assert len(a[0]) == 5 * 252
    with pytest.raises(ValueError, match='cutoff'):
        list(paired_block_paths(frame, **dict(kwargs, training_end='2009-12-31')))


def test_paired_ci_and_failure_gate():
    result = paired_mean_difference_ci([2, 3, 4], [1, 2, 3], seed=1)
    assert result['low'] == result['high'] == result['mean_difference'] == 1
    with pytest.raises(ValueError, match='failed paths'):
        paired_mean_difference_ci([2, np.nan], [1, 1], seed=1)


@pytest.mark.parametrize('years', [5, 10, 20, 30, 40])
def test_required_horizons_share_the_same_seeded_path_prefix(years):
    frame = pd.DataFrame({'X': np.arange(100) / 1000}, index=pd.bdate_range('2010-01-01', periods=100))
    kwargs = dict(paths=1, block_sessions=20, seed=42, training_end='2010-12-31',
                  evaluation_start='2011-01-01', provenance_verified=True)
    short = next(paired_block_paths(frame, years=years, **kwargs))
    long = next(paired_block_paths(frame, years=40, **kwargs))
    assert len(short) == years * 252
    np.testing.assert_array_equal(short, long.iloc[:len(short)])
