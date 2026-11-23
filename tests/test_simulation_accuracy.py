import numpy as np
import pandas as pd
import pytest

from letf import config as cfg
from letf.calibration import simulate_joint_returns_t
from letf.simulation.engine import compute_letf_return_correct, simulate_etf_returns_from_layers
from letf.utils import compute_high_vol_probability


def model(alpha=0.):
    return {'assets': ['X'], 'regimes': {0: {'mu': np.array([.001]),
             'cov': np.array([[.0001]]), 'nu': 5., 'garch_alpha': alpha,
             'garch_beta': .9 if alpha else 0., 'dcc_a': 0., 'dcc_b': 0.}}}


def test_student_t_variance_targets_covariance_once():
    n = 30000
    result = simulate_joint_returns_t(n, np.zeros(n, dtype=int), model(.02), np.random.default_rng(20261123))['X']
    # Prespecified 10% variance tolerance on deterministic 30k draws; tests
    # the actual recursion rather than merely duplicating a scaling formula.
    assert np.var(result) == pytest.approx(.0001, rel=.10)


def test_antithetic_pair_and_no_full_path_mean_adjustment(monkeypatch):
    monkeypatch.setattr(cfg, 'USE_MOMENT_MATCHING', True)
    paths = [simulate_joint_returns_t(500, np.zeros(500, dtype=int), model(),
                                     np.random.default_rng(99), antithetic=a)['X'] for a in (False, True)]
    np.testing.assert_allclose(paths[0] + paths[1], .002, atol=1e-14)
    assert abs(paths[0].mean() - .001) > 1e-7


def test_invalid_garch_dcc_parameters_fail_instead_of_silent_adjustment():
    bad = model(.2)
    with pytest.raises(ValueError, match='stationary'):
        simulate_joint_returns_t(10, np.zeros(10, dtype=int), bad, np.random.default_rng(2))


def test_daily_leverage_is_linear_before_compounding():
    assert compute_letf_return_correct(.01, 3., .99, .01, .0002) == pytest.approx(.03 - .01/252 - .0002)


def test_leveraged_fund_total_loss_is_absorbing():
    cfg_asset = {'leverage': 3., 'expense_ratio': 0., 'tracking_error_base': 0., 'tracking_error_df': 5}
    result = simulate_etf_returns_from_layers('TQQQ', cfg_asset, np.array([-.4, .9, .9]),
                 np.zeros(3, dtype=int), np.full(3, 20.), np.zeros(3), 1, None, None, None)
    np.testing.assert_array_equal(result, [-1., 0., 0.])


def test_stress_signal_prefix_unchanged_by_future_observations():
    first = compute_high_vol_probability([20., 21., 19., 20.])
    extended = compute_high_vol_probability([20., 21., 19., 20., 100., 100., 100., 100.])
    np.testing.assert_array_equal(first, extended[:4])
