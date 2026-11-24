"""Fail-closed invariants and deterministic, lookahead-safe research smoke tests."""
import numpy as np
import pandas as pd
import pytest

from letf.simulation.sector_regime import (
    Scenario, build_market_data, sample_joint_indices, evaluate, manifest, _run_path
)


def market_fixture(seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2017-01-02', periods=720)
    common = rng.normal(.0003, .012, len(dates))
    rets = {
        'QQQ': 1.3 * common + rng.normal(0, .006, len(dates)),
        'SPY': common,
        'XLK': 1.15 * common + rng.normal(0, .007, len(dates)),
        'XLE': .75 * common + rng.normal(0, .012, len(dates)),
    }
    rets['ROM'] = 2 * rets['XLK'] - .0004
    rets['DIG'] = 2 * rets['XLE'] - .0004
    c = pd.DataFrame({k: 100 * np.cumprod(1 + r) for k, r in rets.items()}, index=dates)
    c['^IRX'] = 2.5
    op = c.drop(columns='^IRX').copy()
    vix = pd.Series(np.where(np.arange(len(dates)) % 160 < 65, 35., 16.), index=dates)
    return c, op, vix


def build():
    return build_market_data(*market_fixture(), asof='2019-05-31', sector_pairs={'XLK':'ROM', 'XLE':'DIG'})


def test_missing_symbols_rejected():
    c, o, v = market_fixture()
    with pytest.raises(ValueError, match='Missing required'):
        build_market_data(c.drop(columns='QQQ'), o, v, sector_pairs={'XLK':'ROM'})


def test_future_data_cannot_change_asof_or_source():
    c, o, v = market_fixture()
    pairs = {'XLK':'ROM', 'XLE':'DIG'}
    a = build_market_data(c, o, v, asof='2019-05-31', sector_pairs=pairs)
    c2 = c.copy();o2 = o.copy();v2 = v.copy()
    c2.loc['2019-06-03':,'QQQ'] *= 3
    o2.loc['2019-06-03':,'QQQ'] *= 3
    v2.loc['2019-06-03':] *= 4
    b = build_market_data(c2, o2, v2, asof='2019-05-31', sector_pairs=pairs)
    np.testing.assert_array_equal(a.panel, b.panel)
    np.testing.assert_array_equal(a.seed_closes, b.seed_closes)
    assert a.asof == b.asof


def test_seed_is_deterministic_and_assets_remain_synchronized():
    m = build()
    sc = Scenario(paths=4, years=1, block_days=21, seed=123)
    x = sample_joint_indices(m, sc)
    np.testing.assert_array_equal(x, sample_joint_indices(m, sc))
    assert x.shape == (4, 252)
    # each row is ONE common market index, not independently sampled columns
    assert (x >= 0).all() and (x < len(m.panel)).all()
    assert not np.array_equal(x[0], x[1])


def test_regime_matrices_are_stochastic_and_replay_is_finite():
    m = build()
    np.testing.assert_allclose(m.regime_transition.sum(axis=1), [1., 1.])
    for mode in ('current', 'unconditional', 'persistent_high', 'forced_high', 'forced_low'):
        c = Scenario(paths=3, years=1, block_days=21, mode=mode, transition_mode='semi_markov', seed=5)
        z = evaluate(m, c)
        assert z['valid'] == 3
        assert 0 <= z['p_blend_higher_terminal_wealth'] <= 1
        assert -1 <= z['p50_drawdown_blend'] <= 0
        assert np.isfinite(z['median_annual_log_return_advantage'])


def test_prior_close_logic_does_not_use_future_close():
    m = build()
    # With exactly one step, altering its simulated closing price must not affect
    # either policy's P&L: same prior anchor + same executable open return.
    obs = m.panel.copy()
    x = np.asarray([4], dtype=np.int32)
    params = (m.sector_count, .01, .0002, .0007, .0005)
    a = _run_path(obs, m.seed_closes, x, *params)
    obs[4, len(m.names):2*len(m.names)] *= 3
    b = _run_path(obs, m.seed_closes, x, *params)
    np.testing.assert_allclose(a, b)


def test_cost_stress_is_configurable_without_changing_frozen_rules():
    m = build()
    a = evaluate(m, Scenario(paths=5, years=1, block_days=21, seed=42, funding_spread_bps_annual=100))
    b = evaluate(m, Scenario(paths=5, years=1, block_days=21, seed=42, funding_spread_bps_annual=500))
    assert a['p_baseline_50pct_drawdown'] >= 0
    assert b['valid'] > 0
    assert manifest(m, [Scenario()])['source_sha256_panel']


def test_asof_anchor_rejects_missing():
    c, o, v = market_fixture()
    v.loc['2019-05-31'] = np.nan
    with pytest.raises(ValueError, match='As-of date'):
        build_market_data(c, o, v, asof='2019-05-31', sector_pairs={'XLK':'ROM'})
