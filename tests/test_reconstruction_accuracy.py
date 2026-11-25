from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from letf.reconstruction import (SourceRecord, daily_reset_returns, ReconstructionGate,
                                 validate_reconstruction)
from letf.accuracy import Evidence, UnsupportedModelError


def source():
    return SourceRecord('test fixture', 'https://example.test/input', '2026-11-23', '2026-11-23',
                         'a' * 64, '2020-01-02', 'USD', 'total_return', 'daily', False)


def series(values):
    return pd.Series(values, index=pd.bdate_range('2020-01-02', periods=len(values)))


def test_daily_path_dependence_and_no_second_volatility_charge():
    r = series([.1, -1/11])
    result = daily_reset_returns(r, 3, source=source())
    assert np.prod(1 + r) == pytest.approx(1)
    assert np.prod(1 + result['return']) == pytest.approx(1.3 * (1 - 3/11))
    assert np.prod(1 + result['return']) < 1


def test_observed_returns_no_duplicate_internal_costs():
    r = series([.1, -.1])
    result = daily_reset_returns(r, 1, source=replace(source(), fees_embedded=True), observed_fund_returns=True)
    np.testing.assert_array_equal(result['return'], r)
    assert set(result['evidence']) == {Evidence.OBSERVED.value}
    with pytest.raises(ValueError, match='already include'):
        daily_reset_returns(r, 1, annual_fee=.01, source=source(), observed_fund_returns=True)


def test_calendar_fee_and_financing_accrual():
    r = series([0., 0.])
    result = daily_reset_returns(r, 3, annual_fee=.01, funding_rate_at_previous_close=.05,
                                 borrowing_spread=.005, elapsed_calendar_days=series([1, 3]), source=source())
    assert result['return'].iloc[1] == pytest.approx(-.01 * 3/365 - 2 * .055 * 3/360)


def test_bankrupt_fund_never_recovers_and_costs_stop():
    result = daily_reset_returns(series([-.4, .9, .9]), 3, annual_fee=.01, source=source())
    assert list(result['return']) == [-1, 0, 0]
    assert list(result['fund_fee'].iloc[1:]) == [0, 0]


def test_monthly_or_missing_data_not_fabricated():
    r = pd.Series([.01] * 12, index=pd.date_range('2020-01-31', periods=12, freq='ME'))
    with pytest.raises(UnsupportedModelError, match='monthly'):
        daily_reset_returns(r, 3, source=source())
    with pytest.raises(ValueError, match='nonfinite'):
        daily_reset_returns(series([np.nan]), 3, source=source())
    with pytest.raises(UnsupportedModelError, match='provenance'):
        daily_reset_returns(series([.01]), 3)


def test_frozen_gate_needs_temporal_separation_and_verified_provenance():
    r = series([.001] * 600)
    gate = ReconstructionGate()
    result = validate_reconstruction(r, r, training_end='2019-12-31', gate=gate)
    assert result['numerical_pass'] and not result['passed']
    assert result['evidence'] == Evidence.HYPOTHETICAL.value
    result = validate_reconstruction(r, r, training_end='2019-12-31', gate=gate, provenance_verified=True)
    assert result['passed']
    with pytest.raises(ValueError, match='strictly after'):
        validate_reconstruction(r, r, training_end=r.index[-1], gate=gate)
    assert gate.fingerprint == ReconstructionGate().fingerprint


def test_gate_fails_large_bias_even_with_high_correlation():
    r = series([.001] * 600)
    result = validate_reconstruction(r, r + .001, training_end='2019-12-31', gate=ReconstructionGate())
    assert not result['numerical_pass']
    assert not result['checks']['daily_bias']
def test_gross_swap_spread_and_collateral_credit_are_not_net_borrowing_costs():
    from letf.reconstruction import daily_reset_returns,SourceRecord
    from letf.accuracy import UnsupportedModelError
    import pandas as pd
    import pytest
    idx=pd.to_datetime(['2026-05-28','2026-05-29']);zero=pd.Series(0.,index=idx)
    source=SourceRecord('fixture','https://example.invalid','2026-11-24','fixture','a'*64,'2026-05-28','USD','daily TR','daily',False)
    kw=dict(source=source,annual_fee=.01,funding_rate_at_previous_close=.06,borrowing_spread=.002,
        elapsed_calendar_days=pd.Series([1.,1.],index=idx))
    gross=daily_reset_returns(zero,3,**kw,financed_notional=3.,collateral_income_per_nav=.06,
        financing_source='Dated issuer contract fixture, notional 3 per NAV, collateral income .06 per NAV')
    assert gross.financing.iloc[0]==pytest.approx(3*.062/360-.06/365)
    assert gross['return'].iloc[0]==pytest.approx(-.01/365-3*.062/360+.06/365)
    net=daily_reset_returns(zero,3,**kw)
    assert net.financing.iloc[0]==pytest.approx(2*.062/360)
    with pytest.raises(UnsupportedModelError):daily_reset_returns(zero,3,**kw,financed_notional=3.)
    with pytest.raises(ValueError):daily_reset_returns(zero,3,**kw,collateral_income_per_nav=.06)
    with pytest.raises(ValueError):daily_reset_returns(zero,3,**kw,funding_day_basis=float('nan'))
    with pytest.raises(ValueError):daily_reset_returns(zero,1,source=source,observed_fund_returns=True,
        financed_notional=0.,collateral_income_per_nav=0.,financing_source='fixture')

