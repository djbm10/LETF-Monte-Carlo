import numpy as np
import pandas as pd
import pytest

from letf.macro.counterfactual import dated_statistics,historical_windows,endpoint_real_statistics
from scripts.diagnose_reconstruction import holding_windows


def test_counterfactual_cagr_starts_at_previous_close_not_first_return_date():
    idx=pd.to_datetime(['1933-03-15','1933-03-16'])
    s=dated_statistics(pd.Series([.1,0.],index=idx),pd.Series([12.,1.],index=idx))
    assert s['period_start_close']=='1933-03-03'
    assert s['cagr']==pytest.approx(1.1**(365.2425/13)-1)


def test_counterfactual_purchasing_power_uses_initial_close_month_not_first_return_month():
    cpi=pd.Series([100.,105.,110.],index=pd.to_datetime(['1929-12-31','1930-01-31','1931-12-31']))
    stats=dict(period_start_close='1929-12-31',end='1931-12-31',terminal_index=1.21)
    real=endpoint_real_statistics(stats,cpi)
    years=(pd.Timestamp('1931-12-31')-pd.Timestamp('1929-12-31')).days/365.2425
    assert real['real_cagr']==pytest.approx(1.1**(1/years)-1)
    assert real['inflation_base_month']=='1929-12-31'
    from letf.accuracy import UnsupportedModelError
    with pytest.raises(UnsupportedModelError,match='endpoint'):endpoint_real_statistics(stats,cpi.iloc[1:])


def test_ten_year_error_window_cannot_be_invented_from_six_years():
    idx=pd.bdate_range('2020-01-01',periods=1512)
    actual=pd.Series(.001,index=idx);predicted=pd.Series(.002,index=idx)
    r=holding_windows(actual,predicted)
    assert r['10']['status']=='BLOCKED_INSUFFICIENT_DAILY_COVERAGE'
    assert r['1']['relative_error_p10_p50_p90']==pytest.approx([(1.002/1.001)**252-1]*3)
