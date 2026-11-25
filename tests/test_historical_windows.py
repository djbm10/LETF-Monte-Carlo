import numpy as np
import pandas as pd
import pytest

from letf.macro.counterfactual import dated_statistics,historical_windows
from scripts.diagnose_reconstruction import holding_windows


def test_counterfactual_cagr_starts_at_previous_close_not_first_return_date():
    idx=pd.to_datetime(['1933-03-15','1933-03-16'])
    s=dated_statistics(pd.Series([.1,0.],index=idx),pd.Series([12.,1.],index=idx))
    assert s['period_start_close']=='1933-03-03'
    assert s['cagr']==pytest.approx(1.1**(365.2425/13)-1)


def test_ten_year_error_window_cannot_be_invented_from_six_years():
    idx=pd.bdate_range('2020-01-01',periods=1512)
    actual=pd.Series(.001,index=idx);predicted=pd.Series(.002,index=idx)
    r=holding_windows(actual,predicted)
    assert r['10']['status']=='BLOCKED_INSUFFICIENT_DAILY_COVERAGE'
    assert r['1']['relative_error_p10_p50_p90']==pytest.approx([(1.002/1.001)**252-1]*3)
