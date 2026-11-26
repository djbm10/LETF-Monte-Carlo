import numpy as np
import pandas as pd
import pytest

from scripts.capture_daily_signal_inputs import normalize


def test_original_irx_convention_is_not_french_monthly_or_dff_proxy_and_never_fills_gaps():
    frame=pd.DataFrame({'Close':[3.65,np.nan,4.]},index=pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06']))
    rate=normalize(frame,'^IRX')
    assert rate.iloc[0]==pytest.approx(.0365/252) and rate.isna().sum()==1
    assert normalize(frame,'^VIX').iloc[0]==3.65
    with pytest.raises(ValueError):normalize(frame,'DFF')
