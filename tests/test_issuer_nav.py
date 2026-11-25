import numpy as np
import pandas as pd
import pytest
from letf.fund_nav import issuer_nav_returns
from letf.accuracy import UnsupportedModelError


def test_nav_ex_cash_return_and_prior_close_boundary_are_independently_reconciled():
    idx=pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04'])
    nav=pd.DataFrame({'NAV':[100.,90.,99.]},index=idx)
    rows=[{'ExDate':'2024-01-03','CashDividendPerShare':10.}]
    r,c=issuer_nav_returns(nav,rows,start='2024-01-03',end='2024-01-04')
    np.testing.assert_allclose(r,[0.,.1]);assert c.sum()==10.
    assert np.prod(1+r)==pytest.approx(1.1)
    # No made-up first return and no unsupported special cash.
    with pytest.raises(UnsupportedModelError,match='previous close'):issuer_nav_returns(nav,rows,start='2024-01-02',end='2024-01-04')
    with pytest.raises(UnsupportedModelError,match='Special'):issuer_nav_returns(nav,[dict(rows[0],SpecialPerShare=1.)],start='2024-01-03',end='2024-01-04')
    with pytest.raises(ValueError,match='chronology'):issuer_nav_returns(nav.iloc[::-1],rows,start='2024-01-03',end='2024-01-04')
