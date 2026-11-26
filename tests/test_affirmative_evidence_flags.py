"""A string 'False' must never become trading/tax/provenance evidence."""
from dataclasses import replace

import pandas as pd
import pytest

from letf.accuracy import UnsupportedModelError
from letf.lab.catalog import require_available_at
from letf.reconstruction import validate_reconstruction,ReconstructionGate
from test_derivatives_accuracy import future_book,quote


def test_unknown_evidence_types_cannot_authorize_quotes_1256_or_source_certification():
    ledger,futures,contract=future_book();when,q=quote()
    returns=pd.Series(.001,index=pd.bdate_range('2020-01-01',periods=600))
    for value in ('False','True',0,1):
        with pytest.raises(UnsupportedModelError):futures.trade('2024-01-02',replace(contract,section1256=value),1,5000.)
        with pytest.raises(UnsupportedModelError):replace(q,observed=value).fill(when,'BUY',1)
        with pytest.raises(UnsupportedModelError):require_available_at({'point_in_time_verified':value,'available_at':'2024-01-01T00:00:00Z'},'2024-01-02T00:00:00Z')
        with pytest.raises(ValueError,match='boolean'):validate_reconstruction(returns,returns,training_end='2019-12-31',gate=ReconstructionGate(),provenance_verified=value)
    assert not futures.positions and ledger.accounts['a'].cash==10000.
