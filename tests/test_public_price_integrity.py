"""Public-data ingestion must not convert invalid NAV/index rows into returns.

A documented source with zero NAV can represent actual liquidation, but that
requires separate dated corporate-action/closure evidence. This generic
live-fund source adapter must reject it rather than infer a tradable fill.
"""
import pytest

from letf.lab.public_sources import inspect


@pytest.mark.parametrize("nav", ["0", "-1.5", "NaN"])
def test_live_issuer_nav_rejects_zero_negative_and_missing_values(nav):
    blob = (
        "Date,NAV\n"
        "01/02/2025,10\n"
        f"01/03/2025,{nav}\n"
    ).encode()
    with pytest.raises(ValueError, match="Issuer NAV must be finite and strictly positive"):
        inspect("TQQQ_nav", blob)


@pytest.mark.parametrize("name", ["SP500", "NASDAQ100", "NASDAQXNDX"])
@pytest.mark.parametrize("bad_level", ["0", "-0.5"])
def test_equity_index_levels_reject_nonpositive_values(name, bad_level):
    blob = (
        f"DATE,{name}\n"
        "2025-01-02,100\n"
        f"2025-01-03,{bad_level}\n"
    ).encode()
    with pytest.raises(ValueError, match="Official index level must be strictly positive"):
        inspect(name, blob)


def test_valid_issuer_nav_and_fred_missing_obs_and_negative_rate():
    nav = inspect("TQQQ_nav", b"Date,NAV\n01/02/2025,10\n01/03/2025,11\n")
    assert nav["rows"] == 2
    assert nav["first_observation"] == "2025-01-02"

    # A FRED missing source observation is not fabricated as a zero index value.
    index = inspect("SP500", b"DATE,SP500\n2025-01-02,100\n2025-01-03,.\n")
    assert index["missing_values"] == 1
    assert index["rows"] == 2

    # Negative macro interest rates are not invalid nonpositive INDEX LEVELS.
    rate = inspect("DFF", b"DATE,DFF\n2025-01-02,-0.5\n2025-01-03,0\n")
    assert rate["rows"] == 2
