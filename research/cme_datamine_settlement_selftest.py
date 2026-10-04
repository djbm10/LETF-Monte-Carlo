from __future__ import annotations

import numpy as np
import pandas as pd

import cme_datamine_settlement_backtest as m


def test_roll_gap_not_counted_as_pnl() -> None:
    dates = pd.bdate_range("2024-03-01", "2024-03-18")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({
            "trade_date": dt,
            "contract": "NQH24",
            "expiry": pd.Timestamp("2024-03-15"),
            "settle": 100.0 + i,
            "volume": max(1, 1000 - 100 * i),
            "open_interest": max(1, 2000 - 100 * i),
        })
        rows.append({
            "trade_date": dt,
            "contract": "NQM24",
            "expiry": pd.Timestamp("2024-06-21"),
            "settle": 110.0 + i,
            "volume": 100 + 100 * i,
            "open_interest": 500 + 100 * i,
        })
    z = pd.DataFrame(rows)
    c = m.make_continuous(z, rule="fixed_5bd")
    rolled = c.index[c["rolled"]]
    assert len(rolled) == 1, c[["contract", "rolled"]]
    rd = rolled[0]
    pos = c.index.get_loc(rd)
    prev_dt = c.index[pos - 1]
    old_prev = float(z[(z.trade_date == prev_dt) & (z.contract == "NQH24")].settle.iloc[0])
    old_today = float(z[(z.trade_date == rd) & (z.contract == "NQH24")].settle.iloc[0])
    expected = old_today / old_prev - 1
    assert abs(float(c.loc[rd, "fut_ret"]) - expected) < 1e-12, (c.loc[rd], expected)

    next_dt = c.index[pos + 1]
    new_roll = float(z[(z.trade_date == rd) & (z.contract == "NQM24")].settle.iloc[0])
    new_next = float(z[(z.trade_date == next_dt) & (z.contract == "NQM24")].settle.iloc[0])
    expected_next = new_next / new_roll - 1
    assert abs(float(c.loc[next_dt, "fut_ret"]) - expected_next) < 1e-12, (c.loc[next_dt], expected_next)


def test_frozen_signal_uses_prior_day_only() -> None:
    dates = pd.bdate_range("2023-01-02", periods=260)
    rets = np.where(np.arange(len(dates)) % 2 == 0, 0.002, -0.0005)
    px = pd.Series(100.0 * np.cumprod(1.0 + rets), index=dates)
    crash_i = 220
    px.iloc[crash_i:] = px.iloc[crash_i - 1] * 0.50 * np.cumprod(
        np.full(len(px) - crash_i, 1.0002)
    )
    rf = pd.Series(0.0, index=dates)
    e = m.target_exposure(px, rf, bull=0.35, bear=0.0, cap=3.0)

    # Exposure on the crash date is determined by the prior session only.
    assert e.iloc[crash_i] > 0, e.iloc[crash_i]
    # The crash becomes observable only after that close, so the next session
    # can de-risk. This is the no-look-ahead property required by the freeze.
    assert e.iloc[crash_i + 1] == 0.0, e.iloc[crash_i + 1]


def test_signal_loader_never_backfills_future_data(tmp_path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    src = pd.DataFrame({
        "date": [dates[2], dates[4]],
        "adj_close": [100.0, 101.0],
    })
    p = tmp_path / "signal.csv"
    src.to_csv(p, index=False)
    got = m.load_signal_price(p, dates)
    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == 100.0
    assert got.iloc[3] == 100.0
    assert got.iloc[4] == 101.0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_roll_gap_not_counted_as_pnl()
    test_frozen_signal_uses_prior_day_only()
    with tempfile.TemporaryDirectory() as td:
        test_signal_loader_never_backfills_future_data(Path(td))
    print("CME_DATAMINE_ADAPTER_SELFTEST_PASS")
