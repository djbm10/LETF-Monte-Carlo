# QuantConnect cloud smoke checklist — pre-return gate

These are integration tests, not performance tests. Do not interpret CAGR, Sharpe,
drawdown, alpha, or relative returns from these short smoke windows.

## Reproducible execution path

Preferred execution is the gated GitHub workflow:
`.github/workflows/quantconnect-stage1-smoke.yml`.

Default validated SEC artifact source:
- workflow run: `35946883974`
- event: `[live-sec-smoke] exact-head final data validation`
- SEC fundamentals smoke: passed
- SEC Item 1 network smoke: passed

The workflow requires repository secrets `QC_USER_ID`, `QC_API_TOKEN`, and
`QC_ORGANIZATION_ID`. It runs
`research/free_stage1/quantconnect_cloud_smoke.py` and uploads only a
mechanical go/no-go JSON report. The smoke runner must not inspect or emit
performance statistics.

## 1) Bottleneck Winner end-to-end smoke

Use the small SEC/network smoke bundle generated from the validated live SEC job.

Project: `BottleneckWinnerFreePIT`

Parameters:

```
model=bottleneck_core
top_n=10
min_cross_section=10
start=2024-01-02
end=2024-03-29
fundamentals_url=<smoke SEC CSV URL>
network_url=<smoke network CSV URL>
```

The URL parameters may be replaced with:
`fundamentals_object_key` and `network_object_key`.

Acceptance:
- algorithm initializes without data-source exceptions
- log shows non-zero SEC fundamental and network rows
- at least 10 names have complete bottleneck-core features at the January formation
- every selected row has a CIK, SEC `fund_file_date <= formation_date`, and network snapshot available by formation
- selected-name audit is written to Object Store
- equity orders/fills occur with the frozen 15 bps one-way execution drag
- no current-ticker CIK/GVKEY bridge is introduced
- no performance metric from this smoke is used to alter weights, thresholds, portfolio size, or sample dates

If fewer than 10 names have complete real features in the 18-name bundle, enlarge
the smoke network bundle; do not weaken the full-run feature definitions.

## 2) Real historical LEAPS execution smoke

Project: `RealHistoricalLeaps`

Parameters:

```
underlying=SPY
target_delta=0.80
target_dte=548
allocation=1.00
slippage_bps=5
min_oi=100
start=2023-01-03
end=2024-02-01
```

Acceptance:
- at least one real historical option contract is selected
- selection occurs from a completed daily chain
- fresh entry fills at the following regular-session open, not the selection close
- a six-month/DTE roll sells the old contract first; the replacement is selected from the subsequent completed chain and bought at the following open
- no simultaneous same-open roll sell/buy pair is submitted
- no invalid/rejected option orders
- selection audit contains expiry, strike, DTE, delta, bid, ask, OI, quantity, underlying price and premium budget
- fill audit contains actual symbol, direction, quantity, price and brokerage fee
- actual fills are consistent with quote-side opening execution plus the configured slippage model
- no performance metric from this smoke is used to change the frozen 162-cell grid

## Go/no-go

Only after both smokes pass should the full Stage-1 matrix begin:
- Bottleneck Winner: 15 cells
- LEAPS: 162 cells
- total: 177 frozen runs

Any mechanical bug found by the smokes may be fixed before the return matrix is
observed, but the fix and its rationale must be appended to `frozen_spec.json`
and the smoke re-run before proceeding.
