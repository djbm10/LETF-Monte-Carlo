# QuantConnect execution runbook — Free Stage 1

The research specification and parameter grid were frozen before return results.

## Projects

Create two QuantConnect Cloud projects using the repo code:

1. `BottleneckWinnerFreePIT`
   - source: `research/quantconnect/BottleneckWinnerFreePIT/main.py`
2. `RealHistoricalLeaps`
   - source: `research/quantconnect/RealHistoricalLeaps/main.py`

Do not modify factor weights, sample windows, portfolio sizes, delta targets, DTE targets, allocations, minimum OI, or slippage scenarios after seeing results.

## SEC fundamentals data

Build the full SEC as-filed panel, then compress it to prior-quarter-end CIK snapshots:

```bash
export SEC_USER_AGENT="Douglas research contact@example.com"
python research/free_stage1/sec_fsd_pit.py \
  --start-year 2009 \
  --end-year 2023 \
  --out results/free_stage1/sec_fundamentals

python research/free_stage1/sec_formation_features.py \
  --input results/free_stage1/sec_fundamentals/sec_pit_fundamentals.parquet \
  --formation-dates "$(python - <<'PY'
import pandas as pd
d=pd.date_range('2009-03-31','2023-12-31',freq='QE')
print(','.join(x.strftime('%Y-%m-%d') for x in d))
PY
)" \
  --out results/free_stage1/sec_fundamentals/sec_formation_features.csv
```

The prior-quarter-end snapshot is consumed at the next Jan/Apr/Jul/Oct first-trading-day rebalance. This deliberately excludes filings dated on the rebalance day because SEC bulk filing dates do not provide a reliable before-open/after-close timestamp.

Make `sec_formation_features.csv` available to the QuantConnect Bottleneck project and set:

`fundamentals_url=<location>`

Accounting factor values come from this SEC file. Do not substitute Morningstar financial-statement values for the frozen Stage-1 scoring run.

SEC filing invariant:
- 10-K/A, 10-Q/A, 20-F/A and 40-F/A enter only from their amendment filing date and retain amendment provenance
- annual filings use only qtrs=4 flow facts
- quarterly filings use only qtrs=1 flow facts
- valuation annualizes qtrs=1 revenue by 4 and uses qtrs=4 revenue directly
- other flow-period lengths are treated as missing, not silently substituted

The Bottleneck algorithm also runs a 252-daily-bar warm-up before 2009-01-01 and records price history before applying the current-day eligibility screen.

## Network data

Generate CIK-native network data from SEC filings with:

```bash
export SEC_USER_AGENT="Douglas research contact@example.com"
python research/free_stage1/sec_item1_network.py \
  --start-year 2009 \
  --end-year 2023 \
  --formation-dates "$(python - <<'PY'
import pandas as pd
d=pd.date_range('2009-03-31','2023-12-31',freq='QE')
print(','.join(x.strftime('%Y-%m-%d') for x in d))
PY
)"
```

Audit a stratified sample of Item 1 extracts before using performance results.

The combined output is:
`results/free_stage1/sec_item1_network/network_metrics_all.csv`

Make this file available to the QuantConnect project either via an authorized object-store/upload workflow or a stable URL under your control. Set project parameter:

`network_url=<location>`

For Bottleneck models, both `fundamentals_url` and `network_url` must be populated. Pure momentum does not require either SEC factor file.

Never substitute a current-ticker CIK↔GVKEY match.

## Bottleneck Winner matrix

Frozen grid is in:
`research/free_stage1/quantconnect_run_matrix.json`

15 runs:
- 5 model families
- top 10 / 20 / 40
- 2009-01-01 to 2023-12-31

For every run save:
- complete backtest JSON/statistics
- monthly equity curve
- orders/fills
- holdings at each rebalance
- turnover
- number of eligible/scored securities
- constituent CIKs and signal components at formation date

Evaluate periods without retuning:
- TRAIN: 2009-2014
- VALIDATION: 2015-2019
- FREE_HOLDOUT: 2020-2023

The model/portfolio-size choice for any later live candidate must be determined from train+validation only. Holdout is reported once.

## LEAPS matrix

162 pre-specified runs:
- SPY / QQQ
- delta 0.70 / 0.80 / 0.90
- target DTE 365 / 548 / 730
- premium allocation 25% / 50% / 100%
- slippage 0 / 5 / 10 bps
- min OI 100
- 2012-01-03 to 2026-09-21

Execution invariant:
- contract selection is made from a completed daily option chain
- a fresh entry is submitted as a Market-On-Open order for the next regular session
- a roll is sequenced: old contract exits next open; replacement is selected from that day's completed chain; replacement enters the following open
- this one-session roll gap is intentional and conservative, avoiding same-open sell/buy ordering ambiguity at 100% premium allocation
- selection-time bid/ask and next-open fill must both be retained so same-close execution leakage can be audited

For every run save:
- selected option symbol at each entry
- selection date and actual entry/exit fill dates
- strike / expiry / DTE
- delta at selection
- selection-time bid / ask and actual next-open fill
- OI
- premium paid
- effective portfolio delta exposure
- underlying price
- equity curve and drawdowns

Benchmarks:
- underlying buy-and-hold
- TQQQ/QLD or SPY leveraged ETF implementations over common dates
- S9 / 35-0 research outputs over common dates
- corrected futures implementation when exchange-grade settlement audit is available

## Multiple-testing discipline

Do not report the best CAGR alone.

For Bottleneck Winner report every model × portfolio-size cell and apply:
- validation/holdout separation
- turnover-adjusted returns
- factor/sector attribution
- bootstrap confidence intervals on active returns
- false-discovery-aware interpretation across the 15 strategies

For LEAPS report the full 162-cell surface:
- median and worst result across slippage assumptions
- parameter stability, not one optimized point
- drawdown / terminal-loss / turnover / spread-cost statistics
- common-date comparison against stock and futures

No parameter may be changed because of the 2020-2023 Bottleneck holdout or the full LEAPS result surface. Any changed specification becomes a new research generation with a new untouched holdout.

## Licensed Stage 2

Only after Stage 1 is frozen:
- CRSP + Compustat PIT/Snapshot + official historical linking + TNIC/ETNIC
- OptionMetrics IvyDB US

Stage 2 must use the same definitions and parameter grid, except unavoidable vendor-field mappings documented in a reconciliation table.

Evidence labels:
- FREE_DISCOVERY
- FREE_HOLDOUT
- LICENSED_REPLICATION
- SYNTHETIC_STRESS
