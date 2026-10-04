# Frozen sector-rotation validation specification — 2026-10-04

Status: FROZEN_BEFORE_RESULTS

## Mechanism → Condition X → Outcome Y → Baseline → Falsifier → Test → Decision

Mechanism:
Persistent sector-level earnings revisions, institutional allocation, benchmark-relative flows, and slow-moving macro/capex cycles can cause medium-horizon relative strength to persist across broad liquid sectors.

Who creates the opportunity:
Institutional allocators, benchmarked managers, systematic trend/momentum flows, and investors reacting gradually to persistent sector earnings/capex information.

Condition X:
A sector ranks among the strongest original U.S. Select Sector SPDRs by trailing 6- or 12-month adjusted-price momentum at month-end. Optional risk gate: SPY above its 200-trading-day moving average.

Outcome Y:
The selected sector basket should deliver higher geometric return and/or better risk-adjusted return than SPY after realistic trading costs, with evidence stable across adjacent specifications.

Baseline:
SPY buy-and-hold over identical dates.

Falsifier:
Kill the family if validation performance is not consistently better than SPY across adjacent lookbacks/top-N choices after costs, or if gains are dominated by one narrow cell/regime. Holdout cannot rescue a failed validation family.

## Instruments
Original nine Select Sector SPDRs only:
XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLY, XLB.
Benchmark: SPY.

## Frozen parameter neighborhood
- momentum lookback: 126 or 252 trading sessions
- portfolio breadth: top 1 or top 3 sectors
- risk gate: none or SPY > 200-day SMA
- rebalance: monthly
- signal: completed month-end close
- execution: next trading session close
- weighting: equal weight among selected sectors
- cash return when risk gate is off: 0% in this first conservative proxy
- transaction cost: 10 bps one-way on traded notional at each rebalance

## Frozen sample split
- train: 1999-01-01 through 2008-12-31
- validation: 2009-01-01 through 2016-12-31
- untouched holdout: 2017-01-01 through 2026-10-02

## Family selection rule before holdout
Use train + validation only.
Prefer a parameter neighborhood, not the single top cell.
A family advances only if:
1. at least 3 of the 4 adjacent cells sharing the same risk-gate state have positive validation CAGR excess vs SPY; and
2. median validation CAGR excess is positive; and
3. median validation Sharpe excess is non-negative; and
4. no single cell accounts for the conclusion.

If both gated and ungated families qualify, choose the family with the stronger worst-cell validation CAGR excess; ties go to lower turnover/top-3 breadth.

After family choice, freeze one representative rule at the neighborhood center using no holdout information. Then open the holdout once.
