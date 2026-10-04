# LEAPS Stage-1 disposition — 2026-10-04

Decision: MONITOR
Holdout claim: NONE. The free 2012-2025 surface and pre/post-2020 subperiods have already been inspected, so no portion of that dataset is now represented as an untouched holdout.

## Mechanism → Condition X → Outcome Y → Baseline → Falsifier → Test → Decision

Mechanism:
Deep-ITM long-dated calls can provide capital-efficient equity exposure with bounded premium loss. Any advantage should come from exposure/financing/collateral efficiency, not from an assumed option-market mispricing.

Who creates the opportunity:
Option-market makers and investors price financing, dividends, volatility risk premium, liquidity and convexity into long-dated calls. A durable edge would require the total embedded financing/volatility/liquidity cost to be lower than the utility/value of the capital efficiency in the relevant configuration.

Condition X:
Frozen QQQ/SPY LEAPS grid: delta 0.70/0.80/0.90, DTE 365/548/730, premium allocation 25%/50%/100%, six-month rolls / forced roll below 180 DTE, OI >=100, real quote-side execution plus frozen slippage sensitivity.

Outcome Y:
Higher compounded wealth than underlying buy-and-hold without catastrophic path risk, surviving realistic quote-side execution and an independent options database.

Baseline:
QQQ or SPY buy-and-hold over identical periods, with collateral treatment explicitly accounted for.

Falsifier:
The apparent excess disappears with independent next-open executable quotes, IvyDB/OptionMetrics replication, realistic size/liquidity, or is achieved only through extreme premium allocation and materially worse drawdown.

## Existing evidence
FREE_DISCOVERY_LOCAL_EOD_PROXY completed all 162 frozen cells using real historical SPY/QQQ option-chain data but next-session quote-side CLOSE, not the frozen next-session OPEN.
At 10 bps added slippage, three QQQ moderate-allocation cells exceeded QQQ in both pre/post-2020 subperiods while keeping max drawdown within 10 percentage points of QQQ. SPY had none.
The highest-CAGR 100%-premium cells suffered roughly catastrophic drawdowns (best-CAGR cell about -89.1% max DD), so those are leverage, not evidence of superior probability-adjusted wealth.

## Why not PROMOTE
- no untouched holdout remains inside the already-inspected free dataset;
- execution differs materially from the frozen next-open design;
- no independent IvyDB/OptionMetrics replication;
- no established mispricing mechanism;
- performance improvement can be explained by higher effective equity exposure/capital efficiency.

## Next permitted test
Run the already-frozen family on an independent real historical options source with next-open executable quote semantics. Do not change delta/DTE/allocation/roll/cost parameters based on the free surface.
