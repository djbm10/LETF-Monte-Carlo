# Bottleneck Stage-1 final disposition — 2026-10-04

## Final disposition: PROMOTE TO STAGE 2 — NOT IMPLEMENTATION-GRADE

This disposition was recorded only after the single frozen 2020-2023 holdout run completed. No parameter, factor, top-N, comparator, transaction-cost, universe, staleness, execution, or ranking rule was changed after discovery or after seeing holdout returns.

## Scientific framing

**Mechanism**
Companies occupying economically scarce competitive positions should convert demand acceleration and pricing power into stronger subsequent equity returns than otherwise-similar momentum-selected large-cap firms. Competitive scarcity is measured point-in-time from direct SEC 10-K Item 1 text, while demand/pricing/quality inputs use as-filed SEC fundamentals.

**Condition X**
At quarterly formation dates, a company ranks highly under the frozen Bottleneck factor family using only information available by that date.

**Outcome Y**
Forward portfolio CAGR should exceed the frozen family-specific comparator across a stable neighborhood of top-N portfolio sizes.

**Baselines**
- bottleneck_core vs momentum
- bottleneck_momentum vs momentum
- bottleneck_full vs quality_momentum
- SPY is reported as an absolute investability context baseline, not substituted for the preregistered family comparator.

**Falsifier**
Failure to produce positive holdout CAGR excess at at least 2 of 3 top-N values with positive median excess, after authoritative direct-SEC replication and frozen costs/implementation.

## Authoritative discovery

Direct-SEC network artifact:
- run: 37221395373
- 44 quarterly snapshots, 2009-03-31 through 2019-12-31
- min valid Item 1 CIKs: 452
- median valid Item 1 CIKs: 459
- network rows: 20,203
- direct-SEC filing rows: 6,591
- unique CIKs: 699
- frozen network quality gate: PASS

Direct-SEC discovery:
- run: 37234784573
- holdout status during discovery: SEALED_NOT_READ
- all 15 frozen cells retained
- 15 bps one-way cost unchanged

Family advancement:
- bottleneck_core: 3/3 top-N positive validation excess; median CAGR excess +6.3977%; median Sharpe excess +0.4614.
- bottleneck_momentum: 3/3 positive; median CAGR excess +5.6693%; median Sharpe excess +0.3459.
- bottleneck_full: 3/3 positive; median CAGR excess +4.3313%; median Sharpe excess +0.2716.
- decision: ADVANCE_ALL_15_TO_SINGLE_HOLDOUT_RUN.

## Pre-holdout gates

Sealed direct-SEC holdout network:
- run: 37229757229
- 16 quarterly snapshots, 2020-03-31 through 2023-12-31
- min valid Item 1 CIKs: 470
- median valid Item 1 CIKs: 471
- network rows: 7,548
- manifest status: DATA_QUALITY_ONLY_NO_RETURNS_EXPOSED
- gate: PASS

Sealed price/identity recovery:
- run: 37228861774
- sample: 2020-01-01 through 2023-12-31
- identity mapping session coverage: 100.00%
- effective identity-plus-price coverage: 98.6302%
- frozen gate: 97%
- gate: PASS

## Single untouched holdout

Run: 37235357598
Execution: one all-15-cell run, 2020-01-01 through 2023-12-31.
No prior successful holdout run existed.

### bottleneck_core — CONFIRMED
- positive CAGR excess: 2/3 top-N values
- median holdout CAGR excess: +1.2849%
- median holdout Sharpe excess: +0.0655
- median max-drawdown difference: +0.3645 percentage points
- frozen confirmation rule: PASS
- risk interpretation: RETURN_AND_RISK_ADJUSTED_SUPPORT

By top-N:
- N=10: 5.3652% CAGR vs momentum 6.5269%; excess -1.1616%; Sharpe excess -0.0303.
- N=20: 8.0106% vs 6.7257%; excess +1.2849%; Sharpe excess +0.0655.
- N=40: 11.1236% vs 6.9747%; excess +4.1489%; Sharpe excess +0.1766.

### bottleneck_momentum — NOT CONFIRMED
- positive CAGR excess: 0/3
- median excess: -1.6979%
- median Sharpe excess: -0.0518
- KILL this frozen family specification.

### bottleneck_full — NOT CONFIRMED
- positive CAGR excess: 1/3
- median excess: -2.4332%
- median Sharpe excess: -0.0812
- KILL this frozen family specification.

## Absolute baseline context

SPY holdout benchmark:
- CAGR: 11.7585%
- annualized volatility: 22.6271%
- Sharpe (0 rf): 0.6052
- max drawdown: -33.7173%

Best confirmed core cell, top-40:
- CAGR: 11.1236%
- annualized volatility: 23.5814%
- Sharpe: 0.5654
- max drawdown: -35.2098%

Thus Stage-1 confirms incremental Bottleneck information relative to the preregistered momentum comparator, but it does not show that this free large-cap core portfolio beats a simple SPY allocation on absolute probability-adjusted wealth in the holdout.

## Decision

**PROMOTE Bottleneck core to Stage 2 research.**
Do not promote bottleneck_momentum or bottleneck_full.

Rationale:
1. the core mechanism survived authoritative direct-SEC discovery replication;
2. the frozen core family passed the untouched holdout confirmation rule;
3. positive holdout evidence appears at adjacent portfolio sizes N=20 and N=40 rather than one isolated cell;
4. risk-adjusted comparator evidence is positive at the median;
5. nevertheless the effect is modest in holdout, N=10 fails, and SPY remains superior to the best core cell on CAGR, Sharpe, and max drawdown over 2020-2023.

Next scientific requirement:
Replicate the frozen bottleneck_core family on the stronger full-US licensed/PIT architecture (CRSP + Compustat as-reported/PIT + historical CCM links + TNIC/ETNIC or equivalent network data) without changing the Stage-1 rule because of these holdout results.

This is a research promotion, not a live-trading recommendation.
