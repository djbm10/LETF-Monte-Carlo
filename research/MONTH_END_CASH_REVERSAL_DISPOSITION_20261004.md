# Month-End Institutional Cash-Need Reversal — Discovery Disposition

**Status: KILL**

The 2020+ holdout remains sealed and was not downloaded or inspected.

## Frozen source
Preregistration: `research/MONTH_END_CASH_REVERSAL_PREREG.md`

The discovery CI run was queued behind unrelated research workflows. Under the anti-stall protocol, the exact frozen discovery logic was reproduced with the smallest direct Yahoo-chart probe using SPY adjusted closes only through **2019-12-31**. No scientific parameter was changed.

## Data-quality gate
- Price coverage: 1993-01-29 through 2019-12-31.
- Train event count: 188.
- Validation event count: 132.
- Minimum matched-baseline count: 21.
- Data-quality gate: PASS.
- Holdout: SEALED / NOT DOWNLOADED.

## Frozen return test

| Partition | Hold | Mean event-minus-baseline excess | Positive events |
|---|---:|---:|---:|
| Train | 1d | +0.0514% | 56.9% |
| Train | 3d | -0.0038% | 52.1% |
| Train | 5d | +0.1536% | 57.4% |
| Validation | 1d | +0.1036% | 53.8% |
| Validation | 3d | -0.1341% | 45.0% |
| Validation | 5d | -0.1681% | 50.4% |

Frozen advancement requirements were not met:
- validation positive horizons: **1 of 3** (required >=2);
- median validation mean excess: **-0.1341%** (required >0);
- median train mean excess: **+0.0514%**;
- data quality: PASS.

## Mechanism falsifier

The hypothesized mechanism predicted unusually negative price pressure on the settlement-adjusted cash-raising deadline.

Observed deadline one-day return minus matched same-weekday/PIT-volatility baseline:
- T+3 era: **+0.1221%**, n=292.
- T+2 era: **+0.0148%**, n=28.
- Deadline return minus nearby (-2,-1,+1,+2 session) returns: **+0.0980%**, n=320.

All three signs are opposite the mechanism's required direction.

Therefore the mechanism falsifier fires independently of the return advancement failure.

## Decision

**KILL.** The frozen Stage-1 evidence does not support a settlement-driven pre-month-end forced-selling reversal in SPY under this specification. Per the preregistration, the 2020+ holdout must remain sealed. No thresholds, holding periods, cost assumptions, matching rules, or settlement-date rules may be altered to rescue the hypothesis.

This does not claim that every turn-of-month effect is false. It rejects this specific mechanism-first hypothesis and frozen implementation.
