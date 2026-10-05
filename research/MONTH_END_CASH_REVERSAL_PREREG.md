# Month-End Institutional Cash-Need Reversal — Frozen Discovery Preregistration

Frozen before any discovery returns for this hypothesis are inspected.

## Mechanism
Settlement-driven cash raising by pensions, mutual funds, asset managers, and other institutions with recurring month-end obligations can force sales of liquid equities early enough for cash to settle by month-end. Forced supply should create temporary price pressure near the last eligible sale date and a reversal after the cash need passes.

## Condition X
For each calendar month, let T be the final SPY trading session of the month. Define D as the last SPY trading session whose sale is expected to settle by T under the then-current U.S. equity settlement cycle:
- T+3 before 2017-09-05: D = T minus 3 trading sessions.
- T+2 from 2017-09-05 through discovery end: D = T minus 2 trading sessions.
- T+1 beginning 2024-05-28 is reserved for the untouched holdout only and is not read during discovery.

Trading-session offsets are used as a reproducible public-data approximation to settlement business days. This approximation is fixed before returns are inspected.

## Expected Outcome Y
1. The close-to-close return ending on D is lower than comparable non-month-end returns.
2. Buying SPY at D close and holding 1, 3, or 5 trading sessions produces positive post-cost excess return versus matched non-month-end windows.
3. The pressure date should move later when the settlement cycle shortens from T+3 to T+2.

## Instrument
SPY, using adjusted daily close for return measurement. Discovery downloads end at 2019-12-31; no 2020+ return is fetched.

## Sample split
- TRAIN: 1993-02-01 through 2008-12-31.
- VALIDATION: 2009-01-01 through 2019-12-31.
- UNTOUCHED HOLDOUT: 2020-01-01 through latest available date, opened only after discovery advancement and an immutable frozen specification are recorded.

## Baseline
For each D, matched non-month-end candidate dates must:
- be in the same sample partition;
- have the same weekday as D;
- have the same trailing-20-session realized-volatility quintile computed from information available by the prior close;
- be outside +/- 7 trading sessions of any month-end settlement deadline.

The event-minus-matched-baseline difference is the primary excess-return observation.

## Execution and cost
- Signal/event is known from the calendar before D.
- Entry at D close.
- Exit at the close 1, 3, or 5 trading sessions later.
- Fixed 5 bps round-trip implementation drag is deducted from event strategy returns.
- No leverage is used in Stage 1.

## Frozen parameter neighborhood
Holding horizons = {1, 3, 5} sessions. Primary descriptive horizon = 3 sessions. No other holding periods may be added after viewing discovery results.

## Mechanism falsifier
The mechanism is falsified for Stage 1 if either:
- deadline-day pressure excess is not negative in both T+3 and T+2 discovery-era regimes where sufficient observations exist; or
- the settlement-adjusted deadline does not exhibit more negative pressure than its nearby pre-month-end offsets in aggregate.

## Return advancement rule
The family advances to the holdout only if all are true:
1. validation event-minus-baseline mean excess after cost is positive for at least 2 of 3 frozen holding horizons;
2. median validation mean excess across {1,3,5} is positive;
3. median train mean excess across {1,3,5} is positive;
4. the mechanism falsifier does not fire;
5. data-quality checks pass.

Bootstrap confidence intervals and t-statistics are reported but are not used to change parameters.

## Holdout rule
If the family advances:
1. write an immutable frozen-spec artifact containing this rule and discovery disposition;
2. only then run one holdout job over 2020+ using all 3 frozen horizons;
3. do not change dates, cost, baseline matching, settlement-regime mapping, or holding periods based on holdout results.

If discovery fails, classify KILL or MONITOR as warranted and do not open the holdout.
