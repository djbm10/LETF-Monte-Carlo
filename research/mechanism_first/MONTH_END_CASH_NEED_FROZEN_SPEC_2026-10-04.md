# Frozen specification — Month-End Institutional Cash-Need Reversal

Frozen before any strategy return output is inspected.

## Scientific hypothesis

**Mechanism**
Settlement-driven cash raising by pensions, mutual funds, asset managers, and other institutions with recurring month-end obligations can force sales of liquid equities before the last trade date that still settles by month-end. If part of the selling is non-informational, the temporary price pressure should reverse after the required sale date passes.

**Participant**
Institutional investors with recurring cash obligations.

**Constraint / incentive**
Cash must be available by a scheduled month-end payment date. Missing the last settlement-eligible sale date creates liquidity or failed-payment risk.

**Condition X**
For each calendar month, let T be the final U.S. equity trading session of the month. Determine the latest U.S. equity trading session D whose standard-way settlement would occur no later than T under the settlement regime actually in force on trade date D.

Frozen settlement regimes:
- T+3: 1995-06-07 through 2017-09-04.
- T+2: 2017-09-05 through 2024-05-27.
- T+1: 2024-05-28 onward.

Official SEC support:
- T+3 became effective June 7, 1995.
- T+2 compliance began September 5, 2017.
- T+1 compliance began May 28, 2024.

For this free Stage-1 test, settlement business days are represented by observed U.S. equity trading sessions. This approximation is frozen before returns and must be disclosed.

**Outcome Y**
1. Mechanism diagnostic: the D-1 close to D close return should show negative pressure relative to matched ordinary days.
2. Primary return outcome: a long position entered at D close and exited at the close three trading sessions later should exhibit positive reversal relative to matched non-month-end windows.

The event is calendar-known before D, so a market-on-close order on D does not require same-close price look-ahead.

## Instrument and source

Primary instrument: SPY.
Sample start: 1995-06-07.
Sample end: latest completed session available when the frozen run starts.
Price field: adjusted close / total-return-consistent adjusted history from Yahoo Finance.
No parameter is selected from realized returns.

## Execution and costs

- Entry: event-date D close using a pre-specified MOC order.
- Exit: close three U.S. equity trading sessions after D.
- Frozen all-in trading drag: 5 bps one-way, 10 bps round trip.
- The same cost is applied to event trades and baseline trades.
- No leverage is used in Stage 1.

## Matched baseline

For every date, compute 20-session realized volatility from daily adjusted-close returns using data through the PRIOR session only.

Compute a point-in-time volatility rank using the prior 252 available sessions only and map it to one of five frozen quintiles:
- Q1: [0,20%)
- Q2: [20,40%)
- Q3: [40,60%)
- Q4: [60,80%)
- Q5: [80,100%]

For event date D, matched ordinary baseline dates must:
1. be in the same settlement regime;
2. have the same weekday as D;
3. have the same point-in-time volatility quintile;
4. not be within +/-5 trading sessions of any month-end settlement-deadline event;
5. have a complete three-session forward holding window.

The matched baseline for an event is the mean net three-session return across all eligible ordinary dates satisfying these conditions.

No return-dependent nearest-neighbor count, volatility threshold, or baseline window may be selected after results.

## Pre-registered primary falsifier

The hypothesis is killed if EITHER condition fails:

1. **Return condition:** the aggregate mean event three-session net return minus the matched ordinary-window net return is less than +20 basis points.

2. **Settlement-timing migration condition:** the reversal must move with the settlement regime. For each settlement era, evaluate three calendar-only counterfactual month-end deadlines: one, two, and three trading sessions before T. The actual regime's deadline must produce the largest matched-baseline-adjusted three-session reversal among those three offsets:
   - T+3 era: 3-session-before-T alignment must rank first.
   - T+2 era: 2-session-before-T alignment must rank first.
   - T+1 era: 1-session-before-T alignment must rank first.

Ties fail the timing condition.

This timing test is frozen before return inspection and is intended to distinguish a settlement mechanism from a generic turn-of-month anomaly.

## Statistical reporting

Report without using inference to retune:
- event count by settlement era;
- mean / median / standard deviation of event net 3-day returns;
- matched-baseline mean;
- mean event excess;
- win rate;
- worst event;
- best event;
- 10,000-draw event-level bootstrap 95% CI for mean excess with seed 20261004;
- one-day pre-deadline pressure diagnostic versus matched one-day baseline;
- the three candidate timing-offset excess returns inside each settlement era;
- calendar-year concentration of excess returns;
- SPY buy-and-hold context only; it is not substituted for the matched-window baseline.

## Train / validation / holdout discipline

Because the mechanism includes a structural settlement-rule change, the eras themselves are not tunable parameters. Stage 1 is a mechanism falsification test, not a parameter search.

To preserve validation discipline:
- TRAIN: T+3 era from 1995-06-07 through 2006-12-31.
- VALIDATION: T+3 era from 2007-01-01 through 2017-09-04.
- STRUCTURAL OUT-OF-SAMPLE ERA 1: full T+2 era, 2017-09-05 through 2024-05-27.
- STRUCTURAL OUT-OF-SAMPLE ERA 2: T+1 era, 2024-05-28 onward.

No rule is modified after TRAIN/VALIDATION or after either later settlement era is read.
The primary +20 bp criterion is reported for train, validation, each later era, and full sample, but the final frozen decision uses the full-sample return condition plus the cross-era migration condition already specified above.

## Decision

- **PROMOTE** only if both frozen primary falsifier conditions pass and data quality is adequate.
- **KILL** if either substantive frozen condition fails.
- **MONITOR** only for an evidence-quality failure that prevents a valid test; MONITOR must not be used to rescue a failed empirical result.

## Prohibited after freeze

Do not change:
- settlement-regime dates;
- D definition;
- 3-session holding period;
- 5 bps one-way cost;
- volatility lookback/rank/quintiles;
- +/-5-session event exclusion;
- weekday/regime/quintile baseline match;
- +20 bp hurdle;
- timing-migration rule;
- instrument;
- sample boundaries opportunistically.

No flow filter, Friday filter, funding filter, or cross-sectional ownership extension is permitted in this Stage-1 run.
