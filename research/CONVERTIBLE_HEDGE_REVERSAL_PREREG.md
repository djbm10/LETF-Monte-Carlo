# Convertible Hedge-Pressure Reversal — Frozen Discovery Preregistration

Frozen before any discovery returns for this hypothesis are inspected.

## Mechanism
Convertible issuance causes convert-arbitrage buyers/dealers to acquire convertible exposure and hedge part of its equity delta by shorting the issuer's common stock. That hedge flow can create non-fundamental selling pressure around pricing. If the initial hedge flow is temporary, the common stock should partially reverse after pricing. A simultaneous issuer share repurchase can offset some of the hedge-driven selling.

## Participant / constraint
- Participants: convertible-arbitrage funds, dealers, underwriters.
- Constraint/incentive: hedge equity delta of newly acquired convertible exposure rather than express a discretionary bearish stock view.

## Discovery universe
Historical S&P 500 issuer proxy only, using point-in-time membership keyed by CIK and the existing survivor-aware S&P price panel. This restriction is frozen before returns and is used to avoid a current-survivor ticker universe.

## SEC event definition
Discovery searches SEC EDGAR full text only from 2009-01-01 through 2019-12-31.

Main convertible event:
1. root filing is 8-K or 424B5;
2. matched filing/exhibit text contains a priced/offered convertible-note transaction;
3. text provides parseable aggregate principal amount;
4. text provides parseable initial conversion rate per $1,000 principal or initial conversion price;
5. issuer is an active S&P 500 member by CIK on the filing date;
6. events by the same issuer within 7 calendar days are clustered and only the earliest public pricing record is kept.

Repurchase-offset convertible:
same as above, but the same pricing filing/exhibit explicitly describes a concurrent issuer common-share repurchase/buyback connected to the offering.

No-offset convertible:
convertible event without such concurrent repurchase language.

Straight-debt baseline event:
S&P 500 issuer 8-K/424B5 pricing record for senior notes with parseable principal amount, excluding any filing/exhibit containing convertible/exchangeable-note language.

## Ex-ante hedge-pressure score
For each no-offset convertible:
- conversion-equivalent shares = principal_amount / conversion_price, where conversion_price is taken directly or computed as 1000 / conversion_rate;
- ADV20 shares = mean issuer trading volume over the 20 completed trading sessions ending before the public SEC filing date;
- hedge_pressure_score = conversion_equivalent_shares / ADV20.

This is an ex-ante share-equivalent pressure proxy, not a claim that actual hedge delta equals 1. It incorporates deal size and conversion economics without using subsequent returns.

Train tercile cutoffs for hedge-pressure_score are frozen from TRAIN only and then applied unchanged to VALIDATION.

## Public-information / execution timing
Filing date F is the first public SEC pricing record in the event cluster.
Entry for the reversal trade is the close of the first trading session strictly after F.
All signals/terms and ADV inputs therefore precede entry.
No same-close filing assumption is allowed.

## Outcomes
1. Pricing pressure reaction: issuer return from the close immediately before F through reversal-entry close, minus SPY over the same dates.
2. Reversal trade: long issuer / short SPY, equal notional, from reversal-entry close through 5, 10, or 20 trading sessions later.
3. Deduct 15 bps round-trip implementation cost from each reversal pair return.

Frozen holding-period neighborhood = {5, 10, 20}. No other horizon may be added after discovery is viewed.

## Baselines
Primary: straight-debt pricing events, matched to each convertible by same calendar year and nearest pre-event log(ADV20); if no same-year candidate exists, nearest candidate within +/-1 calendar year.
Secondary: repurchase-offset convertibles, if at least 8 events are available in a partition.
Matched-control outcomes use the same timing and market-adjusted return definitions.

## Sample split
TRAIN: 2009-01-01 through 2014-12-31.
VALIDATION: 2015-01-01 through 2019-12-31.
UNTOUCHED HOLDOUT: 2020-01-01 onward. No 2020+ event or return is fetched during discovery.

## Frozen mechanism falsifier
Mechanism support requires all of:
1. mean no-offset convertible pricing reaction is more negative than its matched straight-debt reaction in TRAIN;
2. the same reaction differential is negative in VALIDATION;
3. high-pressure convertibles have a more negative mean pricing reaction than low-pressure convertibles in both TRAIN and VALIDATION, using TRAIN tercile cutoffs.

Repurchase-offset comparison is reported but is not mandatory if sample <8.

If these conditions fail, the proposed hedge-pressure mechanism is falsified for Stage 1.

## Frozen return advancement rule
The family advances to holdout only if all are true:
1. data-quality gate passes;
2. mechanism falsifier does not fire;
3. no-offset convertible mean net market-neutral reversal return is positive in VALIDATION for at least 2 of 3 horizons;
4. median VALIDATION mean reversal return across {5,10,20} is positive;
5. median TRAIN mean reversal return across {5,10,20} is positive;
6. for at least 2 of 3 horizons, high-pressure VALIDATION returns exceed low-pressure VALIDATION returns.

Data-quality gate:
- >=40 price-resolved no-offset convert events overall;
- >=15 no-offset convert events in VALIDATION;
- >=30 price-resolved straight-debt control events overall;
- >=80% of SEC events that pass term parsing + PIT S&P membership must resolve required pre/post price data;
- >=10 events in each TRAIN pressure tercile.

Bootstrap/t-tests are reported but cannot change parameters.

## Holdout
If discovery advances:
- write an immutable frozen holdout specification;
- then run exactly one 2020+ holdout over all three frozen horizons;
- do not change parsing rules, event clustering, universe, costs, pressure score, tercile cutoffs, matching, or execution timing after seeing holdout results.

If discovery fails, classify KILL or MONITOR and keep holdout sealed.
