# Free PIT Stage 1 — Bottleneck Winner + Real-Options LEAPS

Status: research branch only. No production merge.

## Why this exists

The discovery phase should not depend on CRSP/Compustat/IvyDB licenses. We can do a strong, genuinely point-in-time first-pass using only data that are legally available for $0 in public or cloud-hosted form, then freeze the specification and replicate it on licensed vendor data later.

## Architecture

### A. Bottleneck Winner — free discovery

Primary free implementation:

1. **QuantConnect US Equities + Security Master**
   - survivorship-aware/security-master handling for ticker changes, splits, dividends, mergers and delistings.
   - historical US equity data available in QuantConnect Cloud.
2. **QuantConnect Morningstar US Fundamentals**
   - historical corporate fundamentals in cloud.
   - use only values available at the algorithm date.
   - originally reported observations; no future restatement substitution.
3. **SEC EDGAR 10-K Item 1 text**
   - use filing date as the information timestamp.
   - construct a CIK-native text-similarity competitor network from Item 1 business descriptions.
4. **QuantConnect Symbol.CIK**
   - bridges the CIK-native SEC network directly to point-in-time tradable securities inside LEAN.

This intentionally avoids needing a free CIK↔GVKEY map.

### B. Independent SEC fundamentals replica

For 2009 onward, build a second free fundamentals panel from SEC Financial Statement Data Sets / XBRL:
- as-filed numeric facts
- CIK
- filing date
- fiscal period
- no observation may enter a portfolio before filing date

This is an independent audit of the QuantConnect/Morningstar version, not a replacement for it.

### C. Published TNIC/ETNIC role

Hoberg-Phillips TNIC/ETNIC remains a **methodology and replication benchmark**. Public releases are keyed by GVKEY and are explicitly not lagged. Do not force-match them to SEC using current tickers.

Licensed Stage 2 can use the WRDS historical CIK↔GVKEY/CCM link and run the frozen strategy on official TNIC/ETNIC + CRSP + Compustat PIT.

### D. LEAPS — free discovery

Use QuantConnect/AlgoSeek US Equity Options:
- OPRA-derived historical option quotes
- daily history back to 2012
- daily option-universe Greeks/IV
- open interest
- quote bars so buys/sells incorporate bid/ask

Frozen LEAPS grid:
- underlying: SPY and QQQ first
- target call delta: 0.70, 0.80, 0.90
- target DTE: 365, 548, 730
- premium allocation: 25%, 50%, 100%
- roll every 6 months or when DTE < 180
- minimum open interest: 100 by default
- enter with market order only when a valid quote is present; LEAN's option fill model uses ask-side quote data for buys and bid-side for sells when QuoteBars are available
- benchmarks: underlying buy-and-hold, S9/35-0 research outputs, and later futures

Stage 2 replicates the frozen design on OptionMetrics IvyDB US (1996+) without changing parameters.

## Bottleneck hypothesis

The score must be computable using information known at the formation date.

Candidate components:

- **Demand acceleration:** revenue growth and acceleration
- **Pricing power:** gross/operating margin level and improvement
- **Quality:** operating cash flow / free cash flow, profitability, leverage
- **Supply scarcity / competitive scarcity:** CIK-native Item 1 text network concentration and peer count
- **Momentum:** 6–12 month price momentum and 52-week-high proximity
- **Dilution:** share-count growth penalty
- **Valuation sanity:** enterprise value / sales or earnings/FCF where available

Frozen model families:
1. momentum
2. quality + momentum
3. bottleneck core
4. bottleneck + momentum
5. full bottleneck composite

Portfolio sizes: top 10 / 20 / 40.
Rebalance: quarterly.
Transaction costs: pre-specified, not tuned after holdout results.

## Validation chronology

Free discovery sample:
- build inputs: 2009 onward (SEC XBRL starts in 2009; QuantConnect history is longer)
- training: 2009–2014
- validation: 2015–2019
- untouched free-data holdout: 2020–2023 for TNIC-comparable period, and through latest available date for CIK-native SEC network

The exact dates may be shortened only because a required free dataset is unavailable, never to improve performance.

Then freeze code + hashes before licensed replication.

Licensed replication:
- CRSP survivor-bias-free returns + delistings
- Compustat PIT/Snapshot
- official historical CCM / CIK-GVKEY links
- TNIC/ETNIC
- IvyDB US
- no parameter changes after seeing vendor-data replication

## Evidence labels

Every output must be tagged as one of:
- `FREE_DISCOVERY`
- `FREE_HOLDOUT`
- `LICENSED_REPLICATION`
- `SYNTHETIC_STRESS`

Never call a free discovery result a CRSP/Compustat result or an OptionMetrics result.

## Current hard boundaries

- Public TNIC/ETNIC is GVKEY-based; direct SEC linkage requires a historical CIK↔GVKEY bridge. The free discovery path avoids this by recreating the product-text network in CIK space.
- SEC face-statement datasets are as filed but only cover numeric primary-statement facts; they are not a full Compustat replacement.
- QuantConnect cloud data are excellent for discovery, but the vendor replication remains important because data construction and corporate-action conventions differ.
- QuantConnect options begin broadly in 2012; IvyDB remains valuable for 1996–2011, including the dot-com and GFC windows.
