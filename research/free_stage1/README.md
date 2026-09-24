# Free PIT Stage 1 — Bottleneck Winner + Real-Options LEAPS

Status: research branch only. No production merge.

## Why this exists

The discovery phase should not depend on CRSP/Compustat/IvyDB licenses. We can do a strong, genuinely point-in-time first-pass using only data that are legally available for $0 in public or cloud-hosted form, then freeze the specification and replicate it on licensed vendor data later.

## Architecture

### A. Bottleneck Winner — free discovery

Primary free implementation:

1. **SEC Financial Statement Data Sets / XBRL**
   - accounting factor values are taken from the filing as submitted.
   - SEC filing date is the information timestamp.
   - a compact CIK-native formation panel is built before backtesting; every row enforces `information_date <= snapshot_date`.
2. **SEC EDGAR 10-K Item 1 text**
   - construct a CIK-native TF-IDF competitor network from business descriptions.
   - use only the latest 10-K public by the network snapshot date.
3. **QuantConnect US Equities + Security Master / PIT universe**
   - supplies the historical tradable universe, point-in-time CIK identity, prices, corporate-action handling and returns, including delisted names.
   - contemporaneous market cap is used only as a market input for the valuation ratio; accounting denominators remain SEC as-filed.
4. **QuantConnect Symbol.CIK**
   - bridges both SEC datasets directly to point-in-time tradable securities inside LEAN.

This intentionally avoids needing a free CIK↔GVKEY map.

### B. SEC fundamentals are the primary Stage-1 accounting source

For 2009 onward, `sec_fsd_pit.py` builds the filing-level as-filed panel and
`sec_formation_features.py` compresses it into quarterly CIK snapshots. The
Bottleneck algorithm does not use Morningstar financial-statement values for
factor scoring. QuantConnect is the market/universe engine.

Quarterly SEC/network snapshots use prior-quarter-end cutoffs and are consumed
at the next Jan/Apr/Jul/Oct rebalance. This avoids treating a same-day SEC filing
as if it were known before the market opened.

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
- select contracts from the completed daily chain and execute with next-session Market-On-Open orders, preventing same-close selection/fill leakage

Frozen LEAPS grid:
- underlying: SPY and QQQ first
- target call delta: 0.70, 0.80, 0.90
- target DTE: 365, 548, 730
- premium allocation: 25%, 50%, 100%
- roll every 6 months or when DTE < 180
- minimum open interest: 100 by default
- select only when a valid quote is present; queue the trade for the next regular-session open; LEAN's LatestPriceFillModel uses the next QuoteBar ask open for buys and bid open for sells, with the frozen slippage sensitivity layered on top
- benchmarks: underlying buy-and-hold, S9/35-0 research outputs, and later futures

Stage 2 replicates the frozen design on OptionMetrics IvyDB US (1996+) without changing parameters.

## Bottleneck hypothesis

The score must be computable using information known at the formation date.

Candidate components:

- **Demand signal (`demand_accel` legacy field name):** SEC as-filed YoY revenue growth
- **Pricing power:** SEC YoY gross-margin improvement, with operating-margin improvement as fallback
- **Quality:** SEC FCF margin minus leverage, with operating margin minus leverage as fallback
- **Supply scarcity / competitive scarcity:** CIK-native Item 1 text network concentration and peer count
- **Momentum:** 252-trading-day price momentum and 52-week-high proximity
- **Dilution:** negative SEC YoY share-count growth
- **Valuation sanity:** contemporaneous market cap / annualized SEC as-filed revenue; 10-K qtrs=4 revenue is used directly and 10-Q qtrs=1 revenue is multiplied by 4; other flow-period lengths are rejected

Frozen model families:
1. momentum
2. quality + momentum
3. bottleneck core
4. bottleneck + momentum
5. full bottleneck composite

Portfolio sizes: top 10 / 20 / 40.
Rebalance: quarterly.
Price-history state is primed with a 252-daily-bar warm-up and updated before applying each day's eligibility screen, so 12-month momentum is based on trading observations rather than "eligible-only" days.
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
