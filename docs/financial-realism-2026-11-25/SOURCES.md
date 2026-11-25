# Source and method evidence

Reviewed 2026-11-25. Existing source catalogs and licensing restrictions remain
unchanged. Current revised data are not historical release-time vintages.
The new stress marks are assumptions, not fabricated historical observations.

## Exact prior inputs

| Input | SHA-256 / convention |
|---|---|
| Macro catalog, 52 snapshots | `1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`; French research returns, inflation/rates/production/credit, current/vintage limits preserved |
| Official NAV/index catalog | `2be48a3a975ad9aa8f290b5c3d7f29ce832f436500878913a8ec8bf65ebc61e1`; issuer NAV/cash, Nasdaq-origin price/TR data; not an executable fill tape |
| Quote/DFF manifest | `8a434afe840e8b5f22a0cd1d336cb81b33b672a1a534fb04b70f0900f558bb6c`; prior observed daily prices, actions and funding reference |
| Reference model | `3263d018a49123a2b7659bed42c99e76ce24c914e47f9ba794a73120b8e131e7`; 480-month window ending July 2026, training-only empirical daily templates |

The [previous source catalog discussion](../precision-reliability-2026-11-24/SOURCES.md)
retains each provider, source URL, date bounds, vintage, currency, fees/return
convention and licensing notes. Authentic broad/factor research data support
older counterfactuals; no literal 1930 Nasdaq/TQQQ or Roth IRA is introduced.
The unchanged prior historical counterfactual report remains separate from this
future-calendar stress exercise. No missing monthly/daily path is filled.

## Newly inspected issuer publication

Primary source: [ProShares May 31, 2021 annual report](https://www.proshares.com/globalassets/proshares/documents/summary-prospectus/proshares_summaryprospect_agq.pdf).
Despite its URL filename, the actual 412-page downloaded document is the
2021 geared-funds annual report. Capture SHA-256:
`89149fc0bd2121f0093632815f84616c17e3f4ed3f98b84d7fa4084f3f07b915`.
Raw PDF/text stay local under issuer copyright. Parsing used the existing
isolated pypdf6.4.0 installation, not a new runtime dependency.

Zero-based PDF pages 55, 214 and 215 support the transcribed
[facts](evidence/issuer-2021-facts.json): NAV reinvestment convention, published
returns, dated expenses, stock/futures/swap exposures and nine financing rates.
The statement date is 2021-05-31; typesetting is July 27, 2021. Earliest public
release is unverified. Nothing from this later publication is inserted into
earlier trading signals or the pre-2020 reconstruction fit.

The [reconciliation](evidence/issuer-reconciliation.json) reports all publication
rounding checks, including both index failures, and an independent share-ledger
reinvestment calculation. Reporting-date gross swap financing is not a year-
average expense, contractual spread over DFF or a daily collateral yield.
Current expense tables and a reporting-date portfolio cannot establish an
unobserved historical daily schedule. The old TQQQ CAGR error and evaluation
thresholds are unchanged; already inspected 2020–25 observations are not a new
holdout. Separate issuer publication/API agreement is not a third-party opinion
on all fund mechanics or future returns.

## Legal and execution boundaries

[IRS Publication 550](https://www.irs.gov/publications/p550) and the
[IRS worthless-securities guidance](https://www.irs.gov/faqs/capital-gains-losses-and-sale-of-home/losses-homes-stocks-other-property/losses-homes-stocks-other-property-1)
support blocking a same-day taxable sale inferred only from zero NAV. The deemed
year-end disposition/holding-period issues are not implemented here. See
[LEGAL_SCOPE.md](LEGAL_SCOPE.md). Existing positive-price capital gains, carryovers,
wash/QDI rules and ordinary income use the existing sourced tax engine.

Actual 2024/2025 federal parameters are sourced in the prior
[tax authority table](../integration-2026-11-24/TAX_AND_DATA.md).
[IRS contribution limits](https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-ira-contribution-limits)
and [Publication 590-A](https://www.irs.gov/publications/p590a) support the bounded
contribution examples. Future runs explicitly freeze 2026 eligibility/law;
they are not predictions of future legislation or investor MAGI. Roth owner
history is supplied separately from tax-free internal investment activity.

[NYSE circuit-breaker procedures](https://www.nyse.com/publicdocs/nyse/NYSE_MWCB_FAQ.pdf)
require distinguishing severe structural-break marks from ordinarily executable
broad-market intraday closes. No intraday quote, halt/resumption sequence or
broker margin record is invented. The synthetic fund has a declared terminal
corporate action and cannot reopen/recover. Zero or full unpaid-distribution
recovery are explicit alternative assumptions, with no assigned probability.
Taxable bad-debt treatment is blocked.

Observed TQQQ funded replay uses the previously verified 2024–25 bundle. Its
manifest/hash is embedded in [the comparison](evidence/observed-funded-comparison.json).
Inputs are observed; fills/spreads, household facts, no external wash replacements
and investment policy are modeled. Generic account profiles are not verified
broker-specific promises. Fund costs embedded in observations are never charged
twice. MA fund adjustments, advanced taxes and historical derivatives remain gated.

## Uncertainty interpretation and missing evidence

The exact order-statistic/binomial sampling methods and family/stopping-time
allocations remain as previously registered. They do not quantify data, parameter,
regime, market-structure or future-law error. Scenario ranges are unweighted
illustrations and cannot be interpreted as 10th/90th percentiles or causal effects.
Baseline/reference conditional probabilities remain unapproved research values.

Highest-value missing evidence: exact vintage target-index conventions; dated
daily exposures, funding indices/spreads, collateral and expense/waiver records;
verified broker cash yields/permissions; complete insolvency/distribution tax
records; genuine future outcomes; and approved private off-host storage. Issuer/
exchange exports and public filings are legitimate first options. No paid quote,
subscription entitlement, cheapest guaranteed price or acquired dataset is claimed.
