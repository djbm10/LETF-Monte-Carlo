# Authorities, provenance and unresolved access

Reviewed November 24 US Eastern / November 25 UTC, 2026. The immutable
[original source catalog](../macro-learning-2026-11-24/evidence/source-catalog.json)
and [new catalog](evidence/official-source-catalog.json) retain exact URLs, source
bounds, revisions, fees/return conventions, currency, capture times and hashes.
Current downloads are not historical publication vintages. Public access is not
a license to redistribute raw quotes/index data. Raw snapshots remain local.

## New fund/index evidence

ProShares links its NAV histories from the actual product pages, for example
[TQQQ](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq) and
[UPRO](https://www.proshares.com/our-etfs/leveraged-and-inverse/upro). The linked
CSV convention is current split-restated NAV, not dividend-adjusted/executable
exchange prices. Annual issuer API cash histories are archived for 2006–2025.
Final 2024/25 tax classes continue to use dated ICI files rather than inferring
qualified status from the cash API. Numerical source corrections do not grant
historical fee/swap certification.

[FRED Nasdaq XNDX](https://fred.stlouisfed.org/series/NASDAQXNDX) identifies Nasdaq
as originator and the series as a daily total-return index. Its earliest genuine
observation here is 1999-03-04. [NASDAQ100](https://fred.stlouisfed.org/series/NASDAQ100)
is a separate price-return series. Missing non-session FRED rows are excluded
before returns, with exact issuer-session coverage checks; no observed return is
filled. Nasdaq copyright/pre-approval restrictions are retained. These are
current revised index histories, not decision-time archives.

No exact S&P total-return history was obtained. [FRED SP500](https://fred.stlouisfed.org/series/SP500)
is explicitly price return; it cannot stand in for total return. S&P's
[index search FAQ](https://www.spglobal.com/spdji/zh/documents/IndexSearchFAQ.pdf)
describes limits on website history; an appropriately licensed history is still
needed. SPY remains a labeled proxy in the UPRO/SSO diagnostic.

[ProShares Trust financial statements, May 31, 2026](https://www.proshares.com/globalassets/proshares/documents/annual-reports/annual_proshares_geared_financial_statement.pdf)
show financing on contractual swap notional, excluding commissions (TQQQ printed
p159/PDF p162), alongside collateral/counterparty disclosures. The captured PDF
has a [recorded checksum](evidence/financing-authority.json). This supports
modeling gross financing and collateral income separately. It does not establish
the daily 2010–2025 notional, fee waiver or collateral-income schedule. A generic
net borrowing spread and a swap spread are different model inputs.

## Tax rules implemented in this continuation

| Rule | Dated authoritative source | Independent automated evidence |
|---|---|---|
| Common-share QDI: >60 days within ex-date ±60 days; acquisition excluded, disposition included | [IRS Publication 550 (2025)](https://www.irs.gov/publications/p550), [IRC 1(h)(11)](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section1) | `tests/test_dividend_holding.py`: exact 58/61-day sale counts, partial sales, no duplicated cash |
| Wash tacks do not qualify dividends | [IRC 246(c)(3)(B)](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section246), excluding IRC1223(3) through IRC1(h)(11) | Same fixture gives 365 tacked days and still rejects the early sale |
| January-paid RIC distribution with Q4 declaration/record can be deemed received Dec31 | [IRS 1099-DIV instructions (01/2024)](https://www.irs.gov/instructions/i1099div), [IRC 852(b)(7)](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section852) | `tests/test_year_end_distributions.py`: noncash recognition, January cash once, terminal receivable, actual investor QDI character |
| SPY specific issuer tax year and record/pay amounts | [State Street final primary ICI workbook, captured 2025 edition](https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/SPDR_ICI_Tax_Summary_Primary.xlsx) | Four rows checked against source/vendor amounts; December record and January pay plus final tax-year column M allocation |

The RIC importer requires final issuer tax-year allocation and Q4 record/ex-date
evidence. It does not backdate all January payments or invent a fund election.
The final issuer allocation is relied upon for its tax-year determination; a
separate declaration notice is not independently captured. Unknown mappings,
February payments and incompatible record dates fail explicitly. A receivable
at the endpoint is an asset, not spendable cash or a broker guarantee.

The QDI tracker is an explicit **unhedged common-share** scope. It does not
silently count preferred-share or hedged/risk-reduced days. Forward Section199A
qualification is still gated. Noncash tax reclassification preserves the original
receipt year and cannot amend a tax year already paid through this runner.

Inherited federal/MA/Roth/settlement authorities and limitations remain in
[the foundation tax/data report](../integration-2026-11-24/TAX_AND_DATA.md) and
[the previous source report](../macro-learning-2026-11-24/SOURCES_AND_RESTRICTIONS.md).
Federal/MA supported parameter years are 2024–26, not 1930–present. MA fund
government-obligation exemptions, full household tax returns/AMT/credits,
estimated-tax penalties, UBTI/prohibited transactions and complex derivative
tax interactions remain incomplete or blocked. No California default is added.

## Statistical sources and limits

[Politis/Romano, The Stationary Bootstrap](https://statistics.stanford.edu/technical-reports/stationary-bootstrap)
provides the geometric-block method. A fixed mean of twelve months is registered
here, not optimized on outcomes. A conditional restart is a candidate assumption,
not a theorem that historical crash distributions forecast new monetary regimes.
[NY Fed Staff Report 850](https://www.newyorkfed.org/research/staff_reports/sr850)
supports investigating time-varying distribution risk, not claiming causal
inflation/interest effects from these small retrospective panels.

All variables resample jointly from source months; daily templates preserve
within-month shape and sampled variance. Circular block boundaries, finite
history and synthetic session counts remain model uncertainty. The original
26 forecasts and corrected new comparisons do not become untouched because code
or score units change. Monthly industrial production is not GDP; yields are not
bond returns; ex-post recession classifications are not contemporaneous signals.

## Cheapest legitimate next data routes investigated

No account was provisioned and no purchase made. Prices below are published
retail indications at review time, not quotes or assurance of entitlement.

| Route | Evidence and use | Limitation |
|---|---|---|
| Existing free issuer NAV/cash and FRED Nasdaq | Captured locally in this release | Does not supply dated swap books or exact historical fees/waivers |
| [ThetaData pricing](https://www.thetadata.net/pricing) | Value $40/month lists six years and limited request types; Standard $80/month lists ten years and OPRA NBBO; Pro $160/month lists fourteen years | Request types/actual symbol coverage must be confirmed before purchase. Existing provisioned-free routes from prior research still require user setup. Personal license forbids redistribution/business use. |
| [Massive options pricing](https://massive.com/pricing?frequency=Monthly&product=options) | Basic $0/two years EOD/aggregates; Starter $29; Developer $79/four years; Advanced $199/five-plus years lists quotes | Cheap EOD/aggregate plans do **not** prove historical bid/ask execution. Signup/key needed; no credentials obtained. |
| [Cboe quote intervals](https://datashop.cboe.com/option-quote-intervals) | Historical OPRA NBBO, sizes, strikes/expiries; targeted symbol/time samples before bulk acquisition | Order-specific pricing not verified; timestamp/early-close/adjustment rules still need validation. Futures options excluded. |
| [CME settlement reference](https://www.cmegroup.com/trading/about-settlements.html), [DataMine](https://www.cmegroup.com/market-data/real-time-and-historical-data.html) | Dated contract settlement history, time/sales or depth depending product | Historical entitlements/prices not verified. Settlements alone do not establish intraday margin breaches, executable rolls or old house margins. |
| Genuine ALFRED releases, actual bond TR, licensed S&P history | Needed for prospective/PIT macro, bonds and exact S&P reconstruction | Revised French history cannot certify earlier decision-time availability; Treasury yields cannot replace coupon total returns. |

The lowest-cost published route found for real historical NBBO is a targeted
provider sample or ThetaData Standard's listed $80/month plan, subject to actual
entitlements and license review. This is not a verified cheapest possible quote.
Before spending anything, use free samples to validate the existing quote and
contract schema. Short assignment/collars/straddles still require engineering,
even with perfect data. Full cross-strategy investment ranks remain blocked.
