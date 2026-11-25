# Sources, conventions and uncertainty

Reviewed 2026-11-25. Raw market data/PDFs remain local under provider terms.
Hashes identify the bytes used, not a grant of redistribution rights. A current
download is not an authentic earlier publication vintage. No missing daily
returns, distributions, swap rates or historical tax years were fabricated.

## Preserved input catalogs

| Catalog / SHA-256 | Coverage and use | Limits |
|---|---|---|
| Macro `1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376` | `../macro-learning/audit-output/macro-sources-v4/catalog.json`; 52 captured sources, 17 separately blocked series; French daily broad/factor/industry returns from July 1926; inflation, short rates, production, credit and vintage observations | Current revised research data do not establish historical release-time availability. Missing CPI is not filled. Complete bond total-return/correlation forecasting is not available. |
| Official NAV/index `2be48a3a975ad9aa8f290b5c3d7f29ce832f436500878913a8ec8bf65ebc61e1` | `../tail-validation/audit-output/official-index-nav-v4/catalog.json`; 86 issuer NAV/cash and Nasdaq-origin snapshots | Authentic XNDX observations begin 1999-03-04 here. NAV is not an exchange fill; cash/split conventions require reconciliation. |
| Quote/DFF manifest `8a434afe840e8b5f22a0cd1d336cb81b33b672a1a534fb04b70f0900f558bb6c` | Original `audit-output/public-daily-inputs-network/manifest.json`; daily fund data and observed effective federal funds rates | Vendor restrictions; adjusted ETF returns are not exact index returns. DFF does not identify historical swap spreads. |

The complete prior catalogs retain each provider, URL, genuine observation
bounds, frequency, currency, income convention, publication vintage, capture
time, byte hash and rights notes:
[macro catalog](../macro-learning-2026-11-24/evidence/source-catalog.json),
[official catalog](../tail-validation-2026-11-24/evidence/official-source-catalog.json).
If a prior link is unavailable in a checkout, the retained local catalog and
its exact hash above are authoritative; do not substitute newly revised data.
The baseline study actually trains on the 480-month calendar window
1986-08–2026-07 after release-lag eligibility. Incomplete transitions force
resampling restarts. It does not claim that 1930 data trained this specific
future reference or that the modern Nasdaq existed then.

Primary source entry points include [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
[FRED](https://fred.stlouisfed.org/), [ALFRED](https://alfred.stlouisfed.org/),
[BLS CPI](https://www.bls.gov/cpi/), issuer NAV/cash APIs and the Nasdaq-origin
series identified in the immutable catalogs. They are not interchangeable total-
return or publication conventions. Licensing must be checked per dataset.

## Additional issuer/index review

[Machine-readable source review](evidence/source-review.json) records captured
bytes, timestamps and actual retrieval failures. Current filenames/URLs are
mutable; comparisons must use archived hashes.

| Captured source | SHA-256 | Interpretation |
|---|---|---|
| [Current ProShares statutory prospectus](https://www.proshares.com/globalassets/proshares/prospectuses/statutory_prospectus.pdf) | `c04f079013602d09bd7deaa282b021ef852f7f60b6790925e8706a38c57780c2` | Local byte identity independently re-fetched; general current documentation, not a historical cost invoice |
| [TQQQ issuer summary](https://prod.proshares.com/globalassets/proshares/prospectuses/tqqq_summary_prospectus.pdf) | `b8037fff8b81ad66a74be624378f5613b4e8a1c1e8e389540a8acee466336875` | Actual captured header September 28, 2026; management 0.75%, other 0.19%, gross 0.94%, net 0.78%; contractual cap 0.95% through September 30, 2027; financing/transaction costs are separately relevant |
| [S&P 500 price index through FRED](https://fred.stlouisfed.org/series/SP500) | `494ebff6737d8ba4b14e103400788b97b9e4bcf4d2b2b33e53e748334eb1fe3e` | 2,287 genuine daily closes, 2016-11-25–2025-12-31, USD index points; excludes dividends; not used as a TR replacement |

The [SEC-filed September 27, 2024 summary, supplemented April 7, 2025](https://www.sec.gov/Archives/edgar/data/1174610/000168386325003345/f41532d1.htm)
reports 0.75% management, 0.22% other, 0.97% gross and 0.84% net expenses;
the waiver ran through September 30, 2025. It was read through the web tool;
direct fetch returned 403, so no raw SEC-byte hash is claimed. The 0.95%
contractual cap is not the actual reported 0.84% net ratio. Neither current nor
single-year expense tables prove a complete daily historical fee schedule.

Generic [2019 SAI swap disclosures](https://www.sec.gov/Archives/edgar/data/1174610/000119312519179886/d703387d485apos.htm)
and [2026 SAI disclosures](https://www.sec.gov/Archives/edgar/data/1174610/000117461026000543/f45920d1.htm)
describe reference-asset returns, dividends and financing on contractual notional.
They support investigating gross financing and collateral separately, but do
not identify each historical contract's actual daily exposure, spread or cash
yield. The previously retained [May 31, 2026 financial statement](https://www.proshares.com/globalassets/proshares/documents/annual-reports/annual_proshares_geared_financial_statement.pdf)
has SHA `023caf84103b52366cbe67f5c7d76d1c0aa73dd26a9e88c85c832c251301fae9`;
the prior inspection of printed page 159 supports the same notional distinction.
PDF inspection used `pypdf==6.4.0` in `/tmp`, not a new runtime dependency.

FRED attributes SP500 to S&P Dow Jones Indices LLC, explicitly identifies it as
price-only and limits available daily history to ten years. Its copyright notes
require permission for reproduction. Raw closes are not published in Git.
The direct S&P export returned **403**; Yahoo's attempted `^SP500TR` endpoint
returned **429**; the large SEC annual filing direct download returned **403**.
These are unresolved access results, not a claim the observations do not exist.

[S&P's index-search FAQ](https://www.spglobal.com/spdji/zh/documents/IndexSearchFAQ.pdf)
describes website price/TR exports and subscription history. The
[S&P Marketplace dataset page](https://www.marketplace.spglobal.com/en/datasets/s-p-dow-jones-indices-via-xpressfeed-%28116%29)
describes complimentary daily index access, but provisioning/rights/actual
historical coverage were not obtained here. The cheapest legitimate next step
is to request that provider access or an issuer-supplied reference export;
no anonymous export, price, free historical entitlement or paid purchase is
asserted. A SPY total-return proxy cannot close the exact-index provenance gate.

## Numerical and statistical authorities

[NIST order-statistic definitions](https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm)
and [NIST binomial proportion methods](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
provide the statistical basis. Quantile confidence ranks are computed by exact
binomial inversion, not an asymptotic Gaussian wealth approximation. Independent
test code explicitly sums binomial probabilities to verify coverage. Exact
Clopper–Pearson intervals retain positive upper uncertainty for zero sampled
events. Bonferroni allocation covers every registered quantity and planned look;
it does not require those quantities/looks to be independent. The simulations
themselves must be independent across path draws conditional on the fixed model.

Stationary-bootstrap and empirical-daily-bridge methods are unchanged from the
prior methods report. No new tail distribution, forecast candidate, fitted
regime duration or strategy was chosen using consumed evaluation periods.
Sensitivity cases vary disclosed assumptions and have no posterior weights.

## Account tax authority and restrictions

The new endpoint accelerator delegates to existing `PortfolioLedger` and
`annual_tax`; no tax law was added or extrapolated. It explicitly freezes the
supported 2026 profile. [Rev. Proc. 2025-32](https://www.irs.gov/pub/irs-drop/rp-25-32.pdf)
is the existing dated federal parameter authority; [IRS Publication 550](https://www.irs.gov/publications/p550)
and [NIIT guidance](https://www.irs.gov/newsroom/questions-and-answers-on-the-net-investment-income-tax)
support the existing netting/preferential/NIIT scope. See the full
[tax authority and limits](../integration-2026-11-24/TAX_AND_DATA.md) and
[original rule tests](../accuracy-audit/TAX_RULES.md).

Frozen 2026 law is a hypothetical future-policy assumption, not enacted law for
2031–2066. A Roth existing-balance internal sale is not a contribution or owner
withdrawal. The new accelerator rejects MA, cash flows, distributions, changed
income and unknown law; it cannot make unsupported full household returns valid.
Existing dated 2024–26 canonical cases remain bounded by their prior source gates.

## Required external evidence still missing

1. Complete official target total-return histories and historical constituent/
   dividend conventions, not another ETF's net return series.
2. Daily historical derivative notionals, financing spreads, collateral yields,
   expense changes and waivers for independently reconciled fund NAV attribution.
3. Genuine future released outcomes for the new frozen reference forecasts.
4. Full private off-host source/ensemble backup with restore verification.
5. Dated contract settlements, rolls and broker margin evidence for futures;
   contemporaneous executable option chains, assignments and deliverables for
   options/LEAPS; complete applicable distribution and advanced tax records.

The existing provider/coverage discussion in the prior tax/data report includes
Cboe DataShop and other legitimate derivatives sources. No paid quote was obtained
and no exact cheapest dataset price is claimed. Those strategies remain blocked.
