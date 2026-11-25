# Sources, legal scope and restrictions

Reviewed November 24 US Eastern / November 25 UTC, 2026. The
[machine catalog](evidence/source-catalog.json) records providers, exact URLs,
capture timestamps, requested/confirmed vintages, frequency, genuine bounds,
units, currency/denomination, hashes and failed requests. Raw snapshots stay local;
public download access is not a redistribution license.

## Historical inputs

| Source | Coverage/convention | Use / boundary |
|---|---|---|
| [French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): daily factors, six size/value portfolios, ten industries | Daily July 1926–September 2026, USD research total returns | Broad/growth/HiTec proxies, not executable ETF quotes |
| French monthly factors / RF | Genuine monthly returns; prior-month annualized RF | Funding proxy, not a daily path or swap invoice |
| [BLS CPI via FRED](https://fred.stlouisfed.org/series/CPIAUCNS) | Monthly NSA level from 1913; current/available vintage snapshots | Purchasing power and macro input; missing values retained |
| [Industrial production](https://alfred.stlouisfed.org/series?seid=INDPRO) | Current history and genuine vintages including 1929 | Log growth, not GDP; differing vintage reference bases are not concatenated |
| [Moody's Baa](https://fred.stlouisfed.org/series/BAA)/[Aaa](https://fred.stlouisfed.org/series/AAA) | Monthly annual yields from 1919, vintage gaps explicit | Credit spread, not bond total returns; copyright restrictions |
| [GS10](https://fred.stlouisfed.org/series/GS10), [DFF](https://fred.stlouisfed.org/series/DFF), [DTB3](https://fred.stlouisfed.org/series/DTB3) | Dated yields/rates, actual bounds in catalog | Par yield / funding reference / discount yield, not return tapes |
| [UNRATE](https://fred.stlouisfed.org/series/UNRATE), [GDPC1](https://fred.stlouisfed.org/series/GDPC1), [USREC](https://fred.stlouisfed.org/series/USREC) | Monthly unemployment, quarterly real GDP, ex-post recessions | Collected; GDP not interpolated; recession labels forbidden as contemporaneous predictors |
| [ALFRED](https://alfred.stlouisfed.org/help/downloaddata) | 38 confirmed vintage snapshots | Header and latest-date checks; missing vintages block certification |
| LBMA/FRED gold endpoint | Failed request | BLOCKED, no invented drift or daily path |

52 captures succeeded; 17 requests are blocked: gold and 16 unavailable macro
vintages. The normalized metadata edition corrects noncurrency units and source
bounds without changing observation bytes; parent hashes preserve earlier editions.
French CIZ replaced FIZ in January 2025, with retrospective revisions and changed
monthly dividend reinvestment conventions. RF provider changes in June 2024 from
Ibbotson to ICE BofA. Archived macro data do not make revised equity history PIT.
The genuine 1929 INDPRO snapshot has its old 1923–1925 base and ends November
1929, consistent with the provider's vintage history. No modern base is relabeled.

No credentials, subscriptions or purchases were used. Moody's, ICE/CRSP and vendor
rights remain restrictions; raw data are not committed. Fresh public downloads
can differ from retained hashes. CPI October 2025 remains missing, as documented
by [BLS](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).

## Fund prices, actions and final tax records

The original November 23 Yahoo/yfinance snapshot supplies daily OHLC/adjusted
prices through 2025. Nominal units are restored with dated splits. TQQQ/QLD/SSO's
2025 2-for-1 events are explicit; TMF's earlier reverse split is in its warmup
history. Exact issuer amounts replace rounded vendor cash, and final tax fields
enter ex-post accounting only.

| Issuer | Final records |
|---|---|
| ProShares | [2024 primary ICI](https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_2024.xlsx), [2025 primary ICI](https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_2025.xlsx), exact current issuer API cash comparisons |
| QQQ | [Invesco 2024 ICI](https://www.invesco.com/content/dam/invesco/us/en/documents/tax-document/ce-ici-primary-and-secondary-distribution-file-file-ivz-etf-2024-02-10-2025.xlsx), [2025 ICI](https://www.invesco.com/content/dam/invesco/us/en/documents/tax-documents/ICI-Primary-and-Secondary-and-NRA-File-IVZ-ETF-2025.xlsx) |
| TMF | [Direxion 2024 ICI](https://www.direxion.com/uploads/2024-ICI-Primary-Direxion-ETF.xlsx), [2025 ICI](https://www.direxion.com/uploads/2025-ICI-Primary-Direxion-ETF.xlsx) |
| SPY | [State Street primary ICI](https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/SPDR_ICI_Tax_Summary_Primary.xlsx), captured as 2025 |

The ProShares exact-cash consistency tolerance remains 2e−5. QQQ/TMF separately
verify known split factors against the vendor's 0.001/share rounding quantum;
split factors are never estimated from rounded dividends. Each row uses original
issuer date formats and tax year. SPY's December 19, 2025 ex-date pays January
30, 2026: full-year replay needs deemed-distribution tax-year mapping and later
payable-date coverage. Those missing components block rather than default.

Foreign passthrough, exempt/special-rate gains, ROC, liquidation or year-allocation
fields reject when unimplemented. Complete MA fund-government-obligation review
is missing: the successful fund bundles permit federal-only/TX/FL/NV and explicitly
block MA. See [fund manifests](evidence/fund-source-manifests.json).

## Implemented legal scope

US equity settlement is T+3 from June 7, 1995; T+2 from September 5, 2017; T+1
from May 28, 2024, with actual exchange holidays. Earlier unreviewed versions
block. Sources: [1995 Federal Register](https://www.govinfo.gov/content/pkg/FR-1995-05-26/html/95-12986.htm),
[2017 SEC final rule](https://www.sec.gov/files/rules/final/2017/34-80295.pdf),
[2024 SEC transition](https://www.sec.gov/newsroom/press-releases/2024-62).
Purchase cash is conservatively committed at the trade date; sale proceeds
cannot be reused before settlement. Future sweep treatment remains a disclosed
broker-contract assumption.

The inherited [authority table](../integration-2026-11-24/TAX_AND_DATA.md) covers
2024–26 federal/MA parameters, ST/LT statutory netting/carryovers, NIIT, wash sales,
cross-account Roth disallowance and Section 1256. Older personal tax years and
complete household returns/AMT/credits are unsupported.

New Section 199A scope is verified 2024/25 REIT dividends through a RIC: 20%
deduction, capped at 20% taxable income before deduction excluding net capital
gains/QDI; regular federal taxable income falls but MAGI/full NIIT income does
not. MA does not allow the deduction. PTP/business aggregation and 2026 REIT
worksheet scope remain blocked. Authorities:
[2025 Form 8995 instructions](https://www.irs.gov/pub/irs-prior/i8995--2025.pdf),
[Form 8995](https://www.irs.gov/pub/irs-prior/f8995--2025.pdf),
[MA conformity differences](https://www.mass.gov/info-details/differences-between-ma-and-federal-tax-law-for-personal-income).

Independent examples: $1,000 REIT dividends above $100,000 wages gives a $200
deduction/$176 incremental federal tax at 22%. With $20,000 REIT dividends and
no wages in 2024, the $14,600 standard deduction leaves a $1,080 deduction cap
and $432 federal tax. At the NIIT threshold, $1,000 still adds $38 NIIT.

Investor qualification is distinct from issuer designation. The runner currently
proves >60 held calendar days for QDI and >45 for qualified REIT dividends from
already-held unhedged shares. Future-window-dependent eligibility blocks rather
than being guessed. QQQ's cash-accumulation demo avoids short-held reinvestment
lots. This is a conservative engineering gate, not a declaration that every such
dividend is legally unqualified. Authorities: [IRS Publication 550](https://www.irs.gov/publications/p550)
and Form 8995 holding instructions. Future holding/risk-adjustment schedules
remain necessary to expand scope.

Roth clocks cannot precede 1998. Flows require owner birth/first-year/basis inputs.
Ordinary compliant internal trades have no annual personal investment tax;
dated contribution/MAGI/compensation checks remain separate. Conversions require
supplied pro-rata taxable amounts. The runner blocks taxable/penalized personal
withdrawals needing an external tax schedule; terminal internal sale is not a
qualified withdrawal. Authorities: [590-A](https://www.irs.gov/publications/p590a),
[590-B](https://www.irs.gov/publications/p590b),
[26 CFR 1.408A-6](https://www.govinfo.gov/content/pkg/CFR-2025-title26-vol6/xhtml/CFR-2025-title26-vol6-sec1-408A-6.html).
UBTI/inherited IRAs/prohibited-transaction consequences are not assumed zero when
relevant; they are unsupported. Future frozen-year eligibility is hypothetical.

## Derivatives and brokers

The preserved futures adapter uses dated contracts, shared cash/collateral,
integer sizing, tick/multiplier checks, both roll legs, variation margin, margin
changes and gap liquidations. Qualifying taxable contracts get Section 1256;
Roth internal gains do not. Scenario margins are not historical exchange/house
requirements. Independent dated tapes were not acquired. Continuous charts do
not establish executable fills. The QC prototype compiled; Lean was not run.

Long options require contract-matched timestamped observed bid/ask/depth. Premium
and unsettled T+1 proceeds share the cash ledger. Physical exercise needs a holder
notice, sufficient cash/stock, exact strike delivery, explicit fee and settlement.
The physical settlement version is effective from May 28, 2024. Authorities:
[OCC specifications](https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications),
[OIC exercise](https://www.optionseducation.org/referencelibrary/faq/options-exercise).
European early exercise, naked put delivery, short writing/assignment, interacting
taxable hedge laws and unknown OCC adjusted deliverables reject.

Generic Roth is cash, not unrestricted margin. Existing broker profiles require
actual permissions/minimums and dated futures house margins. Current documents
do not establish old approval. Collateral helpers alone do not implement covered
calls, CSPs, collars or spreads. Theoretical option prices cannot certify fills.

The [preserved data-source investigation](../integration-2026-11-24/TAX_AND_DATA.md)
documents provisioned free ThetaData EOD access and paid/institutional OPRA, Cboe,
OptionMetrics and futures routes. No purchase, entitlement or cost quote is
invented. A free service requiring user provisioning remains unavailable here.
KMLM/managed futures, VBR proxies and leveraged gold retain benchmark, constituent,
roll, daily-path and legal-regime gaps; arbitrary drifts are not substituted.
