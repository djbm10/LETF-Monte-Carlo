# Sources, conventions and rights

Reviewed2026-11-25. The [175-snapshot catalog](evidence/final-v4/final-source-catalog.json)
records exact provider URLs, raw SHA-256, capture time, earliest/latest genuine
observations, conventions, vintage and rights when known. UNKNOWN is retained
rather than inferred. Raw bytes/full ensembles remain private. Mixed catalog
integrity is BLOCKED as an aggregate financial claim because it includes an
explicit provider demonstration sample. No original snapshot was replaced.

| Source | Strongest supported use | Important limit |
|---|---|---|
|[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)|Daily market/size-growth/industry observations from1926; monthly bills, joint macro research|Current revised CIZ/CRSP vintage; early Saturday sessions retained; not executable ETF prices or old publication vintages|
|[FRED/ALFRED](https://fred.stlouisfed.org/legal/)|CPI, production, rates, credit, unemployment; selected old release archives|Series-provider terms differ; yields are not bond total returns; missing CPI/credit months stay incomplete|
|[ProShares tax supplements](https://www.proshares.com/resources/tax-and-filing-documents/tax-supplements)|Final2021–25 ICI classifications, ex/payable dates and split checks|2022 uses`.xls`;2021/22 headers reviewed explicitly; API omission alone cannot establish zero cash|
|[Issuer TQQQ NAV](https://accounts.profunds.com/etfdata/ByFund/TQQQ-historical_nav.csv)|Split-restated NAV through2026-11-24; cash-reinvested valuation|NAV is neither an exchange fill nor automatically dividend-adjusted; earliest NAV differs from tradable listing date|
|[Issuer2026 cash API](https://www.proshares.com/api/distributionsummary/?fund=TQQQ&year=2026)|Three currently reported2026 ex-date cash events|Not final2026 tax classes or independent proof of complete annual payouts|
|[Nasdaq-origin XNDX through FRED](https://fred.stlouisfed.org/series/NASDAQXNDX)|Total-return index observations1999-03-04 onward|Current revised source; exact dated fund target/swap conventions and earlier official histories not certified|
|Existing State Street/Invesco/Direxion final sources|SPY2025, QQQ/TMF2024–25 corporate actions/distributions|Exact source hashes/URLs in preserved bundle manifests; no unreviewed year/state extrapolation|

Important immutable catalogs:

- Original macro:`1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`.
- Refreshed compatible macro:`8a57789fc71a67db3136f8ab5e58239991f36b92e292f248ef8170225a62f345`;
  unchanged parent retained. French returns still end September2026; other
  macro releases reach October/November as individually available.
- Original official index/NAV:`2be48a3a975ad9aa8f290b5c3d7f29ce832f436500878913a8ec8bf65ebc61e1`.
- Original vendor daily:`8a434afe840e8b5f22a0cd1d336cb81b33b672a1a534fb04b70f0900f558bb6c`.

The original live-refresh sandbox DNS failures are retained; registered feeds
were retried with authorized network access into new snapshots. CPI missing
observations remain missing. Provider format changes/garbage numeric text fail
instead of becoming invented zeros. Last-observation date is not release date.

## Dated tax authority

New federal tables use [Rev.Proc.2020-45](https://www.irs.gov/pub/irs-drop/rp-20-45.pdf),
[2021-45](https://www.irs.gov/pub/irs-drop/rp-21-45.pdf) and
[2022-38](https://www.irs.gov/pub/irs-drop/rp-22-38.pdf); Roth limits/phaseouts
use [Notice2020-79](https://www.irs.gov/pub/irs-drop/n-20-79.pdf),
[2021-61](https://www.irs.gov/pub/irs-drop/n-21-61.pdf) and
[2022-55](https://www.irs.gov/pub/irs-drop/n-22-55.pdf).
Massachusetts2023 is constrained to the reviewed
[Form1 instructions](https://www.mass.gov/doc/2023-form-1-instructions/download).
Exact archived PDF hashes are in the existing law-source receipt; implementation
and independent dollar examples are in
[HISTORICAL_TAX_EXTENSION](../HISTORICAL_TAX_EXTENSION.md). Prior2024–26 authorities,
wash/QDI/1256/Roth scope remain in the preserved audit/tax documentation.
No full tax-return certification is implied.

## Derivative evidence and lawful next data request

[Cboe Option EOD Summary](https://datashop.cboe.com/option-eod-summary) and
[quote intervals](https://datashop.cboe.com/option-quote-intervals) advertise
history from January2012. The offered
[free sample217](https://datashop.cboe.com/download/sample/217) was captured:
SHA`981be1aafe6970e798d5be42f049e6c5b6e507f01cb1509ab513e9bc2e08da24`.
It contains two entitlement variants of one2023-08-25 sample, not independent
days. The README describes demonstration/subset/representative records.
Inspection covers32672 rows/34fields without asserting complete historical
execution. [Inspection](evidence/final-v4/final-provider-demo-inspection.json).
The provider notes a quote-size convention change on2026-06-22; nominal depth
columns alone do not establish simultaneous executable liquidity. Index
underlying values may be zero due to entitlements, not actual prices.

Request the smallest licensed **equity/ETF symbol/date subset** with contract
root/strike/expiry/style/settlement/multiplier/OCC deliverables, bid/ask and
sizes with actual timestamps/time zones, volume/open interest, dividends,
exercise/assignment notices and broker permissions. Optional IV/Greeks cannot
replace execution evidence. Current total price is order/entitlement specific
and was not verified; no universal subscription price or cheapest-provider
claim is invented. Free samples cost$0 but do not supply a longitudinal tape.

[CME’s public Daily Bulletin](https://www.cmegroup.com/market-data/daily-bulletin.html)
showed a dated FINAL2026-11-24 equity-futures bulletin; the local PDF transfer
failed/stalled and was cancelled. No source-byte hash or normalized tape is
claimed for that failed download. Public reference reports are not exact
intraday fills. [CME settlement access FAQ](https://www.cmegroup.com/articles/faqs/access-to-cme-group-settlement-data-faq.html)
describes historical access through DataMine/login. Required ES/MES/NQ/MNQ/bond/
gold records: dated contract identity/expiry/notice/delivery calendar, daily
settlements and release time, OHLC/session scope, tick/multiplier conventions,
explicit roll fills/spreads/commissions, collateral yield, effective exchange
and broker house margins, and date-specific IRA approval. Settlement-only
series cannot prove intraday margin survival. No paid account was created.

Secured option mechanics refer to [OCC T+1 memo54580](https://infomemo.theocc.com/infomemos?number=54580),
[OIC assignment](https://www.optionseducation.org/referencelibrary/faq/options-assignment)
and [OIC cash-secured puts](https://www.optionseducation.org/strategies/all-strategies/cash-secured-put).
These explain mechanics, not historical broker approval or a model’s returns.
The original QuantConnect/reconciliation prototypes remain; no new hosted Lean
validation or real dated futures backtest is claimed.

## Index and fund economics

The [2026 TQQQ prospectus](https://www.sec.gov/Archives/edgar/data/1174610/000117461026001035/f46111d1.htm)
references the NDX daily target and separates derivative financing from its
expense table. It cannot establish an earlier historical expense/waiver path.
The prior2021 annual report snapshot has swaps2.23539×NAV, stock0.71878×NAV,
futures0.04605×NAV and gross snapshot swap-cost1.35779%/NAV/year. A reporting-date
notional/rate is not a daily realized funding series; collateral income and
transaction costs matter. Neither a current fee nor a switch of index convention
is used to tune inspected2020–25 or relabel2026 as a prospective holdout.
Issuer rounding discrepancies, raw hashes and all old errors remain visible.

No literal1930 Nasdaq/TQQQ, modern swap market, freely floating modern gold
regime or Roth account is invented. Missing long-run benchmark/constituent/
financing publication evidence remains source uncertainty, separately from
Monte Carlo sampling intervals.


## Additional resolved legal/signal inputs

The archived [2025 CFR1.199A-3(d)(4)(ii)](https://www.govinfo.gov/content/pkg/CFR-2025-title26-vol4/xhtml/CFR-2025-title26-vol4-sec1-199A-3.html)
requires more than45 holding days within91 days for verified RIC Section199A
preferences. The rule applies after2020-08-24; earlier transition elections are
not assumed. New fixtures independently check45/46-day sales, partial lots,
year-end receipt-year retention, splits, ordinary-tax deduction, NIIT gross income
and MA’s separate base. Hedged/preferred/unverified distributions remain gated.
The source lake retains the exact government bytes/hash. Stock-split wash unit
algebra follows the preserved IRS Publication550 basis/replacement rules;
no split itself is modeled as a replacement purchase or a new dollar loss.

Original^VIX and^IRX2024–25 closes were captured with pinned yfinance0.2.66 after
sandbox DNS failure was retried with authorized network access. Signal CSVs have
no missing values; no forward fill, monthly future RF or DFF substitute was used.
IRX Close/100/252 preserves the frozen35/0 selector’s original convention; it is
not certified daily T-bill total return or actual collateral yield. Captured client
DataFrames, not original HTTP response bodies, are archived explicitly as such.
These are current-vintage next-open retrospective signals, not old release archives.
Source snapshot IDs:VIX a5db45db9d04101094cc35e579541076a27df5ac0f7be94a7533a32fd75c3314;
IRX c1221e8b6792e84b073da670737bd352630361a354c8a0d81a8ca402d0fde06e.
