# Historical legal scope added 2026-11-25

Federal regular rate schedules, basic standard deductions and preferential
thresholds now cover **2021–2026**, through the existing canonical tax function.
New primary sources: IRS [Rev Proc 2020-45](https://www.irs.gov/pub/irs-drop/rp-20-45.pdf),
[2021-45](https://www.irs.gov/pub/irs-drop/rp-21-45.pdf), and
[2022-38](https://www.irs.gov/pub/irs-drop/rp-22-38.pdf). Their independently
published bracket-base dollar taxes are used as fixtures. This is the basic
independent-under-65/nonblind taxpayer scope, not a full household Form1040.
Additional aged/blind/dependent rules, credits, AMT, special securities and
Section199A outside prior verified years remain gated or excluded explicitly.

Roth eligibility adds 2021–23 contribution limits and phaseouts using IRS
[Notice2020-79](https://www.irs.gov/pub/irs-drop/n-20-79.pdf),
[2021-61](https://www.irs.gov/pub/irs-drop/n-21-61.pdf),
[2022-55](https://www.irs.gov/pub/irs-drop/n-22-55.pdf). Owner clocks, compensation,
MAGI, age, other-IRA contributions and withdrawal ordering still use `RothHistory`.
No cash-account margin or routine Roth internal tax is introduced.

Massachusetts now additionally covers the **2023** bounded full-year wage/corporate
securities scope. The [2023 Form1 instructions](https://www.mass.gov/doc/2023-form-1-instructions/download)
document $1,000,000 first surtax threshold, 5% ordinary/ordinary-LT income and
8.5% ordinary ST capital gains. Existing independent state netting/carryovers
remain separate from federal loss deductions. MA bank-interest exemptions,
RIC government-obligation adjustments and complex/low-income worksheets are
not silently inferred. The input bundles still block unreviewed fund/state cells.

The previous MA test rejected 2023 solely because no table was implemented. It
now rejects genuinely unreviewed **2022**, and a new independent fixture asserts
the enacted 2023 tax amounts. This changes supported scope, not numerical
acceptance thresholds. All zero-residual tax-disposition tests remain unchanged;
zero NAV is still insufficient evidence of IRC165(g) complete worthlessness.

Original source bytes are private under `audit-output/law-sources-v1`, with
capture dates, URLs and hashes in `receipt.json`. 2022 MA instruction download
returned404; no pre-2023 MA coverage is asserted. Dated ProShares2023 ICI bytes
support new TQQQ/UPRO replays; 2021 layout differences, 2022 missing ICI and
QLD/SSO missing2023 API payouts stay explicit blockers.

The longer actual-source replay exposed JSON failure on positive infinity in
statutory bracket metadata. `account_cli` now labels only unbounded bracket
ceilings `UNBOUNDED`. Nonfinite wealth/tax metrics still raise errors. Calculation
rates, original reconstruction gates, forecasts and evaluated returns are unchanged.
