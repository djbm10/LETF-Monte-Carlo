# Implemented tax rules and explicit exclusions

Audit date: 2026-11-23. These are investment research calculations, not a complete tax-return engine. Tests establish the specified examples and invariants, not legal certification of all investor circumstances.

## Dated federal parameters

`letf/tax/years.py` supports tax years **2024, 2025 and 2026**, for single, married jointly (`married` is a compatibility alias), head of household and married separately. Unknown years/statuses fail. Historical decades cannot silently use these tables. Future scenarios may explicitly freeze an enacted year through `freeze_tax_year`, producing hypothetical results.

Authoritative parameter sources, retrieved 2026-11-23:

- [Rev. Proc. 2023-34](https://www.irs.gov/pub/irs-drop/rp-23-34.pdf), §§2.01, 2.03, 2.13: 2024 rates, capital-gains thresholds and standard deductions.
- [Rev. Proc. 2024-40](https://www.irs.gov/pub/irs-drop/rp-24-40.pdf), §§2.01, 2.03: 2025 rate tables.
- [Rev. Proc. 2025-32](https://www.irs.gov/pub/irs-drop/rp-25-32.pdf), §3.01 and §§4.01, 4.03, 4.14: updated 2025 standard deductions after P.L. 119-21 and 2026 tables. Using the originally announced 2025 deduction would be stale.

Regular federal tax stacks preferred gains and verified qualified dividends above ordinary taxable income. Any standard deduction unused against ordinary income also shelters preferred income. Ordinary dividends and interest are separately identified investment income. An explicitly supplied itemized deduction may replace the standard deduction; the engine does not infer its legal eligibility, SALT cap or investment-interest election.

The API's `ordinary_dividends` input means **nonqualified dividends only**, equivalent to Form 1099-DIV box 1a minus box 1b. `qualified_dividends` is separate. Passing all of box 1a as nonqualified would count qualified dividends twice. Input wages/other ordinary income must likewise exclude the separately supplied dividend/interest amounts.

[IRS NIIT guidance](https://www.irs.gov/individuals/net-investment-income-tax), retrieved 2026-11-23, supports 3.8% on the lesser of net investment income and excess MAGI. The statutory thresholds are not inflation-indexed. The modeled scope excludes foreign-income adjustments, business-income allocation and other Form 8960 adjustments.

Tests: `test_standard_deduction_shelters_preferred_income`, `test_dated_bottom_brackets`, `test_niit_uses_interest_dividends_and_fixed_threshold`, and explicit rejection tests in `tests/test_tax_accuracy.py`.

## Capital losses and lots

[Schedule D instructions](https://www.irs.gov/instructions/i1040sd), lines 6/14 and the Capital Loss Carryover Worksheet; IRC §§1211(b), 1212(b), 1222. Carryovers enter their own short/long categories before cross-netting. There is no taxpayer election to defer an otherwise required loss or select an alternative statutory order. The ordinary-income deduction limit is $3,000, or $1,500 for married separately. Losses absorbed in the carryover worksheet consume short-term losses first. Insufficient taxable income can preserve additional carryovers; it does not create a discretionary deferral election.

The low-level netting API retains an optional income-absorption input for compatibility. The dated ledger supplies it. Arbitrary credits, AMT or other deductions could change the complete return's carryover worksheet; that broader return is not implemented.

FIFO, HIFO and explicit lot identification operate on actual remaining quantities. Non-FIFO execution requires contemporaneous identification with the broker; a simulated choice is not proof that a real broker accepted it. Partial sales leave proportional basis. A holding period longer than one calendar year is long term. Tests include leap-year anniversary boundaries.

Tests: `test_carryovers_keep_character_before_cross_netting`, `test_carry_only_year_uses_loss_deduction`, `test_low_income_carryover_worksheet_preserves_unabsorbed_losses`, `test_separate_filer_loss_limit`, `test_hifo_and_specific_identification`.

## Wash matching

IRC §1091; [IRS Publication 550 (2025), Wash Sales](https://www.irs.gov/publications/p550); [Rev. Rul. 2008-5](https://www.irs.gov/pub/irs-drop/rr-08-05.pdf), published 2008. Retrieved 2026-11-23.

Matching uses actual dates, surviving replacement quantities and purchase order. It includes sale-day transactions and the ±30-calendar-day boundaries. Matched taxable lots receive basis and holding-period adjustments; partial matches split lots. The linked-account owner identity controls matching. A taxable loss replaced inside that owner's Roth is permanently disallowed without increasing Roth basis. Final tax cannot close while replacement purchases remain possible inside an unobserved window.

The `identity` field must be reviewed for substantially identical securities. Similar tickers/indices are not automatically declared substantially identical. Spouse accounts, controlled entities and other relevant activity must be supplied and mapped deliberately. Option/security cross-matching, complex corporate actions during pending matches and short sales are outside the supported matching scope and must not be inferred from this stock/ETF implementation.

Tests reproduce Publication 550's two partial-replacement examples and the numerical Rev. Rul. 2008-5 pattern; they also test same-day matching, calendar boundaries, prior-lot consumption, partial tacking, cross-year activity and idempotence.

## Section 1256

[2025 Form 6781 and instructions](https://www.irs.gov/pub/irs-pdf/f6781.pdf), retrieved 2026-11-23; IRC §1256. A qualifying contract must be affirmatively classified. Futures daily cash settlement and annual mark-to-market share one P&L record; year-end marking does not duplicate amounts already settled. Net gain/loss is allocated 40% short term and 60% long term before Schedule D netting. Closing/rolling explicitly charges each contract leg.

Mixed straddles, hedging transactions, dealer contracts, nonqualifying futures, the elective three-year §1256 loss carryback, and §475 elections are not automatically modeled. Selecting a ticker does not elect a tax regime. Tests: `test_section1256_year_end_sixty_forty_and_no_double_count`, `test_explicit_roll_charges_both_contracts`.

## Roth cash flows

IRC §408A; [Publication 590-A (2025)](https://www.irs.gov/publications/p590a), contribution worksheet 2-2; [Publication 590-B (2025)](https://www.irs.gov/publications/p590b), qualified distributions, conversion recapture and ordering; [Notice 2025-67](https://www.irs.gov/pub/irs-drop/n-25-67.pdf), 2026 limits. Retrieved 2026-11-23.

Routine compliant internal trades generate no annual personal capital-gains tax. Contribution eligibility is tested by year, compensation, MAGI, age and filing status; other IRA contributions reduce the combined cap. One `RothHistory` must aggregate an owner's accounts. Conversion taxable income is an externally determined personal-tax input, kept separate from investment gains. Transfers within that aggregate history do not reset clocks or count as regular contributions.

Distributions consume regular basis, conversions/eligible plan rollovers by year (taxable portion before nontaxable within a year), then earnings. The qualified-distribution five-year clock and individual conversion five-year recapture clocks are distinct. Age 59½ alone does not satisfy the first clock. Disability/death exceptions require explicit caller confirmation; inherited-IRA rules are not supported. Other exception categories, excess-contribution corrections and designated-Roth-plan rollover clocks are rejected.

Tests include age/catch-up limits, phaseout boundaries, excess rejection, nonqualified earnings and conversion penalties, and Publication 590-B's Amelia ordering example. The publication's contribution example prose/table contains inconsistent numbers; the implementation follows the statutory worksheet method rather than that inconsistent prose.

## Payments and excluded law

The new daily account replay accepts explicit tax-payment dates, sells securities if necessary, and includes those new gains in the following tax close. Unpaid terminal taxes are explicit payables; after-tax liquidation metrics require actual modeled terminal sales. A tax-only trade replay does not establish after-tax wealth. Refunds are separate cash events and are not silently reinvested before receipt.

Only federal-only calculations or TX/FL/NV's no-general-income-tax scope are currently accepted. **CA, NY, WA and other state calculations are blocked**, rather than reusing the undated tables in `brackets.py`. Those old constants remain for historical source compatibility and are not certified current law.

AMT, credits, QBI, state conformity, local taxes, aged/blind/dependent deductions, special collectible/§1250 rates, qualified covered calls, straddles (§1092), constructive sales (§1259), option exercise tax (§1234), partnership K-1s, UBTI/UBIT, prohibited transactions, and tax-filing/estimated-tax penalties are not implemented. Products requiring unsupported tax structures and interacting taxable option strategies fail closed where routed through the new APIs. No real filing, election or trade is performed.
