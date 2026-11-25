# Stress marks, trading restrictions and zero-residual tax scope

Reviewed 2026-11-25. This clarification adds restrictions; it does not change
the registered stress returns, weights, tax rates, numerical thresholds or any
forecast. Raw diagnostic results produced before this review remain archived.

[IRS Publication 550](https://www.irs.gov/publications/p550) and the
[IRS worthless-securities FAQ](https://www.irs.gov/faqs/capital-gains-losses-and-sale-of-home/losses-homes-stocks-other-property/losses-homes-stocks-other-property-1)
explain the last-day-of-tax-year deemed disposition for completely worthless
securities. A zero price mark is not evidence of an ordinary same-day sale.
Holding-period character, recovery rights and final liquidation documents may
matter. This continuation does **not** implement all worthless-security,
bankruptcy or complete-liquidation tax rules. It blocks taxable zero-residual
dispositions at both the daily-runner and low-level ledger APIs, before mutation.
Positive-proceeds reviewed cash liquidations and supported ordinary sales retain
their existing treatment. Roth internal zero-residual closure remains supported;
it creates no personal capital-loss deduction or routine capital-gains tax.

Early synthetic taxable zero-price results assumed an ordinary same-day sale.
Those numerical values are not valid tax evidence and must not be released as
supported after-tax outcomes. The final confidence assessment retains these
raw diagnostics, marks the affected rows **BLOCKED**, and verifies the new API
rejection. The original 259 tests are preserved; newly introduced tests that
assumed this unsupported tax disposition are corrected to require rejection.

[NYSE market-wide circuit-breaker procedures](https://www.nyse.com/publicdocs/nyse/NYSE_MWCB_FAQ.pdf)
describe 7%, 13% and 20% S&P 500 decline levels. A prescribed intraday −25% or
−34% broad-market close therefore assumes a change/failure of normal market
structure or an exposure outside those controls. It is a deliberately severe
counterfactual valuation, not a certified executable US broad-index close.
Overnight discontinuities and issuer/counterparty failure are distinct risks.
The workflow does not manufacture intraday quotes, model halt resumptions or
claim exact execution through a market suspension. Portfolio cash and corporate
actions are reconciled under the declared hypothetical timing only. Future
market rules and unforeseen closures are not known.
