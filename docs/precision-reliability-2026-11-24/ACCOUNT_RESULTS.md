# Paired hypothetical account results — not investment forecasts

All entries are exploratory conditional estimates under an **unapproved market
model**. Initial cash $10,000; no contributions, withdrawals, distributions, cash
sweep or intermediate rebalancing; single taxpayer, $100,000 other nominal income,
FEDERAL_ONLY and explicitly frozen 2026 law. Both accounts use identical market
paths and execution assumptions (2 bp spread, 1 bp slippage). Roth is an existing
balance with internal liquidation, not a qualified personal distribution.

The 2,097,152 paired paths per combination are projected through independently
reconciled canonical-ledger piecewise-affine endpoints. They are **not full daily
ledger replays for every path**. Terminal tax/refund payable is included in net
assets; all proceeds/refunds are not necessarily immediately spendable. Exact
daily after-tax drawdowns are unavailable, so only conservative bounds are shown.
All 30 nominal and real-wealth quantile sampling gates pass; their intervals
exclude model/source/parameter/future-policy uncertainty. No dollar-cost ledger
attribution for financing versus management fees is exported for these paths;
both are embedded by the disclosed daily return equation, with separate cost
sensitivities in the report. Actual fund distribution forecasts remain blocked.

The corrected interval allocation covers all six possible upstream sampling
stops, both accounts, both wealth metrics and all quantiles/probabilities. Its
maximum wealth interval error is 0.018446 log units, below the unchanged 0.02
threshold. The initial fixed-look allocation is retained as superseded evidence.

## All five horizons and leverages

Values rounded to whole dollars for readability. These favorable or unfavorable
conditional medians are not model endorsements or account/strategy rankings.

| Years | Leverage | Taxable median nominal | Roth median nominal | Taxable median real | Roth median real | Taxable real-loss frequency | Roth real-loss frequency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1× | $16,484 | $17,628 | $14,296 | $15,287 | 14.78% | 13.58% |
| 5 | 2× | $18,743 | $20,286 | $16,227 | $17,562 | 24.73% | 23.90% |
| 5 | 3× | $19,322 | $20,967 | $16,713 | $18,135 | 32.12% | 31.50% |
| 10 | 1× | $26,859 | $29,834 | $20,269 | $22,513 | 7.42% | 6.46% |
| 10 | 2× | $33,035 | $37,100 | $24,909 | $27,971 | 18.70% | 17.76% |
| 10 | 3× | $32,512 | $36,485 | $24,493 | $27,487 | 29.08% | 28.30% |
| 20 | 1× | $74,331 | $85,684 | $42,529 | $49,020 | 2.11% | 1.70% |
| 20 | 2× | $107,581 | $125,494 | $61,512 | $71,749 | 11.08% | 10.28% |
| 20 | 3× | $97,388 | $112,941 | $55,613 | $64,533 | 23.19% | 22.33% |
| 30 | 1× | $205,629 | $246,243 | $89,226 | $106,834 | 0.64% | 0.48% |
| 30 | 2× | $351,588 | $425,995 | $152,412 | $184,733 | 6.87% | 6.26% |
| 30 | 3× | $292,400 | $353,103 | $126,703 | $152,993 | 18.92% | 18.10% |
| 40 | 1× | $569,175 | $708,550 | $187,199 | $232,960 | 0.20% | 0.14% |
| 40 | 2× | $1,131,355 | $1,446,319 | $371,866 | $475,325 | 4.33% | 3.91% |
| 40 | 3× | $868,106 | $1,100,848 | $285,271 | $361,725 | 15.64% | 14.91% |

## 3× outcome distributions, terminal loss and drawdown bounds

The after-tax drawdown column gives lower/upper bounds on the **median** maximum
drawdown, not a confidence interval and not an exact after-tax daily equity curve.
Extreme nominal terminal outcomes do not prevent very deep interim losses.

| Years | Account | Nominal p10 | Nominal p50 | Nominal p90 | Terminal loss >90% | Median after-tax drawdown bounds |
|---:|---|---:|---:|---:|---:|---|
| 5 | taxable | $4,119 | $19,322 | $59,752 | 0.34% | 63.72%–75.56% |
| 5 | roth | $3,459 | $20,967 | $68,531 | 1.87% | 69.00%–69.04% |
| 10 | taxable | $4,058 | $32,512 | $193,710 | 0.89% | 76.07%–89.90% |
| 10 | roth | $3,398 | $36,485 | $231,564 | 3.07% | 79.80%–79.84% |
| 20 | taxable | $5,108 | $97,388 | $1,384,391 | 1.39% | 85.77%–99.80% |
| 20 | roth | $4,448 | $112,941 | $1,778,387 | 3.41% | 90.59%–90.63% |
| 30 | taxable | $7,743 | $292,400 | $8,367,189 | 1.42% | 91.95%–100.00% |
| 30 | roth | $7,106 | $353,103 | $10,942,164 | 3.04% | 93.89%–93.93% |
| 40 | taxable | $12,228 | $868,106 | $46,016,220 | 1.26% | 94.31%–100.00% |
| 40 | roth | $12,621 | $1,100,848 | $60,350,341 | 2.52% | 95.26%–95.30% |

Complete all-leverage p10/p50/p90 nominal and inflation-adjusted wealth, lump-sum
CAGRs, exact sampling intervals, terminal tax, probability intervals and conditional
inflation subsets are in [paired-account-report.json](evidence/paired-account-report.json).
Inflation comes from the same joint paths. Subsets above 4% / below 2% annualized
inflation describe conditional co-movement, not a causal inflation response.

Nonzero distributions or cash flows, MA, changed tax inputs, intermediate sales
and fund extinction explicitly reject the endpoint accelerator. No recurring-
contribution IRR, actual-fund tax forecast, future tax-law certification, exact
daily after-tax recovery time or calibrated real-world ruin probability is claimed.
The original 1930 counterfactuals and observed fund-account cohorts are separately
documented in the [prior report](../tail-validation-2026-11-24/REPORT.md).
