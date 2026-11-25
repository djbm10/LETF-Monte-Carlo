# Unweighted financial sensitivity results

These are **hypothetical conditions**, not calibrated forecasts. No probabilities
or weights are assigned to cases/anchors. Two reused old paths do not estimate
percentiles, tail frequencies or model confidence. All 600 records and blocked
cases remain in [the confidence assessment](evidence/full-stress-confidence.json).

## One declared anchor, 3× fund plus cash

Anchor root 20261127/block0/path0 was registered before results. Both accounts
start with $10,000 plus $3,000 at each year start, 80% risky/20% cash, 63-session
rebalancing, hypothetical ordinary dividends, zero cash sweep and frozen 2026
federal law. After-tax drawdowns use cash taxes paid and terminal payable, with
cash-flow unitization. They are not daily hypothetical liquidation-tax curves.
The other anchor and all remaining cases are in the JSON; this slice is not a
representative forecast or strategy ranking. The growing contribution budget is
$10,000 + $3,000 × years, so ending wealth is not itself an investment return.

| Years | Assumption | Taxable net ending | Roth net ending | Roth real ending | Roth max drawdown | Roth money-weighted annual return |
|---:|---|---:|---:|---:|---:|---:|
| 5 | reference | $45,148 | $49,079 | $42,874 | 65.27% | 18.56% |
| 5 | daily_extinction_34 | BLOCKED | $10,995 | $9,605 | 91.58% | -20.81% |
| 5 | bear_7y | $7,525 | $6,966 | $5,408 | 89.82% | -31.85% |
| 5 | stagnation_7y_high_rates | $12,321 | $11,825 | $9,180 | 78.39% | -19.03% |
| 5 | inflation_funding_10y | $28,786 | $29,620 | $21,338 | 65.27% | 4.52% |
| 5 | issuer_failure_zero | BLOCKED | $10,955 | $9,570 | 91.75% | -20.90% |
| 5 | higher_cost | $40,158 | $43,137 | $37,683 | 66.16% | 14.88% |
| 10 | reference | $75,437 | $83,115 | $66,423 | 69.38% | 10.84% |
| 10 | daily_extinction_34 | BLOCKED | $25,995 | $20,775 | 91.58% | -6.64% |
| 10 | bear_7y | $19,106 | $17,841 | $10,976 | 97.06% | -12.76% |
| 10 | stagnation_7y_high_rates | $28,756 | $27,154 | $16,706 | 88.78% | -5.96% |
| 10 | inflation_funding_10y | $29,280 | $28,967 | $14,202 | 74.21% | -4.95% |
| 10 | issuer_failure_zero | BLOCKED | $25,955 | $20,742 | 91.75% | -6.67% |
| 10 | higher_cost | $60,636 | $65,150 | $52,066 | 70.49% | 7.26% |
| 20 | reference | $559,193 | $715,554 | $431,241 | 69.38% | 17.29% |
| 20 | daily_extinction_34 | BLOCKED | $55,995 | $33,747 | 91.58% | -1.93% |
| 20 | bear_7y | $219,633 | $236,548 | $109,751 | 97.06% | 9.42% |
| 20 | stagnation_7y_high_rates | $271,498 | $304,888 | $141,459 | 88.78% | 11.26% |
| 20 | inflation_funding_10y | $240,919 | $281,625 | $97,903 | 74.21% | 10.69% |
| 20 | issuer_failure_zero | BLOCKED | $55,955 | $33,722 | 91.75% | -1.94% |
| 20 | higher_cost | $348,377 | $430,192 | $259,263 | 70.49% | 13.71% |
| 30 | reference | $3,557,004 | $5,461,500 | $2,581,854 | 69.38% | 19.15% |
| 30 | daily_extinction_34 | BLOCKED | $85,995 | $40,653 | 91.58% | -0.91% |
| 30 | bear_7y | $1,478,014 | $1,915,004 | $696,948 | 97.06% | 14.46% |
| 30 | stagnation_7y_high_rates | $1,797,630 | $2,420,986 | $881,095 | 88.78% | 15.51% |
| 30 | inflation_funding_10y | $1,607,227 | $2,248,754 | $613,210 | 74.21% | 15.18% |
| 30 | issuer_failure_zero | BLOCKED | $85,955 | $40,634 | 91.75% | -0.91% |
| 30 | higher_cost | $1,673,485 | $2,434,137 | $1,150,707 | 70.49% | 15.54% |
| 40 | reference | $18,413,336 | $33,488,514 | $12,127,816 | 69.38% | 19.33% |
| 40 | daily_extinction_34 | BLOCKED | $115,995 | $42,008 | 91.58% | -0.53% |
| 40 | bear_7y | $7,800,688 | $11,832,140 | $3,298,838 | 97.06% | 15.96% |
| 40 | stagnation_7y_high_rates | $9,451,789 | $14,921,876 | $4,160,266 | 88.78% | 16.71% |
| 40 | inflation_funding_10y | $8,467,781 | $13,870,157 | $2,897,447 | 74.21% | 16.47% |
| 40 | issuer_failure_zero | BLOCKED | $115,955 | $41,993 | 91.75% | -0.53% |
| 40 | higher_cost | $6,485,489 | $10,871,659 | $3,937,155 | 70.49% | 15.68% |

Taxable zero-residual values are intentionally absent: loss timing/character
requires an implemented disposition treatment. Roth fund extinction can leave
reserve cash and later deposits. Such deposits are not fund recovery. The issuer
failure case explicitly writes off unpaid receivables; original full recovery
is retained as a separate unweighted assumption, not a guaranteed payout.

## Existing sampling error versus specification sensitivity

These are the **prior** no-flow/no-distribution fixed-21-session unit-NAV studies,
not the account profile above. The reference passed numerical precision; the
four smaller-budget specification studies retain their failed complete precision
gates. Their point estimates/ranges are not posterior intervals or probabilities.

| Years | Leverage | Reference median unit NAV | Minimum among four specifications | Maximum among four specifications |
|---:|---:|---:|---:|---:|
| 5 | 1× | 1.763 | 1.760 | 1.790 |
| 5 | 2× | 2.029 | 1.785 | 2.215 |
| 5 | 3× | 2.098 | 1.711 | 2.402 |
| 10 | 1× | 2.985 | 2.987 | 3.069 |
| 10 | 2× | 3.711 | 2.890 | 4.333 |
| 10 | 3× | 3.650 | 2.444 | 4.683 |
| 20 | 1× | 8.572 | 8.540 | 9.049 |
| 20 | 2× | 12.554 | 7.631 | 16.980 |
| 20 | 3× | 11.299 | 5.105 | 18.734 |
| 30 | 1× | 24.634 | 24.568 | 26.668 |
| 30 | 2× | 42.617 | 20.222 | 67.124 |
| 30 | 3× | 35.324 | 10.695 | 75.195 |
| 40 | 1× | 70.883 | 70.634 | 78.945 |
| 40 | 2× | 144.690 | 53.477 | 264.788 |
| 40 | 3× | 110.129 | 22.376 | 301.408 |

Within-model log-quantile sampling intervals are stored separately for each
specification. They do not cover source revisions, missing collateral/rate histories,
structural breaks, future policy or rejected forecasting assumptions. No claim
is made that this small set exhausts those uncertainties.

## Acceptance and independent arithmetic

540 supported hypothetical-mechanics cases and 60 explicitly blocked tax cases.
Every supported exported-event terminal identity and unitized drawdown audit passes.
The observed 2024–25 funded TQQQ comparison is a separate observed-input study:
see [REPORT.md](REPORT.md) and [observed account evidence](evidence/observed-funded-comparison.json).
It is not combined with these future scenarios or the earlier unit-NAV distributions.
