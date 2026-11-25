# Numerical sampling results — model conditional only

All rows retain the 0.02 log-quantile tolerance. The original small-prefix
test and the new simultaneous-interval study are different registered
diagnostics; the old failure is not rewritten. The new runs have two independent
seeds × 1,048,576 paths, conditional on an unapproved stationary-bootstrap model.

| Years | Leverage | Original maximum log-quantile discrepancy | Original gate | New maximum simultaneous CI log error | New independent-seed maximum log difference | New gate |
|---:|---:|---:|---|---:|---:|---|
| 5 | 1× | 0.020597 | FAIL | 0.002081 | 0.001186 | PASS |
| 5 | 2× | 0.049670 | FAIL | 0.004311 | 0.002171 | PASS |
| 5 | 3× | 0.068765 | FAIL | 0.007197 | 0.004235 | PASS |
| 10 | 1× | 0.023572 | FAIL | 0.002721 | 0.001434 | PASS |
| 10 | 2× | 0.059977 | FAIL | 0.005725 | 0.003404 | PASS |
| 10 | 3× | 0.098587 | FAIL | 0.009465 | 0.004838 | PASS |
| 20 | 1× | 0.049442 | FAIL | 0.003711 | 0.000938 | PASS |
| 20 | 2× | 0.125722 | FAIL | 0.007723 | 0.003115 | PASS |
| 20 | 3× | 0.199289 | FAIL | 0.012631 | 0.003700 | PASS |
| 30 | 1× | 0.042880 | FAIL | 0.004321 | 0.001305 | PASS |
| 30 | 2× | 0.101548 | FAIL | 0.009355 | 0.003516 | PASS |
| 30 | 3× | 0.189375 | FAIL | 0.014811 | 0.008393 | PASS |
| 40 | 1× | 0.050326 | FAIL | 0.004920 | 0.001621 | PASS |
| 40 | 2× | 0.113289 | FAIL | 0.010471 | 0.003966 | PASS |
| 40 | 3× | 0.183704 | FAIL | 0.016841 | 0.007825 | PASS |

The two columns of errors measure different quantities: original differences
among small prefixes versus a confidence-bound distance from the final estimated
quantile. The new gate also separately checks seed and latest-doubling differences.
The complete [precision report](evidence/precision-report.json) records all three.
Pointwise original pilot intervals are diagnostic, not simultaneous claims.

## Conditional extreme losses

These are probabilities under the finite empirical-template model, **not calibrated
real-world probabilities**. They include ordinary Monte Carlo uncertainty only.
One-session fund extinction is impossible within this enumerated support, so zero
sampled extinction does not show real-world safety. A 90% drawdown is distinct
from mathematical zero NAV.

| Years | Leverage | Terminal unit-NAV loss | Drawdown >90% | Modeled liquidation count |
|---:|---:|---:|---:|---:|
| 5 | 1× | 7.84% | 0.00% | 0 / 2,097,152 |
| 5 | 2× | 19.30% | 0.66% | 0 / 2,097,152 |
| 5 | 3× | 27.84% | 13.36% | 0 / 2,097,152 |
| 10 | 1× | 2.40% | 0.00% | 0 / 2,097,152 |
| 10 | 2× | 12.37% | 2.79% | 0 / 2,097,152 |
| 10 | 3× | 23.42% | 28.60% | 0 / 2,097,152 |
| 20 | 1× | 0.27% | 0.00% | 0 / 2,097,152 |
| 20 | 2× | 5.53% | 7.61% | 0 / 2,097,152 |
| 20 | 3× | 16.44% | 51.74% | 0 / 2,097,152 |
| 30 | 1× | 0.04% | 0.00% | 0 / 2,097,152 |
| 30 | 2× | 2.65% | 12.29% | 0 / 2,097,152 |
| 30 | 3× | 12.00% | 67.39% | 0 / 2,097,152 |
| 40 | 1× | 0.00% | 0.00% | 0 / 2,097,152 |
| 40 | 2× | 1.29% | 16.73% | 0 / 2,097,152 |
| 40 | 3× | 8.92% | 77.97% | 0 / 2,097,152 |

Exact binomial intervals and family allocations are in the linked JSON. The
quantile study targets only the 10th/50th/90th percentiles; it does not certify
the farthest tail or a crash outside training support. The [support matrix](SUPPORT_MATRIX.md)
and [sensitivity study](evidence/specification-sensitivity.json) retain the much
larger structural and source uncertainties.
