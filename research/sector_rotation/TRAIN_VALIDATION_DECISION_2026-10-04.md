# Sector rotation train + validation result — 2026-10-04

Holdout status: SEALED / NOT OPENED

Frozen baseline: SPY buy-and-hold.
Frozen costs: 10 bps one-way.
Frozen family: 6m/12m momentum × top1/top3 × SPY-200DMA gate on/off.

## Mechanism test
Hypothesis: slow institutional/earnings/capex sector rotation should make medium-horizon sector strength persist strongly enough to beat SPY after costs.

## Train
SPY CAGR: -1.46%, max DD -50.76%.
Best strategy train cell: 126d/top1/gated CAGR 15.04%, max DD -17.18%.
Several gated cells strongly improved crisis-period train outcomes.

## Validation
SPY CAGR: 13.95%, Sharpe 0.845, max DD -27.13%.

Frozen cells:
- 126d top1 ungated: CAGR 8.04%, Sharpe 0.530, max DD -24.60%
- 126d top1 gated: CAGR 4.30%, Sharpe 0.378, max DD -25.74%
- 126d top3 ungated: CAGR 11.79%, Sharpe 0.785, max DD -21.85%
- 126d top3 gated: CAGR 7.90%, Sharpe 0.666, max DD -15.42%
- 252d top1 ungated: CAGR 4.96%, Sharpe 0.358, max DD -28.80%
- 252d top1 gated: CAGR 3.92%, Sharpe 0.341, max DD -24.38%
- 252d top3 ungated: CAGR 11.19%, Sharpe 0.751, max DD -22.18%
- 252d top3 gated: CAGR 7.99%, Sharpe 0.694, max DD -17.41%

Validation CAGR excess vs SPY is negative for all 8 cells. The frozen advancement requirement (>=3/4 positive cells within a gate family, positive median CAGR excess, non-negative median Sharpe excess) therefore fails decisively.

## Decision
KILL the tested original-nine-sector monthly relative-momentum family as a candidate for superior probability-adjusted wealth under this frozen specification.

The mechanism may still exist descriptively, and the trend gate reduced drawdowns, but it did not survive the required validation baseline comparison. Do not open the 2017-2026 holdout and do not rescue the family by adding indicators or tuning thresholds.
