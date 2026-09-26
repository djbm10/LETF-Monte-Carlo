# Bottleneck discovery interpretation firewall

Frozen before the first successful full free local Bottleneck discovery surface.

## Discovery sample
- Train: 2009-01-01 through 2014-12-31.
- Validation: 2015-01-01 through 2019-12-31.
- 2020-2023 remains unopened until the discovery interpretation is recorded.
- The full frozen 15-cell surface is always retained; no cell is deleted after seeing returns.

## Comparators
- `bottleneck_core` is compared against `momentum` at the same top-N.
- `bottleneck_momentum` is compared against `momentum` at the same top-N.
- `bottleneck_full` is compared against `quality_momentum` at the same top-N.
- `quality_momentum` is descriptive context, not evidence for the bottleneck thesis by itself.

## Family-level advancement rule
A bottleneck family may advance to the holdout only when:
1. validation CAGR excess versus its comparator is positive for at least 2 of 3 top-N values;
2. the median validation CAGR excess across top-N is positive;
3. the result is not driven by a data-quality failure, missing quarter, stale-price pathology, or materially incomplete cross-section.

Risk-adjusted evidence is reported separately:
- median validation Sharpe excess versus comparator;
- median validation max-drawdown difference versus comparator.

Positive CAGR evidence with negative risk-adjusted evidence is labeled return-only/leverage-like evidence, not robust alpha.

## Holdout rule
If any bottleneck family advances, expose 2020-2023 for all 15 frozen cells in one run. Do not retune weights, top-N, costs, staleness rules, universe rules, or factor definitions between discovery and holdout.

If no bottleneck family advances, the holdout remains sealed and the free Stage-1 bottleneck thesis is recorded as unsupported in this large-cap proxy.
