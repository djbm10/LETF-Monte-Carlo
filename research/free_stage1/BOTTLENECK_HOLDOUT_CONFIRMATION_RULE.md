# Bottleneck holdout confirmation rule

Frozen before any 2020-2023 Bottleneck strategy return is exposed.

## Preconditions
The 2020-2023 holdout may be opened only after:
1. the 2009-2019 discovery family advancement rule has been applied without tuning;
2. at least one bottleneck family advances;
3. the advancing conclusion is reproduced using the authoritative direct-SEC Item 1 parser/network;
4. the sealed 2020-2023 issuer-identity and price-coverage preflight passes the frozen 97% gate;
5. the direct-SEC holdout Item 1 network passes the same minimum/median network coverage gates;
6. no model weight, top-N, transaction cost, staleness rule, universe rule, factor definition, or comparator changes.

## Single holdout run
Expose 2020-01-01 through 2023-12-31 for all 15 frozen cells in one run:
- five model families;
- top-N 10, 20, 40;
- 15 bps one-way trading drag.

Do not delete weak cells after seeing the holdout.

## Family confirmation rule
Evaluate only bottleneck families that advanced in discovery.

A family is confirmed in the holdout when:
1. holdout CAGR excess versus its frozen comparator is positive for at least 2 of 3 top-N values; and
2. median holdout CAGR excess across top-N is positive.

Comparators remain:
- bottleneck_core vs momentum;
- bottleneck_momentum vs momentum;
- bottleneck_full vs quality_momentum.

## Risk-adjusted interpretation
Report separately:
- median holdout Sharpe excess versus comparator;
- median holdout max-drawdown difference versus comparator.

If CAGR confirmation passes but median Sharpe excess is negative, label the result return-only / risk-amplifying rather than robust risk-adjusted alpha.

## Overall conclusion
- If at least one direct-SEC discovery-advanced bottleneck family satisfies the holdout confirmation rule, Stage-1 provides out-of-sample support for the Bottleneck thesis in the free PIT S&P large-cap proxy.
- If none satisfy it, Stage-1 does not provide out-of-sample support in this proxy.
- This remains a large-cap proxy, not the frozen full-US licensed-data replication. Stage 2 is still required for the stronger CRSP/Compustat/TNIC claim.
