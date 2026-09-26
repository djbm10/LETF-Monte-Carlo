# Free Stage-1 execution requests

Exact-head execution request after current CI passed. Run the corrected 162-cell LEAPS EOD real-chain replication, the sharded 2009-2023 S&P SEC/Item-1 input build, and the hybrid PIT S&P historical-price build. Bottleneck return discovery remains capped at 2019; do not expose 2020-2023 holdout in this request.

LEAPS diagnostics rerun requested after adding pre/post-2020 metrics and BIL-excess Sharpe. Strategy rules and frozen 162-cell grid are unchanged.

Compliant SEC shard rerun requested after cancelling the over-parallel build. Network strategy is capped at max-parallel=2 and 3 requests/second per shard; research definitions are unchanged.

Exact-head execution request: rerun corrected 162-cell LEAPS surface, rebuild sharded 2009-2023 SEC/network inputs, and rebuild the coverage-gated WIKI+Yahoo historical S&P price panel. Bottleneck returns remain limited to 2009-2019 discovery until interpretation is frozen.

Rerun gated curated Bottleneck discovery after classifying the SEC-documented 2009 Q1 FSD placeholder as structural-empty source coverage. No future-quarter backfill; all other failed quarters remain fatal. Holdout stays sealed.

Rerun gated Bottleneck discovery after two pre-return mechanical corrections: require full requested top-N breadth and replace obsolete absolute formation-row count with structural PIT coverage checks. Holdout remains sealed.

Rerun gated Bottleneck discovery after freezing the SEC 2009 Q2 structured-data phase-in rule. No future backfill; all mature formations from 2009-09-30 onward must clear the cross-sectional gate; full requested top-N breadth is required.
