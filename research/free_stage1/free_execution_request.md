# Free Stage-1 execution requests

Exact-head execution request after current CI passed. Run the corrected 162-cell LEAPS EOD real-chain replication, the sharded 2009-2023 S&P SEC/Item-1 input build, and the hybrid PIT S&P historical-price build. Bottleneck return discovery remains capped at 2019; do not expose 2020-2023 holdout in this request.

LEAPS diagnostics rerun requested after adding pre/post-2020 metrics and BIL-excess Sharpe. Strategy rules and frozen 162-cell grid are unchanged.

Compliant SEC shard rerun requested after cancelling the over-parallel build. Network strategy is capped at max-parallel=2 and 3 requests/second per shard; research definitions are unchanged.
