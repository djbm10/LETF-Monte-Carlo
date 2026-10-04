# Confirmed-dip overlay disposition — 2026-10-04

Decision: KILL as a probability-adjusted-wealth enhancement to S9.

## Mechanism → Condition X → Outcome Y → Baseline → Falsifier → Test → Decision

Mechanism:
Sharp drawdowns can contain forced/liquidity selling and behavioral overshoot; waiting for a reversal confirmation may identify rebound phases where temporarily increasing exposure earns a recovery premium without taking full falling-knife risk.

Condition X:
Previously frozen confirmed-dip variants (including DIP5_cross20_b20_h60, DIP10_rebound3_b20_h10, DIP5_rebound5_b20_h60) layered on the S9 base.

Outcome Y:
Higher geometric wealth than S9_base while preserving or improving hostile-regime lower-tail CAGR and catastrophic-drawdown probability.

Baseline:
S9_base; NDX_1x reported as secondary context.

Falsifier:
If the dip overlay worsens worst-regime p10 CAGR and materially increases probability/severity of >90% drawdowns relative to S9_base, reject the overlay even if historical/baseline-world CAGR rises.

## Existing completed hostile test
Workflow: dip-hostile-3k.yml
Run: 35523071949
Commit: 95cc37b419e7e878dfb427c49c46af6b3879510d
Conclusion: SUCCESS
Monte Carlo sample: 3,000 paths per stress setting in the fast hostile sensitivity implementation.

40-year baseline/history-like world:
- S9_base median CAGR 18.36%, p10 9.88%, P(>90% DD) 0.93%
- DIP5_cross20_b20_h60 median 19.06%, p10 10.12%, P(>90% DD) 1.80%
- DIP10_rebound3_b20_h10 median 18.42%, p10 9.67%, P(>90% DD) 1.73%
- DIP5_rebound5_b20_h60 median 18.36%, p10 9.30%, P(>90% DD) 2.83%

40-year combined-hostile world:
- S9_base median CAGR -3.56%, p10 -9.01%, P(>90% DD) 70.50%, median DD -94.36%
- DIP5_cross20_b20_h60 median -4.06%, p10 -10.05%, P(>90% DD) 79.30%, median DD -96.06%
- DIP10_rebound3_b20_h10 median -4.34%, p10 -10.41%, P(>90% DD) 79.33%, median DD -96.27%
- DIP5_rebound5_b20_h60 median -5.29%, p10 -11.30%, P(>90% DD) 85.10%, median DD -97.28%

The same ordering persists at 20/40/50-year worst-regime rankings: the dip overlays are consistently below S9_base on lower-tail robustness.

## Decision
KILL the tested confirmed-dip leverage boosts as an addition to S9. Historical upside is insufficient because the overlay increases exactly the path/catastrophic risk the probability-adjusted objective is meant to penalize.

Do not rescue by searching new dip thresholds after this hostile result. A future dip hypothesis must be genuinely new, mechanism-distinct, and pre-frozen before testing.
