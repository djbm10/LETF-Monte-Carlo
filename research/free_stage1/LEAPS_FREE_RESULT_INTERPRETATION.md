# Free LEAPS Stage-1 interpretation

Frozen after the corrected 162-cell free historical-chain surface was replicated
successfully on the exact-head workflow.

## Evidence status

Evidence label: `FREE_DISCOVERY_LOCAL_EOD_PROXY`.

The free dataset contains real historical SPY/QQQ option-chain bid/ask, open
interest, Greeks and underlying prices. The local executor preserves the frozen
delta/DTE/allocation/slippage grid and roll logic, but it cannot reproduce
QuantConnect's next-session option **opening** QuoteBar. Its fill proxy is
therefore completed close `t` selection -> real quote-side close `t+1`
execution. This is a material implementation difference and must remain explicit.

The corrected surface enforces:
- positive, non-crossed quotes;
- equity-call no-arbitrage bounds;
- OI >= 100;
- real next-session ask entry / bid exit;
- 0/5/10 bps added slippage;
- Interactive-Brokers-style low-volume option fees;
- BIL collateral sleeve;
- no-lookahead missing-mark treatment for path statistics.

The exact-head replication completed 162/162 frozen cells and reproduced the
previous corrected surface.

## Main finding

The free evidence does **not** support a claim that deep-ITM LEAPS are a
universal arbitrage or that they dominate the underlying without meaningful
additional risk.

At 10 bps added slippage:
- many LEAPS cells have higher CAGR than their underlying;
- the 100% premium-allocation cells achieve the largest CAGRs but routinely
  experience roughly 73%-98% maximum drawdowns;
- the economically more credible region is 25%-50% premium allocation;
- the strongest risk-aware evidence is concentrated in QQQ, not SPY.

A strict descriptive robustness screen was applied for interpretation only:
1. CAGR exceeds the same underlying in both pre-2020 and post-2020 subperiods;
2. maximum drawdown is no more than 10 percentage points worse than the
   underlying;
3. use the harshest frozen 10 bps slippage case.

Under that screen, QQQ has three qualifying cells and SPY has none.

| Underlying | Delta | Target DTE | Premium allocation | CAGR | Max DD | Pre-2020 CAGR | Post-2020 CAGR | Excess-BIL Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QQQ | 0.70 | 365 | 25% | 21.17% | -38.42% | 20.97% | 21.43% | 0.883 |
| QQQ | 0.90 | 548 | 50% | 23.65% | -40.88% | 23.95% | 23.26% | 0.837 |
| QQQ | 0.90 | 730 | 50% | 20.68% | -39.56% | 20.56% | 20.85% | 0.781 |

QQQ buy-and-hold over the same sample:
- CAGR: 19.59%
- max drawdown: -35.12%
- pre-2020 CAGR: 19.21%
- post-2020 CAGR: 20.09%

SPY buy-and-hold:
- CAGR: 14.78%
- max drawdown: -33.70%
- pre-2020 CAGR: 14.56%
- post-2020 CAGR: 15.08%

This screen was **not** part of the frozen parameter search and must not be used
as a new optimization target. It is a compact way to describe where the already
frozen surface shows return improvement without an extreme drawdown penalty.

## Extreme-allocation result

The highest-CAGR corrected cell is QQQ delta 0.70 / target DTE 548 / 100%
premium allocation. Its CAGR is about 57.6%, but its maximum drawdown is about
-89.1%. This is best interpreted as extreme leveraged-equity exposure, not a
practical low-risk anomaly.

The realized-trade audit indicates that the terminal compounding is not created
by missing daily marks. The strongest cells contain many profitable completed
deep-ITM call trades; however, 100% allocation is more concentrated and carries
catastrophic path risk.

## Robustness notes

At 10 bps:
- QQQ: 18/27 cells beat QQQ CAGR in both pre-2020 and post-2020 periods.
- SPY: 17/27 cells beat SPY CAGR in both subperiods.
- once the additional <=10 percentage-point drawdown condition is imposed,
  QQQ has 3 qualifying cells and SPY has 0.

Slippage sensitivity from 0 to 10 bps is modest relative to the allocation/
leverage effect; allocation is the dominant driver of both CAGR and drawdown.

A representative less-concentrated QQQ cell (delta 0.70, DTE 365, 25%
allocation, 10 bps) had 28 completed trades, about 82% profitable trades, a
median option return of roughly 45%, and the top three positive trades
contributed about 37% of positive realized P&L. The effect is therefore not
explained by a single isolated winner.

## What this does and does not establish

Supported by this free discovery layer:
- real historical-chain evidence that some moderate-allocation QQQ deep-ITM
  LEAPS configurations produced higher compounded returns than QQQ in both
  major subperiods, with a modestly worse drawdown;
- extreme premium allocation creates extreme drawdown and should not be
  confused with alpha;
- the result is sensitive to implementation/exposure choice rather than a
  universal options anomaly.

Not established:
- exact next-open executable returns;
- OptionMetrics/IvyDB replication;
- post-2025-12-16 option-chain evidence;
- live tradability at historical displayed size;
- a causal options-market mispricing explanation.

The next independent validation remains IvyDB US / OptionMetrics or another
institutional options database using the frozen parameter families.
