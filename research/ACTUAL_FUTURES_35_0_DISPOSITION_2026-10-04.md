# Frozen 35/0 actual-futures P&L replication — 2026-10-04

## Final implementation status: MONITOR

The previously promoted strategy family is **not falsified** by futures-derived P&L, but the evidence is not exchange-settlement/contract-chain grade.

### Frozen strategy
No strategy parameter was changed for this replication:
- signal: prior-day QQQ adjusted close vs 200-day SMA;
- risk estimator: prior-day 20-session realized volatility;
- target: 35% annualized portfolio volatility above trend, 0% below;
- exposure cap: 3x;
- same desired exposure used for both proxy and futures P&L.

### Common-period comparison, 2000-09-18 through 2026-10-02
| P&L source | CAGR | Terminal $1 | Max DD | Vol | Sharpe | 10y rolling p10 CAGR | 10y worst CAGR |
|---|---:|---:|---:|---:|---:|---:|---:|
| QQQ adjusted-return proxy | 17.60% | 67.43 | -44.86% | 30.96% | 0.680 | 5.76% | 1.34% |
| Yahoo NQ=F futures-derived continuous | 20.58% | 129.39 | -40.02% | 31.17% | 0.758 | 9.39% | 5.12% |

Both streams use the same average desired exposure (1.664x), 3x cap, and 22.63x notional-exposure turnover/year. Daily QQQ/NQ return correlation is 0.9784. NQ minus QQQ mean return difference is -0.54% annualized; tracking-difference volatility is 5.32% annualized.

### Live MNQ integer-sizing sensitivity, 2019-05 onward
This is an execution sensitivity surface, not a parameter search. It retains the previously used grid:
- starting capital: $25k / $50k / $100k / $250k;
- commission: $2.50 or $4.00 per contract side;
- slippage: 0 / 0.5 / 1.0 tick per side;
- margin constraint: 20% / 30% notional;
- exposure rebalance band: 15 / 30 / 45 percentage points;
- quarterly roll cost: two contract sides, five sessions before third-Friday expiry.

Across all execution cells:
- $25k: median CAGR 41.01%, worst 38.80%; worst max DD -32.48%.
- $50k: median CAGR 39.57%, worst 38.66%; worst max DD -31.64%.
- $100k: median CAGR 38.95%, worst 38.11%; worst max DD -31.79%.
- $250k: median CAGR 39.36%, worst 38.34%; worst max DD -31.23%.

These live-MNQ figures cover only the post-2019 period and therefore are not a long-regime validation.

### Why this is MONITOR, not IMPLEMENTATION-VALIDATED
1. Yahoo NQ=F/MNQ=F are actual futures-derived traded-price series, so this closes the prior ETF-return-as-P&L gap.
2. However Yahoo does not document the underlying dated-contract mapping, settlement field provenance, or roll/back-adjustment rule.
3. A bounded direct probe confirmed Yahoo does not expose expired NQZ24/NQH20/NQZ10 histories through the tested symbols.
4. Stooq's automated CSV endpoint was blocked by browser verification.
5. QuantConnect cannot currently be used from this repository because QC_USER_ID, QC_API_TOKEN and QC_ORGANIZATION_ID are all absent.
6. The repository already contains a correct CME DataMine contract-level settlement adapter with explicit no-look-ahead roll accounting; it cannot be executed until legitimate CME contract-level settlement exports are supplied.

### Decision
**MONITOR.** The 35/0 economic result survives a substantially stronger futures-derived return test and survives integer-MNQ execution sensitivity. It does not earn IMPLEMENTATION-VALIDATED status until an auditable dated-contract/settlement source is run through the existing explicit-roll adapter (or an equivalent documented mapped-contract source).

Runs:
- expired-contract/source probe: 37223530738
- QC credential probe: 37223636997
- frozen 35/0 futures-derived P&L replication: 37223853650

No retuning was performed.
