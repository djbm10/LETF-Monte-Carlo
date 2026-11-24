# Conditional multivariate sector Monte Carlo

## Purpose

This is a **research-only, outcome-neutral** scenario engine comparing frozen Nasdaq **35/0** with **80% 35/0 + 20% risk-controlled actual 2× sector ETFs**. It is not a trading recommendation, a verified dated-futures replication, or an independent holdout. It does not tune either strategy based on simulated performance.

The module `letf.simulation.sector_regime` extends the existing `letf/simulation/bootstrap.py`, `engine.py`, and `random_start.py` concepts to a *joint sector-specific return panel*: SPY, QQQ, ten unleveraged sector signal ETFs, and the corresponding ten actual 2× sector ETF return histories. It is deliberately isolated from the current portfolio engine until exchange-grade futures execution is validated.

## Frozen policies

- Nasdaq: previous close relative to 200-day SMA, prior 20-day realized QQQ volatility, 35% annual volatility target, 3× cap, 0% exposure below trend. The QQQ-exposure model earns Treasury proxy on residual cash and includes cost/financing sensitivity. **This is not CME settlement P&L.**
- Sector: SPY 200-day trend gate, highest 63-session unleveraged sector momentum, actual leveraged ETF 20-day vol targeting to 35% with 1× limit, 21-session selection, with trading costs.
- Sleeve: 80/20, rebalanced every 21 simulated sessions, otherwise drifts.

All features use previously available closing prices. Executable return proxy is open-to-next-open, including its known timing approximation. All strategy signals are recalculated on the **jointly simulated** market path, not sampled from pre-calculated strategy performance. The 21-session "month" is a numerical approximation, not a true exchange calendar month.

## Scenario dimensions

- Starting conditions: current as-of regime, random historical regime, forced high/low, persistent high.
- Regime generator: empirical *semi-Markov* duration spells (sampled independently of block length), or historical monthly-transition *block Markov*.
- Correlated historical return blocks: 21/63/126 days by default; whole rows are synchronized across all instruments.
- Simulation horizons and paired paths: 3/10/20 years by default; thousands of paths are permitted but cannot create independent historical observations.
- Sensitivities: crisis-duration multiplier, financing spread, trading costs, and initial random seed.

## Run locally

```bash
pip install numpy pandas scipy numba pytest yfinance
PYTHONPATH=. python -m pytest -q tests/test_sector_regime.py
# CI-only smoke using fictional market prices (NOT investment evidence)
PYTHONPATH=. python research/run_conditional_sector_mc.py --synthetic-smoke --years 1 --blocks 21 --modes current --transition-modes semi_markov --paths 5
# Real historical data from current provider, with a fully closed explicit as-of session:
PYTHONPATH=. python research/run_conditional_sector_mc.py --fetch-yahoo --asof 2026-11-05 --years 3,10,20 --blocks 21,63,126 --paths 1000 --output results/conditional_sector_research
# Or supply reproducible existing input CSVs:
PYTHONPATH=. python research/run_conditional_sector_mc.py --close prices_close.csv --open prices_open.csv --vix vix.csv --asof 2026-11-05 --years 3,10 --blocks 21,63,126 --paths 1000
```

The CSV price inputs must have dates in the first column, and adjusted price columns for **QQQ, SPY, ^IRX, XLK/ROM, SOXX/USD, XLF/UYG, XLV/RXL, XLE/DIG, XLY/UCC, XLB/UYM, XLI/UXI, XLP/UGE, VNQ/URE**. The VIX input needs a `^VIX` column. The generator fails closed on missing current anchor, duplicated dates, missing instruments, too many missing price days, and inadequate high/low regime samples; it does not fill missing returns with zero.

`--max-scenarios` defaults to 150 so expensive accidental combinatorial jobs fail quickly. GitHub Actions exposes a validated smoke workflow on push and an **explicit manual dispatch** for larger historical runs, with downloadable results, input data and machine-readable provenance in workflow artifacts.

## Research quality gates before strategy promotion

1. Verify **official CME dated-contract settlement and actual roll accounting** for frozen 35/0. QQQ excess-return linearization is a proxy, not implementable futures P&L.
2. Obtain a *survivorship-complete* history of leveraged sectors and record corporate actions, delistings and actual spreads. The current data begin when the required ETFs are all traded; no pre-inception product histories are invented.
3. Validate matched-signal dates versus tradable next-open fills and margin liquidation during crash openings. Costs, borrowing, taxation and financing are approximate.
4. Use **independent historical regimes or forward paper trading** after freezing the code; repeated block shuffles of the same prices are stress tests, not new external evidence.
5. Perform model uncertainty analyses including alternate providers, transition models, historical block sizes, underlying joint distributions, crisis correlations, and simultaneous regime selection without fitting to returns.
6. Pre-register deployment targets: e.g., expected log-return improvement after costs, drawdown reduction, probability of losing acquisition capital, maximum funding needs, and turnover constraints.

## Model changes from November 6 research script

This implementation adds import-safe library design; explicit command-line inputs; parameter grids constrained to **market scenarios** (frozen strategy rules unchanged); independent empirical semi-Markov dwell durations; deterministic paired seeds; correct common asset row sampling; fail-closed data validation; optional Yahoo historical ingestion; data hashes/manifest; and offline regression tests. It may not replicate earlier research results exactly because the regime generator is improved and a positive financing spread is now explicitly modeled. Any result differences are a **model revision**, not proof that earlier statistical probabilities were wrong.
