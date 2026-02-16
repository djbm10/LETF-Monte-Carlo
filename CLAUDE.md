# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a Leveraged ETF (LETF) strategy comparison and simulation engine.
The purpose of the project is to model, backtest, and analyze the behavior of leveraged ETFs under different market regimes, volatility environments, leverage ratios, and rebalancing assumptions.

Primary goals:

- Compare long-term performance of LETFs vs unlevered benchmarks
- Analyze volatility decay / path dependency
- Evaluate drawdown behavior and tail risks
- Test alternative leverage, hedging, and rebalancing rules
- Produce statistically and financially realistic simulations

This is a quantitative finance / research codebase, not a generic application.

## Core Principles

When modifying or generating code, prioritize:

- Financial correctness over syntactic elegance
- Numerical stability over micro-optimizations
- Deterministic reproducibility
- Explicit assumptions
- Transparency of logic

Avoid "black-box" logic or unexplained shortcuts.

## Domain Context (Important)

This project deals with:

- Leveraged ETF mechanics (daily reset leverage)
- Compounding and path dependency
- Volatility drag / convexity effects
- Monte Carlo simulation
- Backtesting with historical returns
- Statistical modeling of market regimes
- Tail risk and drawdown analysis

Assume the user cares about:

- Mathematical accuracy
- Bias avoidance
- Avoiding unrealistic assumptions
- Risk modeling realism

## Modeling Assumptions

Unless explicitly instructed otherwise:

- LETFs reset daily
- Leverage applies to daily returns, not cumulative returns
- Returns compound multiplicatively
- Volatility drag must be preserved
- Simulations must avoid look-ahead bias
- Backtests must not leak future information
- Time series operations must respect index alignment

If assumptions change, state them clearly.

## Numerical & Statistical Rules

Follow these guidelines:

- Avoid unstable cumulative operations
- Guard against divide-by-zero
- Handle NaNs explicitly
- Preserve dtype consistency
- Use vectorized operations when appropriate
- Avoid silently clipping extreme values

Treat randomness carefully:

- Use local RNG objects
- Preserve seeds for reproducibility
- Do not rely on global RNG state

Monte Carlo outputs must be reproducible given a seed.

## Risk & Edge Case Awareness

Always consider:

- Extreme volatility spikes
- Large drawdowns
- Flat / zero-return periods
- Missing data
- Regime transitions
- Negative prices (must never occur)
- Leverage blow-ups
- Numerical overflow / underflow

If a model can fail catastrophically, flag it.

## Code Modification Guidelines

When editing code:

Prefer:
- Minimal, targeted changes
- Backward compatibility when possible
- Clear variable names
- Explicit math
- Comments explaining financial logic
- Preserving interfaces unless instructed

Avoid:
- Large refactors unless requested
- Silent behavioral changes
- Changing leverage definitions without warning
- Removing safeguards
- Over-engineering

## Explanation Style

When explaining code:

- Explain financial intuition
- Explain mathematical mechanics
- Explain numerical implications
- Highlight risks / limitations
- Identify assumptions

Do not give shallow explanations.

## When Debugging

When investigating errors:

- Identify root cause, not just symptom
- Check numerical instability
- Check array alignment / indexing
- Check financial logic validity
- Check regime or simulation assumptions

Explain why the bug occurs.

## When Refactoring

Ensure:

- Financial outputs remain consistent (unless intended)
- Statistical properties preserved
- No hidden logic changes
- Tests updated if behavior changes

State impact of refactor.

## Backtesting Rules

Backtests must:

- Avoid look-ahead bias
- Use correct return alignment
- Handle missing dates
- Preserve compounding logic
- Model leverage realistically
- Reflect transaction costs if present
- Avoid survivorship bias where applicable

Flag unrealistic assumptions.

## Simulation Rules

Monte Carlo / simulations must:

- Clearly state distributional assumptions
- Preserve volatility clustering if modeled
- Avoid unrealistic IID simplifications unless specified
- Preserve leverage path effects
- Allow reproducibility via seed

Explain statistical limitations.

## Performance Guidelines

Optimize only when:

- A bottleneck is identified
- Numerical correctness preserved
- Readability not severely degraded

Never sacrifice correctness for speed silently.

## Testing Philosophy

Tests should cover:

- Edge cases
- Numerical stability
- Extreme volatility scenarios
- Drawdown behavior
- Determinism (seed reproducibility)

Generated tests should validate financial logic, not just execution.

## Communication Expectations

When suggesting changes:

- Explain reasoning
- Identify trade-offs
- Mention risks
- State assumption changes
- Be precise and technically rigorous

## Critical Warning

This project models financial instruments with nonlinear risk.

Incorrect logic may produce:

- Misleading performance results
- Unrealistic risk profiles
- Invalid research conclusions

If uncertain about financial correctness, ask for clarification.

## Running the Code

```bash
# Quick validation (10 sims, 10 years, ~3 seconds)
python quick_test.py

# Full analysis (50 sims, multiple horizons)
python LETF34_analysis.py

# Non-interactive mode (skips prompts, uses defaults)
LETF_NON_INTERACTIVE=1 python quick_test.py

# Direct package entry
python -c "from letf import run; run()"
```

There is no formal test suite, linter, or build system. `quick_test.py` is the primary validation tool — it runs the tax engine's 6 golden tests and executes all strategies.

## Dependencies

```bash
pip install numpy pandas scipy yfinance pandas-datareader tqdm
# Optional performance:
pip install numba    # 10-20x JIT speedup on hot paths
pip install arch     # Professional GARCH estimation
pip install joblib   # Better parallel processing for NumPy arrays
```

## Architecture

### Data Flow

1. **Data acquisition** (`data.py`) — Fama-French (1926–1950) + yfinance (1950–2025) for SPY, QQQ, TLT, VIX, IRX
2. **Calibration** (`calibration.py`) — Fits regime model, DCC-GARCH, funding spreads, tracking residuals; results cached to `corrected_cache_v8/`
3. **Simulation** (`simulation/engine.py`, `mc_runner.py`) — Parallel Monte Carlo with regime path generation, joint return sampling, LETF return calculation, and strategy execution
4. **Tax** (`tax/`) — Capital gains netting (IRC §1211/1212), wash sale rules, lot selection (FIFO/LIFO/HIFO/MINTAX), federal+state brackets
5. **Reporting** (`reporting.py`, `historical.py`) — Percentile statistics, tax impact, historical validation vs real TQQQ data

### Simulation Engines

Two modes controlled by `SIM_ENGINE_MODE` in `config.py`:
- **`institutional_v1`** (default): Multivariate Student-t with DCC-GARCH dynamics
- **`legacy_hybrid`**: 80% block bootstrap from historical data + 20% Student-t noise

### LETF Return Formula

```python
etf_return = leverage * underlying_return - expense_ratio/252 - daily_borrow_cost + tracking_error
```
Volatility drag emerges from geometric compounding — it is **not** subtracted explicitly. Tracking error is **additive** (not multiplicative), AR(1) with Student-t(df=5) innovations, amplified by VIX/regime/move size.

### Key Configuration

All tunable parameters live in `letf/config.py`: simulation engine mode, number of simulations, worker count, variance reduction flags, bootstrap weights, random start settings.

### Strategies

19 strategies in `strategy.py` numbered S1–S19: buy-and-hold benchmarks (TQQQ/SPY/SSO), SMA crossover variants, TQQQ/TMF portfolio rebalancing, volatility targeting (fixed/adaptive/downside/velocity), and advanced strategies (meta-ensemble, crisis alpha, tail risk optimizer).

## Important Conventions

- **Caching**: Calibrated models cached in `corrected_cache_v8/` with date-range filenames. Delete cache files to force recalibration.
- **Multiprocessing safety**: All entry points require `if __name__ == '__main__':` guards. Uses joblib (preferred) or ProcessPoolExecutor.
- **Synthetic data markers**: Historical data includes `<Asset>_IsSynthetic` columns for pre-inception LETF data.
- **Graceful degradation**: Missing optional libraries (numba, arch, cupy) are caught at import and the system falls back to pure NumPy.
- **Debug mode**: Set `DEBUG = True` in `config.py` for verbose error output.
