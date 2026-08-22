# Clean 1926 LETF Walk-Forward Evaluation

Run date: 2026-08-21/22 UTC  
Seed: 20260821  
Monte Carlo paths: 10,000  
Historical source: Kenneth French daily factors, 1926-07-01 through 2026-06-30 (26,274 observations)  
Input SHA-256: `39f9ae1d0e9f575024bc23145980ac270cea508fb67e592578b3f4d65f36d006`

## Guardrails

- Pre-1950 fabricated QQQ/TLT series are **not scored**.
- Pre-1985 TQQQ proxies are **not scored**.
- The 1926 long-history universe is the real Fama-French U.S. market return plus RF, transformed into daily-reset 1x/1.5x/2x/2.5x/3x synthetic leverage.
- Serious simulation uses 10,000 paths and 5/10/20/30/40/50-year horizons.
- S9 grid: SMA 175/200/225; bull vol target 25/30/35/40%; bear vol target 0/5/10/15%.
- Walk-forward selection: trailing 20-year train, next 5-year test.
- Frozen benchmark S9 = 200 SMA / 35% bull / 12% bear; it is outside the optimization grid and was not selected from test results.

## Walk-forward historical result

Across 16 non-overlapping 5-year test folds (1946-2026):
- Dynamic walk-forward-selected S9: **21.05% CAGR**, **-45.9% max drawdown**.
- Frozen S9 200/35/12: **19.38% CAGR**, **-63.4% max drawdown**.
- 1x U.S. market proxy: **11.04% CAGR**, **-54.6% max drawdown**.
- Walk-forward selection beat frozen S9 in 10/16 folds and beat 1x in 14/16 folds.
- Frozen S9 beat 1x in 13/16 folds.
- Most frequently selected grid point: **225 SMA / 40% bull / 0% bear** (6/16 folds).

## 10,000-path Monte Carlo

| Strategy | Horizon | Median CAGR | 10th pct CAGR | P(beat 1x) | P(DD <= -90%) | Median max DD | Median max underwater |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x | 20y | 9.87% | 4.14% | — | 0.00% | -44.0% | 4.6y |
| 2x | 20y | 12.22% | 0.48% | 68.5% | 9.8% | -74.8% | 6.6y |
| 3x + 200 SMA -> cash | 20y | 15.21% | 2.49% | 78.3% | 4.2% | -71.2% | 6.5y |
| Frozen S9 200/35/12 | 20y | 15.96% | 5.57% | 89.5% | 0.14% | -57.8% | 5.4y |
| Modal S9 225/40/0 | 20y | 15.33% | 4.92% | 84.6% | 0.17% | -58.9% | 5.7y |
| 1x | 50y | 9.79% | 6.25% | — | 0.04% | -52.7% | 7.6y |
| 2x | 50y | 12.03% | 4.74% | 77.0% | 26.8% | -84.6% | 12.2y |
| 3x + 200 SMA -> cash | 50y | 15.39% | 7.00% | 88.5% | 14.4% | -80.5% | 11.2y |
| Frozen S9 200/35/12 | 50y | 15.97% | 9.33% | 97.6% | 0.75% | -66.8% | 8.7y |
| Modal S9 225/40/0 | 50y | 15.46% | 8.60% | 94.6% | 1.10% | -68.0% | 9.2y |

## Conclusion

The frozen **S9 200/35/12 survives** this test. The dynamic walk-forward selector has the best realized OOS historical path, but the fixed 200/35/12 rule is stronger than the modal 225/40/0 challenger across the 10,000-path Monte Carlo on median CAGR, 10th-percentile CAGR, probability of beating 1x, and tail drawdown risk.

This does **not** establish a guaranteed 16% future CAGR. The Monte Carlo is a 63-trading-day moving-block bootstrap of the historical U.S. market/RF path, so it preserves local dependence but can still underrepresent structural regime changes. Also, the 1926 benchmark is the Fama-French broad U.S. market, not literally SPY before SPY existed.

`max_recovery_years` in the CSV is best interpreted as maximum underwater duration observed inside each simulated horizon; unresolved drawdowns at the horizon are censored by the horizon end.
