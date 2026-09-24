# Live SEC Stage 1 smoke request

Purpose: trigger the frozen free-PIT workflow's real SEC fundamentals and Item 1 acquisition smoke tests before any return analysis.

This file changes no model definition, factor weight, universe filter, portfolio size, cost assumption, sample window, LEAPS parameter, or holdout boundary.

Requested: 2026-09-23.


## Revalidation

Re-run requested after preserving table text in Item 1 extraction to close known coverage holes in modern inline-XBRL 10-Ks (Amazon/Verizon/Exxon in the prior 18-name smoke). No research specification or portfolio parameter changed.


## Boundary-only table preservation

Re-run requested after narrowing the Item 1 parser fix: preserve only table rows containing Item 1/1A/1B/2 boundaries, while continuing to discard other table text from the TF-IDF corpus. This is a pre-return data-extraction correction only; no research or portfolio parameter changed.


## SEC-first final pre-return revalidation

Re-run requested after the SEC formation-feature builder, SEC-first Bottleneck scoring integration, exact 252-observation momentum rule, and corrected boundary-row regex all passed synthetic CI. This validates the live acquisition layer on the same code generation before any backtest returns are inspected.


## Formation-feature live gate

Re-run requested after extending the live FSD smoke to two Q4 filing vintages and asserting real non-null SEC YoY revenue-growth features in the compact formation panel.


## Final strict-period gate

Re-run requested after enforcing exact SEC flow-period comparability (10-Q qtrs=1, annual qtrs=4), annualizing quarterly revenue for valuation, and fixing 252-day price-history warm-up semantics. This remains pre-return validation; no backtest return surface has been inspected.
