# Engineering decisions

2026-11-25:

1. Preserve `PortfolioLedger` / `run_account_backtest` as the sole account cash,
   lot, settlement, distribution and regular investment-tax implementation.
   Keep `FuturesBook` / `LongOptionBook` attached to that ledger. Legacy research
   calculators remain historical artifacts, never an implicit production route.
2. Build adapters around existing source snapshots and forecast registries.
   Never duplicate, relabel, revise or fit against sealed forecast arrays.
3. Standardize new result evidence labels at the laboratory boundary without
   changing archived reports' legacy labels. Observation is not execution proof.
4. Historical taxes use enacted dated tables only. Modern/future-law scenarios
   must explicitly choose a frozen year; no literal pre-1998 Roth history.
5. Extinction alone is not proof of an IRC165(g) worthless-security tax event.
   Existing 60 blocked cases stay blocked without independently reviewed facts.
6. Prioritize reproducible integration, source integrity and causal accounting
   over more strategies, model complexity or additional million-path studies.
7. The prior local archives are private with restrictive filesystem permissions;
   do not assert cryptographic encryption or off-host survival without evidence.
   The user's baseline description is not a new storage authorization.


- Preserve the original T20/T27 diagnostics. T27 requests a removed crisis-tail
  damping API; adding tail suppression to satisfy its score would not establish
  defensible crash behavior. The legacy sampler is not the canonical macro model.
- Apply original .02 log-quantile/.025 probability-half-width precision goals to
  the full-account study, with all planned looks accounted for; never transfer
  unit-fund accelerator precision to tax/flow/rebalance accounts.
- Retain the earliest same-kind/window forecast within one immutable implementation
  cohort. Later revised forecasts remain archived and cannot be cherry-picked.
- Current refresh dates do not create old publication vintages. No new training
  fingerprint means no replacement model, even if source capture timestamps change.
- Cboe public sample README says demonstration/subset/representative; label the
  sample SOFTWARE_FIXTURE rather than asserting observed executable history.
- Counterfactual CPI uses the month containing the actual initial close, not
  the following first-return month; monthly CPI is still an explicit proxy.
- Zero issuer rows are usable only when tax components, API (if present) and
  vendor cash all affirm zero. QLD2023 fails this, SSO2023 passes. The real2022
  source is binary XLS; a guessed XLSX HTTP404 was not evidence of unavailable data.
- All raw source/ensemble/backup storage stays private local. Hosted optional
  candidate jobs need separately approved private storage; ordinary CI uses fixtures.


- Resolve HFEA’s unhedged RIC holding gap with the actual46-day/91-day rule,
  effective2021 onward without old elections. Keep it distinct from QDI61/121,
  preserve receipt tax year/cash, and verify NIIT/MA bases independently.
- Stock splits restate pending wash replacement units, not original sale shares,
  basis/proceeds or dollar losses. Split itself is not a purchase. Unknown alias
  share conversions and OCC collateral changes remain blocked.
- Original VIX/IRX signals are captured and hashed; IRX/100/252 is preserved,
  avoiding a French monthly-return lookahead or silent DFF replacement.
- The common2025 cohort is now21/21, but uses explicit daily override/cold warmup
  and current-vintage observations. It is retrospective conformance, not a newly
  untouched strategy test or a release of performance rankings.


Final affirmative-evidence guards b3c4e1f reject strings and integers as observed-quote,
Section1256, historical-availability and reconstruction-provenance evidence. Independent
fixture plus reconstruction/source suites:20 passed. Full regression366 passed,
four inherited warnings,65.50s;97 workflow structure checks and supported compile pass.
No reconstruction, tail, legacy or forecast-approval threshold was changed.
