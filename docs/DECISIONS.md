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
