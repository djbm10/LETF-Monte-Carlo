# Free local LEAPS run request

Run the full frozen 162-cell SPY/QQQ LEAPS grid using the preserved free EOD option-chain dataset and the lookahead-free next-close quote-side execution proxy.

## Audited rerun

Re-run the identical 162-cell EOD proxy after adding transaction-level selection/fill/exit audit rows and correcting a dormant collateral rollback path. The prior surface had zero rejected entries, so the rollback correction was not exercised in that run. Research parameters and execution proxy are unchanged.
