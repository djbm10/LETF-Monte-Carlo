# Free local LEAPS execution requests

Run requested after schema probe confirmed real long-dated SPY/QQQ chains, positive bid/ask, OI, and Greeks. Execute all 162 frozen cells on the preserved end-of-day option dataset; do not tune the grid from results.

Final risk-path rerun requested after replacing intrinsic-only missing daily marks with current intrinsic plus last observed non-negative time value. Cash-flow exits with missing quotes remain intrinsic-only; parameter grid remains frozen.

Final integrity rerun requested after enforcing hard American-call no-arbitrage quote bounds: intrinsic <= usable call quote <= underlying. Impossible next-day entry quotes are rejected; exit bids are bounded by exercise value/underlying; daily marks are economically bounded. This is a data-quality correction prompted by impossible path jumps, not return tuning.
