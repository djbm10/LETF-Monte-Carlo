# Recovery execution requests

Run tsaustin former-member metadata + full price recovery. In parallel, build a hybrid HF+SEC Item1 discovery network: use the complete raw HF corpus first, then fetch from SEC only CIKs lacking a valid <=550-day Item1 at one or more 2009-2019 formation dates. Keep the frozen pair density, staleness rule, PIT S&P universe, and coverage gates unchanged.

Hybrid Item1 run requested from the committed exact collector. HF provides 10-K cache coverage; direct SEC fills only issuer/date gaps. Frozen 2009-2019 discovery firewall and network quality bars remain unchanged.

Curated-universe cache-reuse rerun requested. Reuse the preserved HF/direct-SEC filing caches, rebuild on the successful curated CIK membership intervals, and keep the 200/350 Item-1 gates unchanged.
