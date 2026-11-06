# LETF free-data research and evidence gates — 2026-11-06

Status: **Research infrastructure / not strategy-validation or investment advice.** This upgrade does not promote a strategy or loosen any previously sealed holdout.

## Evidence hierarchy

1. **Dated futures settlements (research-only replay):** `free_validation/futures_reconcile.py` replays position P&L from actual dated NQ or MNQ contract settlements, explicit rolls, estimated commissions/ticks, cash carry, integer contract sizing, initial/maintenance margin and forced liquidation. It rejects gaps in **held** contracts and uses QQQ's already-observed close to size NEXT-period exposure. **No official historical CME files are included**. Contract-level settlement-close fills and the margin fractions are approximations; they do not prove live execution.
2. **Independent QuantConnect implementation:** Paste `research/quantconnect_free_futures/main.py` into a Python Cloud backtest. It uses actual dated mapped NQ futures contracts, conservative previous-bar signals and the platform's futures brokerage/margin models. The `Futures.Indices` class and behavior should be checked in the current cloud environment; code has not been run in an authenticated account. US Futures history begins **May 2009** (per AlgoSeek dataset). **Do not download or redistribute licensed data without authorization.** Free Cloud is *not* the paid API and has no live nodes. Compare P&L and trade logs to local model, especially around rolls, gaps and margin calls; *do not assume agreement*.
3. **SEC fundamental history:** `free_validation/sec_pit.py` builds publicly available SEC XBRL snapshots using filing dates rather than period ends. `sec-public-pit.yml` is optional/manual and needs a repository secret `SEC_USER_AGENT` set to a genuine contact. This is free and is **not** CRSP/Compustat or a delisting-safe stock universe. Do not use it to open the sealed Bottleneck Winner holdout.
4. **Prospective freezing:** `free_validation/shadow_signals.py` records a monotone append-only JSON Lines signal journal after completed NYSE sessions. GitHub's `forward-signal-audit.yml` runs late evening Tue–Sat UTC. These are **candidate signals only**, not brokerage fills, and the daily candidate ETF is NOT a promise to rebalance the sector sleeve daily. Source hashes prevent a changing Yahoo adjusted-close history from rewriting old observations. No trading is performed. A full brokerage paper-trading ledger needs account access or a tradable exchange simulator.
5. **Risks:** `free_validation/risk_scenarios.py` applies transparent joint QQQ/futures price shocks (e.g., a crash), costs and alternative margin assumptions, then replays the fixed rule. These are synthetic *counterfactuals*, not actual historical data, and settlement-based shocks still understate intraday liquidation risk.

## How to run

```bash
pip install pandas numpy scipy pytest requests yfinance
pytest -q tests/test_free_data_validation.py
# This requires your OWN complete dated-contract file and QQQ adjusted closes; no synthetic futures chain fabricated:
PYTHONPATH=research python -c 'from free_validation.futures_reconcile import FuturesCosts, replay_contract_settlements; print("Ready for dated CME/approved contracts")'
# SEC data: set SEC_USER_AGENT locally or as a GitHub secret (required, real reply address)
PYTHONPATH=research python -m free_validation.run_sec_pit --cik 320193 --asof 2024-03-01 --tag Revenues
# Indicative next-session signals, appended only if data covers an actual closed market day
python research/free_validation/shadow_signals.py --ledger research/forward_signal_ledger.jsonl
```

## Still needed before a robust decision

- Official CME settlement licensing/entitlement or authorized, contract-by-contract third-party data. Validate NQ/MNQ expiry, settlement currency, multipliers, bid/ask and official margin revisions; do not call a Yahoo NQ continuous series dated-contract evidence.
- Independent vendor reconciliation and prospective broker/paper fills. QuantConnect cloud free historical backtest helps; free tier has no live nodes or CLI/API.
- True point-in-time CRSP/Compustat and survivorship-complete universe for stock-picking claims, or an approved research collaborator. SEC alone does not solve historical delistings.
- Pre-registered candidate universe, complete test ledger, nested or sealed forward tests, multiple-testing controls, and performance versus 35/0 + Treasury at equivalent drawdown/volatility.
- Severe opening gaps, liquidity/spread shock, broker margin hikes, forced liquidation and execution outages. Historic interest rates, options surface, and cross-asset correlation shifts require extra data; don't claim these are fully simulated yet.
- Live paper ledger (actual account or separate execution-quality simulator), risk limits, reviewer signed-off evidence gates.

## No-data / no-account honesty

- CI **passing** demonstrates code invariants, not investment alpha.
- SEC request does not work until `SEC_USER_AGENT` is set. The automated prospective journal begins on future completed sessions; prior historical dates must not be relabeled as forward evidence.
- QuantConnect requires user sign-in for account interaction and licensed cloud datasets cannot be silently copied to GitHub.
