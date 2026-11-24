# Reproduce the integration

Use branch `integrate/financial-engine-2026-11-23`, separate from `main`. Python 3.14.4 was used locally. No optional Numba/arch/GPU path is certified. `requirements-audit.lock` pins installed dependencies, including pandas 2.3.3, NumPy 2.3.5, exchange_calendars 4.11.3, PyYAML 6.0.3, pyarrow 23.0.1 and yfinance 0.2.66.

## Offline tests and preserved research checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.lock
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_workflows.py
.venv/bin/python research/clean_1926_walkforward.py --self-test
.venv/bin/python research/bottleneck_winner_selftest.py
.venv/bin/python research/optionmetrics_leaps_backtest.py \
  --input research/fixtures/optionmetrics_public_sample_20050915.csv \
  --out audit-output/new-option-smoke --smoke-only
.venv/bin/python -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
.venv/bin/python scripts/run_accuracy_audit.py --output audit-output/new-mechanics
```

The last command additionally needs the preserved baseline Git object `b967b2b498c261a2f551cfde9ab1621e5c489c2d`. A shallow clone of only the integration branch may not contain it; fetch `origin audit/financial-accuracy-2026-11-23` first. Its extracted original pure function is used for the before/after diagnostic; no historical evaluation is optimized or overwritten. Legacy `quick_test.py` is expected to stop at its unverified-proxy gate; T20 remains a recorded failure; T27 also fails at the unsupported legacy crisis-damping argument. Passing pytest is not the same as passing these model-confidence gates.

The original main baseline was tested from `/tmp/letf-main-baseline` with:

```bash
/home/djmann/projects/letf-monte-carlo/.venv/bin/python -m pytest \
  -c /dev/null --rootdir . tests -q
```

That explicit isolated configuration matters: a nested worktree without its own pytest configuration can accidentally import its parent audit checkout. Reported 26/124/152 counts use the intended code roots.

## Real stock replay and scheduled flows

The exact local input bundle is:

`/home/djmann/projects/letf-monte-carlo/.worktrees/financial-integration/audit-output/account-demo-inputs`

From the integration worktree, using the existing workspace interpreter if preferred:

```bash
/home/djmann/projects/letf-monte-carlo/.venv/bin/python -m letf.account_cli compare \
  --bundle audit-output/account-demo-inputs --state MA \
  --output audit-output/new-account-replay
/home/djmann/projects/letf-monte-carlo/.venv/bin/python -m letf.account_cli compare \
  --bundle audit-output/account-demo-inputs --state MA \
  --flows docs/integration-2026-11-24/demo-flows.json \
  --output audit-output/new-flow-replay
```

The CLI writes summaries, both account fill journals, paired pre-tax journals, full event lists and daily cash/positions/NAV reconciliations. It verifies input checksums and account-neutral pre-tax fills. Public APIs are `letf.account_cli.compare_accounts`, `letf.account_backtest.run_account_backtest`, `PortfolioLedger` and `RothHistory`; account configuration is not a strategy identifier. Linked-account executions can be supplied to `run_account_backtest` for owner-wide wash matching.

For a fresh independent **different vintage** without credentials:

```bash
.venv/bin/python scripts/capture_account_demo.py --output audit-output/new-stock-inputs
.venv/bin/python -m letf.account_cli compare \
  --bundle audit-output/new-stock-inputs --state MA --output audit-output/new-stock-results
```

A revised vendor capture may change observations and hashes. It is not bitwise reproduction of the frozen result. Original raw vendor inputs are intentionally local/ignored because redistribution rights were not verified. The committed source manifests and derived summaries preserve exact hashes, but someone with a fresh clone still needs the retained source bundle or a separately labeled fresh capture. Tests use self-contained independent fixtures and need neither network nor paid data.

## Real ETF replay from official tax records

Exact local bundles: `audit-output/observed-etf-bundles-issuer/TQQQ` and `.../QLD`.

```bash
.venv/bin/python -m letf.account_cli compare \
  --bundle audit-output/observed-etf-bundles-issuer/TQQQ \
  --strategy buy_hold --state FEDERAL_ONLY \
  --start 2024-01-02 --end 2024-12-31 --output audit-output/new-TQQQ-replay
.venv/bin/python -m letf.account_cli compare \
  --bundle audit-output/observed-etf-bundles-issuer/QLD \
  --strategy buy_hold --state FEDERAL_ONLY \
  --start 2024-01-02 --end 2024-12-31 --output audit-output/new-QLD-replay
```

To rebuild a new bundle from original daily snapshots and fresh issuer captures, create a new issuer directory and capture these public URLs (no purchase/account is required):

```bash
mkdir audit-output/new-issuer
curl -fLsS https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_2024.xlsx \
  -o audit-output/new-issuer/ici_primary_2024.xlsx
curl -fLsS 'https://www.proshares.com/api/distributionsummary/?fund=TQQQ&year=2024' \
  -o audit-output/new-issuer/letf-TQQQ-issuer-2024.json
curl -fLsS 'https://www.proshares.com/api/distributionsummary/?fund=QLD&year=2024' \
  -o audit-output/new-issuer/letf-QLD-issuer-2024.json
.venv/bin/python scripts/build_observed_etf_bundles.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --tax-supplement audit-output/new-issuer/ici_primary_2024.xlsx \
  --issuer-distributions-dir audit-output/new-issuer \
  --output audit-output/new-etf-bundles
```

Original local issuer captures are `/tmp/letf-ici-primary-2024.xlsx`, `/tmp/letf-TQQQ-issuer-2024.json`, `/tmp/letf-QLD-issuer-2024.json`; copies with checksums are retained inside the completed local bundles. Do not substitute current split-restated distributions for original nominal amounts. Current vendor payout rounding was rejected; exact dated issuer amounts are used. Corporate action and distribution coverage is limited to the stated 2024 account period; widening dates fails. MA/state exemption treatment and extra tax components are not authorized by this bundle. SSO/UPRO builder errors are intentional capability gates.

## Frozen reconstruction

```bash
.venv/bin/python scripts/validate_daily_reconstruction.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --output audit-output/new-reconstruction/report.json
```

This re-evaluates the original captured bytes and unchanged thresholds; it does not train on 2020–2025, overwrite prior proxies or change sealed strategy holdouts. New captures use `scripts/capture_daily_audit_inputs.py --output audit-output/new-daily-inputs` and must retain their own vintage. Three numerical passes and two CAGR failures do not override the independent provenance gate.

## Canonical paired scenarios

```bash
.venv/bin/python -m letf.account_scenarios \
  --bundle audit-output/account-demo-inputs \
  --paths 2 --horizons 5 10 20 30 40 --seed 20261124 \
  --freeze-tax-year 2026 --output audit-output/new-canonical-scenarios.json
```

Two paths are a reproducibility/integration demonstration, not an adequate forecast sample. Frozen future taxes, no future stock distributions, calendar projection, conditional bootstrap intervals and missing joint macro/closure models are explicit limitations. The runner refuses ETF/dividend scenarios without joint ex-price/distribution/tax-character paths and refuses unknown frozen tax tables. It calls the **same** daily account engine as historical replay; it does not interpolate tax drag from a separate vectorized ranking.

## Futures/options and CI

`tests/test_research_integration.py` shows canonical dated futures using the existing `FuturesCosts`, `target_exposure`, `validate_contracts` and roll selector with supplied `FuturesContract` definitions. Generic Roth rejects futures. The controlled comparison to the independent research settlement replay uses zero trading costs to isolate P&L/roll identity; separate book tests independently check actual tick, commissions, gap debt and margin changes. No historical tape is implied by the fixture.

The original QuantConnect prototype is unchanged and compile-checked only; its hosted brokerage engine requires a Lean/QC environment and appropriate data. The original network-heavy workflow's TNIC downloads and continuous/discrete historical ranking steps were not used to bypass data-confidence gates or overwrite historical outputs. Repaired workflow validity, offline self-tests and pinned regression CI are separate checks.

Inspect the final branch's GitHub Actions run for `Financial mechanics regression checks`. Record its actual conclusion; a local test log is not remote CI evidence. No force push, main overwrite or automatic PR merge is part of these commands.
