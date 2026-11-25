# Reproduction

Use the development branch/worktree, not the original user's working checkout.
All output directories below must be **new**; repeated commands need new names.
No command places orders, buys data or merges a branch.

## Environment and checks

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/macro-learning
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
$LETF_PY -m pytest -q
$LETF_PY scripts/check_workflows.py
$LETF_PY -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
$LETF_PY scripts/run_accuracy_audit.py --output audit-output/new-mechanics-run
```

For a clean installation, check out `develop/macro-learning-2026-11-24`, create a
Python 3.14 virtual environment and install `requirements-audit.lock`. The lock
is unchanged from the integration foundation. Local tests used Python 3.14.4;
hosted CI used 3.14.10. Set `LETF_PY` to that environment's interpreter.

Raw licensed/vendor inputs are deliberately not embedded in Git. Exact numerical
reproduction requires the retained checksummed local snapshots. Fresh downloads
are **new source vintages**, not guaranteed replicas. The committed
[manifest](evidence/reproduction-manifest.json), [catalog](evidence/source-catalog.json)
and [fund manifests](evidence/fund-source-manifests.json) identify the reported
inputs. A changed/corrupt source must fail hash verification, not be made to fit
the old report.

## Capture and inspect public inputs

```bash
$LETF_PY scripts/capture_macro_sources.py --output audit-output/new-macro-sources
$LETF_PY -m letf.macro.cli catalog --sources audit-output/new-macro-sources
$LETF_PY scripts/capture_daily_audit_inputs.py --output audit-output/new-fund-prices
$LETF_PY scripts/capture_issuer_tax_sources.py --output audit-output/new-issuer-tax
```

Macro capture records failed historical vintage requests individually and does
not interpolate them. The source catalog is mandatory to inspect. Issuer capture
verifies requested 2024/25 schemas and fails incomplete captures. Its unversioned
State Street URL may later serve another year; that is an explicit schema/year
failure. For offline byte reuse, add `--offline-cache` pointing at the retained
issuer directory; it labels archival reuse rather than inventing a network date.

The already retained macro bundle is `audit-output/macro-sources-v4`; v2 is the
original frozen-study input. v4 only corrects/enriches metadata without changing
observations. The normalization is reproducible with:

```bash
$LETF_PY scripts/normalize_macro_catalog.py \
  --sources audit-output/macro-sources-v2 --output audit-output/new-normalized-catalog
```

## Real daily fund comparisons

In this workspace the original quote snapshot is at the absolute path below.
Issuer sources are archived at `audit-output/issuer-tax-sources-v1`.

```bash
$LETF_PY scripts/build_multiyear_fund_bundle.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --issuer-dir audit-output/issuer-tax-sources-v1 \
  --macro-sources audit-output/macro-sources-v4 \
  --symbol TQQQ --output audit-output/new-TQQQ-inputs
$LETF_PY -m letf.account_cli compare \
  --bundle audit-output/new-TQQQ-inputs --strategy buy_hold \
  --start 2024-01-02 --end 2025-12-31 --initial-capital 100000 \
  --ordinary-income 100000 --filing-status single --state FEDERAL_ONLY \
  --output audit-output/new-TQQQ-results
```

Expected net wealth with the reported snapshots: taxable $199,462.76951940503;
Roth $218,422.98633422382. Reconciliations, events, fills and tax calculations are
written alongside `summary.json`. QLD/SSO/UPRO use the same commands with their
symbols. QQQ/TMF use `--buy-hold-initial-only` on the compare command to reproduce
their explicitly different cash-distribution policy. No unsupported strategy
ranking should mix those policies.

CLI cash flows accept `--flows path.json`; Roth flows also require
`--roth-history path.json` containing owner birth date, first tax year and aggregate
basis. These are real inputs, not inferred from the starting balance. The core
API accepts arrays of `Security`, classified `Distribution`, dated split actions,
allocation callback, account profile and explicit tax/payment inputs.
`LegacyAllocator('S12', version='causal_cash_v2')` is callable through that same
API; it is not an independently certified historical strategy.

## Daily counterfactuals and preserved ETF failures

```bash
$LETF_PY -m letf.macro.cli counterfactual \
  --sources audit-output/macro-sources-v4 --start 1930-01-01 \
  --output audit-output/new-1930-counterfactual
$LETF_PY scripts/validate_daily_reconstruction.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --output audit-output/new-reconstruction/report.json
$LETF_PY scripts/diagnose_reconstruction.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --frozen-report audit-output/new-reconstruction/report.json \
  --output audit-output/new-reconstruction/diagnostics.json
```

The last two commands are diagnostic and legitimately report failed gates. They
do not optimize parameters or alter frozen thresholds. The counterfactual report
includes fee/spread sensitivity, named historical eras and overlapping long
holding windows; source daily CSVs remain local.

## Exact frozen walk-forward reproduction

The reported historical study uses **dd460ae**, not the later exploratory version.
An isolated repetition already produced exactly identical 26 forecast IDs,
evaluation records and decisions. To reproduce again, choose unused paths:

```bash
git worktree add --detach /tmp/letf-frozen-macro-source dd460ae
cd /tmp/letf-frozen-macro-source
$LETF_PY -m letf.macro.cli walkforward \
  --sources /home/djmann/projects/letf-monte-carlo/.worktrees/macro-learning/audit-output/macro-sources-v2 \
  --registry /tmp/new-frozen-macro-registry --output /tmp/new-frozen-macro-results
```

The registered 2,000 paths are default. New runs need fresh registries: reusing
consumed selection intervals is intentionally rejected. Repetition establishes
numerical reproducibility and is not additional independent market evidence.
An explicit `--paths` reduction is diagnostic and fails the registered path gate.

## Future account scenarios

Return to the development worktree. This is an execution demonstration, with
too few paths for credible probability estimation:

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/macro-learning
$LETF_PY -m letf.macro.cli explore \
  --sources audit-output/macro-sources-v4 --calibration-end 2026-09-30 \
  --output audit-output/new-retirement --paths 2 --horizons 5 10 20 30 40 \
  --initial-capital 10000 --annual-contribution 1200 \
  --roth-birth-date 1980-01-01 --roth-first-tax-year 2010 --roth-regular-basis 10000 \
  --roth-magi 100000 --annual-compensation 100000 \
  --ordinary-income 100000 --filing-status single --state FEDERAL_ONLY \
  --freeze-tax-year 2026 --rate 0.06 --inflation 0.04 --seed 20261124
$LETF_PY -m letf.macro.cli explore \
  --sources audit-output/macro-sources-v4 --calibration-end 2026-09-30 \
  --output audit-output/new-stagnation --paths 2 --horizons 7 \
  --state FEDERAL_ONLY --freeze-tax-year 2026 --rate 0.06 --inflation 0.04 \
  --production-growth 0 --equity-cagr-constraint 0
$LETF_PY -m letf.macro.cli explore \
  --sources audit-output/macro-sources-v4 --calibration-end 2026-09-30 \
  --output audit-output/new-crash --paths 2 --horizons 1 \
  --state FEDERAL_ONLY --freeze-tax-year 2026 --rate 0.06 --inflation 0.04 \
  --crash-month 6 --crash-return -0.40
$LETF_PY -m letf.macro.cli parameter-sensitivity \
  --sources audit-output/macro-sources-v4 --calibration-end 2026-09-30 \
  --output audit-output/new-parameter-sensitivity --replicates 12 --paths 256
```

The original demonstrations use the byte-identical v2 observations; v4 metadata
changes artifact identifiers, not numeric inputs. Increase path counts only with
runtime/convergence checks; canonical tax-lot/account replays are materially
slower than generating market paths. Default models are unapproved. Bootstrap
families reject independent shock overrides; use the joint conditional model.
`--help` lists fee/spread, volatility, equity drift, stock/bond innovation
correlation, contributions, withdrawals, calibration, model and seed controls.
GDP/recession-probability inputs and advanced derivative comparisons are not
silently mapped onto supported controls.

## Prospective learning

```bash
$LETF_PY -m letf.macro.cli issue --sources audit-output/macro-sources-v4 \
  --registry audit-output/new-prospective-registry --output audit-output/new-issue
$LETF_PY -m letf.macro.cli evaluate-pending --sources audit-output/macro-sources-v4 \
  --registry audit-output/new-prospective-registry --output audit-output/new-evaluation
$LETF_PY -m letf.macro.cli issue --require-existing \
  --sources audit-output/macro-sources-v4 --registry audit-output/new-prospective-registry \
  --output audit-output/new-unchanged-data-check
```

With the same data/code/protocol, the last command produces
`NO_NEW_VERIFIED_TRAINING_DATA`. Current forecasts remain pending, with no champion
or live authority. Archive complete source/registry directories privately with
their hashes. Local deletion and expiring Actions caches are not permanent
research preservation. The optional workflow stays disabled unless the operator
explicitly configures its enable/rights variables and intentional initialization.
