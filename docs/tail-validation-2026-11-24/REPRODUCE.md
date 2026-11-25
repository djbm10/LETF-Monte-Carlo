# Reproduce the continuation

Use the isolated development worktree, Python 3.14 and the unchanged
`requirements-audit.lock`. No command below trades or purchases anything. Output
directories must be **new**; existing source snapshots/registries are not replaced.

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/tail-validation
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
$LETF_PY -m pytest -q
$LETF_PY scripts/check_workflows.py
$LETF_PY -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
```

Expected: 246 tests, four inherited pandas-datareader deprecation warnings; 97
workflow structural checks. CI runs the same pinned fixture suite, not the large
licensed-input studies. A fresh environment needs
`python -m pip install -r requirements-audit.lock`. No new runtime dependency was
added; pypdf6.4.0 was an isolated `/tmp` document-inspection utility only.

## Preserve and reproduce the frozen forecast experiment

Retained source bundle (read-only):
`../macro-learning/audit-output/macro-sources-v4`. Its source catalog hash is
`1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`.
Raw licensed inputs and immutable registry arrays are local, not embedded in Git.
Fresh downloads constitute another source vintage. See the
[manifest](evidence/reproduction-manifest.json) for archived locations and hashes.

Original 26-forecast diagnosis:

```bash
$LETF_PY -m letf.macro.cli diagnose-frozen \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --registry ../macro-learning/audit-output/macro-registry-v1 \
  --prior-report ../macro-learning/audit-output/macro-walkforward-v1/report.json \
  --output audit-output/new-original-diagnosis.json
```

The **initial** new 25 forecasts were generated at `c990b9d`. Reproducing their
exact IDs requires that implementation/protocol, not edited source. Add an
isolated worktree with `git worktree add --detach /tmp/letf-tail-frozen c990b9d`,
then run its `tail-walkforward` command using the same absolute retained source
and prior-report paths, with fresh output/registry directories. That version's
candidate-specific score scaling was defective. Reproducing it is archival
verification, **not** endorsement of the invalid comparison.

At the delivered source, regenerate all monthly/lineage arrays without outcomes:

```bash
$LETF_PY scripts/verify_frozen_tail_paths.py \
  --registry audit-output/tail-registry-v1 \
  --report audit-output/tail-walkforward-v1/report.json \
  --output audit-output/new-frozen-path-check.json
$LETF_PY -m letf.macro.cli rescore-tail \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --registry audit-output/tail-registry-v1 \
  --prior-report audit-output/tail-walkforward-v1/report.json \
  --output audit-output/new-common-scale-diagnostic
```

Expected: 25 exactly reproduced ensembles/lineages; corrected candidates all
rejected. Rescoring uses common baseline units and checks that every original
daily tail/coverage diagnostic remains identical. It writes new evaluation
artifacts but never reconsumes a batch or promotes a model. The registry's earlier
forecasts/decisions are not overwritten. The 26 original forecasts are never
used as fresh candidate-evaluation evidence.

For a **new retrospective** run, `tail-walkforward` with the registered protocol
and new output/registry paths now uses common score units from the start. Repeating
these dates is reproduction, not additional untouched evidence:

```bash
$LETF_PY -m letf.macro.cli tail-walkforward \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --protocol docs/tail-validation-2026-11-24/protocol.json \
  --prior-report ../macro-learning/audit-output/macro-walkforward-v1/report.json \
  --registry audit-output/new-tail-registry --output audit-output/new-tail-study
```

## Official fund/index diagnostic

```bash
$LETF_PY scripts/capture_official_reconstruction_inputs.py \
  --offline-cache audit-output/official-index-nav-v4 \
  --output audit-output/new-archival-official-inputs
$LETF_PY scripts/reconcile_official_reconstruction.py \
  --official audit-output/official-index-nav-v4 \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --output audit-output/new-official-diagnostic/report.json
```

Omit `--offline-cache` only when intentionally capturing a new download vintage.
Expected inspected-period CAGR errors: TQQQ +0.024270821740895565, QLD
+0.006314910576003863, UPRO +0.01024311861460414, SSO −0.014909887718405068.
TQQQ fails; other numerical passes do not satisfy financial-source certification.
No threshold or evaluation date is changed. The original reconstruction
validator and its original results remain available unchanged.

The canonical `daily_reset_returns` API optionally accepts `financed_notional`,
`collateral_income_per_nav`, `collateral_day_basis` and `financing_source`.
Absent those inputs it retains the original net-borrowing convention. Observed
fund returns reject extra internal charges. Historical gross notional/yield
schedules are **not supplied** by these diagnostic input bundles.

## Comparable account cohorts

```bash
$LETF_PY scripts/compare_verified_funds.py \
  --bundle TQQQ=../macro-learning/audit-output/fund-comparisons-v4/TQQQ-inputs \
  --bundle QLD=../macro-learning/audit-output/fund-comparisons-v4/QLD-inputs \
  --bundle SSO=../macro-learning/audit-output/fund-comparisons-v4/SSO-inputs \
  --bundle UPRO=../macro-learning/audit-output/fund-comparisons-v4/UPRO-inputs \
  --bundle QQQ=../macro-learning/audit-output/QQQ-issuer-bundle-v1 \
  --bundle TMF=../macro-learning/audit-output/TMF-issuer-bundle-v1 \
  --reinvest --state FEDERAL_ONLY --output audit-output/new-fair-six
$LETF_PY scripts/build_multiyear_fund_bundle.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --issuer-dir ../macro-learning/audit-output/issuer-tax-sources-v1 \
  --macro-sources ../macro-learning/audit-output/macro-sources-v4 \
  --symbol SPY --output audit-output/new-SPY-2025
```

For the seven-fund cohort, repeat the first command with
`--bundle SPY=audit-output/new-SPY-2025 --start 2025-01-02 --end 2025-12-31` and
another output path. Every fund then uses the same 2025 period. Do not combine it
with the two-year cohort. Omit `--reinvest` to reproduce the separately reported
initial-only cash policy for all funds. MA rejects until fund distribution
adjustments are reviewed. SPY rejects unsupported earlier coverage/signal rules.

All runs export daily reconciliation, pre-tax fills, actual fills, income/tax
events and detailed summary. The core account CLI also exposes
`--qualified-holding-policy track_unhedged_common`. Default `verified_at_ex`
retains the conservative legacy gate. Owner cash flows/withdrawals require the
same separate history/configuration inputs as before.

## Long histories and hypothetical futures

```bash
$LETF_PY -m letf.macro.cli counterfactual \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --start 1930-01-01 --output audit-output/new-preserved-counterfactual
$LETF_PY -m letf.macro.cli sampling-study \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --registry audit-output/unused-sampling-registry \
  --protocol docs/tail-validation-2026-11-24/protocol.json \
  --calibration-end 2026-09-30 --output audit-output/new-unit-sampling
$LETF_PY -m letf.macro.cli explore \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --calibration-end 2026-09-30 --model volatility_stationary_block \
  --paths 1 --horizons 40 --state FEDERAL_ONLY --freeze-tax-year 2026 \
  --output audit-output/new-canonical-40y
```

The counterfactual report reproduces the prior 108 cases exactly with retained
inputs. The sampling study uses two seeds × 4,096 paths and reports **15 failed
quantile-convergence checks**, not tax/account wealth. The final command proves
empirical-tail paths execute through canonical taxable/Roth accounting; one path
cannot certify outcome probabilities. Both remain exploratory/unapproved.

Precision artifacts were frozen at `e4e405a`. Their identity includes the code
hash; the later optional gross-financing API preserves default numeric behavior
but changes that hash. Use the archived implementation to reproduce exact
precision artifact IDs; a current-source run reproduces the numerical study,
not the old code identity.

The existing `conditional_var_t` explorer supports supplied rate/inflation/growth,
crash and covariance interventions, still as hypothetical conditional assumptions.
Bootstrap candidates reject independent rate/asset overrides; they cannot invent
joint causal responses. Existing owner/contribution/withdrawal inputs remain
available; examples are in the [prior guide](../macro-learning-2026-11-24/REPRODUCE.md).

For unchanged legacy failures use the existing `run_forward_realism_tests` API or
`scripts/run_accuracy_audit.py --output audit-output/new-accuracy-run` (also reruns
pytest). Expected realism: 8/10, with T20 and T27 visibly failing. No workflow,
frozen rule, sealed holdout, option quote or futures tape is rewritten to pass.
