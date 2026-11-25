# Reproduce the precision and account studies

Run from the new worktree. All output names below are **new**; existing artifacts
are immutable. No command trades, purchases a dataset, files a return or merges
anything. Source snapshots are licensed local inputs, not embedded in Git.

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/precision-reliability
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
LETF_SOURCES=../macro-learning/audit-output/macro-sources-v4
LETF_PROTOCOL=docs/precision-reliability-2026-11-24/protocol.json
$LETF_PY -m pytest -q
$LETF_PY scripts/check_workflows.py
$LETF_PY -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
git diff --check
```

Expected on the delivered implementation: **259 passed**, four inherited
`pandas_datareader` deprecation warnings; **97 workflow checks pass**. No tests
are skipped/xfailed. Local Python is 3.14.4; the unchanged hosted workflow selects
its own Python version and installs `requirements-audit.lock`. A fresh environment
must install that exact lock (`python -m pip install -r requirements-audit.lock`).
No runtime dependencies were added. PDF review's isolated pypdf6.4.0 is optional
and not needed for these simulations.
The [environment receipt](evidence/environment.json) records exact package,
Python, platform and lockfile identities. Byte-exact sample checks were performed
in this environment; floating-point byte identity across different platforms is
not assumed. Independent numerical conformance also uses explicit tolerances.

## Exact sources versus a fresh download

The retained macro catalog SHA-256 must be
`1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`.
Every captured file is checked against its catalog. A new download may revise
history, drop observations or change availability; it creates a new vintage and
cannot be asserted to reproduce these exact numbers. See [SOURCES.md](SOURCES.md)
and the prior capture guide for legitimate provider retrieval.

The local archive
`audit-output/prospective-reference/sealed-2026-11-25.tar.gz` contains the source
snapshot, forecasts and public commitment, with every member's hash. Its exact
hash and verified count are in [archive-verification.json](evidence/archive-verification.json).
It is not in Git because of licensing. Verify without extracting it:

```bash
$LETF_PY -m letf.macro.prospective_daily verify \
  --output audit-output/prospective-reference/sealed-2026-11-25.tar.gz
```

Expected: `VERIFIED_LOCAL_ARCHIVE`, 62 files, off-host verification false.
An independent reader with Git alone can inspect the aggregate evidence and
run the fixture suite, but cannot regenerate licensed-input numerical studies
without the exact lawful source snapshot. This is an explicit reproducibility
dependency, not an assertion that raw data were published.

## Larger registered precision study

```bash
$LETF_PY -m letf.macro.precision_run \
  --sources "$LETF_SOURCES" --protocol "$LETF_PROTOCOL" \
  --output audit-output/reproduce-precision-v1
$LETF_PY scripts/verify_precision_samples.py \
  --experiment audit-output/reproduce-precision-v1 \
  --output audit-output/reproduce-precision-block-check.json
$LETF_PY scripts/benchmark_precision.py \
  --experiment audit-output/reproduce-precision-v1 \
  --output audit-output/reproduce-same-path-benchmark.json
```

The complete registered study uses about 1.26 GB of sample binaries (decimal),
bounded working memory and approximately 8.3 minutes on this machine. Allow
additional memory for NumPy arrays and sorting; runtime is machine-dependent.
It stops only at a registered checkpoint if all tests pass, otherwise keeps its
failure at the maximum budget. Expected: 1,048,576 paths **per seed**, all 15
combinations pass, model ID
`3263d018a49123a2b7659bed42c99e76ce24c914e47f9ba794a73120b8e131e7`.

Expected immutable sample SHA-256s:

| Root seed | SHA-256 |
|---|---|
| 20261127 | `2eb67cc87a3ad9adadab00d00ac7cb4fa09bb4f71ca149ecd2f151742c0ea79a` |
| 20261128 | `16c264025e65d4e5304e561aa9c098144e258912f4853db3b7cd91788253320d` |

The numeric implementation was frozen at `59d8d02`, protocol at `dcbf011`.
Later persistence/assessment changes do not alter its five hashed numerical
modules. Run/capture timestamps and timing measurements will differ; they are
not numerical failures. The verifier checks complete binary hashes and exactly
regenerates first/last fixed stream blocks for both seeds, not all sample paths.
The benchmark independently executes canonical daily-reset arithmetic; expected
maximum errors are approximately 3.1e-14 in log NAV and 4.2e-15 in drawdown.

To diagnose the **original** unchanged 15 failures rather than reproduce the new
study, use the retained small-sample report and script help:

```bash
$LETF_PY scripts/diagnose_precision_failures.py --help
```

Its numerical diagnosis is already committed in
[original-15-diagnosis.json](evidence/original-15-diagnosis.json). The old
`sampling-study` command in the prior guide still reports its original failures.
Do not overwrite them with a larger sample or change its thresholds.

## Paired canonical account endpoints

```bash
$LETF_PY -m letf.macro.account_precision \
  --precision-report audit-output/reproduce-precision-v1/report.json \
  --output audit-output/reproduce-paired-account.json
```

The command refuses a failed underlying market precision gate or altered binary
hashes. It processes the same 2,097,152 paths per combination for both accounts;
all 30 nominal/real asset-quantile gates should pass. These are canonical-ledger
piecewise-affine endpoint maps under the explicit restricted profile, not full
daily-account replay of each path. [ACCOUNT_RESULTS.md](ACCOUNT_RESULTS.md)
documents tax assumptions, drawdown bounds, loss probabilities and limitations.

Zero cash flows mean lump-sum CAGR is meaningful here; no recurring-contribution
IRR is implied. Nonzero distributions/flows, MA, different tax inputs and actual
fund closure must use supported full daily accounting and a separate convergence
study. Extinct/nonfinite paths are rejected rather than dropped or clipped.

## Separate specification uncertainty

```bash
$LETF_PY scripts/run_precision_sensitivity.py \
  --sources "$LETF_SOURCES" --protocol "$LETF_PROTOCOL" \
  --sensitivity docs/precision-reliability-2026-11-24/sensitivity-protocol.json \
  --output audit-output/reproduce-specification-sensitivity
```

Expected: all four specifications reported, no promotion, all four complete-study
precision gates still false at their smaller budget. The 40-year 3× medians are
approximately 247.95 (shorter history), 116.70 (longer duration), 301.41 (lower
cost) and 22.38 (higher cost), versus reference 110.13. Child error intervals are
retained; these numbers are sensitivity estimates, not confidently weighted
investment outcomes. Cost scenarios share identical market lineages.

## Frozen history and genuine prospective outcomes

```bash
$LETF_PY scripts/verify_frozen_tail_paths.py \
  --registry ../tail-validation/audit-output/tail-registry-v1 \
  --report ../tail-validation/audit-output/tail-walkforward-v1/report.json \
  --output audit-output/reproduce-preserved-25.json
$LETF_PY -m letf.macro.prospective_daily assess \
  --sources "$LETF_SOURCES" \
  --commitment docs/precision-reliability-2026-11-24/evidence/prospective-commitment.json \
  --output audit-output/new-prospective-assessment.json
```

The first regenerates exactly 25 monthly/lineage arrays without outcomes. The
original 26 and all 369 registered files remain hash-identical. The second is
pending before the first genuine target month finishes; later it waits for
complete published macro and daily outcomes. A current source snapshot may have
no released target data yet even after month-end. Use a newly captured verified
outcome vintage then, preserve the issued source snapshot and record its new
catalog hash. There is no production CLI argument for pretending the clock is
in the future, and no automatic model promotion.

To issue an **additional genuinely new** forecast at the actual current time:

```bash
$LETF_PY -m letf.macro.prospective_daily issue \
  --sources "$LETF_SOURCES" --registry audit-output/new-prospective-registry \
  --output audit-output/new-public-commitment.json
$LETF_PY -m letf.macro.prospective_daily seal \
  --sources "$LETF_SOURCES" --registry audit-output/new-prospective-registry \
  --commitment audit-output/new-public-commitment.json \
  --output audit-output/new-private-sealed-snapshot.tar.gz
```

That produces different issue IDs/timestamps, not reproduction of the November
25 commitment. The old public commitment and archive are never replaced. Publish
only appropriate aggregate commitments before target outcomes; keep raw/full
ensembles private under provider rights. Whole-host recovery remains dependent
on an operator-verified private off-host copy; a local seal is not such a backup.

## TQQQ attribution and unchanged historical demonstrations

```bash
$LETF_PY scripts/attribute_reconstruction_bias.py \
  --official ../tail-validation/audit-output/official-index-nav-v4 \
  --prices /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --output audit-output/reproduce-reconstruction-attribution/report.json
```

Expected: XNDX legacy CAGR error +0.024270821740895565, constrained funding
+0.024270606100671888; both fail. The price-only sensitivity has different
income conventions and cannot be promoted based on a better fit. Evaluation
2020–25 is already inspected. The diagnostic keeps the existing constant fee
scenario and reports its source limitations, not actual historical cost invoices.

See [the prior reproduction guide](../tail-validation-2026-11-24/REPRODUCE.md)
for the unchanged 108 daily 1930 counterfactuals, six-/seven-fund comparable
observed-account cohorts, original source-reconciled errors and legacy diagnostics.
Those inputs and scripts remain intact. T20 and T27 were re-run here: T20 scores
59.5075 below its actual 65 gate; T27 calls the unsupported `max_crisis_chain`
argument. Eight of ten pass. No crisis damping or threshold adjustment was added
to obtain a better score.

All outputs above remain `HYPOTHETICAL_STRESS` or bounded observed research as
specified, never certified investable forecasts. A successful reproduction
establishes arithmetic/source identity, not out-of-sample forecasting skill.
