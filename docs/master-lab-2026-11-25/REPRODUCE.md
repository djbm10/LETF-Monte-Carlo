# Reproduce the laboratory checkpoint

Use Python3.14.4 and `requirements-audit.lock`. Existing pins are preserved;
xlrd2.0.2 was added solely for issuer binary XLS. PDF inspection used optional
isolated pypdf6.4.0, not a simulator dependency. New outputs must have new names.
No command buys data, places trades, files taxes or uploads private inputs.

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/master-lab
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
git status --short
git rev-parse HEAD
$LETF_PY -m pytest -q
$LETF_PY scripts/check_workflows.py
$LETF_PY -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
git diff --check
```

Final evidence guard b3c4e1f:366 tests/four inherited warnings,97 workflow structure checks.
The source-dependent financial reports identify engine6641ac0 explicitly.
The original archived `LETF33_analysis.py` still has its pre-existing parse
error and is outside supported application compilation. No source was deleted.
Localhost integration tests require socket permission; do not skip them.

## Portable CLI and local dashboard

```bash
python -m pip install -r requirements-audit.lock
python -m letf.lab demo --output audit-output/new-demo
python -m letf.lab serve --profiles audit-output/new-demo/profiles.json \
  --output-root audit-output/new-dashboard-jobs --port 8765
```

Open `http://127.0.0.1:8765`. This authored fixture needs no market data or
credential. Output is SOFTWARE_FIXTURE. Stop the foreground server normally.
The HTML report is standalone; no CDN or externally deployed service is used.
See [full input/profile/API guide](../RESEARCH_LAB.md).

## Exact observed fund cohort

Private frozen source snapshots are not part of a fresh Git clone. Restore
the verified private archives or use the original capture guides to obtain
legally available inputs; a new provider revision is a different experiment.
The source catalog and generated bundle manifests retain exact raw hashes.

```bash
$LETF_PY scripts/build_multiyear_fund_bundle.py \
  --inputs /home/djmann/projects/letf-monte-carlo/audit-output/public-daily-inputs-network \
  --issuer-dir audit-output/extended-issuer-v1 \
  --macro-sources ../macro-learning/audit-output/macro-sources-v4 \
  --symbol TQQQ --years 2021 2022 2023 2024 2025 \
  --output audit-output/new-TQQQ-2021-inputs
```

Repeat for UPRO/SSO into separately named new directories. The2022 source is
the issuer-linked XLS, not a guessed XLSX;2021/22 use reviewed two-line headers.
Existing exact bundles allow the actual reproduced cohort command:

```bash
$LETF_PY scripts/compare_verified_funds.py \
  --bundle TQQQ=audit-output/extended-accounts-v3/TQQQ-2021 \
  --bundle UPRO=audit-output/extended-accounts-v4/UPRO-2021 \
  --bundle SSO=audit-output/extended-accounts-v4/SSO-2021 \
  --start 2021-01-04 --end 2025-12-31 --state FEDERAL_ONLY --reinvest \
  --output audit-output/new-five-year-cohort
$LETF_PY -m letf.lab observed --config docs/examples/observed-s6.json \
  --output audit-output/new-s6
$LETF_PY -m letf.lab render --report audit-output/new-s6/report.json \
  --output audit-output/new-s6/report.html
```

Expected TQQQ ending taxable216418.1339354625/Roth239721.5817711129;
S6 taxable21927.05952055513/Roth23338.326342158543. No optimistic ranking
or historical fill certification follows from matching these numbers.

## Historical, future and explicit stress modes

```bash
$LETF_PY -m letf.lab counterfactual \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --start 1930-01-01 --end 2025-12-31 --output audit-output/new-1930-v2
$LETF_PY -m letf.lab simulate --config docs/examples/future-exploratory.json \
  --output audit-output/new-future
$LETF_PY -m letf.lab render --report audit-output/new-future/laboratory-report.json \
  --output audit-output/new-future/report.html
```

108 historical counterfactuals;60 full future account replays in the two-path
example. Future sampling gates remain unaccepted; no model is approved.
Existing registered catastrophic/rate/bear stresses are invoked with `lab stress`
as documented in [RESEARCH_LAB](../RESEARCH_LAB.md). Taxable extinction exits2
and keeps BLOCKED rows. Their original600-case assessment remains540 supported
hypothetical mechanics/60 blocked tax dispositions; do not convert it to green.

## Resumable full-account precision, separate from unit-fund precision

```bash
$LETF_PY -m letf.lab account-precision \
  --config audit-output/account-precision-request-v1.json \
  --output audit-output/new-account-precision --max-new-runs 1
# Same command/output resumes unchanged registered source/code/protocol.
$LETF_PY -m letf.lab account-precision \
  --config audit-output/account-precision-request-v1.json \
  --output audit-output/new-account-precision --max-new-runs 3
```

The demonstrated JSON has schema1, `experiment` equal to the future example
without `paths`/`seed`, horizons[5], counts[1,2], root_seeds[20261125,20261126].
It ran at5bbd6ed before the later input/CPI fixes. Exact registered study resume
requires that code version; current code needs a **new namespace**. Completed
reports/receipts are hash-verified; partial/blocked checkpoints are never
silently replaced. Exit2 means pending or failed gates, expected for tiny counts.
For production numerical work register a larger fixed doubling ladder before
running, retain both independent roots and all attempted paths, and budget full
canonical replay time. Original .02 log-quantile/.025 binomial-half-width limits,
all planned-look family coverage and latest-doubling/root checks are retained.
No accelerator is used outside its proven domain. Success would establish only
conditional numerical precision, not forecast skill or future tax correctness.

## Source/forecast/derivative workflows

```bash
$LETF_PY -m letf.lab sources --lake audit-output/source-lake-v1
$LETF_PY -m letf.lab refresh --lake audit-output/new-private-lake \
  --names ff_daily ff_monthly CPIAUCNS INDPRO BAA AAA GS10 UNRATE DTB3 \
  --output audit-output/new-refresh.json
$LETF_PY -m letf.lab cycle --config docs/examples/operations-cycle.json \
  --output audit-output/new-review-only-cycle
$LETF_PY -m letf.lab derivative-inputs --folder PRIVATE_DATED_BUNDLE \
  --decision-at 2025-01-03T15:00:01Z --output audit-output/new-contract-check.json
$LETF_PY -m letf.lab cboe-sample \
  --archive audit-output/free-derivative-samples-v1/cboe-eod-sample217.zip \
  --output audit-output/new-provider-sample-check.json
```

All source/forecast writes are new immutable snapshots. The cycle defaults
review-only; genuine complete future outcomes are required before scoring.
No scheduler was installed. The optional local `scripts/run_lab_cycle.py`
creates a new timestamped run per invocation; configure an operator schedule
only after source rights and private retention have been reviewed.

Fixed-old-parameter TQQQ diagnostic:

```bash
$LETF_PY scripts/diagnose_refreshed_nav.py --lake audit-output/source-lake-v1 \
  --snapshots audit-output/refreshed-nav-snapshots-v1.json \
  --old-report ../tail-validation/audit-output/official-reconstruction-v2/report.json \
  --output audit-output/new-nav-diagnostic.json
```

Expected226 sessions and+2.325029pp annualized error; no504-session gate pass.
Original inspected2020–25 report remains unchanged. Aggregate evidence export:

```bash
$LETF_PY scripts/export_lab_evidence.py --output audit-output/new-aggregate-evidence
```

The exporter needs the private named studies and emits hashes/aggregate reports,
not raw provider quotes/full ensembles. `final-v1` preserves the initial CPI
endpoint convention; `final-v2` supersedes purchasing-power endpoints only.
Exact private source artifacts and final derived hashes are in its manifest.
Recovery procedure and remaining authority prerequisites are in
[LEGAL_SCOPE](LEGAL_SCOPE.md) and [BLOCKERS](../BLOCKERS.md).


## Resolved source/holding/split extensions

`python scripts/capture_daily_signal_inputs.py --lake PRIVATE_LAKE --output NEW`
uses pinned yfinance to capture original^VIX/^IRX2024–25 closes; no fill or DFF/
monthly-riskfree substitution. Original-rule signal manifest is private at
`audit-output/original-rule-signals-v2/manifest.json`. The final21-cell cohort
configuration is `audit-output/cohort-2025-request-v3-signals.json`; run it through
`lab tournament` into a new output. All21 cells complete with common fingerprints,
explicit daily override/cold warmup and no rankings. The original v1 config still
reproduces its source-limited blocks; never overwrite earlier experiment outputs.

Default monthly-wrapper comparison:

```bash
$LETF_PY -m letf.lab tournament --config audit-output/s6-hfea-common-request-v1.json \
  --output audit-output/new-s6-hfea-pair
```

Expected S6 values above; HFEA taxable20409.90308555626/Roth21457.434609841333.
The prior multiyear tournament serialization failure is retained separately;
current code exports explicit paid/unpaid tax dates without dropping rows.
The original jurisdiction/special-fund/unverified holding/zero-NAV gates remain.


## Optional browser verification (not completed in this environment)

```bash
python -m pip install --target /tmp/letf-lab-browser playwright==1.56.0
PYTHONPATH=/tmp/letf-lab-browser PLAYWRIGHT_BROWSERS_PATH=/tmp/letf-lab-browser-binaries \
  python -m playwright install chromium --only-shell
PYTHONPATH=/tmp/letf-lab-browser PLAYWRIGHT_BROWSERS_PATH=/tmp/letf-lab-browser-binaries \
  python scripts/smoke_lab_browser.py --output audit-output/new-browser-fixture
```

Run only where optional package/browser downloads and local sockets are permitted.
The driver forbids outside page requests and closes its browser/server on return.
This environment’s repeated package timeout was retained; screenshots and browser
conformance were not claimed. The simulator requires none of these optional tools.
