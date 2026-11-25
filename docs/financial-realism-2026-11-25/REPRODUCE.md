# Reproduce the financial-realism continuation

Use the isolated worktree and unchanged dependency lock. No command places a
trade, uploads data or merges anything. All output/receipt directories must be
**new**. Exact local source snapshots are required for licensed-input studies;
a fresh provider download is a different vintage, not exact reproduction.

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/financial-realism
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
$LETF_PY -m pytest -q
$LETF_PY scripts/check_workflows.py
$LETF_PY -m compileall -q letf scripts tests research/free_validation research/quantconnect_free_futures
git diff --check
```

Install a fresh compatible environment with
`python -m pip install -r requirements-audit.lock`. Local Python is 3.14.4.
No runtime dependency was added. The final delivery reports the exact full-suite
count and CI SHA; the archived baseline is 259 tests. Warnings are inherited
pandas-datareader deprecations. Hosted CI uses fixture inputs, not the large
local licensed studies. PDF extraction alone uses optional isolated pypdf6.4.0.

## Frozen support audit and full daily stress workflow

```bash
$LETF_PY scripts/audit_catastrophe_support.py \
  --experiment ../precision-reliability/audit-output/precision-v1 \
  --output audit-output/new-support-audit.json
$LETF_PY -m letf.account_stress \
  --experiment ../precision-reliability/audit-output/precision-v1 \
  --protocol docs/financial-realism-2026-11-25/protocol.json \
  --output audit-output/new-full-stress
```

The first checks finite support without new Monte Carlo paths. The second
replays two registered old anchors in all 600 scenario/account combinations.
It writes explicit BLOCKED rows and **returns exit status 2** when any case is
unsupported; that is the expected outcome for taxable zero-residual cases.
Do not remove these cases or suppress the gate to obtain success. Completed
supported rows and source hashes remain available for assessment:

```bash
$LETF_PY scripts/assess_realism_study.py \
  --study audit-output/new-full-stress \
  --precision ../precision-reliability/audit-output/precision-v1/report.json \
  --sensitivity ../precision-reliability/audit-output/specification-sensitivity-v1/report.json \
  --raw-engine-commit YOUR_EXACT_PRODUCER_COMMIT \
  --output audit-output/new-realism-assessment/report.json
```

Expected acceptance: **540 supported hypothetical-mechanics rows and 60 blocked
tax-disposition rows**, no approved model. All supported independent terminal
cash and unitized drawdown identities pass. `--cases` and `--anchors` can select
registered subsets for reproduction/debugging; the report marks the matrix
incomplete rather than implying it ran all cases. No new model selection,
forecast scoring or scenario probabilities occur.

The archived 600-row arithmetic run was produced at
`c165421b81d70787742f2a9b4ad39e204aaf0ed8` before the tax-scope guard. Its same-day
zero-price tax values are **superseded**, not supported results. The confidence
assessment verifies current canonical rejection and omits those tax values.
Roth issuer-default replays examine explicit zero/full unpaid-dividend recovery.
The assessor handles both older full-recovery producers and current zero-recovery
producers, replays the opposite assumption and labels it accurately.

For byte-identical archival raw reproduction, create a separate checkout of
`c165421` and use its CLI with the retained source paths. It preserves the old
limitations; reproducing it is not endorsement. The corrected current CLI is
the supported entry point. Ordinary current-source fixture/account tests never
depend on that unsafe zero-tax assumption.

Model ID remains
`3263d018a49123a2b7659bed42c99e76ce24c914e47f9ba794a73120b8e131e7`.
The exact macro catalog is
`1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`.
Existing fixed-model numerical intervals remain in the prior precision report;
the 600 deterministic cases do not supply new convergence estimates or future
event probabilities. Calendar/cost/enrollment conventions are documented in the
protocol and source manifests. Large studies export daily CSVs/event ledgers
locally; they can take minutes and require disk space. No sample is discarded
for an unattractive outcome.

## Observed funded TQQQ daily accounts

```bash
$LETF_PY scripts/replay_funded_observed.py \
  --bundle ../macro-learning/audit-output/fund-comparisons-v4/TQQQ-inputs \
  --output audit-output/new-funded-observed
```

Expected 2024–25 net assets: taxable **29783.38782724657**, Roth/control
**32386.47031891895**. Taxable annual cash tax is 54.98578721014201, terminal
payable 2490.797402707063; one tax-funding sale settles before payment. Both
accounts receive identical cash contributions and reinvestment policy. Roth
and tax-disabled control pre-tax trades match. Dividends/splits, actual
supported tax years, QDI holding records and independent event/position/return
identities are exported. No double internal fund-cost deduction is made.

The source bundle manifest is embedded in the public result. Inputs are genuine;
execution/household policy are modeled. If a bundle is unavailable, reconstruct
it from the retained original quote/issuer-tax sources using the prior
[bundle reproduction guide](../tail-validation-2026-11-24/REPRODUCE.md), retaining
exact hashes. Do not replace missing tax/distribution records with synthetic ones.

## Issuer publication and financing evidence

The captured private PDF is
`audit-output/issuer-evidence-v1/issuer-annual-candidate.pdf`, hash
`89149fc0bd2121f0093632815f84616c17e3f4ed3f98b84d7fa4084f3f07b915`.
Its source URL and page numbers are in the committed facts JSON. If re-downloading,
verify the bytes; mutable URLs may return another vintage.

```bash
$LETF_PY scripts/reconcile_issuer_2021.py \
  --official ../tail-validation/audit-output/official-index-nav-v4 \
  --facts docs/financial-realism-2026-11-25/evidence/issuer-2021-facts.json \
  --pdf audit-output/issuer-evidence-v1/issuer-annual-candidate.pdf \
  --output audit-output/new-issuer-reconciliation.json
```

Expected: three NAV publication-rounding passes, one index pass and two index
failures. No source thresholds move. Nine swap notionals sum to 26963026907;
weighted floating rate is 0.00607404844774982; annualized gross cost per snapshot
NAV is 0.013577880141697836. The TQQQ reconstruction error remains +2.427 pp,
not relabeled as fixed. Read [SOURCES.md](SOURCES.md) for dated interpretation.

## Preserve genuine prospective forecasting

```bash
$LETF_PY -m letf.macro.prospective_daily assess \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --commitment ../precision-reliability/docs/precision-reliability-2026-11-24/evidence/prospective-commitment.json \
  --output audit-output/new-genuine-future-assessment.json
```

Before any complete target month and released outcomes, expected status is
`PENDING_GENUINE_FUTURE_OUTCOMES` or `PENDING_VERIFIED_OUTCOME_DATA`, no promotion.
Use a newly verified outcome vintage only after it genuinely exists; preserve
issued sources/forecasts. There is no production clock-backdating flag. Never
score the 26+25 consumed historical forecasts as fresh evidence. The frozen
December 2026 onward arrays/commitment are not modified by stress scenarios.

For raw data/private code preservation and recovery, follow the separately
tested [backup procedure](BACKUP_AND_RECOVERY.md). It deliberately cannot claim
off-host success without an approved destination and verified remote readback.
