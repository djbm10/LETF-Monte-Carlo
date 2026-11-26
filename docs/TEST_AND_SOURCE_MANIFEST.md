# Test and source manifest

Base commit: `9a6fb44f4fe95c4df6c4a8bc7598e515a3acc459`.
Pinned dependency file: `requirements-audit.lock` (existing pins preserved;
xlrd2.0.2 added to read the actual issuer2022 binary XLS without macros).
Baseline expected: 282 tests, 97 workflow structural checks; actual repeat is
recorded in the checkpoint when complete.

Prior frozen-artifact hashes are in
`financial-realism-2026-11-25/evidence/preserved-hashes.json` (402 artifacts).
Prior model, source and forecast hashes remain authoritative. Raw inputs,
local archives, full ensembles and account exports stay under ignored private
`audit-output` directories; only reviewed derived reports/metadata enter Git.

Record exact test commands/results, source snapshot hashes and coverage, stage
commits, output classifications and known failures here. Hosted fixture CI is
not a substitute for large local source-dependent economic replays.

Stage 1: `python -m pytest -q tests/test_lab_sources.py`: 10 passed.
Full suite: 292 passed / four inherited warnings (48.86s), private stdout
`audit-output/master-mission-baseline/stage1-tests.txt`. Baseline 282 passed
(54.12s), workflow structure 97 pass. Inventory SHA-256
`9ea899da4c603c7c24f6a984801af2fd7f1c0025058bd7d47d9837fc60bfad14`;
the preserved legacy LETF33 source has a pre-existing parse error at line 1833.
Source-lake import receipt is private under the baseline output directory.

Stage 2: final full suite **309 passed**, four inherited warnings (43.87s).
New independent historical tax/Roth/MA and safe-export fixtures pass.
The old MA unreviewed-year test now uses 2022; the newly sourced2023 case is
asserted independently. No accuracy gate changed. A new test-only exact equality
on 850.0000000000001 was corrected to numeric comparison at ordinary float
precision; the economic tax expectation remains $850.
Primary law hashes, source-lake verification and 2023–25 real TQQQ/UPRO account
results are in `master-lab-2026-11-25/evidence/`. Original high-gain export failure
and partial source imports remain private evidence, not supported results.

Portfolio stage: 315 full tests pass; frozen35/0 source extraction and exact
selector comparison are independent of the adapter. Actual S6 pre-tax trade
quantities differ by at most1.14e-13 shares solely from taxable lot splitting;
dates/actions/prices/commissions/reasons exact. Absolute share/dollar comparison
bounds1e-10/$1e-8 are stricter than canonical reconciliation1e-7/$1e-6.

Product/forecast/derivative-clock integration: **326 passed**, four inherited
warnings (62.17s), 97 workflow checks; unchanged socket test required running
outside sandbox after explicit EPERM. Original environment failure retained at
`audit-output/master-mission-baseline/stage4-product-tests.txt`; final stdout
at `stage4-product-final-tests.txt`. JavaScript syntax checked with Node24.14.1.
Future2-path account demonstration is not a convergence pass; exact conditional
order-statistic intervals remain unbounded where sampling cannot identify tails.

Stage5 integration: **335 passed**, four inherited warnings (35.47s).
New six secured-option independent delivery/collateral fixtures pass alongside
all old derivative tests. Seven additional source/prospective fixtures were
then added; final full-suite result recorded after their integration. No
drawdown/reconstruction/convergence thresholds changed. Updated CI runs a
portable authored SOFTWARE_FIXTURE end-to-end demo; private market inputs are
not uploaded. Optional existing candidate workflow additionally requires an
approved-private-storage setting; all candidate evaluation defaults review-only.


Final engine31f16ddc73789d83142963b4c9125882cb8d05ca: **353 passed**, four inherited
warnings,57.68s;97 workflows; compileall on supported modules and git diff --check
pass. All402 protected artifact hashes still match. Stage7 before final guards
was347; stage8 added provider-sample/mixed-source/registered-account-precision
fixtures for351; final permission/CPI independent fixtures give353. A fixture
retains the original unreviewed MA2022 rejection; no economic gate was weakened.

Current hosted engine checks:16 runs completed SUCCESS, including regression
push45722948239 and PR45722954836. Precursor347-test run45708100675 also succeeded.
Final delivery-head status is recorded separately in the CI receipt; structural97
checks are not97 hosted jobs or97 financial confidence gates. HTTP/dashboard backend checks and
portable fixture report creation passed in hosted CI; no browser visual pass is claimed. No private input is uploaded.

Standalone legacy diagnostics rerun:8/10, T20=59.50749533337072 FAIL, T27unsupported
max_crisis_chain FAIL. TQQQ original+2.427082pp remains FAIL; new226-session2026
fixed-parameter bias+2.325029pp remains uncertified. Full-account tiny registered
study completed4jobs/36replays, FAILED precision as expected with exit2. The older
2×1,048,576-path unit-domain gate remains unchanged, not inherited by accounts.

Current compact aggregate pack:master-lab-2026-11-25/evidence/final-v4/final-manifest.json;
raw private artifact hashes and exporter source hash are included. Version1
preserves the superseded first-return-month CPI base; version2 uses actual initial
close month and repeats S6 exactly at the final engine code. All82 recorded S6
letf source-module hashes match31f16dd; dirty=true reflected documentation/export
work, not uncommitted engine math. Nominal counterfactual108/108 outputs identical.


Late source-backed account integration: engine6641ac06b27c5202a3c5724fbbd88fa7a60a3fae,
**365 passed**, four warnings,82.73s;97 workflow checks. New independent RIC holding
45/46-day, partial/year-end/split, NIIT/MA and strict-proof fixtures pass. Four wash
split fixtures preserve dollar losses, partial prior replacements, reverse units
and permanent Roth losses. Multi-year tournament tax-payment date export is
independently tested. Actual common2025 cohort21/21 after legally captured original
VIX/IRX signals; S6/HFEA common2024–25 dates/21-session flows both reconcile.
Old17/18/20-cell cohorts retained. New source lake175 snapshots includes CFR law
and two current-vintage signal frames. No held-out return/model gate changed.


Delivery UI guard: four unchanged HTTP/dashboard tests pass3.13s; JavaScript
syntax passes Node24.14.1; full365-test rerun77.05s. A new/failed request clears
stale wealth/downloads. Optional real-browser driver exists but is NOT executed:
no browser installed, pinned Playwright1.56.0 package download timed out repeatedly;
agent stopped the retry at10.7/46.3MB. This is an explicit environment/visual
validation gap, not a passed browser check or a disabled baseline test. No runtime
lock dependency was changed by the attempted temporary installation.


Final affirmative-evidence guards b3c4e1f reject strings and integers as observed-quote,
Section1256, historical-availability and reconstruction-provenance evidence. Independent
fixture plus reconstruction/source suites:20 passed. Full regression366 passed,
four inherited warnings,65.50s;97 workflow structure checks and supported compile pass.
No reconstruction, tail, legacy or forecast-approval threshold was changed.
