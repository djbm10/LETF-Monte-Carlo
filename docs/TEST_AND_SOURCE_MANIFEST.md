# Test and source manifest

Base commit: `9a6fb44f4fe95c4df6c4a8bc7598e515a3acc459`.
Pinned dependency file: `requirements-audit.lock` (unchanged).
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
