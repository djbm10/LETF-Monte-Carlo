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
