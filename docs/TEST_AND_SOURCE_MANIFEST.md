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
