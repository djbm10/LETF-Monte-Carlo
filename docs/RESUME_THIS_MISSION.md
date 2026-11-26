# Resume the autonomous LETF laboratory mission

Continue `djbm10/LETF-Monte-Carlo` on
`develop/research-lab-end-to-end-2026-11-25`, review PR41 targeting
`develop/financial-realism-2026-11-25`. Do not restart from the older base or
discard completed work. Financial engine commit:
`6641ac06b27c5202a3c5724fbbd88fa7a60a3fae`; later UI/docs/archive commits are
part of the delivery. Reverify the actual branch head and latest CI first.

```bash
cd /home/djmann/projects/letf-monte-carlo/.worktrees/master-lab
git status --short
git rev-parse HEAD
git worktree list --porcelain
git log --oneline 9a6fb44..HEAD
```

Read `AUTONOMOUS_MASTER_PLAN.md`, `DECISIONS.md`, `PROGRESS_CHECKPOINT.md`,
`BLOCKERS.md`, `TEST_AND_SOURCE_MANIFEST.md`, `RESEARCH_LAB.md` and all
`master-lab-2026-11-25/{REPORT,SOURCES,SUPPORT_MATRIX,REPRODUCE,LEGAL_SCOPE}.md`.
Inspect the final operation/CI receipts and private recovery instructions.
Preserve the original dirty root worktree, all prior worktrees,402 protected
artifacts, original26+25 forecasts, old Dec2026 commitments and three newly
frozen prospective candidates. No paid data, credentials, trades, filings,
private off-host upload, public deployment, main merge or force push.

Implemented: immutable175-snapshot source lake, original VIX/IRX capture,
2021–26 federal/Roth/MA bounded extensions, three-fund2021–25 account cohorts,
multiasset canonical cash/tax portfolios, QDI61/121 and verified RIC199A46/91
holding trackers, split-adjusted wash units preserving dollar losses, shared
derivative collateral and secured Roth option mechanics, dated source schemas,
1930 counterfactual108-case/CPI correction, review-only prospective cycles,
resumable full-account precision, local CLI/dashboard and offline exports.
Final366 tests/97 workflow checks;21/21 current2025 cash adapters execute with
explicit daily override/cold warmup and no rankings. S6/HFEA default21-session
2024–25 flow comparisons reconcile. Do not silently redefine frozen strategies.

Known failures remain scientific/evidence gates, not permission to manipulate:
TQQQ+2.427082pp inspected2020–25 error; new fixed-parameter2026 retrospective
226-session bias+2.325029pp; UPRO exact benchmark/funding provenance; macro
challengers rejected/severe-drawdown failures; legacy T20=59.5075<65 and T27
unsupported damping API; full-account small-sample precision fails;60 taxable
zero-residual dispositions blocked. Original unit-fund million-path precision
does not transfer to general account forecasts. Off-host recovery is unverified.

Next highest-value actions, in order:

1. Reverify final delivery-head hosted CI/recovery receipt if pending. Repeat
   `python -m pytest -q`, `python scripts/check_workflows.py`, supported-module
   compileall and source/402-artifact hash checks. No skip/xfail/threshold edits.
2. Investigate exact dated fund economics and target/index conventions using
   issuer/SEC/Nasdaq/S&P legitimate public records. Keep2020–25 permanently
   INSPECTED; do not choose price/TR/fees merely to improve that curve. Require
   genuinely new prospective evidence before generalization claims.
3. For account numerical precision, register a new fixed domain/count ladder
   with two independent roots in a **new output namespace** and benchmark full
   ledger runtime. Old tiny study at5bbd6ed pins its exact code; current code must
   not overwrite/resume it under changed implementation. No new model ranking.
4. Refresh public source snapshots and run review-only candidate cycle after
   genuine new data. Until complete future months/releases exist, assessments
   must remain PENDING. French data still ended2026September at this checkpoint;
   current refresh alone correctly created no new model. Never erase evidence
   by reinitializing a registry or select a later revision after seeing outcomes.
5. Continue dated derivative input adapters only with real tapes/OCC notices/
   broker permissions; retain theoretical/sample-only performance blocks.
   Complete earlier historical/state/hedged/special-fund tax scope one sourced
   rule/year at a time; current unknown cases must fail closed.
6. If a local browser can be installed legitimately, run optional
   `scripts/smoke_lab_browser.py` with pinned temporary Playwright and local
   Chromium. Its prior installation timed out and was stopped; no browser
   execution/screenshot success was claimed. HTTP and JS syntax checks passed.

Useful commands (all output directories must be NEW):

```bash
LETF_PY=/home/djmann/projects/letf-monte-carlo/.venv/bin/python
$LETF_PY -m letf.lab demo --output audit-output/resumed-demo
$LETF_PY -m letf.lab observed --config docs/examples/observed-s6.json \
  --output audit-output/resumed-s6
$LETF_PY -m letf.lab cycle --config docs/examples/operations-cycle.json \
  --output audit-output/resumed-review-only-cycle
$LETF_PY -m letf.lab account-precision --config NEW_REGISTERED_SPEC.json \
  --output audit-output/resumed-account-precision --max-new-runs 1
```

Fresh clones need private source recovery before observed-input studies. Keep
raw source bytes/ensembles private and verify every hash; relocating paths means
a new profile, not editing sealed source manifests. No code can evaluate an
unobserved future month, infer complete legal worthlessness from a zero NAV, or
verify an unapproved off-host destination. Work on independent supported tasks
when a particular data/authority gate remains blocked. Report actual results
and exact source/code versions, not a plan or an investment-certification claim.
