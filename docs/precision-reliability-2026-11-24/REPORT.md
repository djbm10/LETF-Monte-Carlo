# Precision and reliability continuation — conditional precision, no forecast certification

Reviewed November 25, 2026. Branch: `develop/precision-reliability-2026-11-24`.
Exact parent: `bed720aafa1659101a6f28be96ebad1f3b24701c`. Implementation through
`c529030e6a10163c9c9e16a45c24a50bf2d64cac`; this report is committed afterward.
The delivery response identifies the final documentation SHA and its hosted CI.

**All 15 larger-sample precision gates passed without increasing their error
tolerance. The original 15 failed tests remain unchanged. No market model,
leveraged-fund reconstruction, or future investment probability is certified.**
The account demonstration passes sampling gates for a restricted hypothetical
profile, not for actual funds with missing distributions or future tax laws.

## Implemented changes and preservation

| Commit | Implemented work |
|---|---|
| `dcbf011291d58f510a01e61506de14e5d29b1bc1` | Register larger path counts, independent seeds, simultaneous confidence intervals and stopping rules before running them |
| `59d8d02b5eb8addbb002d27f64bdc3d56140a9e1` | Exact finite-support daily-mechanics acceleration, log NAV and order-statistic intervals |
| `de26edba58d5a935a23b3bfae1c695d17c6e5e12` | Diagnose every original failure; repair new test fixture's missing inflation column |
| `2d80749c5bdd6991a217c3bb66513e38f6e7c8d3` | Atomic forecast artifacts, public prospective commitments, future-only scoring and verified portable source/ensemble archives |
| `2857292336541e95db500243a7dd22a39f5db063` | Publish December 2026–November 2027 reference commitments before outcomes |
| `639cfefc5beca3ba95c73fdac0cf1f834fb844a8` | Restricted canonical-ledger endpoint memoization and paired taxable/Roth projections |
| `3647b090534f64854006df377e130418d363896e` | Predeclare history/duration/cost sensitivities; training-only reconstruction attribution |
| `5d583a71107434e6cb546df8c439cb446d0275a7` | Directory fsync hardening; same-path daily-engine reconciliation on real source inputs |
| `099916518468e797b8a173bba8f1e5f39b48ffd9` | Score elapsed crash/bear prefixes only after complete actual daily outcomes; regenerate frozen sample blocks |
| `1fe70c8c2983dda2185b8b7ec83470be427f37bb` | Protect account intervals across all six possible upstream stopping points |
| `c529030e6a10163c9c9e16a45c24a50bf2d64cac` | Independent regression of the combined account confidence budgets |

The new worktree leaves the original dirty checkout and earlier worktrees in
place. [Preservation verification](evidence/preservation-verification.json)
confirms all **369 registered files** unchanged, including the 26 original and
25 newer forecast registries and original failing Monte Carlo report. All 25
newer monthly ensembles and source-row lineage were regenerated exactly without
loading outcomes: [receipt](evidence/frozen-25-reproduction.json). The original
26 are preserved by hash, not claimed to have been re-fitted or freshly validated.

Only two pre-existing implementation files changed: `macro/resampling.py`
(optional lineage-only generation with the default interface retained) and
`macro/artifacts.py` (crash-safe writes). Existing research, strategy definitions,
workflows, lockfile, prior reports and sealed evaluation thresholds are unchanged.
The README gains a link to this continuation; earlier report sections remain.
There was no merge conflict, force push, main merge, paid purchase or trade.

## Findings ranked by financial importance

| Severity | Finding and likely effect | Result |
|---|---|---|
| Critical | A model can have precise simulated percentiles while missing actual crash and bear-market mechanisms | All candidates remain unapproved; historical rejection preserved; genuine prospective outcomes pending |
| Critical | The finite daily-template support cannot extinguish a 3× fund in one session | Exhaustive support minimum gross return is 0.442; zero simulated liquidation is structural, not evidence of real-world safety |
| High | Fixed 1,024/2,048-path prefixes are too noisy for terminal quantiles, increasingly so at high leverage/long horizons | Larger registered study passes at 1,048,576 paths per independent seed; old tests still fail |
| High | Probability precision was being confused with wealth-percentile precision | Separate exact binomial probability and order-statistic quantile intervals; simultaneous error control across quantities and planned looks |
| High | Repeated daily materialization made adequate sample sizes costly | Exact daily summaries and finite lookup preserve the reference distribution and its tails; no clipping or attractive-path selection |
| High | Plain cumulative products can confuse numerical underflow with product closure | Finite log NAV retained below floating wealth underflow; real daily losses of at least 100% remain absorbing |
| High | Forecast artifacts could be partially published or lost with a local cache | Atomic fsynced artifact identities, portable 62-file archive, Git-hosted aggregate commitments; full off-host/WORM backup remains unverified |
| High | TQQQ's reconstructed compounding bias remains material | +2.427 pp CAGR error; training-only financing attribution does not fix it; no convention selected to manufacture a pass |
| High | A precise account endpoint is narrower than a full account simulation | Canonical-ledger endpoint maps explicitly reject unsupported flows, taxes or extinction; exact daily after-tax drawdowns remain unverified |
| High | Historical financing, index income and fund distributions remain incomplete | No new certification of pre-inception funds, derivatives, MA fund taxes or advanced strategies |

## Monte Carlo diagnosis and numerical before/after

All 15 original failures are individually measured in
[original-15-diagnosis.json](evidence/original-15-diagnosis.json). They occur
across every 5/10/20/30/40-year × 1/2/3× combination. Dispersion of log terminal
wealth increases with horizon and leverage; stable event frequencies do not
imply stable quantiles. The original fixed prefixes do not become accurate just
because a larger master sample is generated. Pilot density-based path-count
estimates are labeled approximations and were not used to move a gate.

The [predeclared protocol](protocol.json) preserves the **0.02 log-unit**
quantile tolerance and **0.025 probability half-width**. Two independent root
seeds, `20261127` and `20261128`, use separate fixed 4,096-path stream blocks.
Scheduling and checkpoint chunk size cannot change a registered path's stream.
There are six planned checkpoints. Exact order-statistic intervals allocate
family error 0.025 across 45 quantiles and all six looks; separate exact binomial
intervals allocate 0.025 across 45 probabilities and all six looks. The union
bound therefore controls combined family noncoverage at no more than 0.05,
conditional on independent draws from this model. Seed differences and latest
doubling differences must also stay within 0.02. These are not confidence
intervals for the true market distribution. [Statistical references](SOURCES.md).

| Paths per seed | Combinations passing / 15 |
|---:|---:|
| Original 4,096 study | 0 — original fixed-prefix test unchanged |
| 32,768 | 0 — first larger checkpoint has no prior doubling |
| 65,536 | 5 |
| 131,072 | 8 |
| 262,144 | 11 |
| 524,288 | 13 |
| 1,048,576 | 15 |

Final paired sample: **2,097,152 paths per combination**, sharing market paths
across leverage and horizons. Maximum simultaneous quantile interval error:
**0.01684057 log units**; maximum independent-seed difference: **0.00839349**.
The same tolerance is applied to every combination, not a separate relaxed
long-horizon threshold. [All 15 before/after values](NUMERICAL_RESULTS.md) and
[full report](evidence/precision-report.json) include every checkpoint and event
interval. Precision of the 10th/50th/90th percentiles does not establish precision
of the 0.1st percentile or an unobserved crisis frequency.

### Efficiency and liquidation behavior

The existing joint stationary bootstrap selects source months; the empirical
bridge chooses a circular daily-template offset. Both have finite support.
`MonthlyLookup` enumerates those shapes and the 72 previous-rate states once,
applies the same daily reset/cost equation, and composes terminal log growth,
peaks and drawdowns exactly. It does not replace daily resetting with monthly
leverage or introduce a new forecast model. Synthetic 21-session months, a
252-session annual cost clock, lagged monthly rates, 1% leveraged fee and 1.5%
borrowing spread remain explicitly hypothetical assumptions.

On authentic training inputs, 128 forty-year paths × three leverages × five
horizons produced **1,920 scalar comparisons** against the canonical daily-reset
engine. Maximum log-NAV error was **3.02e-14** and drawdown error **4.16e-15**.
The amortized evaluation kernel was **8.58×** faster, excluding lookup setup
(4.07 s) and path generation on both sides. The full 2.1-million-path experiment
took about **498 s** here; this is a machine measurement, not a runtime promise.
[Benchmark](evidence/same-path-benchmark.json).

The entire enumerated support has minimum pre-closure daily gross returns
0.8143 / 0.6281 / 0.4420 at 1× / 2× / 3×. No template can create immediate
closure; longer repeated losses can approach zero. Tests independently verify
absorbing closure on sufficiently bad daily returns and finite log NAV below
−2,000 without fictitious liquidation. No sample had numerical underflow or
actual modeled closure in this study. **The very small binomial upper bound
from zero sampled closures is not an upper bound on real future ruin risk.**

Both complete sample binaries match their frozen SHA-256 hashes. First and last
stream blocks, 16,384 paths in total across seeds, regenerate every stored field
exactly: [reproduction evidence](evidence/precision-reproduction.json). The
remaining paths are verified by full-file checksum, not independently regenerated.

### Sampling error versus model and source uncertainty

Four additional specifications were registered before execution in
[sensitivity-protocol.json](sensitivity-protocol.json). All use the existing
reference model family and fixed independent seeds, with 65,536 paths per seed.
They do not all pass the tighter simultaneous precision gates; failures and
intervals remain in [the sensitivity report](evidence/specification-sensitivity.json).
They are sensitivity estimates, not approved replacements or a weighted model
ensemble.

| Specification | 40-year 3× median unit NAV |
|---|---:|
| Reference: 480 months, mean 12-month duration, fee 1%, spread 1.5% | 110.13 |
| Shorter 240-month history | 247.95 |
| Mean duration 24 months | 116.70 |
| Fee 0.5%, spread 0.5% | 301.41 |
| Fee 2%, spread 3% | 22.38 |

This range greatly exceeds within-model Monte Carlo sampling uncertainty. It is
**not** a posterior interval and does not span all source-vintage, structural,
parameter or future-policy uncertainty. No probabilities are assigned to these
specifications. No conditional investment rankings are released.

## Prospective forecasting: frozen now, score only genuine future outcomes

The prior 26 + 25 forecasts remain consumed historical evidence. Original
advanced models retain their 3-of-6 severe-drawdown failures; all four newer
candidates retain their corrected rejections. This continuation changes neither
their parameters nor their evidence labels to obtain a promotion.

Two simple reference forecasts (`joint_block` and `stationary_block`) were issued
at **2026-11-25T12:02:21.843683Z**, 2,000 paths each, for **December 2026 through
November 2027**. Training ends September 2026; unobserved October/November states
are simulated as a blind bridge. No November outcome is scored. The seven-quantile
monthly and daily distributions and crash/drawdown/underwater predictions are
public in [the commitment](evidence/prospective-commitment.json), committed at
`2857292336541e95db500243a7dd22a39f5db063` before any target month.
[Hosted CI for that commitment](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45610877064)
passed. The current result is **PENDING_GENUINE_FUTURE_OUTCOMES**, zero evaluated
months, no promotion. There is no numerical before/after predictive improvement
claim because no genuine prospective outcome exists yet.

Assessment uses proper weighted interval scores with the same training scale
for both references, daily WIS/coverage/lower-tail breaches, and completed prefix
drawdown, underwater and −5%/−10% crash-rate diagnostics. It refuses incomplete
actual daily months using the XNYS session calendar, never fills missing returns,
never backdates an issue and cannot promote a model. Overlapping 1/3/12-month
prefixes are labeled dependent. The test-only artificial future clock verifies
the assessor; it is not observed market evidence. Monthly/credit data release
lags can defer scoring even after the calendar month ends.

Full ensembles and 52 source snapshots are in a **62-file locally verified sealed
archive**. Every member and source hash is verified; partial artifact writes do
not publish valid identities. Directory fsync strengthens persistence across
crashes. [Archive receipt](evidence/archive-verification.json) explicitly reports
**off-host backup false and immutable-storage verification false**. Git preserves
the aggregate commitments off-host, but licensed raw inputs/full ensembles still
need an operator-controlled private off-host backup for whole-host-loss recovery.
No credentials, subscriptions or unapproved external storage were used.

## Leveraged ETF reconciliation: material failure retained

The additional attribution uses the same 2019 training cutoff and already
inspected 2020–25 evaluation, original observation counts and original thresholds.
It is not a new untouched holdout. Dated sources and retrieval failures are in
[SOURCES.md](SOURCES.md) and [source-review.json](evidence/source-review.json).

| Fund / convention | Earlier error, pp | Starting source-reconciled error, pp | This continuation, pp | Status |
|---|---:|---:|---:|---|
| TQQQ, Nasdaq-origin XNDX TR | +2.303 | +2.427082 | +2.427061 with training-only effective funding | FAIL; no material improvement |
| TQQQ, price-only sensitivity | Not the original convention | Not a replacement | −0.707627 | Income/swap/collateral convention unresolved; BLOCKED |
| UPRO | +2.082 | +1.024 | Unchanged | Limited numerical pass; exact target TR provenance BLOCKED |
| QLD / SSO / TMF | Prior limited tests preserved | See prior report | Unchanged | No new financial-source certification |

TQQQ's training-only constrained effective financing coefficient hits essentially
the 2× lower bound; it does not explain the remaining compounding bias. A
price-index sensitivity must not be selected merely because its CAGR passes:
issuer NAV total returns also reflect income and actual derivative terms. Generic
swap disclosures describe dividends and financing on contractual notionals, not
a verified historical daily exposure invoice. The 0.86% fee used in the existing
diagnostic is retained as a scenario, not relabeled an authentic dated schedule.
[Every diagnostic and its failures](evidence/reconstruction-attribution.json).

The September 2026 issuer summary shows 0.94% gross / 0.78% net expenses; the
2024 summary as supplemented April 2025 shows 0.97% / 0.84%. The contractual cap
is not the same as actual net expenses. These dated disclosures demonstrate why
a constant historical cost is uncertain; they do not complete daily fee waivers,
swap notionals, funding spreads or collateral yields for 2010–25. No current
expense ratio is silently backfilled.

An additional official-provider S&P 500 series was captured through FRED, with
genuine observations **2016-11-25–2025-12-31** in this request. It is a **price
index excluding dividends**, with a rolling ten-year history restriction, not
the missing exact total-return target history. It is cataloged, not substituted
for TR. Direct S&P export and SEC filing fetches returned HTTP 403; the attempted
Yahoo TR endpoint returned 429. Missing records are not invented or bypassed.

## Account results after the market precision gate

Only after the registered market study passed, **2,097,152 paired paths per
combination** were projected through `CanonicalTerminalMap`. Every map knot is
an actual `PortfolioLedger` buy, T+1 settlement, terminal sale and tax calculation.
Tax kink locations come from the existing dated tax engine, not fitted tax rates.
Independent interior/kink/random checks and a complete five-year daily-account
runner reconcile the map. No alternative tax engine or cash-balance authority
was created. This is exact endpoint memoization for its supported profile,
**not millions of full daily account-ledger replays**.

The profile is $10,000 initial cash, single, $100,000 other income, federal-only,
explicit frozen 2026 law, 2 bp spread / 1 bp slippage, no sweep, distributions,
contributions or intermediate trades. Roth is an existing balance and internal
sale, not a personal withdrawal. Tax-loss benefits are household receivables,
not immediately spendable brokerage cash. Future calendars are projected regular
sessions; unanticipated closures are not known. Tax is a terminal liability;
this narrow case has no intermediate taxable income requiring annual funding.

All **30 account combinations** passed simultaneous nominal and real-wealth
quantile checks; maximum interval error is **0.018446 log units**. Paired inflation
is taken from the same source-month paths. [Account results](ACCOUNT_RESULTS.md)
contain all five horizons, losses and drawdown bounds; the
[machine-readable report](evidence/paired-account-report.json) contains intervals,
terminal taxes and conditional inflation subsets. Large-outcome floating-point
ledger residuals reached $0.000122 on extreme wealth, within the predeclared
relative numerical tolerance; this is not a universal sub-cent guarantee.

Final review caught a missing multiple-look allocation in the first account
interval report: upstream sample size was selected from six planned checkpoints.
The corrected allocation protects all six possible stopping points as well as
all account quantities. Point wealth and paths are unchanged; the wider maximum
error rises from 0.016802 to 0.018446 and still passes the unchanged 0.02 gate.
The [initial report](evidence/initial-account-fixed-look.json) is retained as
superseded, not advertised as a valid simultaneous post-stopping interval study.

Nonzero distributions/contributions, MA, different tax inputs, intermediate
rebalancing and extinct/nonfinite fund NAV **explicitly reject this accelerator**
and require supported full canonical daily replay. Exact after-tax drawdown
timing is not stored, so the report supplies conservative bounds. Conditional
inflation subsets are observational model sensitivities, not causal forecasts.
These paths cannot establish calibrated actual-fund ruin or outperformance
probabilities. Recurring-contribution Monte Carlo and complete future fund taxes
remain incomplete.

Existing observed six-fund 2024–25 and seven-fund 2025 account demonstrations,
including actual distributions and tax-funded sales, remain unchanged in the
[prior report](../tail-validation-2026-11-24/REPORT.md). The authentic-daily
1930–2026 counterfactuals are also preserved unchanged. They remain hypothetical
mechanics applied to historical research returns, not actual pre-inception ETFs
or a 1930 Roth IRA. No re-ranking is inferred from the restricted new projections.

## Tests, gates and remaining work

Baseline: **246 tests passed**, four dependency deprecations. Final local:
**259 passed**, the same four warnings. **97 workflow checks passed**. Thirteen
additional tests cover independent coverage arithmetic, zero atoms, daily
compounding, stress underflow/closure, stream identities, atomic failed writes,
archives, prospective timing, missing sessions, account stopping-time allocation
and canonical account conformance.
An initial new fixture lacked inflation (one failure); it was fixed rather than
weakening production checks. No old test was deleted, skipped or xfailed.
[Baseline](evidence/baseline-246.txt), [final test stdout](evidence/tests-259.txt).

[Hosted implementation CI](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45613639269)
passed at `5d583a7` (257 tests, 97 workflow checks). Later hosted runs on the final
commit are identified in the delivery response; the earlier successful run is
not presented as testing later commits. The million-path experiments use licensed
local caches and are not run by hosted CI. Both sample-file hashes and registered
block regeneration are checked locally. Reproduction is documented in
[REPRODUCE.md](REPRODUCE.md), and capabilities in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md).

[Prospective-assessment implementation CI](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45666329250)
also passed at `0999165`: 258 tests and 97 workflow checks. It precedes the
account interval correction and is not used as proof of that later change.

The legacy realism function was also rerun: **8/10 pass**. T20 scores
59.5075 against its actual threshold 65; T27 calls an unsupported
`max_crisis_chain` argument. [Actual diagnostic](evidence/legacy-realism-current.json).
No crisis-damping behavior was introduced to improve the diagnostic score.

Unresolved failures: original 15 small-sample tests; original advanced/newer
forecast rejection; TQQQ reconstruction; provenance gates for all reconstructed
funds; legacy T20/T27 realism diagnostics; four lower-budget sensitivity precision
studies; full off-host archival verification. None is waived by passing CI.
Futures, options/LEAPS, advanced tax cases and unsupported strategy executions
remain blocked. They were not expanded to distract from precision.

Highest-value next actions are genuine future scoring after complete released
outcomes, private verified off-host source/ensemble retention, exact target-index
TR access, and dated issuer fee/derivative/collateral records. More simulation
paths can reduce sampling error; they cannot supply the missing market mechanisms,
law, data or evidence.
