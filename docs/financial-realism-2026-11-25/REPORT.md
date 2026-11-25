# Financial realism and predictive-validation continuation

Reviewed November 25, 2026. Branch `develop/financial-realism-2026-11-25`, based
exactly on `1fb825011573748536239c190e07ba9fac19caed`. Implementation through
`381cc1ee8b7abab89d61afb6c65a56a6d7580b97`; subsequent commits contain this
evidence/report. The delivery response identifies the final HEAD and hosted CI.

**This release improves conditional financial mechanics and exposes failures.
It does not establish investment-prediction skill. No forecasting model is
approved. TQQQ reconstruction and actual off-host backup verification remain
blocked. No new forecast model, strategy ID or Monte Carlo probability study
was added.**

## Implemented stages

| Commit | Delivered work |
|---|---|
| `4c64fca` | Register cases, old anchor paths, account assumptions and unchanged numerical gates before execution; preserve 259-test baseline |
| `c165421` | Unweighted catastrophe scenarios, daily-reset costs, actual projected session counts, full canonical daily account replays and independent cash/unit-return audits |
| `7c6070a` | Private research bundle, authenticated safe restore, restrictive permissions, traversal/link/tamper/overwrite protections |
| `f16d3dc` | Official 2021 issuer NAV/index return and dated swap-financing reconciliation; exhaustive old-model support audit |
| `619bcc0` | Block unsupported zero-residual taxable dispositions; preserve Roth internal extinction; document market-halt limits |
| `ce4bd34` | Stream large compressed archives; separate numerical diagnostics from legal/evidence acceptance |
| `b48f006` | Explicit zero recovery on unpaid Roth distributions; principal-withdrawal checks |
| `64b32a5` | Observed TQQQ contributions/reinvestment/annual-tax replay, split-aware independent audit and fixture-date correction |
| `381cc1e` | Stronger valid-manifest recovery tests, blocked-matrix exit status and correct legacy/current recovery-assumption handling |

The original checkout/user edits, historical studies, frozen strategies, workflows,
dependency lock, sealed forecasts and prior failures remain in place. All **402
registered prior artifacts** match their recorded hashes; all **586 tracked
baseline files** were inventoried. Only `accounts.py`, `account_backtest.py` and
the README are intentionally changed pre-existing files. New modules reuse
`PortfolioLedger`, `run_account_backtest`, `daily_reset_returns`, the existing
empirical bridge and frozen model artifacts. [Preservation check](evidence/preservation-check.json).
Generated Serena metadata stays local/ignored and is excluded from research
archives; original tooling metadata is untouched. No user edits were stashed/reset, and no main merge, live trade, paid
purchase, tax filing or raw-data upload occurred.

## Severity-ranked accuracy findings

| Severity | Confirmed issue / financial consequence | Remediation or gate |
|---|---|---|
| Critical | Finite daily-template support cannot produce an extinguishing 3× daily loss, irrespective of simulation count | Exhaustive support enumeration; separate unweighted extreme-loss, persistent-bear, volatility and issuer-failure scenarios |
| Critical | A zero NAV mark could be treated as an ordinary same-day taxable sale, yielding an unsupported tax benefit/holding-period character | Canonical runner and low-level ledger now reject zero-residual taxable disposition before mutation; 60 matrix cases BLOCKED |
| High | Declared unpaid dividends could be assumed recoverable after issuer failure | Explicit Roth close-event writeoff, zero-recovery replay and retained full-recovery sensitivity; taxable bad-debt treatment blocked |
| High | Numerical precision was liable to be read as model/tail reliability | Preserve sampling intervals separately from regime/cost/structure/source/policy sensitivity; no case weights or new probability estimates |
| High | Literal intraday broad-market −25%/−34% marks conflict with normal market-halt assumptions | Treat as structural-break valuations, not certified executable closes; no invented intraday fill tape |
| High | TQQQ financing was fitted without a complete historical actual-notional/rate schedule | Source actual May 2021 swap rates/notionals and fees; original +2.427 pp reconstruction failure retained |
| High | Full-history model/account results lacked independently reconciled flows and actions | Daily contributions, real/hypothetical distributions, annual cash taxes, settlement, rebalancing, splits, unitized drawdown and supported withdrawals now demonstrated together |
| High | A local archive cannot survive loss of its only host | 551-file private bundle and complete recovery drill verified; actual off-host destination/transfer still unapproved |

## Catastrophic-risk representation

[Support enumeration](evidence/support-audit.json) checks all **10,034 finite
daily shapes × 72 lagged rate states** in the unchanged fixed-21-session
reference. Its worst transformed benchmark day is −18.569%; minimum daily gross
fund returns are 0.81431 / 0.62813 / 0.44199 at 1× / 2× / 3×. None can extinguish
the fund in one session. Stationary blocks can concatenate long bears, but they
cannot identify frequencies of shocks outside the empirical support. Zero
modeled extinction therefore says nothing reassuring about actual future ruin.

The [registered protocol](protocol.json) adds ten **unweighted** cases: reference,
−25% daily loss, −34% daily extinction stress, −50% overnight stress, seven-year
bear, seven-year stagnation/high rates, ten-year inflation/funding shock, doubled
daily volatility, issuer failure and higher costs. Daily reset compounding,
ACT/365 fees and ACT/360 net borrowing use the existing mechanics engine.
Volatility effects emerge from daily compounding, with no extra decay charge.

The −34% day extinguishes 3× exposure; −50% overnight extinguishes 2× and 3×
exposure before new opening trades. The issuer-failure case closes a fund even
when the benchmark itself does not crash. Closure is absorbing, with zero
positions and no later fund recovery or post-closure internal fee deductions.
New contributions after closure stay in cash. Twenty percent initial reserve
cash and subsequent contributions mean **fund extinction is not necessarily
zero account wealth**; flow-neutral returns do not count deposits as recovery.

These are stipulated conditions, not assigned event probabilities. Modern
market-wide circuit breakers, suspensions and future rules prevent claims of
exact execution through these hypothetical marks. See [LEGAL_SCOPE.md](LEGAL_SCOPE.md).
A surviving close-to-close path does not establish that no intraday extinction
or collateral threshold was crossed. Intraday evidence and actual issuer terms
would be required to certify that behavior.

## Full daily account matrix and uncertainty

Two old registered first-path anchors are regenerated from the **existing** seeds
20261127/20261128, stream block 0, path index 0. No new random-path sample is
selected or evaluated for forecasting skill. The existing bridge preserves source
monthly totals/variance while mapping to projected XNYS session counts. The first
market row is an explicit zero-return enrollment valuation. This is a different
declared calendar/accrual convention from the earlier synthetic 21-session clock.

The matrix has **600 registered cases**: two anchors × ten assumptions × three
leverages × five horizons × two accounts. Every case uses the same bounded
profile: $10,000 initial, $3,000 contributions at each year's first session,
80% fund/20% cash, 63-session rebalancing, assumed ordinary quarterly 0.5%
distributions with matching ex-price reductions and five-session cash delay,
2 bp full spread/1 bp slippage, zero sweep yield, single/$100,000 other income,
federal-only and **explicitly frozen 2026 tax/eligibility law**. Dividends are
assumptions, not predictions of actual TQQQ or other funds. Interest on reserve
cash remains zero here; a broker-specific sweep is not silently credited at the
funding benchmark.

The raw arithmetic run reconciled all 600, but legal/source acceptance is separate:
**540 supported hypothetical-mechanics cases; 60 taxable extinction cases
BLOCKED**. Original zero-price ordinary-sale tax diagnostics are retained privately
as superseded evidence, excluded from supported after-tax output. The assessment
verifies current canonical API rejection for every affected case. It also reruns
30 Roth issuer-default cases through the daily ledger with zero unpaid-dividend
recovery, retaining full recovery as an unweighted sensitivity.

A separate current-code 30-case reproduction yields the same 15 supported and
15 blocked statuses for its selected issuer-default subset. Maximum supported
ending-wealth difference from the full assessment is 1.82e−12 dollars. Twenty-
seven completed reference replays before/after calendar caching have exactly
equal summaries. These checks establish implementation fidelity, not forecasting
skill. [Current workflow](evidence/current-workflow-reproduction.json),
[cache reconciliation](evidence/calendar-cache-reconciliation.json).

Maximum reported supported residuals: cash **0**, NAV **1.49e−8 dollars**, quantity
**1.46e−11 shares**. The independent exported-event terminal cash identity error
is **0**. Drawdown and underwater time are independently recomputed from flow
valuations/units, not copied from a terminal wealth map. Reported drawdowns use
cash taxes paid during the path plus terminal tax payable; **daily hypothetical
liquidation-tax NAV is not implemented**. Supported owner withdrawals and Roth
principal ordering are tested separately; taxable/penalized personal Roth
withdrawals retain their existing rejection.

[Results and 5–40-year tables](RESULTS.md) and
[complete supported/blocked assessment](evidence/full-stress-confidence.json)
show nominal/real wealth, money-weighted returns, flow-neutral drawdowns,
underwater time, taxes, costs, carryovers and source hashes. Two named anchors
are not a confidence sample. Their ranges/failures do not estimate real-world
loss or outperformance probabilities.

Numerical error remains the earlier fixed-model uncertainty: maximum registered
log-quantile CI error 0.016841. Historical-regime/duration/cost uncertainty is
separate: the prior 40-year 3× median unit NAV ranges from **22.38 to 301.41**
across registered specifications, versus reference 110.13. Those smaller-budget
specification studies still fail their complete precision gates. These ranges
have no posterior weights or coverage guarantee. They also have a different
no-flow/no-distribution unit-NAV profile from the full daily account matrix.
Source-vintage, future tax law, market structure, collateral and model errors
are not identified by a narrow Monte Carlo interval.

## Observed TQQQ account comparison

An existing constant 100% allocation is replayed on **2024-01-02–2025-12-31**,
with real reviewed quotes, splits and classified issuer distributions. Both
alternatives receive $10,000 initial capital plus $3,000 in January 2024 and
January 2025 and reinvest every 63 sessions. Actual supported 2024/2025 federal
tables are used, with $100,000 ordinary income, federal-only and the same costs.
Qualified distributions use actual unhedged common-share holding records.

| Account | Net ending wealth | Annual tax cash paid | Terminal tax payable | Money-weighted annual return | Unitized max drawdown |
|---|---:|---:|---:|---:|---:|
| Taxable | $29,783.39 | $54.99 | $2,490.80 | 40.37% | 57.98% |
| Roth internal liquidation | $32,386.47 | $0 | $0 | 46.85% | 57.98% |
| Tax-disabled account-neutral control | $32,386.47 | $0 | $0 | 46.85% | 57.98% |

Roth/control pre-tax trades match exactly within declared floating tolerances.
The actual taxable case has one tax-funding sale settling before payment.
All daily cash/position identities and split-adjusted terminal inventories
reconcile independently. Contributions are not labeled lump-sum CAGR; the
money-weighted return is reported separately. Actual fund expenses/financing
remain embedded in observed prices and are not deducted again.
[Full comparison](evidence/observed-funded-comparison.json).

These are alternative single-account household scenarios, not simultaneous owner
trades in multiple accounts. Unobserved external wash replacements are explicitly
assumed absent. Roth uses an existing balance plus checked contributions; internal
sale is not a qualified personal withdrawal. Execution uses assumed costs at
observed daily opens/closes, not actual NBBO fills. MA fund adjustments, full
household credits/AMT, estimated-tax safe-harbor penalties and complex hedged-QDI
tax rules remain outside this bounded demonstration. No strategy ranking or new
historical holdout validation is inferred from these known market years.

## Actual fund-mechanics evidence

The downloaded **May 31, 2021 ProShares annual report** is pinned by SHA-256
`89149fc0bd2121f0093632815f84616c17e3f4ed3f98b84d7fa4084f3f07b915`.
It identifies ex-date NAV reinvestment, dated expense disclosures and actual
swap exposures/rates. [Facts and page numbers](evidence/issuer-2021-facts.json),
[reconciliation](evidence/issuer-reconciliation.json).

| Holding period ending May 2021 | NAV API vs published annualized return error, pp | Official-origin XNDX vs published index error, pp |
|---|---:|---:|
| 1 year | −0.001579 — pass | +0.003245 — pass |
| 5 years | −0.002399 — pass | +0.011522 — FAIL |
| 10 years | +0.004740 — pass | +0.016272 — FAIL |

The predeclared publication-rounding limit is ±0.005 percentage points. A separate
share-reinvestment calculation agrees with compounded NAV returns to about
2.2e−15 relative error. These overlapping periods check a separate issuer
publication against its API; they are not independent forecast eras or a third-
party audit opinion. The index discrepancies are not waived.

Nine swaps reconcile exactly to $26.963 billion notional against $12.062 billion
NAV at that snapshot. The notional-weighted floating financing rate is **0.6074%**,
implying **1.3578% annualized gross swap financing per NAV at the reporting date**.
This excludes commissions, collateral income and other positions and is not
the year's realized expense. Stock exposure was about 71.88%, swaps 223.54%
and futures 4.60%. A single reporting-date mix cannot be backfilled as daily
historical exposures or interpreted as a contractual spread over federal funds.

The October 2020 prospectus disclosure in that annual report shows 0.98% gross /
0.95% net operating expenses, not the constant 0.86% used in the preserved
reconstruction diagnostic. Dated disclosure is not a daily realized invoice.
**TQQQ's +2.427 pp CAGR error remains unchanged and failed.** UPRO's exact target
TR provenance remains blocked; QLD/SSO/TMF receive no new certification. Daily
derivative notionals, financing benchmarks/spreads, collateral cash yields,
fee/waiver history and exact historical index conventions are still needed.
Neither changing to a better-fitting price-only index nor fitting to the already
inspected 2020–25 period is accepted as a repair.

## Prospective forecasts and research recovery

The 26 original + 25 newer historical forecasts and December 2026–November 2027
commitments are unchanged. The current prospective assessment is
**PENDING_GENUINE_FUTURE_OUTCOMES**, zero months evaluated, no promotion.
[Receipt](evidence/prospective-assessment.json). New stresses do not alter those
distributions, invent probabilities or turn consumed history into untouched data.

A private **551-file, 814,575,466-byte** archive contains original/later/future
forecasts, exact macro/index/price/tax sources and full precision binaries.
Archive SHA-256:
`e002457010331b9aea760a01286e59dcad97f6aadaac5070fb56a27d6547a707`.
A fresh local recovery checks every member, permissions, 52 source snapshots,
26+25+2 forecast artifacts, 53 forecast models and all precision sample bytes.
Unsafe paths, links, credentials, tampering and replacement of existing targets
fail explicitly. Large compressed archives are verified/restored sequentially,
avoiding repeated decompression. [Receipts](evidence/recovery-semantic-check.json).

**Actual off-host verification remains blocked: no approved private destination
was supplied.** The local bundle does not protect against losing this host.
The [backup/recovery procedure](BACKUP_AND_RECOVERY.md) is concrete and locally
tested; it requires an approved destination, remote checksum/readback and a new
restore drill before claiming an off-host backup. Raw data are not in Git or
uploaded to an unapproved service. Provider rights, access and retention remain
destination-specific requirements.

The new issuer publication and exact observed-fund input bundle are preserved
in a separately verified 11-file private delta. Replaying the restored bundle
reproduces the complete observed-account report exactly.
[Recovery replay](evidence/restored-observed-reproduction.json).

A separate private Git bundle preserves 86 committed refs through `efbdb48`,
including the original unpublished audit history. A clean isolated clone passes
object-integrity checks and restores all seven required historical commits.
The original checkout's 14 tracked user edits/deletions are also preserved in a
private patch and reproduced byte-for-byte only in that recovery clone.
Untracked user/tool directories are excluded; the original checkout is unchanged.
These additional artifacts are also local, with no off-host protection claim.
[History/edit recovery](evidence/git-history-recovery.json).

## Tests, CI and remaining blockers

Baseline **259 passed**, final **282 passed**, with the same four inherited
dependency warnings; **97 workflow checks pass**. [Test stdout](evidence/tests-282.txt).
[Hosted CI](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45685106917)
passed all 282 tests and 97 workflow checks at `efbdb48`; the delivery response links CI for the final documentation
commit. Hosted CI runs the pinned fixture suite, not local licensed-input studies.
Compilation and whitespace checks pass. No tests were skipped, xfailed, removed
or weakened to obtain a green result.

Intermediate failures remain auditable: a holiday-bound calendar construction
error, serialization of unbounded bracket metadata, a NumPy boolean conversion
and a test-only date-index mismatch were corrected. A new fixture's unsupported
zero-price tax-sale expectation was replaced with the stricter source-backed
rejection; the original 259 tests are unchanged. The interrupted slow pre-cache
study and raw pre-tax-guard diagnostics are retained, never presented as fresh
predictive validation. Unchanged legacy T20/T27 still fail: 8/10 realism checks
pass. The original 15 small-sample failures, reconstruction gates and rejected
macro candidates remain visible.

Futures, options/LEAPS, advanced tax cases, broker-specific IRA permissions and
unsupported historical classifications remain blocked. Cash sweep/collateral
yield, halt/reopening fills and daily liquidation-tax valuations are not silently
invented. See [support matrix](SUPPORT_MATRIX.md), [sources](SOURCES.md),
[numerical results](RESULTS.md) and [reproduction commands](REPRODUCE.md).
