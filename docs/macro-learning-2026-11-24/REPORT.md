# Historical macro learning: implemented system and evidence

Development branch: `develop/macro-learning-2026-11-24`, based exactly on
`553ce1af5e7d02846882cf7cb3c0fc3d8d59a7bd`. Scientific implementation:
`7096a17998094bd8d83ee0ab25e76478893f63ec`; workflow continuity checks:
`36d70cc026698cc693cce66fb3e6d3d6723aeab0`. Later delivery commits add this report
and evidence. Nothing was merged into main.

The branch implements historical account replay, daily counterfactual exposures
back to 1930, three competing macro models, forecast-first walk-forward scoring,
a prospective model registry, controlled retraining and a canonical account
scenario explorer. **No candidate model passed promotion. This is a bounded
research platform, not a certified investment-performance or tax-filing system.**
Some acceptance criteria remain blocked by data; others require further
engineering. They are separated below and in the [support matrix](SUPPORT_MATRIX.md).

Implementation stages are reviewable in Git: `51249bf` (multi-year issuer replay),
`dd460ae` (frozen macro study), `1bc76bf` (tax/signals/liquidation/legacy adapters),
`62e9693` (long option delivery), `6839af4` (QQQ/TMF), `f753984` (scenario/registry),
`e52e339` (historical provenance/diagnostics), `f9a4e52`/`36d70cc` (workflow
continuity/syntax), `b17e093` (issuer capture), `b7782f3` (Roth input separation),
`cff938c` (option cash/depth), `506fc11` (dated settlement) and `7096a17` (fixture
correction). No merge conflicts arose: this mission branched directly from the
already reconciled foundation and did not replace main or the original checkout.

## Preservation and validation

The original integration/audit commits remain ancestors or preserved branches.
Development used a separate worktree; the original working checkout and its
uncommitted user edits were not checked out or rewritten. The [foundation
inventory](evidence/foundation-inventory.json) records all tracked files read and
hashed, an AST scan and targeted manual review. It does not assert line-by-line
financial certification of every archived study.

The [preservation check](evidence/preservation.json) confirms **276 protected
tracked files unchanged** from the supplied base: existing research, frozen and
sealed documents, data, workflows, legacy strategy/configuration and dependency
lock. New captures/results use new directories. No sealed study was rerun or
overwritten. The model study was independently reproduced in a detached checkout
of its frozen source: **all 26 forecast IDs, evaluation records and model
decisions matched exactly**. This is numerical reproduction, not new market
evidence. [Reproduction receipt](evidence/walkforward-reproduction.json).

| Check | Result |
|---|---|
| Supplied foundation | 152 passed, four warnings, 49.69 seconds |
| Final local suite | **221 passed**, four pandas-datareader deprecation warnings, 31.38 seconds |
| Hosted regression CI | **221 passed**, four warnings, 30.84 seconds, Python 3.14.10 |
| Workflow structure / duplicate keys | **97 workflows pass** |
| Active package, scripts, tests, free futures research and QC prototype compilation | PASS |
| Independent saved fund CSV reconciliation | Six funds, both accounts: PASS |
| Canonical future account scenarios | 80 retirement + 16 stagnation + 16 crash replays completed |
| Original T20 | **FAIL**, 59.5074953334 below unchanged 65 threshold |
| Original T27 | **FAIL**, unsupported `sample_returns(max_crisis_chain=...)` argument |
| TQQQ / UPRO reconstruction | **FAIL**, unchanged CAGR thresholds |
| Macro promotion | **REJECTED**, tail, statistical and provenance gates fail |

[Local log](evidence/pytest-final.txt), [hosted run 45525817221](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45525817221),
[CI metadata](evidence/hosted-ci.json), [mechanics report](evidence/mechanics-final.json).
The optional retraining job is implemented but disabled by default; its remote
scheduled operation was not activated or certified. YAML checking caught and
corrected an intermediate conditional-quoting error before publication.

Intermediate [CI run 45525663121](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45525663121)
failed five newly added settlement fixtures: a default 20-year calendar excluded
1998 and `Timestamp` lookups did not match ledger `date` indices. Explicit calendar
bounds/date keys fixed the fixtures; all five financial settlement assertions
remain. [Failure excerpt](evidence/intermediate-ci-fixture-failure.txt).

The previously requested failed run
[45311541299](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45311541299)
remains documented in the [preserved integration report](../integration-2026-11-24/REPORT.md):
a duplicate YAML job key prevented jobs from starting. There were no original
runtime test logs to reinterpret as a financial-model failure.

## Confirmed defects repaired in this mission

| Severity | Defect / effect | Implemented correction |
|---|---|---|
| High | Real fund tax components blocked SSO/UPRO; REIT dividends cannot simply be ordinary income with no Section 199A review | Bounded 2024–25 qualified REIT-dividend deduction, taxable-income cap, NIIT distinction and MA nonconformity; independent examples in `test_section199a.py` |
| High | Nominal split jumps could enter strategy signals and create artificial crashes | Causal split-normalized price history and cash-distribution total-return reference in `account_backtest.py`; execution remains nominal |
| High | Zero-NAV fund loss had no executable liquidation path | Sourced open/close liquidation events through `PortfolioLedger`; no repurchases or resurrection of closed units; correct total-loss return with later contributions |
| High | Option-sale cash was immediately reusable in a cash IRA | T+1 unsettled proceeds; prior settlements mature before new buying-power checks |
| High | Earlier equity replay assumed T+2 before its 2017 introduction | Dated T+3/T+2/T+1 rules, holiday-transition tests and purchase delivery dates; unreviewed pre-June-1995 settlement rejects |
| High | Missing option depth behaved like unlimited depth; zero-bid closing could create commission debt | Integer/nonmissing depth validation and settled-cash commission funding with independent rejection tests |
| High | Physical long exercise was absent | Explicit holder notice, exact strike delivery through the shared stock ledger, sufficient cash/owned shares, settlement and exercise fee; unsupported OCC adjustments reject |
| High | Historical learning/forecast audit trail was absent | Immutable forecasts before scoring, fixed protocol, proper scores, disjoint-era tests, consumed evidence windows and guarded model registry |
| High | Naively dropping missing macro months could fabricate VAR transitions | New exploratory/prospective version fits only adjacent complete observations and bootstraps complete contiguous blocks; no missing CPI interpolation |
| Medium | Short numerical debit residue was treated as a margin loan for sweep interest | Roundoff tolerance only; actual cash debit and negative sweep assumptions still reject |
| Medium | A string `False` verification flag evaluated as true | Strict tax-verification parsing |
| Medium | Scenario/CLI cash flows invented owner age/basis/clock history | Explicit owner history required; invalid pre-1998 Roth clocks reject |
| High | Ordinary income was used as Roth-eligible compensation | Separate explicit MAGI and eligible compensation; pension/interest income cannot silently authorize contributions |
| Medium | Old counterfactual CAGR excluded its first return interval; proxy provenance did not identify each input file | Dated previous-close boundary and individual source hashes; earlier report retained |
| Medium | S8/S11–S19 were disconnected from canonical execution | Versioned allocation adapters with independent legacy-return checks and canonical replay tests; S16 look-ahead and >100% cash allocations gated/versioned |

Core files: [`account_backtest.py`](../../letf/account_backtest.py),
[`accounts.py`](../../letf/accounts.py), [`tax/marginal.py`](../../letf/tax/marginal.py),
[`derivatives.py`](../../letf/derivatives.py), [`strategy_allocators.py`](../../letf/strategy_allocators.py),
[`macro/`](../../letf/macro/). The old strategy implementation/configuration is unchanged.

## Real fund account comparison

2024-01-02 through 2025-12-31, $100,000 existing account balance, single filer,
$100,000 annual ordinary income, explicitly **federal-only**, zero commission,
2 bp full spread plus 1 bp slippage, actual issuer cash and split actions,
annual tax payment funding and terminal internal security liquidation. These
are daily observed OHLC with assumed fills, not actual NBBO executions. No fund
expense or financing charge is subtracted again from observed fund prices.

| Fund | Taxable net account wealth | Roth account wealth | Taxable real CAGR | Roth real CAGR | Distribution policy |
|---|---:|---:|---:|---:|---|
| TQQQ | $199,462.77 | $218,422.99 | 37.872% | 44.289% | Reinvest |
| QLD | $176,021.22 | $189,479.45 | 29.500% | 34.370% | Reinvest |
| SSO | $170,886.98 | $183,530.45 | 27.594% | 32.240% | Reinvest |
| UPRO | $201,078.12 | $220,111.05 | 38.430% | 44.847% | Reinvest |
| QQQ | $144,792.24 | $152,696.76 | 17.429% | 20.598% | Accumulate cash |
| TMF | $64,037.36 | $64,245.66 | −21.971% | −21.844% | Accumulate cash |

**This is not a six-fund ranking:** the two distribution policies differ.
For each paired account comparison, pre-tax fill CSVs match exactly. Tax-funded
sales and foregone reinvestment can subsequently change actual taxable trades.
The largest independent daily NAV identity residual across these fund replays
is below $0.000000001; every terminal position is zero within numerical tolerance.
[Independent checks](evidence/independent-fund-reconciliation.json),
[source manifests](evidence/fund-source-manifests.json), detailed results:
[TQQQ](evidence/fund-TQQQ.json), [QLD](evidence/fund-QLD.json),
[SSO](evidence/fund-SSO.json), [UPRO](evidence/fund-UPRO.json),
[QQQ](evidence/fund-QQQ.json), [TMF](evidence/fund-TMF.json).

For TQQQ, 2024 investment tax of $422.91 is paid in April 2025 and terminal
2025 payable is $18,096.57. Roth routine internal investment tax is zero.
Liquidation proceeds can still be unsettled; net assets and spendable settled
cash are reported separately. Roth wealth here is **inside the account**;
personal distribution is not executed. Recurring contribution runs report IRR,
unitized time-weighted returns and wealth separately, rather than cash-flow-
inflated lump-sum CAGR. [Tax/source scope](SOURCES_AND_RESTRICTIONS.md).

## Leveraged reconstruction: failures remain failures

Same frozen vendor inputs, training through 2019, inspected 2020–2025 evaluation,
same numerical gate hash. All five before/after error records match exactly.
The extra diagnostics do not refit to the evaluation period.

| Fund | CAGR error before, pp | After, pp | Numerical gate | Overall certification |
|---|---:|---:|---|---|
| TQQQ | +2.303367 | +2.303367 | FAIL | BLOCKED |
| QLD | +0.553906 | +0.553906 | PASS | BLOCKED provenance |
| UPRO | +2.082450 | +2.082450 | FAIL | BLOCKED |
| SSO | +0.536560 | +0.536560 | PASS | BLOCKED provenance |
| TMF | +0.546444 | +0.546444 | PASS | BLOCKED provenance |

The one-constant-spread fit reaches its nonnegative boundary for both failing
3× equity products. Net-fee QQQ/SPY/TLT proxies, funding conventions, changing
expense waivers, swaps/collateral and NAV-versus-market effects remain unresolved
explanations, not demonstrated corrections. Official target-index daily total
returns and dated product financing/expense evidence are needed. The current
[Nasdaq XNDX history](https://indexes.nasdaq.com/Index/History/XNDX) identifies
the proper total-return target but did not supply an independently licensed,
checksummed full history in this work.

[Unchanged-gate result](evidence/reconstruction-after.json),
[nonregression check](evidence/reconstruction-nonregression.json),
[rate/crisis and 1/3/5/10-year diagnostics](evidence/reconstruction-diagnostics.json).
Ten-year evaluation windows are blocked by the six-year evaluation coverage.
2020–2025 is never called untouched again.

## Daily 1930 counterfactuals

Authentic French daily research returns, including historical Saturday sessions,
run from 1930-01-02 through **2026-09-30**, the last retained observation, not
today's market close. Each underlying's genuine source begins 1926-07-01.
The 1933 closure uses independent dated evidence and accrues the full calendar
gap. Daily leverage/compounding creates path decay; no separate decay debit.

Central 3× scenario: 1% annual fund fee and 1.5% borrowing spread over a prior-
month bill-return funding proxy. The 108 Cartesian fee/spread/leverage cases
are assumptions, not 108 independent observations.

| Underlying research exposure | Nominal CAGR | Real CAGR | Max drawdown | 3× CAGR across fee/spread assumptions |
|---|---:|---:|---:|---:|
| Broad US market | 9.182% | 5.869% | 99.714% | 4.848%–11.982% |
| Large growth / low book-to-market | 8.470% | 5.179% | 99.600% | 4.165%–11.252% |
| Small growth / low book-to-market | 2.555% | −0.557% | 99.796% | −1.520%–5.188% |
| Historical HiTec industry | 3.071% | −0.056% | 99.984% | −1.023%–5.717% |

These are **HYPOTHETICAL_STRESS counterfactual products** on observed research
inputs, not observed ETFs, literal pre-1957 S&P 500, pre-1985 Nasdaq-100, or
1930 Roth accounts. The report includes nine era slices and 92/87/77/67/57
overlapping 5/10/20/30/40-year windows per central exposure. Overlap is explicit;
their quantiles are historical-window summaries, not forecast probabilities.
[Full counterfactual evidence](evidence/counterfactual-1930.json).

## Historical learning results

The protocol was committed before the study. Source version `dd460ae` trains
on past observations only, freezes every forecast, then loads evaluation data.
26 forecasts × 2,000 paths cover nine declared origins. The 1929 VAR is blocked
for insufficient training; 2021 is incomplete; overlapping eras stay diagnostic.
Six complete common disjoint eras enter comparison. [Methods](METHODS.md).

| Model versus joint block baseline | Mean CRPS improvement | 97.5% paired era loss-difference interval | Drawdown-tail breach | Decision |
|---|---:|---:|---:|---|
| Regime block | 0.262% | [−0.00677, +0.00143] | 3/6 eras | REJECT |
| Conditional VAR / Student-t | 3.721% | [−0.11142, +0.02423] | 3/6 eras | REJECT |

Neither confidence interval establishes improvement. Tail breaches exceed the
registered 30% ceiling and current revised equity history blocks historical
point-in-time promotion. The VAR's equity CRPS is **0.664763 versus 0.664420**
for the baseline, slightly worse; rate and credit forecasts explain most average
improvement. Both bootstrap variants miss severe Depression losses; stagflation,
2000s and the financial crisis expose additional tail failures.
[All eras, marginal scores, coverage, crashes and correlations](evidence/walkforward-v1.json).

The later missing-observation-aware exploratory/prospective variant is a distinct
implementation. It has not earned the older version's historical validation by
inheritance. No post-evaluation parameter tuning is presented as an improvement.

## Future explorer and controlled retraining

The explorer jointly simulates equity, inflation, short rates, production and
credit, creates explicitly synthetic conditional daily paths, then runs every
strategy/account through `run_account_backtest` and `PortfolioLedger`. Paired
market paths cover 1×/2×/3× broad-market counterfactuals and 60% equity/40% cash.
Government bonds, futures and options are not silently substituted into that set.

The retirement demonstration has two paired paths, five horizons, four sleeves
and two accounts: **80 completed replays**, cash residual zero and maximum NAV
residual $0.0000000168. It supplies owner history, $10,000 initial balance and
$1,200 annual contributions, 6% rates, 4% inflation, and explicit frozen 2026
tax/contribution rules. Two paths verify execution, not probability convergence.
[Path-level wealth, real wealth, IRR, drawdown, taxes and cost attribution](evidence/future-retirement.json).

Separate [seven-year stagnation](evidence/seven-year-stagnation.json) fixes monthly
equity total return at zero, rates at 6%, inflation at 4%, and production growth
at zero. The 3× Roth sleeve ends near $1,463 from $10,000. A separate
[−40% one-day stress](evidence/crash-extinction.json) permanently liquidates the
3× fund; a tax-loss benefit can remain in taxable account net assets without
resurrecting the fund. All results are conditional scenarios, not calibrated
real-world probabilities.

Probability outputs include Wilson Monte Carlo sampling intervals; 0/2 events
still permits an upper interval near 65.8%. Twelve training-only joint-block
refits with 256 paths each quantify parameter sensitivity separately: the
40-year cumulative-equity-log-return **conditional medians** range from −1.62
to +6.81 at the across-refit 10th/90th percentiles. This is neither an account
wealth interval nor a full model-error bound.
[Sensitivity evidence](evidence/parameter-sensitivity.json).

The prospective pipeline issued three immutable 2,000-path candidates for
December 2026 onward, refused duplicate retraining on unchanged inputs, and
returned pending when no outcomes existed. **No champion was created.**
[Issue](evidence/prospective-issue.json), [deduplication](evidence/prospective-deduplication.json),
[evaluation](evidence/prospective-evaluation.json). The optional workflow checks
registry continuity against the latest successful run, refuses absent/stale
history, evaluates old cohorts, then freezes candidates. Cache/artifact retention
is not a permanent archive; operator-managed durable preservation remains required.

## Remaining boundaries

Data/evidence: official daily index/NAV/funding histories; genuine dated futures
tapes and house margins; contract-level historical option quotes, depth,
adjustment memos and assignment evidence; older publication vintages; usable
gold history and product/regime terms. [Source catalog and restrictions](SOURCES_AND_RESTRICTIONS.md).

Engineering: pre-2024 personal tax tables, full household returns/AMT/credits,
estimated-tax safe-harbor calculations, forward QDI reclassification, year-end
distribution tax mapping, automatic TLH, per-order specific-lot selection in the
strategy CLI, tax-funded derivative replay, short assignment/spreads/collars,
interacting derivative taxes, UBTI/prohibited-transaction consequences, complete
bond return/curve mechanics and large-ensemble performance/convergence work.
These are **not all paid-data blockers**. SPY's captured 2025 tax record includes
a December distribution payable January 2026; its complete year-end account
comparison remains blocked rather than assigning that income to a guessed year.

No all-strategy ranking is released. No actual options/futures investment return,
1930 personal-account backtest or future investability claim passed certification.
The working reproduction commands are in [REPRODUCE.md](REPRODUCE.md).
