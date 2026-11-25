# Financial engine continuation: implemented, tested, not investment-certified

Reviewed November 24 US Eastern / November 25 UTC, 2026. Branch:
`develop/tail-validation-2026-11-24`, based exactly on
`140535f1ce8d80e31118a72b43aa9a83b4633c19`. No main merge, trades, purchases,
credential use, or replacement of earlier research inputs/holdouts occurred.

**The full mission remains incomplete. No macro model is approved, no fund
reconstruction has complete financial-source certification, and no long-horizon
after-tax investment probability is certified.** This release adds tested
mechanics, observed-fund account comparisons and falsifiable validation. The
[support matrix](SUPPORT_MATRIX.md) separates implementations from blockers.

## Implemented stages and preservation

| Commit | Delivered change |
|---|---|
| `4b1bb26` | Register protocol before candidate evaluation; 221-test baseline and tracked-file hashes |
| `c990b9d` | Four bootstrap candidates, source-row lineage, empirical daily bridge, daily tail scores, 25 frozen forecasts |
| `a873c1c` | Actual forward common-share QDI holding tracker; consistent observed-fund cohort workflow |
| `66fb46f` | Official Nasdaq index/issuer NAV ingestion and diagnostics; statutory wash/QDI refinement |
| `e4e405a` | Common score normalization, empirical bridge in canonical accounts, profiled array reads, 4,096-path precision workflow |
| `c1971be` | Separate deemed year-end dividend tax recognition from cash; SPY 2025 account support |
| `bf4d5ef` | Sourced gross financing/collateral income, strict day-basis validation, frozen-path reproduction CLI |

The subsequent documentation commit contains the evidence and this report.
Use `git log 140535f..HEAD --oneline` for exact delivered hashes; the final
response identifies the final branch HEAD and its hosted CI run.

The [preservation check](evidence/preservation.json) compares all 496 original
tracked files with the registered baseline. Eleven existing implementation/test
files changed intentionally. Original research, workflows, data files, strategy
definitions, prior reports, frozen thresholds and dependency lock are unchanged.
The original checkout remains on its original branch with the user's dirty files
left in place. There were no cherry-pick conflicts: this is a descendant branch,
not a replacement checkout or new integration of main.

## Severity-ranked findings

| Severity | Finding and effect | Disposition / code |
|---|---|---|
| Critical | Candidate-specific score denominators made longer training histories appear 24–26% better without a comparable loss unit | Corrected in [tail_study.py](../../letf/macro/tail_study.py). This error was introduced and caught in this continuation. Initial invalid comparison remains archived; corrected candidates are worse. |
| Critical | Monthly forecast scores cannot certify daily-reset leverage tail risk | Added [daily_tails.py](../../letf/macro/daily_tails.py), authentic training shapes/variance, crash/drawdown/underwater scores. Daily scenarios remain hypothetical. |
| High | Original advanced macro models understated drawdown risk and long bears | Diagnosed all 26 original artifacts read-only; no parameters changed to fit those outcomes. All new candidates rejected. |
| High | Net-fee QQQ/SPY proxies and exchange prices confound exact target-index versus NAV tracking | Added issuer NAV/cash and Nasdaq-origin TR index. TQQQ CAGR still fails; S&P exact-index provenance still missing. |
| High | Qualified-dividend gate could only prove ownership already completed at ex-date | Optional actual calendar-day tracker now handles forward qualification, partial sales, splits and year-end reclassification. Default conservative interface retained. |
| High | Wash-sale tacked holding days are inapplicable to QDI qualification | Actual acquisition dates used, following IRC 1(h)(11)/246(c)(3)(B), independently tested. No tacked-day shortcut. |
| High | January cash payment and prior-year deemed receipt were inseparable | Added separate noncash tax/cash events; SPY December 2025 income recognized in 2025 without making January cash spendable. |
| High | Comparing different reinvestment policies gave incompatible fund/account results | Added consistent six-fund 2024–25 and seven-fund 2025 cohorts, independently reconciled. |
| High | A net borrowing formula does not describe every swap-financed fund | Canonical daily-reset API now also supports sourced gross notional financing and separate collateral income, with distinct day bases. No fitted historical cost schedule is invented. |
| High | Small future demonstrations did not establish Monte Carlo convergence | Ran two 4,096-path ensembles, paired 1×/2×/3× paths across five horizons. **All 15 quantile-convergence gates fail.** Account-wealth convergence remains untested. |
| High | Historical taxes, MA fund adjustments, option tax/assignment and dated futures evidence remain incomplete | Explicitly blocked; retained interfaces and tests are not presented as completed historical capabilities. |

## Macro diagnosis and corrected results

[Original diagnosis](evidence/original-26-diagnosis.json) verifies and inspects all
26 prior forecasts. For 1970–74 the realized monthly-close drawdown was 46.36%;
original baseline/regime/VAR 95th percentiles were 26.89%/24.98%/26.92%.
For 2000–04 the realized 57-month underwater spell exceeded the respective
35/37/37-month 95th percentiles. Both advanced models retained their original
3-of-6 drawdown breaches and rejection. Three-month blocks interrupt long bears;
the VAR's constant innovation covariance and regime bootstrap's short blocks do
not identify persistent conditional daily volatility. Neither has an actual
bond-return variable; stock/bond forecast certification is blocked.

Five additional retrospective windows cover 1955–59, 1965–69, 1975–79, 1980–84
and 1995–99. All 25 forecasts contain 2,000 paths, frozen before outcome loading.
Their intervals do not overlap earlier consumed forecast intervals. They are
human-known history, including previously published counterfactual summaries,
**not pristine holdouts**. The original 26 remain diagnostics only.

| Candidate vs three-month/240-month-history baseline | Corrected CRPS improvement | Paired mean loss difference, 98.75% era-bootstrap interval | Decision |
|---|---:|---|---|
| Same three-month blocks, longer history | −6.88% | [−0.1002, +0.2205] | Reject |
| Twelve-month blocks, longer history | −8.30% | [−0.0931, +0.2407] | Reject |
| Stationary geometric blocks, mean 12 months | −6.81% | [−0.0984, +0.2139] | Reject |
| Volatility-conditioned stationary blocks | −2.61% | [−0.1221, +0.1672] | Reject |

All candidates fail improvement, paired-CI and historical-vintage gates. Five
eras provide wide uncertainty. The corrected comparison uses the **same baseline
training scale for each candidate at an origin**, for all five macro variables
and daily marginal CRPS. Neither rescaling nor changed code creates another
independent evaluation. The [initial invalid result](evidence/initial-invalid-scale-comparison.json)
and [corrected consumed-evidence result](evidence/common-scale-comparison.json)
are both retained. The correction never promotes or consumes the batch again.

On these same monthly paths, the fixed Gaussian daily reference breaches the
crash-frequency 95th percentile in 2/5 eras; empirical variance templates breach
in 0/5. Empirical daily interval coverage is about 95%; daily drawdown breaches
are 0/5. These paired diagnostics isolate the bridge assumption. They **do not**
prove success in the earlier failed severe-crash eras or calibrated future tail
probabilities. Zero breaches in five eras still allows substantial uncertainty.

The daily bridge preserves monthly log totals and sampled realized variance,
uses only training observations, and retains joint source-month lineage. Missing
rows force restarts; no gap is filled. Circular stationary boundaries and
21-session synthetic forecast months are explicit assumptions. The conditional
restart has a 25% uniform mixture, so past extremes never lose all sampling
support. No paths are removed for unattractive outcomes.

All 25 monthly ensembles and lineage arrays were regenerated exactly without
loading outcomes: [receipts](evidence/frozen-path-reproduction.json). This is
numerical reproducibility, not independent financial validation. The existing
prospective registry/scheduler remains research-only; future outcomes are still
needed. New tail-study gates are required on registered tail batches, and the
existing monthly-model approval cannot certify daily fund implementation.

## Leveraged-fund source reconciliation

86 new local snapshots: four issuer NAV histories, 80 annual cash-history API
responses and two Nasdaq-origin indices. The [catalog](evidence/official-source-catalog.json)
records checksums, actual observation bounds, capture/reuse timestamps, provider,
frequency, currency and copyright restrictions. Raw inputs are not redistributed.
Nasdaq XNDX genuinely begins 1999-03-04 in this source; it is not a 1930 Nasdaq.

The following comparison preserves the 2019 training cutoff, 2020–25 evaluation,
training observation counts, constant-fee scenarios, nonnegative training-only
spread fitting and original numerical thresholds. The evaluation is already
inspected, so these are **source-reconciliation diagnostics, not fresh OOS proof**.

| Fund | Original CAGR error, percentage points | Source-reconciled error, points | Reconciled daily RMSE | Numerical gate / full certification |
|---|---:|---:|---:|---|
| TQQQ | +2.303 | +2.427 | 0.0003485 | FAIL / BLOCKED |
| QLD | +0.554 | +0.631 | 0.0002820 | Pass / BLOCKED |
| UPRO | +2.082 | +1.024 | 0.0021639 | Pass / BLOCKED |
| SSO | +0.537 | −1.491 | 0.0013049 | Pass / BLOCKED |
| TMF | +0.546 | Unchanged | 0.0024981 | Prior limited pass / BLOCKED |

TQQQ's daily error falls substantially but its compounding bias persists. UPRO's
new numerical pass reflects issuer NAV/cash replacing market-price observations
and the resulting training spread; it still uses a **SPY proxy**, not an official
S&P TR index. No claim that UPRO's financial-source failure is fixed follows.
NAV and exchange prices are different valuation series, not interchangeable fills.
[Full errors and annual/holding-window results](evidence/official-reconstruction.json).

Issuer NAV total returns add ex-date cash using a common split-restated unit
convention. NAV already embeds fund costs. No fund expense/funding charge is
subtracted from observed returns. The optional gross-financing API charges
contractual financed notional and credits exposure-weighted collateral income;
it does not assume every product obtains collateral yield equal to its financing
rate. Dated daily notionals, fee waivers and historical collateral returns remain
missing. The [2026 issuer financing authority](evidence/financing-authority.json)
supports the distinction, not a retrospective cost invoice.

## Comparable observed account experiments

All six funds now support the **same reinvestment rule** for 2024-01-02 through
2025-12-31: $100,000 initial balance, federal-only, single, $100,000 other ordinary
income, 2 bp quoted spread/1 bp slippage, no commission/cash yield, actual issuer
distributions, US settlement, prior-close decisions and terminal sale. Annual
2024 tax settlement occurs in April 2025; security sales fund it when needed.
The initial-only cash policy is also reproduced as a separate cohort.

| Fund | Taxable net assets | Roth internal net assets |
|---|---:|---:|
| QQQ | $144,980.96 | $152,961.30 |
| TQQQ | $199,462.77 | $218,422.99 |
| QLD | $176,021.22 | $189,479.45 |
| SSO | $170,886.98 | $183,530.45 |
| UPRO | $201,078.12 | $220,111.05 |
| TMF | $63,361.91 | $63,587.48 |

[Six-fund results](evidence/fair-six-reinvest.json) include nominal/real CAGR,
drawdown, underwater time, actual modeled taxes, tax funding, transaction costs,
terminal liabilities and reconciliation. Account-neutral pre-tax fills match
exactly for both configurations; taxes can alter taxable funding/reinvestment.
Independent exported identities reconcile each daily cash/position/NAV and
terminal net payable. Actual fills remain assumed daily OHLC, not observed NBBO.

A second, common **2025-01-02–2025-12-31 seven-fund cohort** adds SPY. This is a
shorter, generally short-term holding experiment and is not mixed into the
two-year table. SPY ends at taxable $113,333.13 / Roth $116,982.41, including a
$341.03 distribution receivable in each account. The December 19 ex-date/record
distribution pays January 30, 2026; final issuer 2025 tax allocation recognizes
it December 31. It is **not spendable January cash at the 2025 endpoint**.
[All seven funds and assumptions](evidence/fair-seven-2025.json).

Roth values are assets after internal sale, **not a personal qualified withdrawal**.
Contributions and owner withdrawal clocks remain separate inputs. Tax losses can
produce a household tax benefit under the supplied wage/refund assumptions;
negative modeled tax is disclosed rather than floored. MA fund-government-
obligation adjustments are still unverified, so these fund bundles reject MA.
Historical tax years before 2024, full household returns and future tax laws are
not silently inferred. No cross-fund investment ranking is certified.

## 1930 counterfactuals and long-horizon precision

Re-running the existing 108 daily counterfactual cases reproduces the entire
original report exactly. Authentic inputs begin July 1926; simulated returns run
1930-01-02 through 2026-09-30, including genuine historical Saturday sessions and
the sourced 1933 closure. No daily path is inferred from monthly observations.

| Hypothetical 3× exposure | Central CAGR | Maximum drawdown | CAGR range across disclosed costs |
|---|---:|---:|---:|
| Broad US research market | 9.18% | 99.714% | 4.85%–11.98% |
| Large growth by historical book/market | 8.47% | 99.600% | 4.16%–11.25% |
| Small growth by historical book/market | 2.55% | 99.796% | −1.52%–5.19% |
| Historical HiTec industry | 3.07% | 99.984% | −1.02%–5.72% |

Central costs are 1% annual fund fee and 1.5% financing spread; the ranges use
0.5%/1%/2% fees and 0.5%/1.5%/3% spreads, not statistical confidence intervals.
These are **HYPOTHETICAL_STRESS**, not literal TQQQ, issuer track records or 1930
personal accounts. Roth did not exist. Legal availability and historical modern
swap financing are not asserted. [Reproduction/sensitivity evidence](evidence/counterfactual-preservation.json).

The new precision study uses 4,096 paths per seed, two independent seeds, common
joint monthly/daily paths across leverage and 5/10/20/30/40-year horizons. Every
1024/2048/4096 prefix is retained. All 15 unit-NAV quantile checks fail the fixed
0.02 maximum log-quantile difference; all binomial half-width checks pass.
For 40-year 3× NAV the largest nested-prefix difference is about 0.112 log units
and separate-seed difference 0.184. Precision is not adequate for confident
quantile/ranking claims. [Full results](evidence/unit-sampling-4096.json).

This precision calculation calls the existing daily-reset engine and calculates
**fund-unit NAV only**. It adds no parallel account/tax engine. Thousands of
after-tax account replays were not completed and must not be inferred. The
empirical bridge does execute through the canonical account explorer, including
a reconciled 40-year/four-allocation/two-account single-path demonstration.
That execution demonstration is not account-wealth convergence.
Zero modeled liquidations do not bound losses from an unseen catastrophe:
finite historical resampling cannot identify tail behavior outside its training
shapes. Binomial sampling precision is not real-world risk calibration.

## Verification and remaining work

Baseline: 221 tests. Final implementation: **246 passing tests**, four inherited
dependency deprecation warnings. All 97 workflow YAML checks pass. Compile checks
cover package/scripts/tests and both existing futures research prototypes; Lean
was not executed. The [test log](evidence/pytest-final.txt) and
[reproduction guide](REPRODUCE.md) identify exact commands/dependencies.

Hosted [CI run 45536567040](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45536567040)
verified `c1971be` with 245 tests and 97 workflow checks; the subsequent financing
fixture adds one test. [CI run 45537576913](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45537576913)
passed on implementation `bf4d5ef088a128535966826662e6b1faef32e72b` with 246 tests
and 97 workflow checks. The final response links the delivered HEAD's hosted run.
Large source-dependent studies were run locally; the offline CI fixture suite
does not certify their financial inputs or execute a paid-data backtest.

Legacy realism remains **8/10**: T20 = 59.507495333 < 65; T27 calls an unsupported
`max_crisis_chain` keyword. Neither test was disabled or weakened. Inserting
selective crisis suppression to obtain a passing T27 score is not a correction.
[Preserved diagnostics](evidence/legacy-diagnostics.json).

Additional code remains necessary for short-option writing/assignment, collars
and spreads, interacting derivative tax laws, complete historical federal/MA
versions, estimated-tax penalties, MA fund exemptions, special distributions,
and scalable account-wealth convergence. Data alone does not complete those
features. Dated futures tapes/house margins, authentic option NBBO/OCC records,
actual bond TR, point-in-time macro/equity releases and full dated fund costs are
also missing. [Sources and least-cost access investigation](SOURCES.md).

The highest-value next action is to preserve genuine **prospective daily and
monthly forecasts with their source vintages**, and evaluate later outcomes
without retuning or reusing these consumed eras. For reconstruction, acquire
dated swap notional/collateral/fee schedules and an official S&P TR history before
claiming the remaining bias is explained. Faster or more optimistic curves do
not resolve those evidence gaps.
