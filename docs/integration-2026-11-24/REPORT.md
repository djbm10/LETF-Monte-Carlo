# Financial-engine integration and evidence report

2026-11-24. Branch: `integrate/financial-engine-2026-11-23`. Final implementation commit: `11f57067a2b47a03f60893fc09af3794647db86e`. The report/evidence commit follows it on this branch. This supersedes the **status** of the preserved November 23 audit; those historical documents and results remain unchanged.

**Delivered: an integrated, tested cash-account engine and runnable stock/ETF taxable-versus-Roth comparisons. Comprehensive financial certification remains BLOCKED.** Mechanical correctness, observed source data, reconstruction confidence and investable execution are separate claims. Missing engineering is distinguished from missing paid data below.

## Repository reconciliation

Updated GitHub `main` was verified at `fe8e1b28cb0275396a0b9b6c30a3ab29b1e25365`. The checkout initially pointed `origin` at TomerButbul and `fork` at djbm10. Both were preserved: djbm10 is now `origin`; the original remote is `upstream-original`. This work did not modify main or merge a PR. During final verification its scheduled bot independently advanced main to `79fcaa6e240dcaa2ede7fe3069d4cf0bc8bb1390`, appending one prospective signal observation. This was merged into the integration branch as `9b895306e21d889dd8136dd307d7f66cefc9dfe9`; all old journal rows remain an exact prefix and no financial code changed.

The original unpublished audit commits were found in local branch/reflog history; they were **not recreated from summaries**:

| Original audit commit | Reviewed integration cherry-pick |
|---|---|
| `9fb4a83ee1e3dcc9b07fd60afdb81368a7541eef` | `190f660` |
| `05c1b49fc43648513295dc31d58e7ad10db98519` | `90df3ce` |
| `96af497cc5a9432b3298623fb16a69ce069153fa` | `fe016fc` |

They descend from `b967b2b498c261a2f551cfde9ab1621e5c489c2d`. Original branch `audit/financial-accuracy-2026-11-23` remains at `96af497`. A complete local ref bundle, working-tree patch, status/reflogs and user-edit archive are in `/tmp/letf-integration-preservation/`. The original checkout's `LETF35_analysis.py` edits, deleted tracked bytecode and `.claude`/`.serena` files were left in place. Integration work lives in `.worktrees/financial-integration`.

**Conflict report:** the second cherry-pick conflicted in `letf/__init__.py` and `letf/validation.py`, where its realism diagnostics were absent from current main. Reviewed audit diagnostic blocks were retained; the exact original audit `letf/simulation/realism.py` and required verbosity configuration were restored. No newer sector-regime module, research rule or sealed result was replaced. Subsequent main compatibility fix: the audit runner now reports the absent baseline-only `validation_metadata.json`, rather than crashing or fabricating it.

Additional implementation stages: `b376f83` canonical accounts/MA/scenarios; `bfed77e` workflow/PIT repair; `221b574` derivative research adapters and evidence gates; `2ea5dca` issuer-reconciled ETF inputs; `11f5706` daily exchange-session completeness gate. [Protected-input evidence](evidence/protected-inputs.json) verifies 100 existing research/result/frozen files against origin/main. [Inventory](evidence/source-inventory.json) records machine source/document scans and syntax findings. This is not a claim of exhaustive manual certification of every archived line.

## Confirmed defects and implemented repairs

| Priority | Defect / location | Financial or operational effect | Implemented disposition |
|---|---|---|---|
| Critical | Previously unpublished corrected tax/simulation engine | Main still had statutory-netting, wash-lot, fictitious financing and variance defects | Original three commits ported with regression tests; original history retained |
| High | `clean-1926-walkforward.yml`: repeated `bottleneck_leaps_futures` job | Workflow cannot start | Removed redundant duplicate while preserving distinct steps; added offline clean self-test and duplicate-key check across all 96 workflows |
| High | `research/bottleneck_winner_tournament.py`: `.at`/`.lt` column attributes | PIT self-test crashes before ranking | Explicit column indexing; independent balance-sheet example |
| High | Same file: second, global momentum `.shift(1)` overwrote grouped calculation | First row of another security could inherit the previous security's momentum | Retained the grouped transform; independent two-security boundary test |
| High | Account replay: same-day cash recycling / payment timing lacked exchange chronology | Excess buying power and reinvestment of cash needed for taxes | XNYS T+2/T+1 settlement, early funding sales, reserves and payment events |
| High | No end-to-end contribution/withdrawal replay | Contribution-boosted “CAGR”; unavailable cash for withdrawals | Scheduled flows, owner Roth history, settlement-funded withdrawals, unitized TWR and separate XIRR |
| High | No MA state engine / federal loss carry used for state | Wrong state rate, wage loss deduction and carryforwards | Explicit dated MA subset; separate ST/LT state carryovers and Schedule B ordering |
| High | `research/optionmetrics_leaps_backtest.py`: missing held quote marked zero | Asset could disappear, followed by fabricated reinvestment | Hard error; no missing quotes dropped or replaced |
| High | Same options module: implicit strike scale, ~21-day interest and excessive roll budget | Wrong contracts, invented cash accrual, negative cash | Explicit scale; missing-rate-path error; ask/bid, commissions and cash cap; scenario-only labels |
| High | Long-option valuation accepted an unrelated contract quote | Arbitrary portfolio value | Contract identity and expiry checks; independent wrong-contract test |
| High | Reconstruction treated a vendor checksum as verified financial provenance | False “validated” confidence despite undocumented target/fee conventions | Numerical gates preserved; source/term certification separated and withheld |
| High | Vendor historical ETF payouts rounded; current issuer payouts split-restated | Wrong distributions, nominal shares and tax basis | Dated issuer ICI amounts replace vendor payout amounts; exact issuer restatement factor reconciles to dated split actions |
| Medium | Sweep interest credited using cash settling today for yesterday | Unearned interest | Accrue only on the prior settled balance |
| Medium | Missing interior CPI observations implicitly prevented valid endpoint metrics | No report, or incentive to invent interpolation | Endpoint real CAGR with explicit missing-count disclosure; no invented monthly path |

The original audit's tax, GARCH, liquidation, observed-cost and historical-window repairs are detailed in [its preserved report](../accuracy-audit/AUDIT.md). They remain part of this branch. Legacy entry points that cannot support after-tax rankings still fail explicitly.

## Real daily account demonstrations

Identical instrument definitions, prior-close allocation rules, next-open cost assumptions, accounting and terminal close liquidation are used for both accounts. Account type is an independent configuration. Tax payments cause subsequent taxable quantities to diverge; “identical pre-tax trades” is tested with the tax switch off and broker restrictions retained.

BRK-B: 752 observed sessions from 2023-01-03 through 2025-12-31; 2023 is warmup. Trading covers 502 sessions, 2024-01-02–2025-12-31, SMA50, $100,000 existing capital, $100,000 ordinary wage income, single full-year MA resident. Spread 2 bp full width plus 1 bp slippage per side, zero commissions and cash yield. No personal borrowing. No distribution/split was found through capture; the issuer's [2024](https://www.berkshirehathaway.com/2024ar/2024ar.pdf) and [2025](https://www.berkshirehathaway.com/2025ar/2025ar.pdf) annual reports support the no-dividend review.

| Result | Taxable MA | Roth cash |
|---|---:|---:|
| Pre-tax paired fills / terminal wealth | 38 / $110,976.54 | 38 / $110,976.54 |
| Ending wealth, net accrued tax/refund | $106,907.44 | $110,976.54 |
| After-tax nominal CAGR | 3.4031% | 5.3566% |
| Endpoint real CAGR | 0.8723% | 2.7780% |
| Maximum drawdown | 21.4090% | 17.9109% |
| Maximum underwater sessions | 188 | 188 |
| Tax paid 2025-04-15 for 2024 | $5,084.04 | $0 |
| Terminal 2025 tax payable (negative is receivable) | −$660.00 | $0 |
| Explicit trading costs | $845.53 | $858.91 |

Tax funding sells on 2025-04-14, settling on April 15. The sale itself creates 2025 tax events. Initial ordinary income taxes are external salary costs; the portfolio pays incremental investment taxes. Annual-payment treatment assumes enough external salary withholding for applicable safe harbors; the engine does **not** calculate estimated-tax penalties. The terminal $660 is an accrued household tax benefit, not settled brokerage cash. Terminal sale proceeds are likewise reported separately as unsettled. See [full tax rows and results](evidence/account-demo.json).

With a $1,000 contribution on 2024-01-03 and $500 withdrawal on 2025-06-20, terminal wealth is $107,463.86 taxable / $111,577.46 Roth. Cash-flow IRRs are 3.4014% / 5.3591%; lump-sum CAGR is deliberately null. The Roth starts with supplied existing regular contribution basis; it is not modeled as a new $100,000 annual contribution. [Cash-flow results](evidence/account-demo-flows.json).

Real ETF buy-and-hold uses the same engine: 2024-01-02 open to 2024-12-31 close, federal-only, same $100,000 capital/income and execution costs. **These are open-to-close strategy periods, not issuer calendar-year closing-price returns.** Taxable ST liquidation applies because holding is less than one year.

| ETF | Equal pre-tax fills | Roth terminal wealth | Taxable net terminal wealth | Terminal federal liability |
|---|---:|---:|---:|---:|
| TQQQ | 6 | $162,505.85 | $147,841.44 | $14,664.41 |
| QLD | 6 | $145,355.61 | $134,805.86 | $10,549.75 |

Dated [ProShares 2024 ICI primary supplement](https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_2024.xlsx) supplies cash/tax character and payable dates; current [TQQQ issuer distribution records](https://www.proshares.com/api/distributionsummary/?fund=TQQQ&year=2024) establish the restated convention. All relevant TQQQ/QLD dated/current payout ratios equal 2, matching the recorded 2025 split. Nominal market observations are scaled back by that constant, and original source hashes remain in the bundle. No model funding/management fees are charged to observed fund prices. Investor qualified-dividend holdings are checked; unresolved future holding periods reject. SSO/UPRO 2024 supplements contain additional tax components and are explicitly blocked by this builder. State distribution exemptions are not inferred. [TQQQ results](evidence/TQQQ-account-demo.json), [QLD results](evidence/QLD-account-demo.json), [build gates](evidence/etf-build-report.json).

All reported daily cash residuals are zero; NAV residuals are at most $6e−11. Independent quantity replay also checks the lot ledger. Deterministic fixture tests separately calculate expected taxable/Roth wealth, settlement, contribution returns and cross-account Roth loss disallowance. Observed daily quotes plus assumed fills remain **HYPOTHETICAL_STRESS execution**, and investment-performance certification stays **BLOCKED**.

CPI endpoints are observed BLS/FRED CPI-U NSA levels. October 2025 is missing for 23 trading rows. BLS confirms it [did not publish that CPI release following the funding lapse](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm). Only endpoint real returns are calculated; no real drawdown path or inflation-adjusted ETF return is invented where CPI was not supplied.

## Reconstruction: before versus after

The exact original CSV bytes, training cutoff 2019-12-31, evaluation window 2020–2025 and frozen threshold fingerprint `fa62413c93af6fb5eccf9b9743c4ae3837b9c38a2e0c082b31ede64e1c73f512` were reused. No held-out result was used to tune parameters or thresholds.

| Fund | Earlier CAGR error, pp | Integrated CAGR error, pp | Frozen numerical result | Financial provenance certification |
|---|---:|---:|---|---|
| TQQQ | +2.303367 | +2.303367 | FAIL (>2 pp) | BLOCKED |
| QLD | +0.553906 | +0.553906 | PASS | BLOCKED |
| UPRO | +2.082450 | +2.082450 | FAIL (>2 pp) | BLOCKED |
| SSO | +0.536560 | +0.536560 | PASS | BLOCKED |
| TMF | +0.546444 | +0.546444 | PASS | BLOCKED |

The equations did not change. Financial certification is tighter: a checksum validates captured bytes, not official index/NAV conventions, contemporaneous fund expenses or swap terms. Prior numerical passes are preserved as numerical passes. The report does not call these pre-inception series observed or fully validated reconstructions.

Measured diagnosis: TQQQ residual daily bias is +1.0060 bp and UPRO +0.6443 bp; their nonnegative training-fitted borrowing spreads hit zero. TQQQ yearly CAGR errors range +0.56 to +4.54 pp; UPRO +0.83 to +4.10 pp. Evaluation-only fee/spread tuning would contaminate the frozen test, so it was not performed. ETF benchmark returns embed benchmark fees; targets are indices; fixed product fees, DFF proxy funding and market-price/NAV differences remain unseparated causes. The issuer [TQQQ prospectus](https://prod.proshares.com/globalassets/proshares/prospectuses/tqqq_summary_prospectus.pdf) identifies derivative financing/transaction costs as relevant expenses. Attribution requires actual target-index TR/price conventions, dated issuer NAV/distributions, historical swap/reference-rate spreads and fee-waiver schedules. The new issuer payout reconciliation also exposes rounded vendor actions; it does not by itself prove that adjusted-return reconstruction errors are repaired.

[Full errors](evidence/reconstruction-integrated.json) include years, calm/high-volatility regimes, drawdowns, volatility and rolling 21/63/252/1260-session holding periods. Daily-reset pre-inception proxies are still generated from daily paths where available, separately labeled `HYPOTHETICAL_STRESS`. Their parameters are fitted after the hypothetical early period, so they are counterfactual analysis, **not point-in-time trading inputs**. No daily path is fabricated from monthly data.

## Futures, options and long-horizon status

`letf/research_adapters.py` reuses `research/free_validation/futures_reconcile.py` validation, frozen 35/0 signal and dated selector. It connects to the same account cash, `FuturesBook` margin reservations/variation settlement and Section 1256 income. Explicit contract expiry, multiplier, tick and sourced house margins are mandatory. Controlled roll fixtures reconcile to the independent research replay with prior-session signals; no continuous-price roll jump becomes P&L. Margin-call/house-increase/gap-debt tests pass. Current limitations include one margin vintage per adapter contract, weekday roll selection, assumed settlement fills, unknown intraday breaches and no account-level derivative tax-payment schedule. Thus this is **scenario mechanics**, not a certified historical taxable/Roth futures comparison. QuantConnect's original prototype remains byte-identical and compiles; hosted Lean execution and actual independent CME tapes were not run.

The published OptionMetrics sample contains six rows on 2005-09-15, including two qualifying long calls. It validates parsing/contract definitions only. Exact quote timestamps, depth and longitudinal coverage are absent; no fictional timestamp/quote size is assigned. Restricted approved-Roth long options use the shared cash book, real bid/ask fields and commissions. Taxable options, assignment, adjusted deliverables, physical exercise and interacting hedge taxes remain blocked. Short-option collateral helpers are not complete covered-call/CSP/collar/spread strategies. See [data and legal scope](TAX_AND_DATA.md).

The canonical scenario CLI completed **20 account runs**: two independent seeded block paths × five horizons × two account types. Seed `20261124`, block 21 sessions, BRK-B joint overnight/session returns, same path per paired account. Future federal law is explicitly frozen at 2026; projected exchange sessions and no-future-dividend stock assumptions are hypothetical. [Path-level results and paired bootstrap intervals](evidence/canonical-scenarios.json) are engineering demonstrations. Two paths cannot establish convergence or source/model uncertainty. No historical 40-year fund/account claim or cross-strategy ranking is released.

## Validation and remaining failures

| Check | Result |
|---|---|
| Updated main alone, isolated root/import config | 26 tests passed |
| Original prior audit | 98 tests passed, preserved historical evidence |
| Main plus original audit picks | 124 tests passed together |
| Final integrated suite | **152 passed**, four pandas-datareader deprecation warnings; [log](evidence/pytest-final.txt) |
| Independent cash/tax/settlement/flow/research boundary fixtures | Included in the 152 tests |
| All 96 workflow YAML files | Duplicate mapping/jobs structure check PASS |
| Frozen clean walk-forward offline self-test | PASS, 24 synthetic MC paths; no sealed historical evaluation rerun |
| Corrected PIT self-test | PASS, 1,152 fixture rows / 384 monthly rows |
| OptionMetrics public sample smoke | PASS as schema test; historical performance BLOCKED |
| Active package/scripts/tests/free-validation/QC compilation | PASS |
| Original T20 realism gate | **FAIL, 59.5075 <65**, preserved |
| Legacy T27 crisis-damping diagnostic | **FAIL**: calls `sample_returns(max_crisis_chain=...)`, which the retained bootstrap interface does not accept |
| Legacy `quick_test.py` | Explicit unverified historical-proxy gate; no false ranking |
| TQQQ / UPRO reconstruction CAGR gate | **FAIL**, thresholds unchanged |
| Hosted regression CI | **SUCCESS**, [run 45495063247](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45495063247) on `cfeeae3`: 152 passed, four warnings; all 96 workflow structures pass |

[Actual failed run 45311541299](https://github.com/djbm10/LETF-Monte-Carlo/actions/runs/45311541299) failed before jobs: GitHub returned no jobs and zero check runs; `gh run view` identified a workflow-file issue. The duplicated job key was confirmed in the checked-in file. After making it parse, the local PIT self-test exposed the additional pandas attribute crash described above. There were **no runtime logs from the original failed run** to attribute to a numerical test. [Metadata](evidence/failed-workflow.json).

T27 is an additional unresolved legacy diagnostic/interface mismatch in `letf/validation.py:590`. Its requested improvement depends on suppressing repeated severe crisis blocks. That score target is not a reason to add unsupported return selection to the canonical unbiased sampler; no threshold was relaxed and the failure is retained in the mechanics report.

Remaining **engineering** includes pre-2024 tax tables; full personal/state returns (AMT, credits, special assets, government-obligation passthrough, Section 199A and residency adjustments); explicit estimated-tax safe-harbor/penalty calculations; automatic TLH strategy; fully specified per-order specific-lot selection in the strategy runner; dynamic forward qualified-dividend reclassification; tax-funded derivative replay; S8 and S11–S19 migration; exercise/assignment and interacting derivative tax laws; UBTI/prohibited-transaction consequences; full inflation/stress/funding/correlation scenario framework and convergence studies. These are not all paid-data blockers.

Remaining **evidence/access** includes authenticated broker-specific historical permissions/margin vintages, independent dated CME prices, longitudinal option quotes/actions and historical issuer index/fee/swap inputs. [Support matrix](SUPPORT_MATRIX.md) gives individual strategy status and genuine captured start dates. No real trades, purchases, credentials, tax filings, deployments, main merge or sealed-holdout writes occurred.

Publication and the independently appended upstream journal reconciliation are recorded in [publication evidence](evidence/publication.json). The post-merge test log is [here](evidence/pytest-post-merge.txt); no financial source code differs from the hosted passing revision.

Reproduction commands and exact-input limitations are in [REPRODUCE.md](REPRODUCE.md). Dependencies are pinned in `requirements-audit.lock`. New output directories/files are required; commands refuse to overwrite evaluation evidence.
