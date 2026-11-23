# Financial accuracy audit and remediation

Date: 2026-11-23. Branch: `audit/financial-accuracy-2026-11-23`.

**Status: substantial implemented remediation; comprehensive mission not yet verified.** Mechanical tests pass, but some reconstruction and original model-confidence gates fail. Full historical taxable/Roth rankings are withheld. A passing unit suite is not a certificate that every strategy, tax regime or broker is modeled accurately.

## Scope actually examined

The initial inventory read and statically scanned 99 source/document/metadata files comprising 285,288 lines, including 86 Python files. The inventory records hashes and symbol/risk locations in [source-inventory-before.json](source-inventory-before.json). The package's approximately 11,800 lines received focused review, along with instructions, prior validation reports, entry points and targeted parts of the 37,759-line `LETF35_analysis.py`. Archived `src/` versions received automated structural/risk triage, **not an exhaustive manual line-by-line financial certification**. This distinction matters.

There was no `AGENTS.md` in this repository, no existing automated tests in `tests/`, no existing workflow and no dependency lock. `CLAUDE.md` was read. The named files `research/free_validation/futures_reconcile.py` and `research/quantconnect_free_futures/main.py` do not exist in this checkout; their reconciliation prototypes could not be inspected or preserved here. No substitute is presented as those originals.

Baseline commit: `b967b2b498c261a2f551cfde9ab1621e5c489c2d`. Existing modifications to `LETF35_analysis.py`, deleted tracked bytecode, `.claude/` and `.serena/` were preserved. Original research caches and `validation_metadata.json` were not changed. The latter already recorded `validation_pass: false` before this audit. The archived `src/LETF33_analysis.py` contains a syntax error at line 1833; it was recorded rather than rewriting a historical version.

## Severity-ranked findings

| Severity | Confirmed finding and original location | Likely financial impact | Disposition |
|---|---|---|---|
| Critical | `tax/engine.py:compute_capital_gains` cross-netted before including character-specific carryovers and offered invented loss-ordering elections | Wrong gain character, omitted annual deductions, wrong carryforwards | Replaced with statutory ordering; illegal configurations reject |
| Critical | `tax/wash_sale.py:process_all_wash_sales` did not consume lots on gain sales, could use future buys as basis, ignored same-day matches, selected only one replacement | Incorrect deductible losses, basis and tax character across repeated trades | Replaced by shared dated ledger; IRS partial-replacement and Roth-loss examples tested |
| Critical | `integration.py:process_trades_with_wired_engine` used trading ordinals for ±30-day rules and 365-day holding tests | Wrong wash windows/year-end tax and delayed long-term status | Requires actual dates or supplied market-calendar map |
| Critical | Same integration estimated borrowing from trade frequency and deducted the resulting fictitious interest | Understated taxes even in cash-funded LETF trades; double treats fund leverage as personal borrowing | Removed; no automatic personal-interest deduction |
| Critical | Original reporting interpolated taxes from at most 50 paths, subtracted liabilities only at terminal wealth and treated benchmark journals as tax-free | Biased after-tax rankings and compounding; missing benchmark liquidation tax | Old ranking/batch APIs fail closed; new dated account replay tracks actual cash and payables |
| Critical | `data.py` treated S&P price-index returns as SPY; fitted Nasdaq/bond proxies used future overlap; `IsSynthetic` sometimes flipped false merely after inception without fund data | Missing dividends, invented historical assets, look-ahead and false validation | Actual fund capture, observed/proxy labels, explicit exploratory opt-in; uncertified historical rankings gated |
| High | `calibration.py:simulate_joint_returns_t` mixed raw-t variance with a unit-variance GARCH intercept | Inflated long-run volatility and nonlinear leveraged losses | Unit-variance innovations and covariance targeting; before/after test |
| High | Same function shifted each realized path toward its full-sample mean | Altered Monte Carlo distribution and artificial horizon behavior | Removed; antithetic and prefix/variance tests |
| High | Simulation clipped losses to −99.9%, then allowed recovery, and substituted zero-wealth results after exceptions | Suppressed fund extinction and contaminated distributions | Absorbing hypothetical liquidation, hard error gates, no omitted failure paths |
| High | Cash in the stressed regime earned arbitrary −2% while borrowing used another positive rate path | Inconsistent defensive returns and financing | Uses same short-rate path for simulated cash/borrowing |
| High | No Roth cash-flow basis/eligibility/clock model or linked-account loss model | Routine Roth taxes or misreported distribution tax/eligibility | Owner-level Roth history and cross-account ledger added; exclusions explicit |
| High | No executable dated futures/margin or quote-driven option account mechanics in package | Fractional/excess buying power, missing variation margin/collateral, unexecutable option returns | Shared-cash futures and restricted long-option mechanisms added; unsupported capabilities reject |
| High | `validation.py` used ~50% annual low-volatility drift to force a favorable realism fixture score | Circular “validation” conceals unrealistic assumptions | Removed favorable drift; T20 now fails and is retained |
| High | `letf/__init__.py` scored newly fabricated Gaussian proxy paths instead of actual generated paths | Confidence gate did not validate the simulations being reported | Actual path diagnostic metadata used |
| Medium | `historical.py` dropped synthetic rows before rolling calculations and used N−1 price intervals for N sessions | Fictitious contiguous holding periods and annualization errors | Preserve rows, reject contaminated windows, use full endpoints/calendar duration |
| Medium | Unknown state/filing values silently fell back to CA/single; NIIT thresholds inflated | Wrong jurisdiction and tax rates | Explicit state/year/status gates; fixed NIIT thresholds |
| Medium | SPEC_ID silently used MINTAX; partial replacement holding periods shared across unaffected shares | Selection not actually specified, misclassified sales | Explicit IDs and split lots |
| Medium | Cache filenames used only endpoint years | Different date windows could reuse stale data/models | Full dates and accuracy-version suffix |
| Medium | SSO configured inception as 2006-07-11 | Genuine early observations discarded | Corrected to issuer fund inception 2006-06-19; captured trading data starts 2006-06-21 |

File locations above name the original symbols, since replacements changed line numbers. Detailed source authority and rule-to-test mapping are in [TAX_RULES.md](TAX_RULES.md). Data, costs and broker assumptions are in [DATA_AND_EXECUTION.md](DATA_AND_EXECUTION.md).

## Implemented stages

1. Preserved original hashes/baselines, created a branch and pinned the tested environment.
2. Corrected netting, dated federal regular-tax parameters, deductions and NIIT; replaced independent wash replay with the shared stock/ETF ledger; added Roth histories.
3. Added cash execution, lot selection, settlement availability, dividends, splits, loss harvesting, explicit tax payments and terminal liquidation replay. S1–S7/S9/S10 have a conservative account-backed allocation implementation. New allocation behavior requires full indicator warmup, correct two-sided SMA-band hysteresis and weights no greater than one; these variants are not silently equated with the legacy overleveraged versions.
4. Added dated futures, daily cash settlement, Section 1256 allocation, margin schedules, gap/forced-liquidation mechanics and restricted real-quote long-option Roth handling. Added dated conservative broker capability subsets.
5. Corrected Monte Carlo variance/path-boundary errors; added joint paired block paths and paired confidence intervals; repaired historical-window logic.
6. Captured public daily inputs, froze thresholds and evaluated reconstructions. Added reproducible reports and CI test configuration.

The legacy strategy generator and monolith remain available as research history; they are not silently converted into an audited account simulator. Entry points that formerly printed unsupported account rankings now stop explicitly. `fetch_historical_data(allow_hypothetical=True)` retains exploratory legacy data loading with labels. This is an intentional safety change, not a claim that the old full workflow passes.

## Before versus after

| Fixed case | Before | After |
|---|---:|---:|
| Year with only $5,000 ST carryover and sufficient income | $0 deduction; $5,000 carried | $3,000 deduction; $2,000 carried |
| $10k ST gain, $10k LT loss and $10k ST carryover | $10k ST carryover, no deduction | $7k LT carryover after $3k deduction |
| Single 2024, $60k LTCG, no ordinary income, federal only | $1,946.25 tax | $0 regular federal tax |
| Student-t/GARCH variance, target .0001, 30k fixed-seed draws | .0001218886 (+21.89%, FAIL) | .0001016380 (+1.64%, PASS) |
| Broad after-tax strategy rankings without dated execution | Plausible-looking ranking | Explicit unsupported/error gate |

The first three are financial-rule corrections, not favorable-performance calibration. Tax liability can rise or fall in other scenarios. The variance tolerance (10%) was fixed before the diagnostic; original code fails it. The six preserved baseline numerical netting cases still pass after removing the invalid “election” interpretation; the original baseline is kept in [baseline.json](baseline.json).

Reconstruction on 1,508 evaluation sessions per product: QLD, SSO and TMF passed all numerical thresholds; TQQQ and UPRO failed the 2 pp CAGR-error gate at 2.303 pp and 2.082 pp. No threshold was relaxed. A serialization-only issue in the first report write was corrected without changing model parameters, input bytes, split or thresholds. Abandoned artifacts remain separate; the completed report is `reconstruction-v1/report.json`.

## Verification results and failures

- The new pytest suite passed: **98 tests**, with four deprecation warnings from the legacy pandas-datareader dependency. See the latest [mechanics report](evidence/mechanics-validation-final.json) and [test output](evidence/pytest-final.txt).
- Preserved netting regression outcomes: 6/6 pass. New tests add independent IRS examples and failing-before edge cases.
- Existing zero-drift and flat-market numerical validations pass after distinguishing approximate log decay from simple median return.
- Original forward-realism suite: **T20 fails**, score 59.5075 after removing the favorable-drift fixture. This failure is retained; it is not counted as financial-model certification.
- Original `quick_test.py`: **fails intentionally** at the default unverified historical-data gate, after passing netting regressions. It does not complete a 10-path strategy comparison.
- `git diff --check` and compilation of the active package, scripts and tests were run. Archived syntax errors are excluded from active-package certification.
- No actual futures settlement series, historical option chain, issuer NAV tape, point-in-time tax table before 2024, state-tax engine beyond the stated scope, or actual personal broker approval was verified.
- The new CI workflow was authored; a remote GitHub Actions run was not observed.
- Numba, arch-library estimation and GPU execution variants were not installed or validated in this audit environment.

## Remaining work; not a production-ready certification

The requested full all-strategy 1970s–2025, 5/10/20/30/40-year taxable/Roth comparison is **not established**. No after-tax return table was fabricated. The detailed [support matrix](SUPPORT_MATRIX.md) distinguishes data availability from complete account confidence.

Missing implementation includes complete state tax, AMT/credits, exact estimated-tax timing, all historical tax years, interacting derivative tax laws, option exercise/assignment and adjusted deliverables, short-option/advanced spread execution, a complete sector/universe survivor-bias model, conversion pro-rata determination, UBTI/UBIT and prohibited-transaction consequences. Advanced legacy strategy allocation and execution remain unverified. These are engineering/legal-model gaps, not all paid-data blockers.

Missing evidence includes historical point-in-time expense/swap schedules, issuer reconciliation, futures contracts/margins and granular option quotes. Paid data and broker account information may be needed for some historical claims. No purchases, filings, live orders, service deployment or holdout overwrites occurred.

## Reproduce

From the repository root, using Python 3.14 (tested 3.14.4):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.lock
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_accuracy_audit.py --output audit-output/new-mechanics-run
```

The audit runner rechecks the original pure GARCH function from the preserved baseline Git object without running historical entry points. Output directories must be new. Existing validation artifacts are never overwritten.

For exact reconstruction reproduction, retain the captured input CSVs whose hashes are in the manifest:

```bash
.venv/bin/python scripts/validate_daily_reconstruction.py \
  --inputs audit-output/public-daily-inputs-network \
  --output audit-output/new-reconstruction-run/report.json
```

To obtain a fresh **different vintage**, explicitly capture into a new directory:

```bash
.venv/bin/python scripts/capture_daily_audit_inputs.py --output audit-output/new-public-capture
```

Fresh-vintage results are not bitwise replacements for this frozen capture. They must keep their own hashes, timestamp and reports.
