# Run the investment research laboratory

This entry point composes the existing account, source, macro, stress and
forecast-registry modules. It never starts another cash/tax ledger or trades.
The local dashboard binds **127.0.0.1 only**. No raw/private input is uploaded.

## Portable demonstration, no private inputs

```bash
python -m pip install -r requirements-audit.lock
python -m letf.lab demo --output audit-output/new-demo
python -m letf.lab serve --profiles audit-output/new-demo/profiles.json \
  --output-root audit-output/new-dashboard-jobs --port 8765
```

Open `http://127.0.0.1:8765`. The demo is **SOFTWARE_FIXTURE**, not actual market
history or a forecast. It includes full daily taxable/Roth comparison, ordinary
distributions, an explicit cash yield, contributions, annual tax funding, costs,
liquidation, independent reconciliation and HTML/CSV/JSON exports. An offline
report is also at `audit-output/new-demo/report.html`; it needs no external CDN.

The dashboard selects only operator-registered private data profiles. It permits
dated input, capital, allocation/account/strategy JSON, ordinary income, state,
cash APR and annual contribution overrides. MAGI and compensation are **separate
inputs**, never inferred from ordinary income. Blank contribution override keeps
the JSON flows; zero removes contributions; a positive value generates each
year's first reviewed session, including the first year. Downloaded configuration
contains the actual generated dated flows. Unsupported combinations are BLOCKED.
Progress/cancel applies before the next canonical daily event. Cancelled runs
produce no completed wealth or tax result. This is local software, not a deployed
public web service or a background task continuing after this process stops.

## Actual source-reviewed portfolios

```bash
python -m letf.lab observed --config docs/examples/observed-s6.json \
  --output audit-output/new-observed-s6
python -m letf.lab render --report audit-output/new-observed-s6/report.json \
  --output audit-output/new-observed-s6/report.html
python -m letf.lab strategies
```

The example references existing private source snapshots relative to the mission
worktree. A fresh Git clone does not contain licensed/raw histories. Obtain the
reviewed bundles using the prior source capture/build guides or restore the
private archive; missing files produce an error. Never substitute adjusted-close
prices for nominal fills or invent classifications. Every bundle needs complete
reviewed distributions, splits, coverage, provider URLs and source checksums.

`fixed_weights/cash_v1` accepts nonnegative weights totaling at most1; residual
stays cash. S1–S19 preserve original source definitions; `legacy_vector_v0` is
archived research only, `legacy_rules_v1` is the existing canonical adapter,
and `causal_cash_v2` explicitly uses prior-close TR signals and cash caps.
Unsupported borrowing/current-close S16 variants reject. S6 retains 60/40
TQQQ/TMF with its21-session default. HFEA's explicit cash wrapper is55/45
UPRO/TMF. Frozen35/0 preserves the independently checked200SMA/35%/20-day/3x
selector; its ETF wrapper changes the vehicle and needs QQQ and reviewed daily
RF observations. It is not represented as identical futures/fractional P&L.

Comparison controls require exact dates/actions/prices/commissions/reasons;
absolute quantity/notional differences may be at most1e-10 shares/$1e-8, tighter
than the canonical reconciliation limits. This accommodates lot-splitting
roundoff, not economically different trades. Actual taxable trades may differ
because taxes consume cash. Paired accounts are alternatives, not simultaneous
owner trades; outside replacements require an explicitly reconciled household
run rather than an assumed loss deduction.

Each run exports source and implementation hashes, inputs, daily cash/positions,
flow-neutral curves, events, pre-tax controls, annual tax calculations, explicit
costs and independent identity checks. Contributions use money-weighted returns;
lump-sum CAGR is not reported for contribution-inflated wealth. CPI interior
gaps stay missing: real terminal wealth needs observed endpoints and real IRR
needs observed CPI at every cash-flow date. Daily liquidation-tax NAV is not
implemented. Generic cash restrictions remain in force. Actual fund fees and
financing embedded in observed prices are never charged again.

## 1930 counterfactuals, macro scenarios and unweighted stresses

```bash
python -m letf.lab counterfactual \
  --sources ../macro-learning/audit-output/macro-sources-v4 \
  --start 1930-01-01 --end 2025-12-31 --output audit-output/new-1930-history
python -m letf.lab simulate --config docs/examples/future-exploratory.json \
  --output audit-output/new-future-scenarios
python -m letf.lab stress \
  --experiment ../precision-reliability/audit-output/precision-v1 \
  --cases bear_7y stagnation_7y_high_rates issuer_failure_zero \
  --anchors 1 --output audit-output/new-unweighted-stresses
```

Counterfactuals reuse authentic French daily broad/growth/industry histories,
the1933 closure record and disclosed lagged bill-rate financing proxies, with
108 leverage/fee/spread configurations and overlapping historical windows.
These are **HISTORICAL_COUNTERFACTUAL**, not literal1930 S&P500/Nasdaq/TQQQ
or Roth products. The known source vintage and proxy ambiguity remain visible.

Future results are **CONDITIONAL_STRESS**. The example uses the unchanged
stationary reference and two demonstration paths for5/10/20/30/40-year full
canonical account replays. It does **not** certify account convergence. Exact
Bonferroni order-statistic sampling intervals are separate from regime, source,
funding, daily bridge and future-tax-law uncertainty. Missing independent-seed,
doubling and account-tail precision evidence cannot be replaced by the prior
million-path unit-fund precision result. No investment model is approved.

For rate/inflation/growth/correlation constraints, choose the existing
`conditional_var_t` explicitly; it remains rejected/unapproved. Bootstrap
overrides without coherent conditional mechanics fail instead of independently
changing asset returns. Named multi-year/catastrophe stresses use the existing
registered old anchors and have **no assigned probabilities**. A zero-price
taxable closure remains BLOCKED without reviewed legal disposition evidence;
the stress command preserves all such cases and exits2.

## Source lake and controlled updating

```bash
python -m letf.lab ingest --lake audit-output/new-lake \
  --folder ../macro-learning/audit-output/macro-sources-v4 --namespace macro
python -m letf.lab refresh --lake audit-output/new-lake \
  --names DFF NASDAQXNDX TQQQ_nav --output audit-output/new-refresh.json
python -m letf.lab sources --lake audit-output/new-lake
python -m letf.lab cycle --config docs/examples/operations-cycle.json \
  --output audit-output/new-operations-cycle
```

Source roots must be private0700. Captures use registered HTTPS providers,
retain raw bytes/gaps and never replace a prior snapshot. Revised current data
are not historical publication vintages. A source checksum does not confer
redistribution rights or certify a price, tax class or model. New training
catalogs materialize into new directories; original snapshots remain unchanged.

The cycle verifies the frozen commitment, assesses only complete genuinely
elapsed outcomes, optionally issues unchanged candidate model types, and scores
in **REVIEW_ONLY** mode. No silent promotion or strategy change. New registries
require explicit current-date-only intent; lost historical caches cannot reset
rejected evidence. Each cycle archives/restores the exact inputs and candidate
arrays locally; it never selects an off-host destination. A manual invocation
is not a claim a scheduled job is installed. An optional local scheduler can run:

```bash
python scripts/run_lab_cycle.py --config docs/examples/operations-cycle.json \
  --output-root audit-output/local-scheduled-cycles
```

This creates a new timestamped output on every invocation. Review source rights,
release coverage and local retention before adding it to an operator-managed
cron/systemd schedule. No credentials, paid feed, external deployment or private
data transfer is provisioned by this software.


## Derivative source boundaries

`python -m letf.lab derivative-inputs --folder PRIVATE_BUNDLE --output NEW.json`
validates schema1 `manifest.json`, SHA-256 CSV sources, dated contracts,
settlements or option bid/ask/depth and timezone-aware publication instants.
Use `--decision-at` to reject records not available at the decision. The Python
`quote_at` lookup enforces publication availability; the existing canonical
`OptionQuote.fill` additionally checks quote age, depth and spreads. Missing
margins, unknown OCC adjustments and continuous charts cannot become fills.
The adapter returns the existing contract/quote objects, never another ledger.

`SecuredOptionBook` supplies a bounded standard100-share American physical
covered-call/CSP mechanics subset in an explicitly approved Roth profile.
Premium settles T+1 and is offset by a short liquidation-ask liability. Full
strike cash or settled owned shares are reserved across all books/orders.
Actual assignment and final nonassignment require supplied notices; moneyness
alone cannot prove either. Adjustments, taxable writing/straddles/spreads and
unverified broker approval remain blocked. Sources reviewed2026-11-25:
[OCC T+1 memo54580](https://infomemo.theocc.com/infomemos?number=54580),
[OIC assignment](https://www.optionseducation.org/referencelibrary/faq/options-assignment),
[OIC cash-secured put](https://www.optionseducation.org/strategies/all-strategies/cash-secured-put).
Fixture tests establish mechanics only; no actual historical option study is certified.

The optional hosted candidate workflow requires both source-rights and approved
private-storage settings. Review the repository/storage visibility before enabling
it; no new permission or credential is configured here. Evaluation defaults to
review-only. Later same-kind/window forecast revisions are retained but excluded
from prospective selection in favor of the earliest immutable issuance.


## Registered full-account numerical precision

`python -m letf.lab account-precision --config STUDY.json --output NEW_PRIVATE_DIR
--max-new-runs 1` preregisters exact source/code, at least two distinct root seeds
and fixed doubling counts. Reinvoke with the same inputs/code to resume new runs;
completed reports are hash-checked and never overwritten. Immutable assessments
cover exact quantile/binomial intervals, independent-root differences and latest
doubling stability under the original .02/.025 goals. Incomplete or blocked paths
cannot disappear from a passing sample. Exit2 means pending/failed precision.
This uses the full canonical ledger, not a terminal tax lookup accelerator, and
never approves a forecast model. See the checkpoint reproduction guide for schema
and the deliberately failing tiny-count acceptance demonstration.


The dashboard clears prior wealth/results when a new request starts, including
when input validation rejects it. HTTP controls/cancellation/export and JS syntax
are verified. Optional `scripts/smoke_lab_browser.py --output NEW` drives only an
authored fixture through a loopback browser, filters/accounts/curves/downloads
and an unsupported-state rejection. It requires separately installed Playwright
and Chromium. This environment’s optional pinned installation timed out; no
browser execution or screenshot success is claimed. It is not a runtime dependency.
