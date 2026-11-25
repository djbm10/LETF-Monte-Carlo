Leveraged ETF Monte Carlo simulation testing multiple LETF strategies

## Financial realism and validation (2026-11-25)

The [financial-realism report](docs/financial-realism-2026-11-25/REPORT.md)
adds unweighted catastrophic-risk scenarios and complete daily account replays.
Its 600-case assessment accepts 540 bounded hypothetical-mechanics cases and
blocks 60 unreviewed taxable extinction cases. A real 2024–25 funded TQQQ
comparison reconciles dividends, contributions, splits and tax-funded sales.
Official 2021 issuer NAV/financing checks expose remaining index/cost gaps.
No model is forecast-approved; TQQQ reconstruction still fails. Private local
recovery is verified, while actual off-host storage awaits an approved destination.
See [results](docs/financial-realism-2026-11-25/RESULTS.md),
[support](docs/financial-realism-2026-11-25/SUPPORT_MATRIX.md),
[reproduction](docs/financial-realism-2026-11-25/REPRODUCE.md) and
[private recovery](docs/financial-realism-2026-11-25/BACKUP_AND_RECOVERY.md).

## Precision and reliability continuation (2026-11-25)

The [latest engineering report](docs/precision-reliability-2026-11-24/REPORT.md)
diagnoses all 15 original Monte Carlo failures and records a predeclared larger
study passing conditional sampling gates with two independent seeds × 1,048,576
paths. Original failures, rejected models and frozen forecasts are preserved.
Restricted canonical-ledger taxable/Roth endpoints reconcile; genuinely future
reference forecasts are publicly frozen and await outcomes. TQQQ reconstruction
and financial-source certification still fail. No model is investment-approved.
See [numerical results](docs/precision-reliability-2026-11-24/NUMERICAL_RESULTS.md),
[account scope/results](docs/precision-reliability-2026-11-24/ACCOUNT_RESULTS.md),
[support matrix](docs/precision-reliability-2026-11-24/SUPPORT_MATRIX.md) and
[reproduction commands](docs/precision-reliability-2026-11-24/REPRODUCE.md).

## Historical macro learning (2026-11-24)

The [implemented research system and evidence](docs/macro-learning-2026-11-24/REPORT.md)
adds daily 1930 counterfactuals, three joint macro models, immutable walk-forward
forecasts, controlled prospective retraining and account-based future scenarios.
Six real-fund account comparisons reconcile; 221 tests pass. Both macro candidates
were rejected, and the TQQQ/UPRO reconstruction failures remain visible.
See [methods](docs/macro-learning-2026-11-24/METHODS.md),
[supported combinations](docs/macro-learning-2026-11-24/SUPPORT_MATRIX.md),
[sources/restrictions](docs/macro-learning-2026-11-24/SOURCES_AND_RESTRICTIONS.md)
and [reproduction commands](docs/macro-learning-2026-11-24/REPRODUCE.md).
No model or historical derivative strategy is certified for investment use.

## Integrated account engine (2026-11-24)

The [engineering report](docs/integration-2026-11-24/REPORT.md) records repository
reconciliation, implemented fixes, real BRK-B/TQQQ/QLD taxable-versus-Roth replays,
152 passing tests, unchanged reconstruction failures and remaining capability gaps.
See the [support matrix](docs/integration-2026-11-24/SUPPORT_MATRIX.md),
[tax/data scope](docs/integration-2026-11-24/TAX_AND_DATA.md) and
[reproduction commands](docs/integration-2026-11-24/REPRODUCE.md).

```bash
.venv/bin/python -m letf.account_cli compare --bundle audit-output/account-demo-inputs \
  --state MA --output audit-output/new-comparison
```

Source bundles require explicit capture/review. Observed data with assumed
execution is not certified investment performance; unsupported scenarios fail.

# Accuracy audit (2026-11-23)

Read the [implemented audit and remaining limitations](docs/accuracy-audit/AUDIT.md),
[support matrix](docs/accuracy-audit/SUPPORT_MATRIX.md), [tax rules](docs/accuracy-audit/TAX_RULES.md)
and [data/execution assumptions](docs/accuracy-audit/DATA_AND_EXECUTION.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.lock
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_accuracy_audit.py --output audit-output/new-run
```

Legacy historical loaders and after-tax rankings are gated where their accuracy
cannot be established. The audit preserves existing research and does not claim
complete tax, derivatives, brokerage or 40-year historical strategy certification.
