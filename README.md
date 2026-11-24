Leveraged ETF Monte Carlo simulation testing multiple LETF strategies

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
