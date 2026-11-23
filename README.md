Leveraged ETF Monte Carlo simulation testing multiple LETF strategies
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
