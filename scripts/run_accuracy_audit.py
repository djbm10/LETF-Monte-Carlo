"""Offline reproducible audit checks. Does not run unsupported strategy rankings.

Run from the repository root with the pinned virtual environment. Results are
new files only; immutable original validation metadata and caches are untouched.
"""
import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from letf.tax.engine import compute_capital_gains, TaxpayerElections, run_golden_tests
from letf.tax.marginal import calculate_comprehensive_tax_v6
from letf.calibration import simulate_joint_returns_t
from letf.utils import nearest_psd_matrix
from letf import config as cfg
from letf.validation import validate_zero_drift_vol_drag, validate_flat_market_decay, run_forward_realism_tests


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'python': platform.python_version(), 'seed': 20261123, 'strategy_comparisons_run': False,
              'reason': 'Relevant historical data/execution/tax gates are incomplete or failed',
              'tests': {}, 'before_after': {}}
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], text=True, capture_output=True)
    (output / 'pytest.txt').write_text(result.stdout + result.stderr)
    report['tests']['pytest'] = {'exit_code': result.returncode, 'output': result.stdout.strip()}
    baseline = json.loads(Path('docs/accuracy-audit/baseline.json').read_text())
    for name, args in {'character': (10000, 0, 0, 10000, 10000, 0),
                       'carry_only': (0, 0, 0, 0, 5000, 0)}.items():
        current = asdict(compute_capital_gains(*args, TaxpayerElections()))
        keys = ('taxable_st', 'taxable_lt', 'st_loss_cf_out', 'lt_loss_cf_out', 'capital_loss_deduction')
        report['before_after'][name] = {'before': {k: baseline[name][k] for k in keys},
                                      'after': {k: current[k] for k in keys}}
    report['before_after']['unused_deduction_tax'] = {
        'before': baseline['unused_deduction']['total_tax'],
        'after': calculate_comprehensive_tax_v6(0, 60000, 0, include_state=False, tax_year=2024)['total_tax']}
    # Re-execute just the original pure simulation function for a paired
    # diagnostic. No import-time entry points or original research reruns.
    original = subprocess.check_output(['git', 'show', 'b967b2b498c261a2f551cfde9ab1621e5c489c2d:letf/calibration.py'], text=True)
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate_joint_returns_t')
    source = ast.get_source_segment(original, node)
    namespace = {'np': np, 'Dict': dict, 'nearest_psd_matrix': nearest_psd_matrix,
                 'cfg': SimpleNamespace(USE_MOMENT_MATCHING=True)}
    exec(compile(source, '<preserved-baseline-function>', 'exec'), namespace)
    n = 30000
    model = {'assets': ['X'], 'regimes': {0: {'mu': np.array([.001]), 'cov': np.array([[.0001]]),
              'nu': 5., 'garch_alpha': .02, 'garch_beta': .9, 'dcc_a': 0., 'dcc_b': 0.}}}
    old = namespace['simulate_joint_returns_t'](n, np.zeros(n, dtype=int), model, np.random.default_rng(20261123))['X']
    new = simulate_joint_returns_t(n, np.zeros(n, dtype=int), model, np.random.default_rng(20261123))['X']
    report['before_after']['student_t_garch_variance'] = {'target': .0001, 'n': n,
         'before': float(np.var(old)), 'after': float(np.var(new)), 'relative_tolerance': .10,
         'before_passed': bool(abs(np.var(old) / .0001 - 1) <= .10),
         'after_passed': bool(abs(np.var(new) / .0001 - 1) <= .10)}
    for name, check in [('netting_regressions', run_golden_tests), ('zero_drift', validate_zero_drift_vol_drag),
                        ('flat_market', validate_flat_market_decay), ('legacy_forward_realism', run_forward_realism_tests)]:
        stream = io.StringIO()
        try:
            with redirect_stdout(stream):
                values = check()
            report['tests'][name] = values
        except Exception as exc:
            report['tests'][name] = {'error': repr(exc)}
        (output / f'{name}.txt').write_text(stream.getvalue())
    # Intentional fail-closed behavior of the former network smoke script is
    # recorded separately from unit-test success.
    quick = subprocess.run([sys.executable, 'quick_test.py'], text=True, capture_output=True)
    (output / 'legacy_quick_test.txt').write_text(quick.stdout + quick.stderr)
    report['tests']['legacy_quick_test'] = {'exit_code': quick.returncode,
                                           'expected_accuracy_gate': 'unverified proxies' in quick.stderr}
    hashes = {}
    for name in ('LETF35_analysis.py', 'validation_metadata.json', 'docs/accuracy-audit/reconstruction-gate.json',
                 'requirements-audit.lock'):
        # Updated main removed some baseline-only metadata. Its absence is an
        # audit fact, not a reason to manufacture a replacement sealed report.
        hashes[name] = sha256(Path(name).read_bytes()).hexdigest() if Path(name).is_file() else None
    report['baseline_files_absent_from_integrated_checkout'] = [name for name, value in hashes.items() if value is None]
    for path in sorted(Path('data').rglob('*.pkl')):
        hashes[str(path)] = sha256(path.read_bytes()).hexdigest()
    report['protected_input_sha256'] = hashes
    # A useful suite can pass while the original model-confidence gate fails.
    report['implementation_tests_passed'] = result.returncode == 0
    report['full_mission_verified'] = False
    (output / 'report.json').write_text(json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    print(json.dumps({'implementation_tests_passed': report['implementation_tests_passed'],
                      'before_after': report['before_after'], 'full_mission_verified': False}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
