#!/usr/bin/env python3
"""
LETF34 mini runner.

Runs the same LETF34 engine with lighter defaults for faster iteration.
No core logic is edited in LETF34_analysis.py.

Usage:
  python LETF34mini.py

Optional environment overrides:
  LETF34_MINI_NUM_SIMS=20
  LETF34_MINI_WORKERS=2
  LETF34_MINI_FAST_VALIDATION=1
  LETF34_MINI_DEEP_REALISM=0
  LETF34_MINI_ENABLE_STAGE_AB=0
  LETF34_MINI_ENABLE_STAGE_C=0
  LETF34_MINI_REAL_ONLY_MODE=unconditional
  LETF34_MINI_REPEAT_RUNS=1
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

import LETF34_analysis as core


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return max(int(raw), int(min_value))
    except Exception:
        return int(default)


def _quick_zero_drift_vol_drag(n_sims: int = 1500) -> Dict[str, Any]:
    annual_vol = 0.15
    daily_std = annual_vol / np.sqrt(252.0)
    leverage = 3.0
    n_days = 252
    rng = np.random.default_rng(42)

    sim_returns = np.empty(int(n_sims), dtype=float)
    for i in range(int(n_sims)):
        daily_returns = rng.normal(0.0, daily_std, n_days)
        leveraged_returns = leverage * daily_returns
        sim_returns[i] = np.prod(1.0 + leveraged_returns) - 1.0

    expected_drag = -0.5 * leverage ** 2 * annual_vol ** 2
    actual_drag = float(np.median(sim_returns))
    tol = abs(expected_drag) * 0.20
    test_pass = bool(abs(actual_drag - expected_drag) < tol)

    return {
        "test_passed": test_pass,
        "expected_drag": float(expected_drag),
        "actual_drag": float(actual_drag),
        "error_pct": float(abs(actual_drag - expected_drag) * 100.0),
        "mini_profile": True,
        "n_sims": int(n_sims),
    }


def _quick_flat_market_decay(n_sims: int = 1200) -> Dict[str, Any]:
    annual_vol = 0.15
    daily_std = annual_vol / np.sqrt(252.0)
    n_days = 252
    out: Dict[str, Any] = {"mini_profile": True, "n_sims": int(n_sims)}
    all_passed = True

    for leverage in (2.0, 3.0):
        rng = np.random.default_rng(42 + int(leverage))
        sim_returns = np.empty(int(n_sims), dtype=float)
        for i in range(int(n_sims)):
            daily_returns = rng.normal(0.0, daily_std, n_days)
            leveraged_returns = leverage * daily_returns
            sim_returns[i] = np.prod(1.0 + leveraged_returns) - 1.0

        expected_drag = -0.5 * leverage ** 2 * annual_vol ** 2
        actual_median = float(np.median(sim_returns))
        actual_mean = float(np.mean(sim_returns))
        actual_std = float(np.std(sim_returns))
        error = abs(actual_median - expected_drag)
        passed = bool(error < 0.03)
        all_passed = all_passed and passed
        out[f"{leverage}x"] = {
            "expected": float(expected_drag),
            "actual_median": actual_median,
            "actual_mean": actual_mean,
            "actual_std": actual_std,
            "error": float(error),
            "passed": passed,
        }

    out["all_passed"] = bool(all_passed)
    return out


def _run_validation_tests_mini(
    df: Optional["core.pd.DataFrame"] = None,
    regime_model: Optional[Dict[str, Any]] = None,
    walk_forward_df: Optional["core.pd.DataFrame"] = None,
) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print("RUNNING VALIDATION TESTS (MINI PROFILE)")
    print(f"{'='*80}\n")

    include_deep_realism = _env_bool("LETF34_MINI_DEEP_REALISM", False)
    results: Dict[str, Any] = {}

    calibration_report = {}
    synthetic_risk_report = {}
    spy_crash_tuning = {}
    letf_tail_tuning = {}
    if isinstance(regime_model, dict):
        calibration_report = regime_model.get("calibration_report", {}) or {}
        synthetic_risk_report = regime_model.get("synthetic_model_risk", {}) or {}
        spy_crash_tuning = regime_model.get("spy_crash_tuning", {}) or {}
        letf_tail_tuning = regime_model.get("letf_tail_tuning", {}) or {}
        if calibration_report:
            results["calibration_details"] = dict(calibration_report)
        if synthetic_risk_report:
            results["synthetic_model_risk"] = dict(synthetic_risk_report)
        if spy_crash_tuning:
            results["spy_crash_tuning"] = dict(spy_crash_tuning)
        if letf_tail_tuning:
            results["letf_tail_tuning"] = dict(letf_tail_tuning)

    results["zero_drift_test"] = _quick_zero_drift_vol_drag(
        n_sims=_env_int("LETF34_MINI_ZERO_DRIFT_SIMS", 1500, min_value=200)
    )
    results["flat_market_test"] = _quick_flat_market_decay(
        n_sims=_env_int("LETF34_MINI_FLAT_TEST_SIMS", 1200, min_value=200)
    )

    if regime_model is not None:
        results["institutional_sanity"] = core.run_institutional_sanity_checks(
            regime_model=regime_model,
            funding_model=regime_model.get("funding_model", {}),
            tracking_residual_model=regime_model.get("tracking_residual_model", {}),
        )

    if include_deep_realism and df is not None and regime_model is not None:
        results["max_drawdown_real_vs_sim"] = core.validate_simulated_max_drawdowns_vs_real(
            df=df,
            regime_model=regime_model,
        )
        results["volatility_real_vs_sim"] = core.validate_simulated_volatility_vs_real(
            df=df,
            regime_model=regime_model,
        )
        results["leveraged_real_fit"] = core.validate_leveraged_real_fit_metrics(
            df=df,
            regime_model=regime_model,
            years=5,
            save_plots=False,
        )

    results["leverage_levels_stress_test"] = core.validate_leverage_level_stress_test(
        n_sims=_env_int("LETF34_MINI_LEVERAGE_STRESS_SIMS", 600, min_value=100),
        n_days=252,
        annual_mu=0.08,
        annual_vol=0.18,
    )

    results["mini_profile"] = {
        "enabled": True,
        "deep_realism": bool(include_deep_realism),
    }

    with open(core.VALIDATION_RESULTS, "w") as f:
        json.dump(results, f, indent=2)

    print("Validation profile: MINI")
    print(f"  Deep realism checks: {'ON' if include_deep_realism else 'OFF'}")
    print(f"  Results saved to: {core.VALIDATION_RESULTS}")
    return results


def apply_mini_profile() -> None:
    default_workers = max(1, min(int(getattr(core, "N_WORKERS", 1)), 2))
    default_sims = 20

    core.N_WORKERS = _env_int("LETF34_MINI_WORKERS", default_workers, min_value=1)
    core.NUM_SIMULATIONS = _env_int("LETF34_MINI_NUM_SIMS", default_sims, min_value=4)

    core.ENABLE_TWO_STAGE_TAIL_TUNING = _env_bool("LETF34_MINI_ENABLE_STAGE_AB", False)
    core.ENABLE_STAGE_C_EXEC_TE_TUNING = _env_bool("LETF34_MINI_ENABLE_STAGE_C", False)

    core.REAL_ONLY_VALIDATION_MODE = os.getenv(
        "LETF34_MINI_REAL_ONLY_MODE", "unconditional"
    ).strip() or "unconditional"
    core.REAL_ONLY_REPEATABILITY_RUNS = _env_int(
        "LETF34_MINI_REPEAT_RUNS", 1, min_value=1
    )

    if _env_bool("LETF34_MINI_FAST_VALIDATION", True):
        core.run_validation_tests = _run_validation_tests_mini

    print(f"\n{'='*80}")
    print("LETF34 MINI PROFILE")
    print(f"{'='*80}")
    print(f"Workers: {core.N_WORKERS}")
    print(f"Monte Carlo sims per horizon: {core.NUM_SIMULATIONS}")
    print(f"Stage A/B tuning: {'ON' if core.ENABLE_TWO_STAGE_TAIL_TUNING else 'OFF'}")
    print(f"Stage C tuning: {'ON' if core.ENABLE_STAGE_C_EXEC_TE_TUNING else 'OFF'}")
    print(f"Real-only validation mode: {core.REAL_ONLY_VALIDATION_MODE}")
    print(f"Repeatability runs: {core.REAL_ONLY_REPEATABILITY_RUNS}")
    print(
        f"Fast validation: {'ON' if _env_bool('LETF34_MINI_FAST_VALIDATION', True) else 'OFF'}"
    )
    print(f"{'='*80}\n")


def main() -> None:
    apply_mini_profile()

    print("\n### VALIDATING TAX ENGINE ###\n")
    try:
        core.run_golden_tests(trace_failures=True)
        print("\n✓ Tax engine validated - proceeding with simulation\n")
    except Exception as exc:
        print(f"\n⛔ GOLDEN TESTS FAILED: {exc}")
        print("⛔ STOPPING - System is broken")
        raise SystemExit(1) from exc

    core.main()


if __name__ == "__main__":
    main()
