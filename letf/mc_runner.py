import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List
import pandas as pd
from letf import config as cfg
from letf.utils import load_cache
from letf.simulation.bootstrap import create_bootstrap_sampler
from letf.simulation.engine import simulate_single_path_fixed
from letf.calibration import (
    calibrate_joint_return_model, calibrate_funding_spread_model,
    calibrate_stress_state_model, calibrate_tracking_residual_model
)

# Try to use joblib (better for large NumPy arrays), fall back to ProcessPoolExecutor
try:
    from joblib import Parallel, delayed
    USE_JOBLIB = True
except ImportError:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    USE_JOBLIB = False
    print("Warning: joblib not available, using ProcessPoolExecutor (install joblib for better performance)")


def parallel_monte_carlo_fixed(strategy_ids, time_horizon, regime_model, correlation_matrices,
                               historical_df: pd.DataFrame = None):
    """
    Parallel Monte Carlo with all fixes.

    UPGRADE: Now supports block bootstrap from historical data for fat-tailed returns.

    Args:
        strategy_ids: List of strategy IDs to simulate
        time_horizon: Simulation horizon in years
        regime_model: Calibrated regime model
        correlation_matrices: Calibrated correlation matrices
        historical_df: Optional historical data for block bootstrap
    """
    print(f"\n{'='*80}")
    print(f"MONTE CARLO: {cfg.NUM_SIMULATIONS:,} sims x {time_horizon}Y")
    print(f"{'='*80}")

    # Check for Numba JIT compilation
    try:
        from numba import jit
        print(f"  Numba JIT: ENABLED (10-20x speedup on hot paths)")
    except ImportError:
        print(f"  Numba JIT: not available (install with 'pip install numba' for 10-20x speedup)")

    # Check for arch library
    try:
        from arch import arch_model
        print(f"  arch library: ENABLED (professional GARCH estimation)")
    except ImportError:
        print(f"  arch library: not available (install with 'pip install arch' for better GARCH)")

    # Check for GPU acceleration
    if cfg.USE_GPU:
        try:
            import cupy as cp
            gpu_info = cp.cuda.Device().attributes
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"  GPU acceleration: ENABLED (device: {gpu_name})")
        except ImportError:
            print(f"  WARNING: USE_GPU=True but CuPy not installed. Install with:")
            print(f"           pip install cupy-cuda11x  # For CUDA 11.x")
            print(f"           pip install cupy-cuda12x  # For CUDA 12.x")
            print(f"  Falling back to CPU execution.")
            cfg.USE_GPU = False  # Disable if not available
        except Exception as e:
            print(f"  WARNING: GPU initialization failed: {e}")
            print(f"  Falling back to CPU execution.")
            cfg.USE_GPU = False

    regime_model = regime_model.copy()  # local mutable copy for worker payload enrichments
    sim_cfg = cfg.get_simulation_config()

    # ========================================================================
    # CREATE BOOTSTRAP SAMPLER (if historical data provided)
    # ========================================================================
    if historical_df is not None and sim_cfg.use_block_bootstrap:
        print(f"\n  Creating block bootstrap sampler for fat-tailed returns...")
        bootstrap_sampler = create_bootstrap_sampler(historical_df)

        # Add to regime_model so it's passed to simulation workers
        regime_model['bootstrap_sampler'] = bootstrap_sampler

        print(f"  [OK] Bootstrap sampler ready (block size: {cfg.BOOTSTRAP_BLOCK_SIZE} days)")
        print(f"  [OK] Student-t df: {cfg.STUDENT_T_DF} (fat tails)")
        print(f"  [OK] Bootstrap weight: {sim_cfg.bootstrap_weight*100:.0f}% historical + {(1-sim_cfg.bootstrap_weight)*100:.0f}% noise")
    else:
        print(f"\n  Using parametric Student-t returns (no historical data for bootstrap)")

    print(f"\n  Simulation engine mode: {sim_cfg.engine_mode}")
    if sim_cfg.engine_mode == 'institutional_v1':
        print("  [OK] Institutional v1: joint multivariate t returns + state-linked funding + calibrated residual TE")
        print("  [OK] Added latent stress channels: liquidity, credit, and crisis jump overlay")
    else:
        print(f"\n  Using parametric Student-t returns (no historical data for bootstrap)")


    # ========================================================================
    # RANDOMIZED START DATE CONFIGURATION
    # ========================================================================
    if cfg.USE_RANDOM_START:
        print(f"\n  Randomized start dates ENABLED:")
        print(f"    Method: {cfg.RANDOM_START_METHOD}")

        if cfg.RANDOM_START_METHOD == 'regime_only':
            print(f"    Start regime probabilities: {cfg.START_REGIME_PROBABILITIES}")
        elif cfg.RANDOM_START_METHOD == 'offset':
            print(f"    Buffer: {cfg.RANDOM_START_BUFFER_YEARS} years ({int(cfg.RANDOM_START_BUFFER_YEARS*252)} days)")
        elif cfg.RANDOM_START_METHOD == 'historical_anchor':
            print(f"    Min history for anchor: {cfg.MIN_HISTORY_FOR_ANCHOR} years")

            # Pass historical data for anchor point selection
            if historical_df is not None:
                regime_model['historical_df_for_anchors'] = historical_df
                print(f"    Historical data available: {len(historical_df)} days")
            else:
                print(f"    WARNING: No historical data - will fallback to regime_only")

        if cfg.RANDOMIZE_INITIAL_VIX:
            print(f"    Initial VIX ranges: Low vol {cfg.INITIAL_VIX_RANGE[0]}, High vol {cfg.INITIAL_VIX_RANGE[1]}")
    else:
        print(f"\n  Randomized start dates DISABLED (fixed start in low vol)")

    all_results = {sid: [] for sid in strategy_ids}

    # ========================================================================
    # ANTITHETIC VARIATES (optional, 30-50% variance reduction)
    # ========================================================================
    if cfg.USE_ANTITHETIC_VARIATES:
        print(f"  Antithetic variates ENABLED (30-50% variance reduction)")
        print(f"  Pairing simulations: (0,1), (2,3), ...\n")
        # Ensure even number of simulations for proper pairing
        effective_sims = cfg.NUM_SIMULATIONS if cfg.NUM_SIMULATIONS % 2 == 0 else cfg.NUM_SIMULATIONS + 1
    else:
        effective_sims = cfg.NUM_SIMULATIONS

    # Use joblib if available (better for NumPy arrays), otherwise ProcessPoolExecutor
    if USE_JOBLIB:
        # Joblib with progress bar
        print(f"  Using joblib with {cfg.N_WORKERS} workers\n")

        if cfg.USE_ANTITHETIC_VARIATES:
            # Create pairs: even sim_ids get antithetic=False, odd get antithetic=True
            # Odd simulations use same base sim_id as even (for same regime path)
            sim_args_list = []
            for sim_id in range(0, effective_sims, 2):
                # Normal path
                sim_args_list.append((sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids, False))
                # Antithetic path (uses same sim_id for regime path consistency)
                sim_args_list.append((sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids, True))

            results_list = Parallel(n_jobs=cfg.N_WORKERS, backend='loky', verbose=0)(
                delayed(simulate_single_path_fixed)(args)
                for args in tqdm(sim_args_list[:cfg.NUM_SIMULATIONS], desc=f"{time_horizon}Y MC", unit="sim")
            )
        else:
            results_list = Parallel(n_jobs=cfg.N_WORKERS, backend='loky', verbose=0)(
                delayed(simulate_single_path_fixed)(
                    (sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids)
                )
                for sim_id in tqdm(range(cfg.NUM_SIMULATIONS), desc=f"{time_horizon}Y MC", unit="sim")
            )

        # Organize results by strategy
        for path_results in results_list:
            if path_results is not None:  # Skip failed simulations
                for sid in strategy_ids:
                    if sid in path_results:
                        all_results[sid].append(path_results[sid])

    else:
        # Fallback to ProcessPoolExecutor
        print(f"  Using ProcessPoolExecutor with {cfg.N_WORKERS} workers\n")

        if cfg.USE_ANTITHETIC_VARIATES:
            # Create antithetic pairs
            sim_args = []
            for sim_id in range(0, effective_sims, 2):
                sim_args.append((sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids, False))
                sim_args.append((sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids, True))
            sim_args = sim_args[:cfg.NUM_SIMULATIONS]  # Trim to requested count
        else:
            sim_args = [
                (sim_id, time_horizon, regime_model, correlation_matrices, strategy_ids)
                for sim_id in range(cfg.NUM_SIMULATIONS)
            ]

        with ProcessPoolExecutor(max_workers=cfg.N_WORKERS) as executor:
            futures = {executor.submit(simulate_single_path_fixed, arg): i
                      for i, arg in enumerate(sim_args)}

            with tqdm(total=cfg.NUM_SIMULATIONS, desc=f"{time_horizon}Y MC", unit="sim") as pbar:
                for future in as_completed(futures):
                    try:
                        path_results = future.result(timeout=300)  # 5 minute timeout per simulation
                        if path_results is not None:
                            for sid in strategy_ids:
                                if sid in path_results:
                                    all_results[sid].append(path_results[sid])
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n[!] Simulation error: {e}")
                        import traceback
                        traceback.print_exc()
                        pbar.update(1)

    return all_results
