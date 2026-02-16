import numpy as np
import pandas as pd
from typing import Dict, Optional
from letf import config as cfg


def select_random_start_regime(rng: np.random.Generator = None) -> int:
    """
    Randomly select a starting regime based on configured probabilities.

    This prevents overfitting to strategies that only work when starting
    in a specific market environment (e.g., always starting in low vol).

    Args:
        rng: Random number generator (for reproducibility)

    Returns:
        Regime ID (0 = low vol, 1 = high vol)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Extract regimes and probabilities
    regimes = list(cfg.START_REGIME_PROBABILITIES.keys())
    probs = [cfg.START_REGIME_PROBABILITIES[r] for r in regimes]

    # Normalize probabilities (in case they don't sum to 1)
    probs = np.array(probs) / sum(probs)

    return rng.choice(regimes, p=probs)


def select_random_start_offset(buffer_days: int, rng: np.random.Generator = None) -> int:
    """
    Select a random starting offset within a generated buffer.

    We generate extra history (buffer_days), then pick a random point
    to start the actual simulation. This tests robustness to different
    entry points in the market cycle.

    Args:
        buffer_days: Number of buffer days generated
        rng: Random number generator

    Returns:
        Day index to start simulation (0 to buffer_days-1)
    """
    if rng is None:
        rng = np.random.default_rng()

    if buffer_days <= 0:
        return 0

    # Don't start in the very first few days (need some history for indicators)
    min_offset = min(50, buffer_days // 4)  # At least 50 days or 25% of buffer

    return rng.integers(min_offset, buffer_days)


def get_historical_anchor_conditions(historical_df: pd.DataFrame,
                                     min_history_years: float = 2.0,
                                     rng: np.random.Generator = None) -> Dict:
    """
    Sample starting conditions from actual historical data.

    This anchors simulations to real market conditions that actually occurred,
    providing the most realistic test of strategy robustness.

    Args:
        historical_df: DataFrame with historical market data
        min_history_years: Minimum years of history before sampling
        rng: Random number generator

    Returns:
        Dict with starting conditions:
        - 'regime': Starting regime (inferred from VIX)
        - 'vix': Starting VIX level
        - 'spy_ret_20d': Recent 20-day SPY return (for momentum context)
        - 'vol_20d': Recent 20-day volatility
        - 'date': Date of the anchor point (for reference)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Calculate minimum days of history needed
    min_days = int(min_history_years * 252)

    # Valid anchor points (exclude first min_days and last year for forward sim)
    n_days = len(historical_df)
    valid_start = min_days
    valid_end = n_days - 252  # Leave at least 1 year for forward simulation

    if valid_end <= valid_start:
        # Not enough data - return default conditions
        return {
            'regime': 0,
            'vix': 18,
            'spy_ret_20d': 0.02,
            'vol_20d': 0.15,
            'date': None
        }

    # Select random anchor point
    anchor_idx = rng.integers(valid_start, valid_end)

    # Extract conditions at anchor point
    anchor_row = historical_df.iloc[anchor_idx]

    # Infer regime from VIX
    vix = anchor_row.get('VIX', 20)
    regime = 0 if vix < 25 else 1

    # Get recent market context
    lookback = min(20, anchor_idx)
    recent_returns = historical_df['SPY_Ret'].iloc[anchor_idx-lookback:anchor_idx]

    spy_ret_20d = recent_returns.sum() if len(recent_returns) > 0 else 0
    vol_20d = recent_returns.std() * np.sqrt(252) if len(recent_returns) > 0 else 0.15

    # Get date for reference
    anchor_date = historical_df.index[anchor_idx] if hasattr(historical_df.index[anchor_idx], 'strftime') else None

    return {
        'regime': regime,
        'vix': vix,
        'spy_ret_20d': spy_ret_20d,
        'vol_20d': vol_20d,
        'date': anchor_date
    }


def apply_random_start_conditions(sim_id: int, sim_days: int,
                                  regime_model: Dict,
                                  historical_df: pd.DataFrame = None) -> Dict:
    """
    Apply randomized starting conditions for a simulation.

    This is the main function called by simulate_single_path_fixed()
    to set up randomized initial conditions.

    Args:
        sim_id: Simulation ID (for reproducible randomization)
        sim_days: Number of simulation days
        regime_model: Regime parameters
        historical_df: Optional historical data for anchor points

    Returns:
        Dict with:
        - 'start_regime': Initial regime
        - 'initial_vix': Starting VIX level
        - 'buffer_days': Extra days to generate (for offset method)
        - 'start_offset': Day to start actual simulation
        - 'start_method': Which method was used
        - 'anchor_date': Historical anchor date (if applicable)
    """
    # Create reproducible RNG for this simulation
    rng = np.random.default_rng(sim_id + 99999)

    result = {
        'start_regime': 0,  # Default: low vol
        'initial_vix': 15,  # Default VIX
        'buffer_days': 0,
        'start_offset': 0,
        'start_method': 'default',
        'anchor_date': None
    }

    if not cfg.USE_RANDOM_START:
        return result

    # Apply randomization based on configured method
    if cfg.RANDOM_START_METHOD == 'regime_only':
        # Simple: just randomize starting regime
        result['start_regime'] = select_random_start_regime(rng)
        result['start_method'] = 'regime_only'

    elif cfg.RANDOM_START_METHOD == 'offset':
        # Generate extra buffer, start at random point
        buffer_days = int(cfg.RANDOM_START_BUFFER_YEARS * 252)
        start_offset = select_random_start_offset(buffer_days, rng)

        result['buffer_days'] = buffer_days
        result['start_offset'] = start_offset
        result['start_method'] = 'offset'

        # The starting regime will be determined by the regime at start_offset
        # (handled in the main simulation function)

    elif cfg.RANDOM_START_METHOD == 'historical_anchor':
        # Anchor to actual historical conditions
        if historical_df is not None and len(historical_df) > 252 * cfg.MIN_HISTORY_FOR_ANCHOR:
            anchor = get_historical_anchor_conditions(historical_df, cfg.MIN_HISTORY_FOR_ANCHOR, rng)

            result['start_regime'] = anchor['regime']
            result['initial_vix'] = anchor['vix']
            result['anchor_date'] = anchor['date']
            result['start_method'] = 'historical_anchor'
        else:
            # Fallback to regime_only if no historical data
            result['start_regime'] = select_random_start_regime(rng)
            result['start_method'] = 'regime_only_fallback'

    # Randomize initial VIX if enabled
    if cfg.RANDOMIZE_INITIAL_VIX and result['start_method'] != 'historical_anchor':
        regime = result['start_regime']
        vix_low, vix_high = cfg.INITIAL_VIX_RANGE[regime]
        result['initial_vix'] = rng.uniform(vix_low, vix_high)

    return result
