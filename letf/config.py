from pathlib import Path
from dataclasses import dataclass
import multiprocessing
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Extended backtest using Fama-French data back to 1926
# This includes the Great Depression (1929-1932) for stress testing!
DATA_START_DATE = "1926-07-01"  # Earliest available data (Fama-French)
DATA_END_DATE = "2025-12-31"    # Latest available data
INITIAL_CAPITAL = 10000

# The actual analysis start date (can be changed by user)
# These will be set by get_start_date_interactive()
ANALYSIS_START_DATE = "1926-07-01"  # Default: use all available data
ANALYSIS_END_DATE = "2025-12-31"    # Default: use all available data

# Cutoff date for switching from Fama-French to yfinance
# yfinance has better data (NASDAQ, VIX, etc.) from 1950 onward
FAMA_FRENCH_END_DATE = "1949-12-31"
YFINANCE_START_DATE = "1950-01-01"

TIME_HORIZONS = [1, 2, 5, 10, 20, 30, 40, 50]

# Predefined start date options for quick selection
START_DATE_OPTIONS = {
    1: {
        'date': '1926-07-01',
        'name': 'Full History',
        'description': 'Includes Great Depression, WWII, all major events'
    },
    2: {
        'date': '1950-01-01',
        'name': 'Post-WWII',
        'description': 'More reliable data, excludes pre-war period'
    },
    3: {
        'date': '1980-01-01',
        'name': 'Modern Era',
        'description': 'After stagflation, more relevant to today'
    },
    4: {
        'date': '2000-01-01',
        'name': '21st Century',
        'description': 'Includes dot-com crash, 2008 crisis, COVID'
    },
    5: {
        'date': '2010-01-01',
        'name': 'Post-Crisis',
        'description': 'TQQQ real data begins, bull market era'
    },
    6: {
        'date': '2015-01-01',
        'name': 'Recent History',
        'description': 'Last 10 years only'
    }
}

ASSETS = {
    'TQQQ': {
        'name': '3x NASDAQ-100',
        'inception': '2010-02-11',
        'leverage': 3.0,
        'expense_ratio': 0.0086,
        'underlying': 'QQQ',
        'proxy_index': '^IXIC',
        'beta_to_spy': 1.0,  # QQQ returns already generated independently by joint model
        'tracking_error_base': 0.0002,  # 2 bps in low vol
        'tracking_error_df': 5,  # t-distribution degrees of freedom
        'borrow_spread': 0.0075,  # 0.75% spread above risk-free rate (realistic for equity swaps)
    },
    'UPRO': {
        'name': '3x S&P 500',
        'inception': '2009-06-25',
        'leverage': 3.0,
        'expense_ratio': 0.0091,
        'underlying': 'SPY',
        'proxy_index': '^GSPC',
        'beta_to_spy': 1.0,
        'tracking_error_base': 0.00015,
        'tracking_error_df': 5,
        'borrow_spread': 0.0060,  # 0.60% spread (S&P 500 is more liquid, cheaper to borrow)
    },
    'SSO': {
        'name': '2x S&P 500',
        'inception': '2006-07-11',
        'leverage': 2.0,
        'expense_ratio': 0.0089,
        'underlying': 'SPY',
        'proxy_index': '^GSPC',
        'beta_to_spy': 1.0,
        'tracking_error_base': 0.0001,
        'tracking_error_df': 5,
        'borrow_spread': 0.0050,  # 0.50% spread (2x funds often get better rates)
    },
    'TMF': {
        'name': '3x 20Y Treasury',
        'inception': '2009-04-16',
        'leverage': 3.0,
        'expense_ratio': 0.0108,
        'underlying': 'TLT',
        'proxy_index': '^TNX',
        'beta_to_spy': -0.3,
        'tracking_error_base': 0.0003,
        'tracking_error_df': 5,
        'borrow_spread': 0.0040,  # 0.40% spread (Treasuries are very liquid collateral)
    },
    'SPY': {
        'name': 'S&P 500 (No Leverage)',
        'inception': '1993-01-29',
        'leverage': 1.0,
        'expense_ratio': 0.000945,  # 0.0945% (updated 2025)
        'underlying': 'SPY',
        'proxy_index': '^GSPC',
        'beta_to_spy': 1.0,
        'tracking_error_base': 0.00005,
        'tracking_error_df': 10,
        'borrow_spread': 0.0,  # No leverage = no borrowing
    },
    'QQQ': {
        'name': 'NASDAQ-100 (No Leverage)',
        'inception': '1999-03-10',
        'leverage': 1.0,
        'expense_ratio': 0.0020,  # 0.20% QQQ expense ratio
        'underlying': 'QQQ',
        'proxy_index': '^IXIC',
        'beta_to_spy': 1.0,  # beta applied inside map_underlying_series_for_asset
        'tracking_error_base': 0.00005,
        'tracking_error_df': 10,
        'borrow_spread': 0.0,  # No leverage = no borrowing
    }
}

# Alias used by simulate_single_path_fixed() layered simulation
LETF_CONFIGS = ASSETS

# Transaction costs - realistic values
BASE_SPREAD_BPS = {0: 2, 1: 8}  # Low vol / High vol
REBALANCE_COST_PER_DOLLAR = 0.0001

# Risk-free rate by regime
CASH_RATE_BY_REGIME = {
    0: 0.010,  # Low vol: normal rates
    1: -0.020   # High vol: Fed cuts
}

# Monte Carlo parameters
N_WORKERS = max(1, multiprocessing.cpu_count() - 2)
NUM_SIMULATIONS = 200

# Variance reduction techniques (NEW - for improved accuracy)
USE_ANTITHETIC_VARIATES = True   # 30-50% variance reduction, minimal cost
USE_MOMENT_MATCHING = True       # Improves long-horizon numerical stability
USE_LATIN_HYPERCUBE = False      # 20-40% variance reduction via quasi-Monte Carlo (QMC)
                                  # NOTE: Disable when using antithetic variates (incompatible)

# Debugging and logging
DEBUG = False                     # Set to True for verbose error messages and stack traces

# GPU acceleration (requires NVIDIA GPU and CuPy)
USE_GPU = False                   # Set to True to use GPU acceleration via CuPy
                                  # Requires: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)
                                  # Expected speedup: 50-100x for large simulations (100k+ paths)

# Regime parameters (FIX: 2 REGIMES BASED ON VOLATILITY)
N_REGIMES = 2
REGIME_NAMES = {0: 'Low Vol', 1: 'High Vol'}

# Minimum regime durations (trading days)
MIN_REGIME_DURATION = {
    0: 60,   # Low vol: minimum ~3 months
    1: 20    # High vol: minimum ~1 month
}

# Cache
CACHE_DIR = Path("corrected_cache_v8")

# NOTE: Cache filenames will be set dynamically based on selected dates
# These are just defaults - actual filenames set in get_cache_filenames()
DATA_CACHE = CACHE_DIR / "historical_data.pkl"
REGIME_MODEL_CACHE = CACHE_DIR / "regime_model.pkl"
CORRELATION_CACHE = CACHE_DIR / "correlations.pkl"
VALIDATION_RESULTS = CACHE_DIR / "validation_results.json"


def init_cache():
    """Create the cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def get_cache_filenames(start_date: str, end_date: str):
    """
    Generate cache filenames that include the date range.
    This ensures different date selections use different caches.
    """
    # Create a short hash from the dates for the filename
    date_suffix = f"{start_date[:4]}_{end_date[:4]}"

    return {
        'data': CACHE_DIR / f"historical_data_{date_suffix}.pkl",
        'regime': CACHE_DIR / f"regime_model_{date_suffix}.pkl",
        'correlation': CACHE_DIR / f"correlations_{date_suffix}.pkl",
        'validation': CACHE_DIR / f"validation_results_{date_suffix}.json"
    }


def clear_all_caches():
    """Clear all cache files to force fresh data load."""
    import shutil
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print("All caches cleared")


BOOTSTRAP_BLOCK_MIN = 21
BOOTSTRAP_BLOCK_MAX = 168  # ~8 months max (safer than 252)
BOOTSTRAP_BLOCK_MEAN = 84  # target average block length
BOOTSTRAP_BLOCK_SIZE = BOOTSTRAP_BLOCK_MAX

# Reduce directional persistence (regime-dependent)
BOOTSTRAP_MOMENTUM_BIAS_BY_REGIME = {
    0: 0.54,   # low vol: modest trend continuation
    1: 0.505   # high vol: near-random continuation
}

# Degrees of freedom for Student-t noise
# Lower = fatter tails, more extreme events
#
# df=5 gives excess kurtosis ~6, matching empirical daily equity returns (3-5).
# With GARCH dynamics already producing time-varying heavy tails,
# df=5 avoids compounding tail thickness through two independent channels.
# df=4 had undefined kurtosis of squared returns and over-represented extremes.
STUDENT_T_DF = 5

# How much of the return comes from bootstrap vs Student-t noise
#
# 0.80 = 80% historical bootstrap + 20% correlated Student-t noise
#
# WHY 80/20 INSTEAD OF 70/30:
# The previous 30% noise was calibrated to match synthetic+historical data
# that included the Great Depression destroying a hypothetical TQQQ.
# With the borrow cost bug fixed and real QQQ blocks in place, the
# bootstrap pool itself carries proper volatility and tail risk.
#
# 20% noise still allows for "what if" scenarios beyond exact historical
# replay, while keeping the simulation grounded in real market behavior.
# This leans slightly pessimistic because:
#   - Borrow costs are now fully applied (~10.5%/yr drag for TQQQ)
#   - Real QQQ blocks carry tech-sector crash risk (2000-2002, 2022)
#   - Noise is correlated (no free diversification benefit)
BOOTSTRAP_WEIGHT = 0.80


# Whether to use block bootstrap (True) or parametric generation (False)
# Set to False to revert to old behavior for comparison
USE_BLOCK_BOOTSTRAP = True

# Simulation engine mode:
# - 'legacy_hybrid': existing block-bootstrap + calibrated overlays
# - 'institutional_v1': latent-state-style joint multivariate t core
SIM_ENGINE_MODE = 'institutional_v1'


@dataclass(frozen=True)
class SimulationConfig:
    """Centralized simulation engine configuration used by orchestration/validation."""
    engine_mode: str
    use_block_bootstrap: bool
    bootstrap_weight: float


def get_simulation_config() -> SimulationConfig:
    """Return the active simulation configuration in one canonical object."""
    return SimulationConfig(
        engine_mode=SIM_ENGINE_MODE,
        use_block_bootstrap=USE_BLOCK_BOOTSTRAP,
        bootstrap_weight=float(BOOTSTRAP_WEIGHT)
    )

# Structural model caches (institutional_v1)
JOINT_RETURN_MODEL_CACHE = CACHE_DIR / "joint_return_model.pkl"
FUNDING_MODEL_CACHE = CACHE_DIR / "funding_spread_model.pkl"
TRACKING_RESIDUAL_CACHE = CACHE_DIR / "tracking_residual_model.pkl"
STRESS_STATE_CACHE = CACHE_DIR / "stress_state_model.pkl"

# Cache for bootstrap data (processed historical returns by regime)
BOOTSTRAP_CACHE = CACHE_DIR / "bootstrap_data.pkl"


# ============================================================================
# RANDOMIZED START DATE CONFIGURATION
# ============================================================================
#
# To avoid overfitting to specific starting conditions, we randomize:
# 1. Which regime the simulation starts in
# 2. Where in a generated history the simulation begins
# 3. Optionally anchor to actual historical market conditions
#
# This tests strategy robustness across different market entry points.

# Master switch for start date randomization
USE_RANDOM_START = True

# Method for randomizing start conditions:
# - 'regime_only': Random starting regime (low vol or high vol)
# - 'offset': Generate extra history, start at random point
# - 'historical_anchor': Sample from actual historical starting conditions
RANDOM_START_METHOD = 'offset'

# For 'regime_only' method:
# Probability of starting in each regime (should sum to 1.0)
# Default reflects historical: ~80% of time in low vol, ~20% in high vol
START_REGIME_PROBABILITIES = {
    0: 0.80,  # Low volatility regime
    1: 0.20   # High volatility regime
}

# For 'offset' method:
# How much extra history to generate (in years)
# We generate this much extra, then pick a random start point
RANDOM_START_BUFFER_YEARS = 5

# For 'historical_anchor' method:
# Minimum years of historical data needed before an anchor point
# (We won't start at Day 1 of history - need some history to establish conditions)
MIN_HISTORY_FOR_ANCHOR = 2  # Years

# Whether to also randomize VIX starting level based on regime
RANDOMIZE_INITIAL_VIX = True

# VIX ranges by regime for random initialization
INITIAL_VIX_RANGE = {
    0: (12, 20),   # Low vol: VIX typically 12-20
    1: (25, 45)    # High vol: VIX typically 25-45
}

# Track which start method was used (for analysis)
# This will be stored in results for debugging
TRACK_START_CONDITIONS = True

# Strategy definitions
STRATEGIES = {
    'S1': {'name': 'TQQQ Buy Hold', 'type': 'benchmark', 'asset': 'TQQQ'},
    'S2': {'name': 'SPY Buy Hold', 'type': 'benchmark', 'asset': 'SPY'},
    'S3': {'name': 'SSO BuyHold (2x)', 'type': 'benchmark', 'asset': 'SSO'},
    'S4': {'name': '200-SMA Simple', 'type': 'sma', 'asset': 'TQQQ', 'sma_period': 200},
    'S5': {'name': 'SMA +/-2% Band', 'type': 'sma_band', 'asset': 'TQQQ', 'sma_period': 200, 'band': 0.02},
    'S6': {'name': '60/40 TQQQ/TMF', 'type': 'portfolio', 'assets': {'TQQQ': 0.6, 'TMF': 0.4}, 'rebalance_freq': 21},
    'S7': {'name': 'Vol Targeting (20%)', 'type': 'vol_targeting', 'asset': 'TQQQ', 'target_vol': 0.20, 'lookback': 20},
    'S8': {'name': 'Composite Regime', 'type': 'composite', 'asset': 'TQQQ', 'defensive_asset': 'SPY','sma_period': 200, 'rsi_period': 14, 'vix_threshold': 25.0},
    'S9': {'name': 'Adaptive Vol Target', 'type': 'adaptive_vol', 'asset': 'TQQQ', 'bull_target': 0.35, 'bear_target': 0.12, 'lookback': 20, 'sma_period': 200},
    'S10': {
        'name': 'Sortino Optimize',
        'type': 'downside_vol',
        'asset': 'TQQQ',
        'target_downside_vol': 0.15, # Target 15% downside deviation
        'lookback': 20
    },
    'S11': {
        'name': 'Hyper-Convex',
        'type': 'convex_vol',
        'asset': 'TQQQ',
        'target_vol': 0.25,
        'power': 1.2,
        'sma_period': 200
    },
    'S12': {
        'name': 'Vol-Velocity',
        'type': 'vol_velocity',
        'asset': 'TQQQ',
        'target_vol': 0.22
    },
    'S13': {
        'name': 'VoV Momentum',
        'type': 'vol_mom',
        'asset': 'TQQQ',
        'target_vol': 0.25
    },
    'S14': {
        'name': 'Skewness-Adjusted',
        'type': 'skew_convex',
        'asset': 'TQQQ',
        'target_vol': 0.25
    },
    'S15': {
        'name': 'Meta-Ensemble',
        'type': 'meta_ensemble',
        'asset': 'TQQQ',
        'target_vol': 0.28  # Slightly higher target due to better defense
    },
    'S16': {
        'name': 'Crisis Alpha',
        'type': 'regime_asymmetric',
        'asset': 'TQQQ',
        'base_target_vol': 0.30,        # Aggressive base
        'crisis_target_vol': 0.08,      # Defensive in crisis
        'vix_alarm_level': 25,          # Warning threshold
        'vol_expansion_threshold': 1.5, # If realized vol > 1.5x historical, crisis mode
        'lookback_fast': 5,
        'lookback_slow': 60
    },
    'S17': {
        'name': 'Tail Risk Optimizer',
        'type': 'skew_kelly',
        'asset': 'TQQQ',
        'base_target_vol': 0.30,
        'skew_lookback': 60,
        'vol_lookback': 20,
        'kelly_fraction': 0.7
    },
    'S18': {
        'name': 'Mom. Vol Conv.',
        'type': 'mom_vol_convergence',
        'asset': 'TQQQ',
        'base_target_vol': 0.28,
        'momentum_lookback': 126,
        'vol_fast': 10,
        'vol_slow': 60,
        'momentum_threshold': 0.05
    },
    'S19': {
        'name': 'Conviction Compounder',
        'type': 'conviction_compounder',
        'asset': 'TQQQ',
        'base_target_vol': 0.32,
        'momentum_lookback': 126,
        'vol_lookback': 20,
        'trend_sma': 100,
        'rebalance_threshold': 0.05  # Only rebalance if >5% change
    },
}


def print_banner():
    """Print the startup banner with version info and applied fixes."""
    print(f"\n{'='*80}")
    print(f"CORRECTED LEVERAGED ETF ANALYSIS v8.1 (BUG FIX: Regime Mismatch)")
    print(f"{'='*80}")
    print(f"FUNDAMENTAL FIXES APPLIED:")
    print(f"  1. Volatility drag: Correct -0.5*L*(L-1)*sigma^2 formula")
    print(f"  2. Tracking error: Multiplicative with AR(1) and fat tails")
    print(f"  3. Regime model: Fit to VOLATILITY (not returns)")
    print(f"  4. Portfolio rebalancing: Track leverage drift")
    print(f"  5. Removed jumps: Continuous diffusion sufficient")
    print(f"  6. Correlation dynamics: Time-varying by regime")
    print(f"  7. Realistic tracking in crisis: Non-linear liquidity impact")
    print(f"  8. Pre-inception data: Labeled as SYNTHETIC")
    print(f"  9. BUG FIX: Regime path mismatch handled in validation")
    print(f"  10. Mimmics ROTH IRA... no tax, but fees on trades, etc")
    print(f"{'='*80}")
    print(f"System: {N_WORKERS} workers, {NUM_SIMULATIONS} sims/horizon")
    print(f"{'='*80}\n")


def init_plotting():
    """Initialize matplotlib/seaborn plot styling."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['figure.dpi'] = 100
