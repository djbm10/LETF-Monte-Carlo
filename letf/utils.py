import numpy as np
import pandas as pd
import pickle
from typing import Dict, Optional
from letf import config as cfg


def save_cache(data, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Cache save failed: {e}")

def load_cache(filepath):
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None

def get_max_underwater_days(equity_curve):
    """Calculates the longest period (in trading days) the strategy was in a drawdown."""
    hwm = equity_curve.cummax()
    underwater = equity_curve < hwm

    # Calculate run lengths of True values (underwater days)
    # This magic converts [F, T, T, T, F, T] into counts of consecutive Trues
    check_series = underwater.astype(int)
    # Group consecutive 1s and 0s
    groups = check_series.ne(check_series.shift()).cumsum()
    # Sum the 1s in each group
    run_lengths = check_series.groupby(groups).sum()

    if run_lengths.empty:
        return 0
    return run_lengths.max()

def nearest_psd_matrix(corr_matrix):
    """Project correlation matrix to nearest positive semi-definite matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues[eigenvalues < 1e-8] = 1e-8

    corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Normalize to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)

    return corr_psd

def compute_high_vol_probability(vix_series, realized_vol=None, term_spread=None, smoothing=0.94):
    """
    Probabilistic high-volatility regime score in [0, 1].

    Uses a smooth logistic model on VIX, realized volatility, and term structure,
    then applies EWMA smoothing to reduce brittle day-to-day flips.
    """
    vix = np.asarray(vix_series, dtype=float)
    n = len(vix)

    if realized_vol is None:
        rv = pd.Series(vix).rolling(20, min_periods=5).std().bfill().fillna(0).values / 100.0
    else:
        rv = np.asarray(realized_vol, dtype=float)
        rv = pd.Series(rv).ffill().bfill().fillna(np.nanmedian(rv)).values

    if term_spread is None:
        ts = np.zeros(n)
    else:
        ts = np.asarray(term_spread, dtype=float)
        ts = pd.Series(ts).ffill().bfill().fillna(0.0).values

    # Logistic score: higher VIX, higher realized vol, and flatter/inverted curve
    # imply higher stress probability.
    logit = (
        -4.0
        + 0.22 * (np.nan_to_num(vix, nan=20.0) - 20.0)
        + 6.5 * (np.nan_to_num(rv, nan=0.18) - 0.18)
        + 0.10 * np.clip(-ts, -5, 5)
    )
    raw_p = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    smoothed_p = np.zeros(n)
    if n > 0:
        smoothed_p[0] = raw_p[0]
    for i in range(1, n):
        smoothed_p[i] = smoothing * smoothed_p[i - 1] + (1 - smoothing) * raw_p[i]

    return np.clip(smoothed_p, 0.001, 0.999)


def infer_regime_from_vix(vix_series, realized_vol=None, term_spread=None, hysteresis=0.08):
    """
    Infer regime using probabilistic stress score with hysteresis.

    This avoids brittle single-threshold switching at VIX=25 by combining VIX,
    realized vol, and optional term-structure information.

    Used when validating against historical data or when regime_path is missing.
    """
    p_high = compute_high_vol_probability(
        vix_series=vix_series,
        realized_vol=realized_vol,
        term_spread=term_spread
    )

    enter_high = 0.50 + hysteresis / 2
    exit_high = 0.50 - hysteresis / 2

    regimes = np.zeros(len(p_high), dtype=int)
    if len(p_high) == 0:
        return regimes

    current = 1 if p_high[0] >= 0.50 else 0
    regimes[0] = current
    for i in range(1, len(p_high)):
        if current == 0 and p_high[i] >= enter_high:
            current = 1
        elif current == 1 and p_high[i] <= exit_high:
            current = 0
        regimes[i] = current

    return regimes


def fill_missing_with_dynamic_factor(df: pd.DataFrame, target_col: str, factor_col: str,
                                     default_beta: float, seed: int = 1234) -> pd.Series:
    """Fill missing returns using overlap-calibrated dynamic beta + residual sampling."""
    if target_col not in df.columns:
        df[target_col] = np.nan

    target = df[target_col].copy()
    factor = df[factor_col].copy()

    valid = target.notna() & factor.notna()
    if valid.sum() < 40:
        return target.fillna(default_beta * factor)

    cov = target.rolling(252, min_periods=40).cov(factor)
    var = factor.rolling(252, min_periods=40).var()
    beta = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    beta = beta.clip(-3.0, 3.0).ffill().bfill().fillna(default_beta)

    alpha = (target - beta * factor).rolling(252, min_periods=40).mean()
    alpha = alpha.ffill().bfill().fillna(0.0)

    fitted = alpha + beta * factor
    residuals = (target - fitted)[valid].dropna().values
    missing = target.isna() & factor.notna()

    if len(residuals) > 20 and missing.any():
        rng = np.random.default_rng(seed)
        sampled_resid = rng.choice(residuals, size=missing.sum(), replace=True)
        target.loc[missing] = fitted.loc[missing].values + sampled_resid
    else:
        target.loc[missing] = fitted.loc[missing]

    return target


# ============================================================================
# DYNAMIC BORROWING COST CALCULATION
# ============================================================================

def calculate_daily_borrow_cost(leverage: float, risk_free_rate: float,
                                 spread: float) -> float:
    """
    Calculate the daily borrowing cost for a leveraged ETF.

    HOW LEVERAGED ETFs WORK:
    - A 3x fund needs to borrow 2x its capital to get 3x exposure
    - A 2x fund needs to borrow 1x its capital to get 2x exposure
    - A 1x fund (like SPY) borrows nothing

    The borrowing cost is:
        annual_cost = (leverage - 1) x (risk_free_rate + spread)
        daily_cost = annual_cost / 252

    Args:
        leverage: The fund's leverage ratio (e.g., 3.0 for TQQQ)
        risk_free_rate: Annual risk-free rate as decimal (e.g., 0.05 for 5%)
        spread: Annual spread above risk-free rate (e.g., 0.0075 for 0.75%)

    Returns:
        Daily borrowing cost as a decimal (e.g., 0.0002 for 2 bps/day)

    Example:
        # TQQQ with 5% rates and 0.75% spread:
        # Borrows 2x capital at 5.75% = 11.5% annual drag
        # Daily: 11.5% / 252 = 0.046% per day

        daily_cost = calculate_daily_borrow_cost(3.0, 0.05, 0.0075)
        # Returns: 0.000456 (about 4.6 bps per day)
    """

    # How much leverage is borrowed (not your own capital)
    borrowed_leverage = leverage - 1.0

    # If no borrowing (1x fund), no cost
    if borrowed_leverage <= 0:
        return 0.0

    # Total annual borrowing rate (floored at 0: can't earn money by borrowing)
    annual_borrow_rate = max(risk_free_rate + spread, 0.0)

    # Annual cost = borrowed amount x rate
    annual_cost = borrowed_leverage * annual_borrow_rate

    # Convert to daily (252 trading days)
    daily_cost = annual_cost / 252

    return daily_cost


def get_borrow_cost_series(df: pd.DataFrame, leverage: float,
                           spread: float) -> pd.Series:
    """
    Create a time series of daily borrowing costs based on historical rates.

    This uses the IRX (3-month T-bill rate) as a proxy for SOFR/short-term rates.

    Args:
        df: DataFrame with 'IRX' column (interest rate in percentage, e.g., 5.0 for 5%)
        leverage: Fund's leverage ratio
        spread: Annual spread above risk-free rate

    Returns:
        Series of daily borrowing costs (as decimals)
    """

    # IRX is in percentage points (e.g., 5.0 means 5%)
    # Convert to decimal (0.05)
    risk_free_rate = df['IRX'] / 100.0

    # Calculate daily borrow cost for each day
    borrowed_leverage = leverage - 1.0

    if borrowed_leverage <= 0:
        return pd.Series(0.0, index=df.index)

    # Annual cost = borrowed_leverage x (risk_free + spread)
    annual_cost = borrowed_leverage * (risk_free_rate + spread)

    # Daily cost
    daily_cost = annual_cost / 252

    return daily_cost
