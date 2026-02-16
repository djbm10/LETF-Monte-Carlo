import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from scipy.stats import t as student_t
from typing import Dict, Optional
from letf import config as cfg
from letf.utils import save_cache, load_cache, fill_missing_with_dynamic_factor, get_borrow_cost_series


def fetch_fama_french_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily market returns from the Fama-French Data Library.

    The Fama-French data goes back to July 1926, giving us nearly 100 years
    of market history including:
    - The Great Depression (1929-1932)
    - World War II (1941-1945)
    - Post-war boom (1945-1950)

    Data source: Kenneth French's website (via pandas_datareader)
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Args:
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'

    Returns:
        DataFrame with columns:
        - SPY_Ret: Daily market return (decimal, e.g., 0.01 for 1%)
        - RF: Daily risk-free rate (decimal)
        - SPY_Price: Synthetic price series (for compatibility)
    """

    print("  [INFO]Fetching Fama-French daily data (1926-present)...")

    try:
        # Fetch the Fama-French 3 Factors (Daily)
        # This includes Mkt-RF, SMB, HML, and RF
        ff_data = web.DataReader(
            'F-F_Research_Data_Factors_daily',
            'famafrench',
            start=start_date,
            end=end_date
        )

        # The data is returned as a dict with one DataFrame
        # Key '0' contains the daily data
        ff_df = ff_data[0].copy()

        print(f"  [OK]Fama-French data retrieved: {len(ff_df):,} days")
        print(f"    Date range: {ff_df.index[0]} to {ff_df.index[-1]}")

    except Exception as e:
        print(f"  [ERR]Error fetching Fama-French data: {e}")
        print("  Attempting alternative download method...")

        try:
            # Alternative: Direct CSV download from Kenneth French's website
            import urllib.request
            import zipfile
            import io

            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

            # Download the zip file
            with urllib.request.urlopen(url) as response:
                zip_data = response.read()

            # Extract and read the CSV
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV')][0]
                with z.open(csv_name) as f:
                    # Skip header rows and read data
                    ff_df = pd.read_csv(f, skiprows=3)

            # Clean up the data
            ff_df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
            ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
            ff_df.set_index('Date', inplace=True)
            ff_df = ff_df.apply(pd.to_numeric, errors='coerce')

            # Filter to date range
            ff_df = ff_df.loc[start_date:end_date]

            print(f"  [OK]Fama-French data (alternative): {len(ff_df):,} days")

        except Exception as e2:
            print(f"  [ERR]Alternative download also failed: {e2}")
            return None

    # Create output DataFrame
    result = pd.DataFrame(index=ff_df.index)

    # Convert from percentage points to decimals
    # Fama-French data is in percentage points (e.g., 1.5 = 1.5%)
    # We need decimals (e.g., 0.015)

    # Market return = Mkt-RF + RF
    result['SPY_Ret'] = (ff_df['Mkt-RF'] + ff_df['RF']) / 100.0

    # Risk-free rate (for borrowing costs)
    result['RF'] = ff_df['RF'] / 100.0

    # Annualized risk-free rate (for IRX proxy)
    # IRX is typically quoted as annual percentage
    result['IRX'] = ff_df['RF'] * 252 / 100.0 * 100  # Convert daily to annual, as percentage

    # Create synthetic price series (for compatibility with existing code)
    # Start at 100 and compound returns
    result['SPY_Price'] = (1 + result['SPY_Ret']).cumprod() * 100

    # For pre-1950, we don't have NASDAQ, so approximate it
    # Historical beta of NASDAQ to S&P is roughly 1.2-1.3
    # This is a rough approximation for synthetic LETF calculation
    result['NASDAQ_Ret'] = result['SPY_Ret'] * 1.25
    result['QQQ_Ret'] = result['NASDAQ_Ret']  # QQQ tracks NASDAQ

    # Synthetic VIX (volatility index didn't exist before 1990)
    # Approximate using 20-day rolling volatility
    rolling_vol = result['SPY_Ret'].rolling(20).std() * np.sqrt(252) * 100
    result['VIX'] = rolling_vol.fillna(20.0)  # Default to 20 if not enough data

    # Treasury returns approximation for pre-1950
    # Use inverse correlation to equity as rough proxy
    # Historical correlation of long bonds to stocks is roughly -0.2 to -0.3
    result['TLT_Ret'] = result['SPY_Ret'] * -0.25 + result['RF']

    # Mark all Fama-French data as from this source
    result['Data_Source'] = 'Fama-French'

    print(f"  [OK]Fama-French data processed")
    print(f"    Sample return (first day): {result['SPY_Ret'].iloc[0]*100:.2f}%")
    print(f"    Sample RF (first day): {result['RF'].iloc[0]*100:.4f}%")

    return result


def combine_data_sources(ff_data: pd.DataFrame, yf_data: pd.DataFrame,
                         cutoff_date: str = "1950-01-01") -> pd.DataFrame:
    """
    Combine Fama-French (pre-1950) and yfinance (1950+) data.

    This function:
    1. Uses Fama-French data for dates BEFORE cutoff_date
    2. Uses yfinance data for dates ON or AFTER cutoff_date
    3. Ensures smooth transition at the cutoff

    Args:
        ff_data: DataFrame from fetch_fama_french_data()
        yf_data: DataFrame from yfinance processing
        cutoff_date: Date to switch from FF to yfinance

    Returns:
        Combined DataFrame with full history
    """

    print(f"\n  [INFO]Combining data sources at {cutoff_date}...")

    cutoff = pd.to_datetime(cutoff_date)

    # Filter each dataset
    ff_before = ff_data[ff_data.index < cutoff].copy()
    yf_after = yf_data[yf_data.index >= cutoff].copy()

    print(f"    Fama-French (pre-{cutoff_date}): {len(ff_before):,} days")
    print(f"    yfinance ({cutoff_date}+): {len(yf_after):,} days")

    # Mark data sources
    if 'Data_Source' not in ff_before.columns:
        ff_before['Data_Source'] = 'Fama-French'
    if 'Data_Source' not in yf_after.columns:
        yf_after['Data_Source'] = 'yfinance'

    # Align columns - use yfinance columns as the standard
    # Add any missing columns to ff_before with NaN
    for col in yf_after.columns:
        if col not in ff_before.columns:
            ff_before[col] = np.nan

    # For columns in ff_before but not yf_after, they'll be kept
    # This is fine - we'll have full data where available

    # Concatenate
    combined = pd.concat([ff_before, yf_after], axis=0)
    combined = combined.sort_index()

    # Remove any duplicate dates (prefer yfinance data)
    combined = combined[~combined.index.duplicated(keep='last')]

    # Recalculate price series to be continuous
    # The SPY_Price needs to be continuous across the boundary
    if 'SPY_Ret' in combined.columns:
        combined['SPY_Price'] = (1 + combined['SPY_Ret'].fillna(0)).cumprod() * 100

    print(f"    Combined total: {len(combined):,} days")
    print(f"    Date range: {combined.index[0].date()} to {combined.index[-1].date()}")

    # Verify the transition
    transition_idx = combined.index.get_indexer([cutoff], method='nearest')[0]
    if transition_idx > 0 and transition_idx < len(combined) - 1:
        before_ret = combined.iloc[transition_idx - 1]['SPY_Ret']
        after_ret = combined.iloc[transition_idx]['SPY_Ret']
        print(f"    Transition check: day before = {before_ret*100:.2f}%, day of = {after_ret*100:.2f}%")

    return combined

def fetch_historical_data():
    """
    Fetch historical data with CORRECT volatility drag implementation.

    ENHANCED: Now combines two data sources for maximum history:
    - Fama-French (1926-1949): Academic-quality market returns
    - yfinance (1950-present): Full market data including NASDAQ, VIX, etc.

    This gives us nearly 100 years of data including the Great Depression!

    FIX: Pre-2010 TQQQ data is SYNTHETIC and clearly labeled.
    """
    cached = load_cache(cfg.DATA_CACHE)
    if cached is not None:
        print("[OK]Using cached data")
        return cached

    print(f"\n{'='*80}")
    print("FETCHING HISTORICAL DATA (EXTENDED HISTORY)")
    print(f"{'='*80}\n")

    # ========================================================================
    # STEP 1: Fetch Fama-French data for pre-1950 period
    # ========================================================================
    print("  PHASE 1: Fama-French Data (1926-1949)")
    print("  " + "-"*50)

    ff_data = fetch_fama_french_data(cfg.DATA_START_DATE, cfg.FAMA_FRENCH_END_DATE)

    if ff_data is None:
        print("  [WARN] Fama-French data unavailable, using yfinance only")
        use_fama_french = False
    else:
        use_fama_french = True
        print(f"  [OK]Fama-French: {len(ff_data):,} days (1926-1949)")

    # ========================================================================
    # STEP 2: Fetch yfinance data for 1950+ period
    # ========================================================================
    print(f"\n  PHASE 2: yfinance Data ({cfg.YFINANCE_START_DATE}+)")
    print("  " + "-"*50)

    print("  Downloading market data from Yahoo Finance...")
    tickers = ['^GSPC', '^IXIC', '^VIX', '^IRX', '^TNX', 'TLT', 'QQQ', 'TQQQ', 'UPRO', 'SSO']

    try:
        data = yf.download(tickers, start=cfg.YFINANCE_START_DATE, end=cfg.DATA_END_DATE,
                          progress=False, auto_adjust=True)
        print("  [OK]yfinance data downloaded")
    except Exception as e:
        print(f"  [ERR]Error: {e}")
        if not use_fama_french:
            return None
        # If we have FF data, we can still proceed with limited data
        data = None

    # Process yfinance data
    yf_df = pd.DataFrame()

    if data is not None:
        # S&P 500
        if '^GSPC' in data['Close'].columns:
            yf_df['SPY_Price'] = data['Close']['^GSPC']
            yf_df['SPY_Ret'] = yf_df['SPY_Price'].pct_change()
        else:
            print("  [WARN] No S&P 500 data from yfinance")
            if not use_fama_french:
                return None

        # NASDAQ
        if '^IXIC' in data['Close'].columns:
            yf_df['NASDAQ_Price'] = data['Close']['^IXIC']
            yf_df['NASDAQ_Ret'] = yf_df['NASDAQ_Price'].pct_change()
        else:
            yf_df['NASDAQ_Ret'] = yf_df['SPY_Ret'] * 1.3 if 'SPY_Ret' in yf_df.columns else np.nan

        # QQQ (for TQQQ validation)
        if 'QQQ' in data['Close'].columns:
            yf_df['QQQ_Price'] = data['Close']['QQQ']
            yf_df['QQQ_Ret'] = yf_df['QQQ_Price'].pct_change()
        else:
            yf_df['QQQ_Ret'] = yf_df['NASDAQ_Ret'] if 'NASDAQ_Ret' in yf_df.columns else np.nan

                # Real TQQQ / UPRO / SSO prices (for true post-inception history)
        if 'TQQQ' in data['Close'].columns:
            yf_df['TQQQ_Real_Price'] = data['Close']['TQQQ']
            yf_df['TQQQ_Real_Ret'] = yf_df['TQQQ_Real_Price'].pct_change()

        if 'UPRO' in data['Close'].columns:
            yf_df['UPRO_Real_Price'] = data['Close']['UPRO']
            yf_df['UPRO_Real_Ret'] = yf_df['UPRO_Real_Price'].pct_change()

        if 'SSO' in data['Close'].columns:
            yf_df['SSO_Real_Price'] = data['Close']['SSO']
            yf_df['SSO_Real_Ret'] = yf_df['SSO_Real_Price'].pct_change()


        # VIX
        if '^VIX' in data['Close'].columns:
            yf_df['VIX'] = data['Close']['^VIX']
        else:
            yf_df['VIX'] = np.nan

        # Fill VIX gaps with rolling volatility
        if 'SPY_Ret' in yf_df.columns:
            spy_vol_20d = yf_df['SPY_Ret'].rolling(20).std() * np.sqrt(252) * 100
            yf_df['VIX'] = yf_df['VIX'].fillna(spy_vol_20d).fillna(20.0)

        # Interest rates
        if '^IRX' in data['Close'].columns:
            yf_df['IRX'] = data['Close']['^IRX']
        yf_df['IRX'] = yf_df['IRX'].fillna(4.5) if 'IRX' in yf_df.columns else 4.5
        yf_df['Cash_Ret'] = yf_df['IRX'] / 100 / 252

        # Treasury data for TMF
        if 'TLT' in data['Close'].columns:
            yf_df['TLT_Price'] = data['Close']['TLT']
            yf_df['TLT_Ret'] = yf_df['TLT_Price'].pct_change()
        else:
            if '^TNX' in data['Close'].columns:
                yf_df['TNX'] = data['Close']['^TNX']
                yf_df['TLT_Ret'] = -yf_df['TNX'].diff() * 0.15
            else:
                yf_df['TLT_Ret'] = yf_df['SPY_Ret'] * -0.3 if 'SPY_Ret' in yf_df.columns else np.nan

        # Mark data source
        yf_df['Data_Source'] = 'yfinance'

        print(f"  [OK]yfinance: {len(yf_df):,} days ({cfg.YFINANCE_START_DATE}+)")

    # ========================================================================
    # STEP 3: Combine data sources
    # ========================================================================
    if use_fama_french and len(yf_df) > 0:
        print(f"\n  PHASE 3: Combining Data Sources")
        print("  " + "-"*50)
        df = combine_data_sources(ff_data, yf_df, cfg.YFINANCE_START_DATE)
    elif use_fama_french:
        print("  Using Fama-French data only")
        df = ff_data
    else:
        print("  Using yfinance data only")
        df = yf_df

    # Ensure we have SPY_Ret
    if 'SPY_Ret' not in df.columns or df['SPY_Ret'].isna().all():
        print("[ERR]No valid market return data")
        return None

    # ========================================================================
    # STEP 4: Fill in missing columns for older periods
    # ========================================================================
    print(f"\n  PHASE 4: Filling Missing Data for Older Periods")
    print("  " + "-"*50)

    # NASDAQ (didn't exist before 1971): dynamic factor model vs SPY
    if 'NASDAQ_Ret' not in df.columns:
        df['NASDAQ_Ret'] = np.nan
    if df['NASDAQ_Ret'].isna().any():
        df['NASDAQ_Ret'] = fill_missing_with_dynamic_factor(
            df, target_col='NASDAQ_Ret', factor_col='SPY_Ret', default_beta=1.25, seed=1101
        )
        print("  [OK]NASDAQ filled via overlap-calibrated dynamic factor model")

    # QQQ (didn't exist before 1999): dynamic factor model vs NASDAQ
    if 'QQQ_Ret' not in df.columns:
        df['QQQ_Ret'] = np.nan
    if df['QQQ_Ret'].isna().any():
        df['QQQ_Ret'] = fill_missing_with_dynamic_factor(
            df, target_col='QQQ_Ret', factor_col='NASDAQ_Ret', default_beta=1.0, seed=1102
        )
        print("  [OK]QQQ filled via overlap-calibrated dynamic factor model")

    # VIX (didn't exist before 1990)
    if 'VIX' not in df.columns:
        df['VIX'] = np.nan
    if df['VIX'].isna().any():
        spy_vol = df['SPY_Ret'].rolling(20).std() * np.sqrt(252) * 100
        df['VIX'] = df['VIX'].fillna(spy_vol).fillna(20.0)
        print("  [OK]VIX approximated using rolling volatility for pre-1990")

    # Interest rates
    if 'IRX' not in df.columns:
        df['IRX'] = np.nan
    if df['IRX'].isna().any():
        # Prefer Fama-French RF, then infer from TNX slope, then smooth backfill.
        if 'RF' in df.columns:
            df['IRX'] = df['IRX'].fillna(df['RF'] * 252 * 100)
        if 'TNX' in df.columns:
            inferred_irx = (0.55 * df['TNX']).clip(lower=0.0)
            df['IRX'] = df['IRX'].fillna(inferred_irx)
        df['IRX'] = df['IRX'].interpolate(limit_direction='both').ffill().fillna(3.0)
        print("  [OK]Interest rates filled from RF/term-structure interpolation")

    if 'Cash_Ret' not in df.columns:
        df['Cash_Ret'] = df['IRX'] / 100 / 252

    # Treasury returns
    if 'TLT_Ret' not in df.columns:
        df['TLT_Ret'] = np.nan
    if df['TLT_Ret'].isna().any():
        tlt_filled = fill_missing_with_dynamic_factor(
            df, target_col='TLT_Ret', factor_col='SPY_Ret', default_beta=-0.20, seed=1103
        )
        rf_daily = df['IRX'] / 100 / 252
        df['TLT_Ret'] = tlt_filled.fillna(rf_daily)
        print("  [OK]Treasury returns filled via dynamic factor model + carry")

    # ========================================================================
    # STEP 5: Verify data quality
    # ========================================================================
    print(f"\n  PHASE 5: Data Quality Check")
    print("  " + "-"*50)
    print(f"    Total trading days: {len(df):,}")
    print(f"    Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"    Years of data: {(df.index[-1] - df.index[0]).days / 365.25:.2f}")

    # Count data by source
    if 'Data_Source' in df.columns:
        source_counts = df['Data_Source'].value_counts()
        for source, count in source_counts.items():
            print(f"    {source}: {count:,} days")

    # Check for any remaining NaN in critical columns
    critical_cols = ['SPY_Ret', 'VIX', 'IRX']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    [WARN] {col} has {nan_count} missing values")

    print("  Reconstructing leveraged returns with CORRECT volatility drag...")

    # FIX #1: CORRECT VOLATILITY DRAG
    # Key insight: For daily-rebalanced LETFs, volatility drag emerges from
    # GEOMETRIC COMPOUNDING, not from subtracting a drag term each day.
    #
    # Daily return = L * underlying_return - expenses
    # The -0.5*L*(L-1)*sigma^2 drag appears in the EXPECTED (arithmetic mean) return
    # over time due to Jensen's inequality, not as a daily cost.

    for asset_id, config in cfg.ASSETS.items():
        leverage = config['leverage']
        expense_ratio = config['expense_ratio']
        beta = config['beta_to_spy']

        # Get underlying returns
        if asset_id == 'TQQQ':
            underlying_ret = df['QQQ_Ret']
        elif asset_id in ['UPRO', 'SSO', 'SPY']:
            underlying_ret = df['SPY_Ret']
        elif asset_id == 'TMF':
            underlying_ret = df['TLT_Ret']
        else:
            underlying_ret = df['SPY_Ret']

        # Apply beta if needed
        # NOTE: Only apply beta when using SPY as proxy for an asset
        # TQQQ uses QQQ_Ret directly, which is already more volatile than SPY
        # SSO/UPRO use SPY_Ret with beta=1.0, so no multiplication needed
        if beta != 1.0 and asset_id not in ['TMF', 'TQQQ']:
            underlying_ret = underlying_ret * beta

        # Daily expense (fixed)
        daily_expense = expense_ratio / 252  # Trading days, not calendar

        # Dynamic borrowing cost based on current interest rates
        # This is more realistic than a fixed cost because rates change over time
        borrow_spread = config.get('borrow_spread', 0.0)

        # Get time-varying borrow cost (uses IRX as proxy for short-term rates)
        daily_borrow_cost_series = get_borrow_cost_series(df, leverage, borrow_spread)

        # Gross leveraged return (drag emerges from compounding)
        gross_return = leverage * underlying_ret

        # Net return BEFORE tracking error
        # Now subtracts DYNAMIC borrowing cost that varies with interest rates
        net_return_before_te = gross_return - daily_expense - daily_borrow_cost_series

        # FIX #2: TRACKING ERROR - Multiplicative with AR(1) and fat tails
        # Generate tracking error series
        tracking_error_base = config['tracking_error_base']
        df_param = config['tracking_error_df']

        # Fixed seed: synthetic pre-inception returns are for display/charting only.
        # No calibration, bootstrap, or MC simulation reads these columns.
        te_rng = np.random.default_rng(42 + ord(asset_id[0]))

        # VIX-scaled tracking error (higher vol = worse tracking)
        vix_multiplier = (df['VIX'] / 20.0) ** 1.5  # Non-linear in crisis

        # AR(1) process with t-distributed innovations
        te_series = np.zeros(len(df))
        rho = 0.3  # Autocorrelation

        for i in range(1, len(df)):
            # Fat-tailed innovation
            innovation = student_t.rvs(df=df_param, random_state=te_rng) * tracking_error_base * vix_multiplier.iloc[i]

            # Also scales with return magnitude (liquidity impact)
            if not pd.isna(underlying_ret.iloc[i]):
                move_multiplier = 1 + 10 * abs(underlying_ret.iloc[i])
                innovation *= move_multiplier

            # AR(1)
            te_series[i] = rho * te_series[i-1] + innovation

        # Tracking error is MULTIPLICATIVE (funds don't perfectly replicate)
        synthetic_ret = (1 + net_return_before_te) * (1 + te_series) - 1

        # Start with synthetic series (works for full history if no real data exists)
        df[f'{asset_id}_Ret'] = synthetic_ret
        df[f'{asset_id}_Price'] = (1 + synthetic_ret.fillna(0)).cumprod() * 100

        # If real ETF data exists, overwrite post-inception with real data
        inception_date = pd.to_datetime(config['inception'])
        real_price_col = f'{asset_id}_Real_Price'
        real_ret_col = f'{asset_id}_Real_Ret'

        if real_price_col in df.columns and real_ret_col in df.columns:
            real_mask = (df.index >= inception_date) & df[real_price_col].notna()

            if real_mask.any():
                # Replace returns with real post-inception returns
                df.loc[real_mask, f'{asset_id}_Ret'] = df.loc[real_mask, real_ret_col]

                # Scale synthetic pre-inception prices so they connect smoothly
                pre_mask = ~real_mask
                if pre_mask.any():
                    pre_prices = (1 + df.loc[pre_mask, f'{asset_id}_Ret'].fillna(0)).cumprod()
                    first_real_price = df.loc[real_mask, real_price_col].iloc[0]
                    scale = first_real_price / pre_prices.iloc[-1]
                    df.loc[pre_mask, f'{asset_id}_Price'] = pre_prices * scale

                # Overwrite post-inception prices with real prices
                df.loc[real_mask, f'{asset_id}_Price'] = df.loc[real_mask, real_price_col]

            # True synthetic flag (only before real data exists)
            df[f'{asset_id}_IsSynthetic'] = ~real_mask
        else:
            # Fallback: all data is synthetic before inception
            df[f'{asset_id}_IsSynthetic'] = df.index < inception_date

    # Technical indicators
    print("  Computing technical indicators...")
    ref_price = df['SPY_Price']

    df['SMA200'] = ref_price.rolling(200, min_periods=1).mean()

    # ========================================================================
    # ENHANCED VOLATILITY MODEL (EWMA + Regime-Conditional)
    # ========================================================================
    # Instead of simple rolling std, use exponentially weighted moving average
    # This gives more weight to recent data and captures volatility clustering

    # EWMA volatility (more responsive to recent changes)
    df['Market_Vol_EWMA'] = df['SPY_Ret'].ewm(span=20, adjust=False).std() * np.sqrt(252)

    # Keep rolling vol for backwards compatibility
    df['Market_Vol_20d'] = df['SPY_Ret'].rolling(20).std() * np.sqrt(252)

    # Use EWMA as primary vol measure (more accurate for LETFs)
    df['Market_Vol'] = df['Market_Vol_EWMA']

    # Clean - use the user-selected analysis date range
    df = df.loc[cfg.ANALYSIS_START_DATE:cfg.ANALYSIS_END_DATE].copy()
    df.dropna(subset=['SPY_Ret', 'VIX'], inplace=True)

    print(f"\n{'='*80}")
    print("DATA SUMMARY")
    print(f"{'='*80}")
    print(f"[OK]Data ready: {len(df):,} trading days ({len(df)/252:.2f} years)")
    print(f"  Full period: {df.index[0].date()} to {df.index[-1].date()}")

    # Show data source breakdown
    if 'Data_Source' in df.columns:
        print(f"\n  Data Sources:")
        ff_days = (df['Data_Source'] == 'Fama-French').sum()
        yf_days = (df['Data_Source'] == 'yfinance').sum()
        if ff_days > 0:
            ff_start = df[df['Data_Source'] == 'Fama-French'].index[0].date()
            ff_end = df[df['Data_Source'] == 'Fama-French'].index[-1].date()
            print(f"    Fama-French: {ff_days:,} days ({ff_start} to {ff_end})")
        if yf_days > 0:
            yf_start = df[df['Data_Source'] == 'yfinance'].index[0].date()
            yf_end = df[df['Data_Source'] == 'yfinance'].index[-1].date()
            print(f"    yfinance:    {yf_days:,} days ({yf_start} to {yf_end})")

    # Historical highlights
    print(f"\n  Historical Events Covered:")
    if df.index[0].year <= 1929:
        print(f"    [OK]Great Depression (1929-1932)")
    if df.index[0].year <= 1941:
        print(f"    [OK]World War II (1941-1945)")
    if df.index[0].year <= 1973:
        print(f"    [OK]Oil Crisis (1973-1974)")
    if df.index[0].year <= 1987:
        print(f"    [OK]Black Monday (1987)")
    if df.index[0].year <= 2000:
        print(f"    [OK]Dot-com Crash (2000-2002)")
    if df.index[0].year <= 2008:
        print(f"    [OK]Financial Crisis (2008-2009)")
    if df.index[0].year <= 2020:
        print(f"    [OK]COVID Crash (2020)")

    # Count synthetic vs real data
    for asset_id in ['TQQQ', 'UPRO', 'SSO']:
        if f'{asset_id}_IsSynthetic' in df.columns:
            n_synthetic = df[f'{asset_id}_IsSynthetic'].sum()
            n_real = (~df[f'{asset_id}_IsSynthetic']).sum()
            print(f"  {asset_id}: {n_real:,} real days, {n_synthetic:,} SYNTHETIC days")

    # Verify SPY geometric mean
    spy_annual_returns = df['SPY_Ret'].groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
    spy_geo_mean = np.exp(np.mean(np.log(1 + spy_annual_returns))) - 1
    print(f"  Historical SPY geometric mean: {spy_geo_mean*100:.2f}%/year")

    print(f"\n[WARN]  WARNING: Pre-inception LETF data is SYNTHETIC simulation.")
    print(f"  Do NOT treat pre-2010 TQQQ results as historical validation!")

    save_cache(df, cfg.DATA_CACHE)
    return df
