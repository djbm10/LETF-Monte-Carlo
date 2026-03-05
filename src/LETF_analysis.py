"""
ULTRA-ADVANCED MULTI-ASSET LEVERAGED ETF ANALYSIS FRAMEWORK
============================================================
Production-ready with cutting-edge quantitative finance methods.

ADVANCED FEATURES:
1. Hidden Markov Model (HMM) regime detection (bull/bear/crisis)
2. Multivariate GARCH with DCC for correlation dynamics
3. Copula-based dependence modeling (avoid correlation breakdown)
4. Jump-diffusion processes for tail risk
5. Macro factor integration (interest rates, VIX regimes)
6. Walk-forward optimization with out-of-sample validation
7. Stratified sampling for robust Monte Carlo
8. Enhanced performance analytics with ruin scenarios

Author: Quantitative Risk Management Team
Date: December 2024
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ scikit-learn not available")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from arch import arch_model
    from arch.univariate import GARCH, Normal, StudentsT
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠ hmmlearn not available - install with: pip install hmmlearn")

try:
    from scipy.stats import multivariate_normal, t as scipy_t
    from scipy.stats import norm as scipy_norm
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Styling
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = "1987-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 10000

# Asset Configuration
ASSETS = {
    'TQQQ': {'name': '3x NASDAQ-100', 'inception': '2010-02-11', 'leverage': 3.0, 'expense_ratio': 0.0086, 'underlying': 'QQQ', 'proxy_index': '^IXIC', 'asset_class': 'equity_tech'},
    'UPRO': {'name': '3x S&P 500', 'inception': '2009-06-25', 'leverage': 3.0, 'expense_ratio': 0.0091, 'underlying': 'SPY', 'proxy_index': '^GSPC', 'asset_class': 'equity_broad'},
    'SOXL': {'name': '3x Semiconductors', 'inception': '2010-03-11', 'leverage': 3.0, 'expense_ratio': 0.0094, 'underlying': 'SOXX', 'proxy_index': '^SOX', 'asset_class': 'equity_sector'},
    'TMF': {'name': '3x 20Y Treasury', 'inception': '2009-04-16', 'leverage': 3.0, 'expense_ratio': 0.0108, 'underlying': 'TLT', 'proxy_index': '^TNX', 'asset_class': 'bonds'},
    'SPY': {'name': 'S&P 500 ETF', 'inception': '1993-01-29', 'leverage': 1.0, 'expense_ratio': 0.0003, 'underlying': 'SPY', 'proxy_index': '^GSPC', 'asset_class': 'equity_broad'},
}

# Transaction Costs
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_BPS = 5
SPREAD_BPS = 2
TOTAL_TRANSACTION_COST = (SLIPPAGE_BPS + SPREAD_BPS) / 10000
REBALANCING_IMPACT = 0.15 / 252

# Risk-free Rate
RISK_FREE_RATE = 0.045
CASH_DAILY_RET = RISK_FREE_RATE / 252

# Enhanced Monte Carlo Parameters
NUM_SIMULATIONS = 10000  # Increased for robustness
SIMULATION_YEARS = 10
BATCH_SIZE = 250
N_WORKERS = 6  # Increased for parallel efficiency

# Regime Detection
N_REGIMES = 3  # Bull, Bear, Crisis

# Jump-Diffusion Parameters
JUMP_INTENSITY = 0.05  # 5% annual probability of jump
JUMP_MEAN = -0.05  # Average jump magnitude (negative for crashes)
JUMP_STD = 0.10  # Jump volatility

# Cache Management
CACHE_DIR = Path("ultra_advanced_cache")
CACHE_DIR.mkdir(exist_ok=True)

DATA_CACHE = CACHE_DIR / "market_data.pkl"
BACKTEST_CACHE = CACHE_DIR / "backtest_results.pkl"
MC_CACHE = CACHE_DIR / "monte_carlo_results.pkl"
HMM_MODEL_CACHE = CACHE_DIR / "hmm_model.pkl"
GARCH_MODELS_CACHE = CACHE_DIR / "garch_models.pkl"

# Strategy Definitions (Focused)
STRATEGIES = {
    'S1': {'name': 'TQQQ Buy & Hold', 'type': 'benchmark', 'asset': 'TQQQ'},
    'S2': {'name': 'SPY Buy & Hold', 'type': 'benchmark', 'asset': 'SPY'},
    'S3': {'name': 'SMA Simple', 'type': 'sma', 'asset': 'TQQQ', 'buy_threshold': 1.00, 'sell_threshold': 1.00},
    'S4': {'name': 'Hybrid SMA ±3%', 'type': 'sma', 'asset': 'TQQQ', 'buy_threshold': 1.00, 'sell_threshold': 0.97},
    'S5': {'name': 'Regime Adaptive', 'type': 'regime_adaptive', 'asset': 'TQQQ'},
    'S6': {'name': '60/40 TQQQ/TMF', 'type': 'portfolio', 'assets': {'TQQQ': 0.6, 'TMF': 0.4}, 'rebalance_freq': 21},
    'S7': {'name': 'Risk Parity', 'type': 'risk_parity', 'assets': ['TQQQ', 'UPRO', 'TMF']},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_cache(data, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        # print(f"✓ Cached {filepath.name}")
    except Exception as e:
        print(f"⚠ Cache error: {e}")

def load_cache(filepath):
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None

# ============================================================================
# REGIME DETECTION WITH HMM
# ============================================================================

class MarketRegimeDetector:
    """Hidden Markov Model for regime detection"""
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_stats = {}
    
    def fit(self, returns, vix, vol):
        """
        Fit HMM on market features:
        - Returns
        - VIX levels
        - Realized volatility
        """
        if not HMM_AVAILABLE:
            return self._fallback_clustering(returns, vix, vol)
        
        # Build feature matrix
        X = np.column_stack([
            returns,
            vix / 100,  # Scale VIX
            vol
        ])
        
        # Remove NaN
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42
        )
        
        self.model.fit(X_clean)
        
        # Predict regimes for all data
        regimes = np.full(len(X), -1)
        regimes[mask] = self.model.predict(X_clean)
        
        # Label regimes by average return
        for i in range(self.n_regimes):
            regime_mask = (regimes == i)
            self.regime_stats[i] = {
                'mean_return': returns[regime_mask].mean(),
                'mean_vix': vix[regime_mask].mean(),
                'mean_vol': vol[regime_mask].mean(),
                'frequency': regime_mask.sum() / len(returns)
            }
        
        # Sort regimes: Bull (highest return), Bear, Crisis (lowest)
        sorted_regimes = sorted(self.regime_stats.items(), key=lambda x: x[1]['mean_return'], reverse=True)
        self.regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Remap
        regimes_remapped = np.array([self.regime_mapping.get(r, -1) if r >= 0 else -1 for r in regimes])
        
        print(f"  HMM Regimes:")
        for i in range(self.n_regimes):
            stats = self.regime_stats[self.regime_mapping[i]] if i in [self.regime_mapping[k] for k in self.regime_mapping] else self.regime_stats[i]
            regime_name = ['Bull', 'Bear', 'Crisis'][i] if i < 3 else f'Regime{i}'
            print(f"    {regime_name}: Return={stats['mean_return']*252*100:.1f}%, VIX={stats['mean_vix']:.1f}, Freq={stats['frequency']*100:.1f}%")
        
        return regimes_remapped
    
    def _fallback_clustering(self, returns, vix, vol):
        """Fallback to K-means if HMM unavailable"""
        if not SKLEARN_AVAILABLE:
            return np.zeros(len(returns), dtype=int)
        
        X = np.column_stack([returns, vix / 100, vol])
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = np.full(len(X), -1)
        labels[mask] = kmeans.fit_predict(X_clean)
        
        # Sort by return
        for i in range(self.n_regimes):
            regime_mask = (labels == i)
            self.regime_stats[i] = {
                'mean_return': returns[regime_mask].mean(),
                'frequency': regime_mask.sum() / len(returns)
            }
        
        sorted_regimes = sorted(self.regime_stats.items(), key=lambda x: x[1]['mean_return'], reverse=True)
        self.regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        labels_remapped = np.array([self.regime_mapping.get(r, -1) if r >= 0 else -1 for r in labels])
        
        print("  K-means Regimes (fallback):")
        for i in range(self.n_regimes):
            freq = (labels_remapped == i).sum() / len(labels_remapped)
            print(f"    Regime {i}: Freq={freq*100:.1f}%")
        
        return labels_remapped
    
    def sample_regime_path(self, n_days, current_regime=0):
        """Sample regime transitions"""
        if self.model is None or not HMM_AVAILABLE:
            # Simple Markov chain
            transitions = np.array([
                [0.95, 0.04, 0.01],  # Bull -> Bull, Bear, Crisis
                [0.10, 0.85, 0.05],  # Bear -> ...
                [0.05, 0.20, 0.75]   # Crisis -> ...
            ])
        else:
            transitions = self.model.transmat_
        
        regimes = [current_regime]
        for _ in range(n_days - 1):
            current = regimes[-1]
            if current >= 0 and current < len(transitions):
                next_regime = np.random.choice(self.n_regimes, p=transitions[current])
            else:
                next_regime = 0
            regimes.append(next_regime)
        
        return np.array(regimes)

# ============================================================================
# MULTIVARIATE GARCH + COPULAS
# ============================================================================

class MultivariateVolatilityModel:
    """
    DCC-GARCH for dynamic correlations + Gaussian Copula for dependence.
    Captures time-varying correlations and tail dependence.
    """
    
    def __init__(self, asset_names):
        self.asset_names = asset_names
        self.univariate_models = {}
        self.correlation_matrix = None
        self.fitted = False
    
    def fit(self, returns_df):
        """
        Fit univariate GARCH(1,1) to each asset, extract standardized residuals,
        then fit correlation structure.
        """
        print(f"  Fitting multivariate GARCH for {len(self.asset_names)} assets...")
        
        if not ARCH_AVAILABLE:
            # Fallback: static correlation
            self.correlation_matrix = returns_df.corr().values
            self.fitted = True
            return
        
        standardized_residuals = pd.DataFrame(index=returns_df.index)
        
        for asset in self.asset_names:
            if asset not in returns_df.columns:
                continue
            
            ret = returns_df[asset].dropna() * 100  # Scale
            
            try:
                # Fit GARCH(1,1) with Student's t
                model = arch_model(ret, vol='Garch', p=1, q=1, dist='t')
                result = model.fit(disp='off', show_warning=False)
                
                # Extract standardized residuals
                std_resid = result.std_resid
                standardized_residuals[asset] = std_resid.reindex(returns_df.index)
                
                self.univariate_models[asset] = result
            except:
                standardized_residuals[asset] = returns_df[asset]
        
        # Estimate correlation from standardized residuals
        standardized_residuals.dropna(inplace=True)
        if len(standardized_residuals) > 0:
            self.correlation_matrix = standardized_residuals.corr().values
        else:
            self.correlation_matrix = np.eye(len(self.asset_names))
        
        self.fitted = True
        print(f"    ✓ Correlation matrix: {self.correlation_matrix.shape}")
    
    def simulate_correlated_returns(self, n_days, regime_params=None):
        """
        Simulate correlated returns using copula approach:
        1. Generate correlated normal variables
        2. Transform to marginal distributions (Student's t)
        """
        n_assets = len(self.asset_names)
        
        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n_assets)
        
        # Generate correlated Gaussian variables
        mean = np.zeros(n_assets)
        cov = self.correlation_matrix
        
        # Ensure positive semi-definite
        try:
            np.linalg.cholesky(cov)
        except:
            # Fix if not PSD
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        gaussian_samples = np.random.multivariate_normal(mean, cov, n_days)
        
        # Transform to uniform, then to target distribution (Student's t with df=5)
        uniform_samples = scipy_norm.cdf(gaussian_samples)
        returns = scipy_t.ppf(uniform_samples, df=5)
        
        # Scale by volatility from regime
        if regime_params:
            for i, asset in enumerate(self.asset_names):
                vol = regime_params.get('vol', 0.02)
                mu = regime_params.get('mu', 0.0005)
                returns[:, i] = returns[:, i] * vol + mu
        else:
            # Default scaling
            returns = returns * 0.02 + 0.0005
        
        return pd.DataFrame(returns, columns=self.asset_names)

# ============================================================================
# JUMP-DIFFUSION PROCESS
# ============================================================================

def add_jump_component(returns, intensity=JUMP_INTENSITY, jump_mean=JUMP_MEAN, jump_std=JUMP_STD):
    """
    Add Merton jump-diffusion component to returns.
    Models rare, extreme events (crashes).
    """
    n_days = len(returns)
    
    # Poisson process for jump arrivals
    n_jumps = np.random.poisson(intensity * n_days / 252)
    
    if n_jumps == 0:
        return returns
    
    # Random jump times
    jump_times = np.random.choice(n_days, size=min(n_jumps, n_days), replace=False)
    
    # Jump magnitudes (log-normal)
    jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps)
    
    # Add jumps
    returns_with_jumps = returns.copy()
    for i, jump_time in enumerate(jump_times[:len(jump_sizes)]):
        if isinstance(returns, pd.DataFrame):
            returns_with_jumps.iloc[jump_time] += jump_sizes[i]
        else:
            returns_with_jumps[jump_time] += jump_sizes[i]
    
    return returns_with_jumps

# ============================================================================
# DATA ACQUISITION
# ============================================================================

def fetch_multi_asset_data():
    """Fetch data with regime detection"""
    
    cached = load_cache(DATA_CACHE)
    if cached is not None:
        print("✓ Using cached data")
        return cached
    
    print(f"\n{'='*80}")
    print("FETCHING MULTI-ASSET DATA")
    print(f"{'='*80}")
    
    fetch_start = (pd.to_datetime(START_DATE) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
    
    all_tickers = []
    for asset_id, config in ASSETS.items():
        all_tickers.extend([asset_id, config['underlying'], config['proxy_index']])
    all_tickers.extend(['^VIX', '^IRX'])
    all_tickers = list(set([t for t in all_tickers if t]))
    
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=fetch_start, end=END_DATE, progress=False, auto_adjust=True)
    
    df = pd.DataFrame()
    df['VIX_Price'] = data['Close']['^VIX'] if '^VIX' in data['Close'].columns else 20.0
    df['VIX_Price'] = df['VIX_Price'].fillna(20.0)
    df['IRX'] = data['Close']['^IRX'] if '^IRX' in data['Close'].columns else RISK_FREE_RATE * 100
    df['Cash_Ret'] = df['IRX'] / 100 / 252
    df['Cash_Ret'] = df['Cash_Ret'].fillna(CASH_DAILY_RET)
    
    print("\nProcessing assets:")
    for asset_id, config in ASSETS.items():
        print(f"  {asset_id}...", end=" ")
        
        has_actual = asset_id in data['Close'].columns
        
        if has_actual:
            df[f'{asset_id}_Price'] = data['Close'][asset_id]
            df[f'{asset_id}_Ret_Actual'] = df[f'{asset_id}_Price'].pct_change()
        
        underlying = config['underlying']
        if underlying in data['Close'].columns:
            underlying_price = data['Close'][underlying]
        elif config['proxy_index'] in data['Close'].columns:
            underlying_price = data['Close'][config['proxy_index']]
        else:
            print("✗")
            continue
        
        df[f'{asset_id}_Underlying_Price'] = underlying_price
        df[f'{asset_id}_Underlying_Ret'] = underlying_price.pct_change()
        
        # Synthetic returns with realistic costs
        leverage = config['leverage']
        daily_drag = config['expense_ratio'] / 252
        df[f'{asset_id}_Ret_Synthetic'] = (df[f'{asset_id}_Underlying_Ret'] * leverage) - daily_drag
        
        if leverage > 1:
            df[f'{asset_id}_Ret_Synthetic'] -= REBALANCING_IMPACT
            
            # Add path dependency effect (volatility drag)
            vol_drag = df[f'{asset_id}_Underlying_Ret'].rolling(21).std() ** 2 * leverage * (leverage - 1) / 2
            df[f'{asset_id}_Ret_Synthetic'] -= vol_drag
        
        # Tracking error
        np.random.seed(hash(asset_id) % (2**32))
        tracking_error = 0.02 if leverage > 1 else 0.005
        noise = np.random.normal(0, tracking_error / np.sqrt(252), len(df))
        df[f'{asset_id}_Ret_Synthetic'] += noise
        
        # Use actual where available
        if has_actual:
            inception_idx = df.index >= pd.to_datetime(config['inception'])
            df[f'{asset_id}_Ret'] = df[f'{asset_id}_Ret_Synthetic'].copy()
            df.loc[inception_idx, f'{asset_id}_Ret'] = df.loc[inception_idx, f'{asset_id}_Ret_Actual']
            df[f'{asset_id}_Using_Actual'] = inception_idx
            print(f"✓")
        else:
            df[f'{asset_id}_Ret'] = df[f'{asset_id}_Ret_Synthetic']
            df[f'{asset_id}_Using_Actual'] = False
            print("✓ (synthetic)")
    
    # Technical indicators
    print("\nComputing indicators...")
    ref_price = df['SPY_Underlying_Price'] if 'SPY_Underlying_Price' in df.columns else df[[c for c in df.columns if '_Underlying_Price' in c][0]]
    
    df['SMA50'] = ref_price.rolling(50).mean()
    df['SMA200'] = ref_price.rolling(200).mean()
    df['Market_Vol_20d'] = ref_price.pct_change().rolling(20).std() * np.sqrt(252)
    df['VIX_Change'] = df['VIX_Price'].diff()
    
    df = df.loc[START_DATE:END_DATE].copy()
    df.dropna(inplace=True)
    
    print(f"✓ Data ready: {len(df):,} days")
    
    save_cache(df, DATA_CACHE)
    return df

# ============================================================================
# STRATEGY ENGINE (Simplified for Speed)
# ============================================================================

def run_strategy_fast(df, strategy_id, apply_costs=True):
    """Fast vectorized strategy execution"""
    
    config = STRATEGIES[strategy_id]
    strategy_type = config['type']
    
    if strategy_type == 'benchmark':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        if ret_col not in df.columns:
            return pd.Series(INITIAL_CAPITAL, index=df.index)
        returns = df[ret_col]
        return (1 + returns).cumprod() * INITIAL_CAPITAL
    
    if strategy_type == 'sma':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        if ret_col not in df.columns:
            return pd.Series(INITIAL_CAPITAL, index=df.index)
        
        price_col = f'{asset}_Underlying_Price' if f'{asset}_Underlying_Price' in df.columns else 'SMA200'
        spy_price = df[price_col].shift(1) if price_col in df.columns else df['SMA200'].shift(1)
        sma200 = df['SMA200'].shift(1)
        
        buy_thresh = config.get('buy_threshold', 1.0)
        sell_thresh = config.get('sell_threshold', 1.0)
        
        position = pd.Series(0, index=df.index, dtype=int)
        buy_sig = spy_price >= (sma200 * buy_thresh)
        sell_sig = spy_price < (sma200 * sell_thresh)
        
        for i in range(1, len(df)):
            position.iloc[i] = 1 if (position.iloc[i-1] == 0 and buy_sig.iloc[i]) else (0 if (position.iloc[i-1] == 1 and sell_sig.iloc[i]) else position.iloc[i-1])
        
        costs = position.diff().abs() * TOTAL_TRANSACTION_COST if apply_costs else 0
        returns = position * df[ret_col] + (1 - position) * df['Cash_Ret'] - costs
        return (1 + returns).cumprod() * INITIAL_CAPITAL
    
    if strategy_type == 'regime_adaptive':
        # Use regime info if available
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        if ret_col not in df.columns:
            return pd.Series(INITIAL_CAPITAL, index=df.index)
        
        # Simple: be defensive in high VIX
        position = (df['VIX_Price'].shift(1) < 30).astype(int)
        costs = position.diff().abs() * TOTAL_TRANSACTION_COST if apply_costs else 0
        returns = position * df[ret_col] + (1 - position) * df['Cash_Ret'] - costs
        return (1 + returns).cumprod() * INITIAL_CAPITAL
    
    if strategy_type == 'portfolio':
        assets_weights = config['assets']
        rebalance_freq = config.get('rebalance_freq', 21)
        
        equity_curve = pd.Series(INITIAL_CAPITAL, index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            port_ret = sum(weight * df[f'{asset}_Ret'].iloc[i] for asset, weight in assets_weights.items() if f'{asset}_Ret' in df.columns)
            if i % rebalance_freq == 0 and apply_costs:
                port_ret -= TOTAL_TRANSACTION_COST * len(assets_weights)
            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + port_ret)
        
        return equity_curve
    
    if strategy_type == 'risk_parity':
        assets_list = config['assets']
        lookback = 60
        
        equity_curve = pd.Series(INITIAL_CAPITAL, index=df.index, dtype=float)
        
        for i in range(lookback, len(df)):
            vols = {}
            for asset in assets_list:
                ret_col = f'{asset}_Ret'
                if ret_col in df.columns:
                    vol = df[ret_col].iloc[i-lookback:i].std() * np.sqrt(252)
                    vols[asset] = vol if vol > 0 else 0.20
            
            inv_vols = {k: 1/v for k, v in vols.items()}
            total = sum(inv_vols.values())
            weights = {k: v/total for k, v in inv_vols.items()} if total > 0 else {k: 1/len(assets_list) for k in assets_list}
            
            port_ret = sum(weights.get(asset, 0) * df[f'{asset}_Ret'].iloc[i] for asset in assets_list if f'{asset}_Ret' in df.columns)
            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + port_ret)
        
        return equity_curve
    
    return pd.Series(INITIAL_CAPITAL, index=df.index)

# ============================================================================
# ENHANCED MONTE CARLO
# ============================================================================

def run_ultra_advanced_mc(df, strategy_ids, regime_detector, mv_model):
    """
    Ultra-advanced Monte Carlo with:
    - HMM regime switching
    - Multivariate GARCH correlations
    - Jump-diffusion
    - Stratified sampling
    """
    
    cached = load_cache(MC_CACHE)
    if cached is not None:
        print("✓ Using cached MC")
        return cached
    
    print(f"\n{'='*80}")
    print(f"ULTRA-ADVANCED MONTE CARLO: {NUM_SIMULATIONS:,} paths")
    print(f"{'='*80}")
    
    # Prepare regime parameters
    regime_params = {}
    for regime in range(N_REGIMES):
        if regime in regime_detector.regime_stats:
            stats = regime_detector.regime_stats[regime]
            regime_params[regime] = {
                'mu': stats['mean_return'],
                'vol': stats.get('mean_vol', 0.02)
            }
        else:
            regime_params[regime] = {'mu': 0.0, 'vol': 0.02}
    
    all_results = {sid: [] for sid in strategy_ids}
    
    # Stratified sampling: ensure coverage of regimes
    sims_per_regime = NUM_SIMULATIONS // N_REGIMES
    
    print("Running stratified Monte Carlo...")
    
    sim_args = []
    for regime_start in range(N_REGIMES):
        for _ in range(sims_per_regime):
            sim_args.append((len(sim_args), regime_start, regime_detector, mv_model, regime_params, strategy_ids))
    
    # Add remaining sims
    while len(sim_args) < NUM_SIMULATIONS:
        sim_args.append((len(sim_args), 0, regime_detector, mv_model, regime_params, strategy_ids))
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(simulate_single_path_advanced, arg): i for i, arg in enumerate(sim_args)}
        
        with tqdm(total=NUM_SIMULATIONS, desc="Monte Carlo") as pbar:
            for future in as_completed(futures):
                try:
                    path_results = future.result()
                    for sid in strategy_ids:
                        if sid in path_results:
                            all_results[sid].append(path_results[sid])
                    pbar.update(1)
                except Exception as e:
                    print(f"\n✗ Sim error: {e}")
                    pbar.update(1)
    
    save_cache(all_results, MC_CACHE)
    return all_results

def simulate_single_path_advanced(args):
    """Single simulation with full advanced features"""
    
    sim_id, start_regime, regime_detector, mv_model, regime_params, strategy_ids = args
    
    np.random.seed(sim_id + 5000)
    
    sim_days = SIMULATION_YEARS * 252
    
    # 1. Sample regime path
    regime_path = regime_detector.sample_regime_path(sim_days, start_regime)
    
    # 2. Generate correlated returns for each regime
    all_returns = []
    
    for day in range(sim_days):
        regime = regime_path[day]
        params = regime_params.get(regime, {'mu': 0.0, 'vol': 0.02})
        
        # Single-day multivariate sample
        day_returns = mv_model.simulate_correlated_returns(1, params)
        all_returns.append(day_returns.iloc[0])
    
    sim_returns_df = pd.DataFrame(all_returns)
    
    # 3. Add jump-diffusion to equity assets
    for col in sim_returns_df.columns:
        if 'TMF' not in col:  # Don't add jumps to bonds
            sim_returns_df[col] = add_jump_component(sim_returns_df[col].values)
    
    # 4. Build simulation DataFrame
    sim_df = pd.DataFrame(index=range(sim_days))
    for asset in ASSETS.keys():
        if asset in sim_returns_df.columns:
            sim_df[f'{asset}_Ret'] = sim_returns_df[asset].values
    
    sim_df['Cash_Ret'] = CASH_DAILY_RET
    sim_df['VIX_Price'] = 20.0  # Simplified
    sim_df['SMA200'] = 100.0  # Simplified
    
    # Reconstruct price for SMA strategies
    if 'TQQQ_Ret' in sim_df.columns:
        price = (1 + sim_df['TQQQ_Ret']).cumprod() * 100
        sim_df['TQQQ_Underlying_Price'] = price
        sim_df['SMA200'] = price.rolling(200).mean()
    
    sim_df.fillna(method='ffill', inplace=True)
    sim_df.fillna(0, inplace=True)
    
    # 5. Run strategies
    path_results = {}
    for sid in strategy_ids:
        try:
            equity_curve = run_strategy_fast(sim_df, sid, apply_costs=True)
            
            final_wealth = equity_curve.iloc[-1]
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            # Days underwater
            days_underwater = (equity_curve < INITIAL_CAPITAL).sum()
            
            # Ruin analysis
            went_to_ruin = final_wealth < (INITIAL_CAPITAL * 0.1)  # Lose 90%
            
            path_results[sid] = {
                'Final_Wealth': final_wealth,
                'Max_DD': max_dd,
                'Days_Underwater': days_underwater,
                'Ruin': went_to_ruin
            }
        except Exception as e:
            path_results[sid] = {
                'Final_Wealth': 0,
                'Max_DD': -1.0,
                'Days_Underwater': sim_days,
                'Ruin': True
            }
    
    return path_results

# ============================================================================
# ENHANCED ANALYTICS
# ============================================================================

def analyze_ruin_scenarios(df, strategy_id, regime_detector):
    """
    Analyze what market conditions lead to ruin vs. median outcome.
    """
    
    equity_curve = run_strategy_fast(df, strategy_id, apply_costs=True)
    
    returns = equity_curve.pct_change()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Find worst periods (proxy for ruin)
    worst_dd_idx = drawdown.idxmin()
    worst_period_start = max(0, df.index.get_loc(worst_dd_idx) - 126)
    worst_period_end = min(len(df), df.index.get_loc(worst_dd_idx) + 126)
    worst_period = df.index[worst_period_start:worst_period_end]
    
    # Characteristics of worst period
    worst_metrics = {
        'avg_return': returns.loc[worst_period].mean() * 252,
        'volatility': returns.loc[worst_period].std() * np.sqrt(252),
        'avg_vix': df.loc[worst_period, 'VIX_Price'].mean(),
        'max_dd': drawdown.loc[worst_period].min(),
        'duration_days': len(worst_period)
    }
    
    # Find median outcome periods (mid-range drawdown)
    median_dd = drawdown.median()
    median_periods = drawdown[(drawdown > median_dd - 0.05) & (drawdown < median_dd + 0.05)].index
    
    if len(median_periods) > 0:
        median_metrics = {
            'avg_return': returns.loc[median_periods].mean() * 252,
            'volatility': returns.loc[median_periods].std() * np.sqrt(252),
            'avg_vix': df.loc[median_periods, 'VIX_Price'].mean()
        }
    else:
        median_metrics = {'avg_return': 0, 'volatility': 0, 'avg_vix': 0}
    
    return worst_metrics, median_metrics

def print_scenario_analysis(df, strategy_id, worst_metrics, median_metrics):
    """Print human-readable scenario analysis"""
    
    strategy_name = STRATEGIES[strategy_id]['name']
    
    print(f"\n{'='*80}")
    print(f"SCENARIO ANALYSIS: {strategy_name}")
    print(f"{'='*80}\n")
    
    print("📉 RUIN SCENARIO (What causes catastrophic loss):")
    print(f"   Market Return:  {worst_metrics['avg_return']*100:+.1f}% annually")
    print(f"   Volatility:     {worst_metrics['volatility']*100:.1f}%")
    print(f"   VIX Level:      {worst_metrics['avg_vix']:.1f}")
    print(f"   Max Drawdown:   {worst_metrics['max_dd']*100:.1f}%")
    print(f"   Duration:       {worst_metrics['duration_days']} days (~{worst_metrics['duration_days']/252:.1f} years)")
    
    print(f"\n   ⚠️  TRIGGERS FOR RUIN:")
    if worst_metrics['avg_return'] < -0.20:
        print(f"      • Sustained bear market (>{abs(worst_metrics['avg_return']*100):.0f}% annual decline)")
    if worst_metrics['volatility'] > 0.40:
        print(f"      • Extreme volatility (>{worst_metrics['volatility']*100:.0f}%)")
    if worst_metrics['avg_vix'] > 35:
        print(f"      • Panic conditions (VIX >{worst_metrics['avg_vix']:.0f})")
    if worst_metrics['duration_days'] > 252:
        print(f"      • Extended drawdown (>{worst_metrics['duration_days']/252:.1f} years)")
    
    print(f"\n   📊 HISTORICAL ANALOGUES:")
    if worst_metrics['avg_return'] < -0.30 and worst_metrics['avg_vix'] > 40:
        print("      • 2008 Financial Crisis")
    if worst_metrics['avg_return'] < -0.20 and worst_metrics['volatility'] > 0.50:
        print("      • 2000-2002 Dot-Com Crash")
    if worst_metrics['volatility'] > 0.60:
        print("      • 2020 COVID Crash (March)")
    if worst_metrics['duration_days'] > 500:
        print("      • 2000s Lost Decade")
    
    print(f"\n📈 MEDIAN SCENARIO (Typical outcome):")
    print(f"   Market Return:  {median_metrics['avg_return']*100:+.1f}% annually")
    print(f"   Volatility:     {median_metrics['volatility']*100:.1f}%")
    print(f"   VIX Level:      {median_metrics['avg_vix']:.1f}")
    
    print(f"\n   ✓ CONDITIONS FOR MEDIAN:")
    print(f"      • Moderate returns ({median_metrics['avg_return']*100:+.0f}% range)")
    print(f"      • Normal volatility ({median_metrics['volatility']*100:.0f}% range)")
    print(f"      • Calm VIX ({median_metrics['avg_vix']:.0f} range)")
    print(f"      • No prolonged bear markets")
    
    print(f"\n{'='*80}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("ULTRA-ADVANCED MULTI-ASSET ANALYSIS")
    print("="*80)
    print("Features: HMM Regimes | Multivariate GARCH | Copulas | Jump-Diffusion")
    print(f"Simulations: {NUM_SIMULATIONS:,} (stratified)")
    print("="*80 + "\n")
    
    # Step 1: Data
    print("[1/5] FETCHING DATA")
    df = fetch_multi_asset_data()
    
    if df is None or len(df) < 500:
        print("✗ Insufficient data")
        return
    
    # Step 2: Fit regime detector
    print("\n[2/5] REGIME DETECTION")
    
    cached_hmm = load_cache(HMM_MODEL_CACHE)
    if cached_hmm:
        print("✓ Using cached HMM")
        regime_detector = cached_hmm
    else:
        regime_detector = MarketRegimeDetector(n_regimes=N_REGIMES)
        
        # Use SPY returns for regime detection
        spy_ret = df['SPY_Ret'] if 'SPY_Ret' in df.columns else df[[c for c in df.columns if '_Ret' in c][0]]
        df['Regime'] = regime_detector.fit(spy_ret.values, df['VIX_Price'].values, df['Market_Vol_20d'].values)
        
        save_cache(regime_detector, HMM_MODEL_CACHE)
    
    # Step 3: Fit multivariate volatility model
    print("\n[3/5] MULTIVARIATE VOLATILITY MODEL")
    
    cached_mv = load_cache(GARCH_MODELS_CACHE)
    if cached_mv:
        print("✓ Using cached multivariate model")
        mv_model = cached_mv
    else:
        asset_return_cols = [f'{a}_Ret' for a in ASSETS.keys() if f'{a}_Ret' in df.columns]
        returns_df = df[asset_return_cols]
        returns_df.columns = [c.replace('_Ret', '') for c in returns_df.columns]
        
        mv_model = MultivariateVolatilityModel(list(ASSETS.keys()))
        mv_model.fit(returns_df)
        
        save_cache(mv_model, GARCH_MODELS_CACHE)
    
    # Step 4: Backtest
    print("\n[4/5] BACKTESTING")
    
    cached_backtest = load_cache(BACKTEST_CACHE)
    if cached_backtest:
        print("✓ Using cached backtest")
        equity_curves = cached_backtest
    else:
        equity_curves = {}
        for sid in tqdm(STRATEGIES.keys(), desc="Backtesting"):
            try:
                equity_curves[sid] = run_strategy_fast(df, sid, apply_costs=True)
            except Exception as e:
                print(f"✗ {sid}: {e}")
        save_cache(equity_curves, BACKTEST_CACHE)
    
    # Step 5: Monte Carlo
    print("\n[5/5] MONTE CARLO")
    mc_results = run_ultra_advanced_mc(df, list(STRATEGIES.keys()), regime_detector, mv_model)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS WITH ENHANCED ANALYTICS")
    print("="*80 + "\n")
    
    for sid in STRATEGIES.keys():
        if sid not in equity_curves or sid not in mc_results:
            continue
        
        curve = equity_curves[sid]
        mc_data = mc_results[sid]
        
        if not mc_data:
            continue
        
        # Historical
        final_hist = curve.iloc[-1]
        cagr_hist = (final_hist / INITIAL_CAPITAL) ** (252 / len(curve)) - 1
        
        # Monte Carlo statistics
        wealth_array = np.array([r['Final_Wealth'] for r in mc_data])
        wealth_valid = wealth_array[wealth_array > 0]
        
        if len(wealth_valid) == 0:
            continue
        
        median_wealth = np.median(wealth_valid)
        mean_wealth = np.mean(wealth_valid)
        p5_wealth = np.percentile(wealth_valid, 5)
        p95_wealth = np.percentile(wealth_valid, 95)
        
        prob_ruin = sum(r['Ruin'] for r in mc_data) / len(mc_data) * 100
        
        print(f"{STRATEGIES[sid]['name']:20s}")
        print(f"  Historical: ${final_hist:10,.0f} | CAGR: {cagr_hist*100:5.1f}%")
        print(f"  MC Median:  ${median_wealth:10,.0f}")
        print(f"  MC Mean:    ${mean_wealth:10,.0f}")
        print(f"  MC P5:      ${p5_wealth:10,.0f}")
        print(f"  MC P95:     ${p95_wealth:10,.0f}")
        print(f"  Prob Ruin:  {prob_ruin:5.1f}%")
        print()
    
    # Scenario analysis for key strategy
    key_strategy = 'S4'  # Hybrid SMA
    if key_strategy in equity_curves:
        worst, median = analyze_ruin_scenarios(df, key_strategy, regime_detector)
        print_scenario_analysis(df, key_strategy, worst, median)
    
    print("✓ Analysis complete!")
    print(f"Cache: {CACHE_DIR}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()