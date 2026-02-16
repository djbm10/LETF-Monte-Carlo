"""
High-Performance Numerical Computing for Financial Applications
================================================================

This comprehensive guide covers:
1. Numba (JIT compilation) for speeding up NumPy code
2. JAX for autodiff and GPU acceleration
3. CuPy for GPU-accelerated NumPy
4. SciPy optimization routines
5. Vectorization best practices
6. Memory efficiency for large simulations
7. Parallel processing (multiprocessing vs joblib vs dask)

With specific examples for financial Monte Carlo simulations and time series modeling.
"""

import numpy as np
import time
from typing import Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. NUMBA - JIT COMPILATION FOR NUMPY CODE
# ============================================================================
"""
Numba compiles Python functions to machine code using LLVM, providing
significant speedups for numerical computations with minimal code changes.

Key benefits:
- 10-100x speedup for numerical loops
- GPU acceleration support
- Automatic parallelization
- Works seamlessly with NumPy arrays
"""

try:
    from numba import jit, njit, vectorize, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")


if NUMBA_AVAILABLE:
    # Example 1: Monte Carlo Option Pricing with Numba
    @njit(parallel=True)
    def monte_carlo_option_numba(S0, K, T, r, sigma, num_sims, num_steps):
        """
        Price European call option using Monte Carlo simulation with Numba JIT.

        Parameters:
        -----------
        S0 : float - Initial stock price
        K : float - Strike price
        T : float - Time to maturity (years)
        r : float - Risk-free rate
        sigma : float - Volatility
        num_sims : int - Number of simulation paths
        num_steps : int - Number of time steps

        Returns:
        --------
        float - Option price
        """
        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)
        discount = np.exp(-r * T)

        payoffs = np.zeros(num_sims)

        # Parallel loop over simulations
        for i in prange(num_sims):
            S = S0
            for j in range(num_steps):
                # Geometric Brownian Motion
                z = np.random.standard_normal()
                S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)

            # Calculate payoff
            payoffs[i] = max(S - K, 0.0)

        return discount * np.mean(payoffs)


    # Example 2: GARCH(1,1) Volatility Estimation with Numba
    @njit
    def garch_11_loglikelihood_numba(params, returns):
        """
        Calculate log-likelihood for GARCH(1,1) model.

        Parameters:
        -----------
        params : array - [omega, alpha, beta]
        returns : array - Return series

        Returns:
        --------
        float - Negative log-likelihood
        """
        omega, alpha, beta = params
        n = len(returns)

        # Initial variance
        sigma2 = np.var(returns)
        log_likelihood = 0.0

        for t in range(n):
            # GARCH(1,1): sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) +
                                     returns[t]**2 / sigma2)

            sigma2 = omega + alpha * returns[t]**2 + beta * sigma2

        return -log_likelihood


    # Example 3: Vectorized Black-Scholes with Numba
    @vectorize(['float64(float64, float64, float64, float64, float64)'],
               target='parallel')
    def black_scholes_call_numba(S, K, T, r, sigma):
        """
        Vectorized Black-Scholes formula for European call options.
        Automatically parallelized across CPU cores.
        """
        from math import sqrt, log, exp, erf

        def norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


    # Example 4: Time Series Correlation Matrix (Optimized)
    @njit(parallel=True)
    def rolling_correlation_numba(returns, window):
        """
        Calculate rolling correlation matrix for multiple assets.
        """
        n_assets, n_periods = returns.shape
        n_windows = n_periods - window + 1

        correlations = np.zeros((n_windows, n_assets, n_assets))

        for w in prange(n_windows):
            window_data = returns[:, w:w+window]

            # Calculate correlation matrix for this window
            for i in range(n_assets):
                for j in range(i, n_assets):
                    corr = np.corrcoef(window_data[i], window_data[j])[0, 1]
                    correlations[w, i, j] = corr
                    correlations[w, j, i] = corr

        return correlations


# ============================================================================
# 2. JAX - AUTOMATIC DIFFERENTIATION AND GPU ACCELERATION
# ============================================================================
"""
JAX provides automatic differentiation, GPU/TPU acceleration, and JIT compilation.

Key benefits:
- Automatic differentiation (grad, jacobian, hessian)
- GPU/TPU support with minimal code changes
- Functional programming paradigm
- vmap for automatic vectorization
- Composable transformations
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, value_and_grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available. Install with: pip install jax jaxlib")


if JAX_AVAILABLE:
    # Example 1: Portfolio Optimization with Automatic Differentiation
    @jit
    def portfolio_variance(weights, cov_matrix):
        """Calculate portfolio variance."""
        return jnp.dot(weights, jnp.dot(cov_matrix, weights))


    @jit
    def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
        """
        Calculate negative Sharpe ratio for optimization.
        """
        portfolio_return = jnp.dot(weights, returns)
        portfolio_std = jnp.sqrt(portfolio_variance(weights, cov_matrix))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe  # Negative for minimization


    # Gradient of Sharpe ratio w.r.t. weights
    grad_sharpe = jit(grad(portfolio_sharpe_ratio))


    # Example 2: Monte Carlo with JAX (GPU-accelerated)
    @jit
    def monte_carlo_european_jax(key, S0, K, T, r, sigma, num_sims, num_steps):
        """
        GPU-accelerated Monte Carlo option pricing with JAX.
        """
        dt = T / num_steps
        sqrt_dt = jnp.sqrt(dt)

        # Generate all random numbers at once (vectorized)
        keys = jax.random.split(key, num_sims)

        def simulate_path(key):
            z = jax.random.normal(key, shape=(num_steps,))
            log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
            S_T = S0 * jnp.exp(jnp.sum(log_returns))
            return jnp.maximum(S_T - K, 0.0)

        # Vectorize across all paths
        payoffs = vmap(simulate_path)(keys)
        option_price = jnp.exp(-r * T) * jnp.mean(payoffs)

        return option_price


    # Example 3: Neural Network for Time Series Prediction
    def init_network_params(layer_sizes, key):
        """Initialize neural network parameters."""
        keys = jax.random.split(key, len(layer_sizes))
        params = []

        for i in range(len(layer_sizes) - 1):
            w_key, b_key = jax.random.split(keys[i])
            params.append({
                'W': jax.random.normal(w_key, (layer_sizes[i], layer_sizes[i+1])) * 0.1,
                'b': jax.random.normal(b_key, (layer_sizes[i+1],)) * 0.1
            })

        return params


    @jit
    def forward_pass(params, x):
        """Forward pass through neural network."""
        for layer in params[:-1]:
            x = jnp.tanh(jnp.dot(x, layer['W']) + layer['b'])

        # Linear output layer
        x = jnp.dot(x, params[-1]['W']) + params[-1]['b']
        return x


    @jit
    def mse_loss(params, x, y):
        """Mean squared error loss."""
        pred = forward_pass(params, x)
        return jnp.mean((pred - y)**2)


    # Gradient of loss w.r.t. all parameters
    grad_mse = jit(grad(mse_loss))


    # Example 4: Implied Volatility Calculation with Autodiff
    @jit
    def black_scholes_call_jax(S, K, T, r, sigma):
        """Black-Scholes call option price (JAX version)."""
        from jax.scipy.stats import norm

        d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)

        return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


    @jit
    def implied_volatility_objective(sigma, S, K, T, r, market_price):
        """Objective function for implied volatility calculation."""
        model_price = black_scholes_call_jax(S, K, T, r, sigma)
        return (model_price - market_price)**2


    # Gradient for Newton-Raphson iteration
    grad_iv = jit(grad(implied_volatility_objective))


# ============================================================================
# 3. CUPY - GPU-ACCELERATED NUMPY
# ============================================================================
"""
CuPy is a NumPy-compatible library for GPU-accelerated computing.

Key benefits:
- Drop-in replacement for NumPy
- 10-100x speedup for large arrays
- Custom CUDA kernels support
- Seamless CPU-GPU transfer
"""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda12x")


if CUPY_AVAILABLE:
    # Example 1: GPU-Accelerated Covariance Matrix
    def calculate_covariance_gpu(returns_cpu):
        """
        Calculate covariance matrix on GPU.

        Parameters:
        -----------
        returns_cpu : numpy array (n_assets, n_periods)

        Returns:
        --------
        numpy array - Covariance matrix
        """
        # Transfer to GPU
        returns_gpu = cp.array(returns_cpu)

        # Calculate covariance on GPU
        cov_gpu = cp.cov(returns_gpu)

        # Transfer back to CPU
        return cp.asnumpy(cov_gpu)


    # Example 2: Custom GPU Kernel for Black-Scholes
    black_scholes_kernel = cp.ElementwiseKernel(
        'float64 S, float64 K, float64 T, float64 r, float64 sigma',
        'float64 call_price',
        '''
        const double sqrt_T = sqrt(T);
        const double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
        const double d2 = d1 - sigma * sqrt_T;

        // Approximate normal CDF
        const double a1 = 0.31938153;
        const double a2 = -0.356563782;
        const double a3 = 1.781477937;
        const double a4 = -1.821255978;
        const double a5 = 1.330274429;

        double cnd_d1, cnd_d2;
        double k1 = 1.0 / (1.0 + 0.2316419 * abs(d1));
        cnd_d1 = 1.0 - exp(-d1 * d1 / 2.0) / sqrt(2.0 * M_PI) *
                 (a1 * k1 + a2 * k1 * k1 + a3 * k1 * k1 * k1 +
                  a4 * k1 * k1 * k1 * k1 + a5 * k1 * k1 * k1 * k1 * k1);
        if (d1 < 0) cnd_d1 = 1.0 - cnd_d1;

        double k2 = 1.0 / (1.0 + 0.2316419 * abs(d2));
        cnd_d2 = 1.0 - exp(-d2 * d2 / 2.0) / sqrt(2.0 * M_PI) *
                 (a1 * k2 + a2 * k2 * k2 + a3 * k2 * k2 * k2 +
                  a4 * k2 * k2 * k2 * k2 + a5 * k2 * k2 * k2 * k2 * k2);
        if (d2 < 0) cnd_d2 = 1.0 - cnd_d2;

        call_price = S * cnd_d1 - K * exp(-r * T) * cnd_d2;
        ''',
        'black_scholes'
    )


    # Example 3: Large-Scale Portfolio Simulation
    def gpu_portfolio_simulation(n_assets, n_scenarios, n_periods):
        """
        Simulate portfolio returns on GPU.
        """
        # Generate random returns on GPU
        returns = cp.random.normal(0.08/252, 0.2/cp.sqrt(252),
                                  (n_assets, n_scenarios, n_periods))

        # Equal weights
        weights = cp.ones(n_assets) / n_assets

        # Calculate portfolio returns
        portfolio_returns = cp.tensordot(weights, returns, axes=([0], [0]))

        # Calculate cumulative returns
        cumulative_returns = cp.cumprod(1 + portfolio_returns, axis=1)

        return cp.asnumpy(cumulative_returns)


# ============================================================================
# 4. SCIPY OPTIMIZATION ROUTINES
# ============================================================================
"""
SciPy provides robust optimization algorithms for parameter estimation and
portfolio optimization.

Key methods:
- minimize: General-purpose optimization
- least_squares: Nonlinear least squares
- curve_fit: Curve fitting
- differential_evolution: Global optimization
"""

from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.stats import norm


# Example 1: GARCH Model Parameter Estimation
def garch_11_variance(params, returns, initial_var=None):
    """
    Calculate variance series for GARCH(1,1) model.
    """
    omega, alpha, beta = params
    n = len(returns)

    variance = np.zeros(n)
    variance[0] = initial_var if initial_var else np.var(returns)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

    return variance


def garch_11_negative_loglikelihood(params, returns):
    """
    Negative log-likelihood for GARCH(1,1).
    """
    if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[1] + params[2] >= 1:
        return 1e10  # Invalid parameters

    variance = garch_11_variance(params, returns)

    # Avoid log(0)
    variance = np.maximum(variance, 1e-10)

    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) +
                                   returns**2 / variance)

    return -log_likelihood


def estimate_garch_parameters(returns):
    """
    Estimate GARCH(1,1) parameters using maximum likelihood.
    """
    # Initial guess
    initial_params = [0.01, 0.1, 0.8]

    # Optimize
    result = minimize(
        garch_11_negative_loglikelihood,
        initial_params,
        args=(returns,),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (0, 1), (0, 1)]
    )

    return result.x


# Example 2: Portfolio Optimization (Mean-Variance)
def portfolio_optimization_scipy(expected_returns, cov_matrix, target_return=None):
    """
    Optimize portfolio weights using mean-variance optimization.

    Parameters:
    -----------
    expected_returns : array - Expected returns for each asset
    cov_matrix : array - Covariance matrix
    target_return : float - Target portfolio return (optional)

    Returns:
    --------
    array - Optimal weights
    """
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return np.dot(weights, np.dot(cov_matrix, weights))

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, expected_returns) - target_return
        })

    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Initial guess
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# Example 3: Calibrating Option Pricing Model
def heston_characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, tau, r):
    """
    Heston model characteristic function for option pricing.
    """
    # Complex calculations for Heston model
    xi = kappa - sigma * rho * 1j * phi
    d = np.sqrt(xi**2 + sigma**2 * (phi**2 + 1j * phi))
    g = (xi - d) / (xi + d)

    C = r * 1j * phi * tau + (kappa * theta / sigma**2) * (
        (xi - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )

    D = ((xi - d) / sigma**2) * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


def calibrate_heston_model(market_prices, strikes, maturities, S0, r):
    """
    Calibrate Heston model to market option prices.
    """
    def objective(params):
        v0, kappa, theta, sigma, rho = params

        model_prices = []
        for K, T, market_price in zip(strikes, maturities, market_prices):
            # Simplified pricing (would need full implementation)
            model_price = S0 * 0.1  # Placeholder
            model_prices.append(model_price)

        return np.sum((np.array(model_prices) - np.array(market_prices))**2)

    # Parameter bounds: v0, kappa, theta, sigma, rho
    bounds = [(0.01, 1), (0.1, 10), (0.01, 1), (0.1, 1), (-0.99, 0.99)]

    result = differential_evolution(objective, bounds, maxiter=100)

    return result.x


# ============================================================================
# 5. VECTORIZATION BEST PRACTICES
# ============================================================================
"""
Vectorization eliminates slow Python loops by using NumPy's optimized C code.

Key principles:
1. Avoid explicit loops - use array operations
2. Use broadcasting for operations on different shapes
3. Generate all random numbers at once
4. Use NumPy functions (sum, mean, std) instead of Python equivalents
5. Pre-allocate arrays when possible
"""


# BAD: Slow loop-based approach
def monte_carlo_slow(S0, K, T, r, sigma, num_sims, num_steps):
    """Slow implementation with Python loops."""
    dt = T / num_steps
    payoffs = []

    for _ in range(num_sims):
        S = S0
        for _ in range(num_steps):
            z = np.random.standard_normal()
            S = S * np.exp((r - 0.5 * sigma**2) * dt +
                          sigma * np.sqrt(dt) * z)
        payoffs.append(max(S - K, 0))

    return np.exp(-r * T) * np.mean(payoffs)


# GOOD: Vectorized approach
def monte_carlo_vectorized(S0, K, T, r, sigma, num_sims, num_steps):
    """Fast vectorized implementation."""
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Generate ALL random numbers at once
    z = np.random.standard_normal((num_sims, num_steps))

    # Calculate log returns for all paths simultaneously
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z

    # Calculate final prices (cumulative product)
    S_T = S0 * np.exp(np.sum(log_returns, axis=1))

    # Calculate payoffs (vectorized max)
    payoffs = np.maximum(S_T - K, 0)

    return np.exp(-r * T) * np.mean(payoffs)


# Example: Broadcasting for Correlation Analysis
def calculate_pairwise_correlations_vectorized(returns):
    """
    Calculate all pairwise correlations efficiently using broadcasting.

    Parameters:
    -----------
    returns : array (n_assets, n_periods)

    Returns:
    --------
    array (n_assets, n_assets) - Correlation matrix
    """
    # Standardize returns
    means = np.mean(returns, axis=1, keepdims=True)
    stds = np.std(returns, axis=1, keepdims=True)
    standardized = (returns - means) / stds

    # Use matrix multiplication for correlation
    n_periods = returns.shape[1]
    correlation_matrix = np.dot(standardized, standardized.T) / n_periods

    return correlation_matrix


# ============================================================================
# 6. MEMORY EFFICIENCY FOR LARGE SIMULATIONS
# ============================================================================
"""
Memory optimization strategies:

1. Use appropriate dtypes (float32 vs float64)
2. Process data in chunks
3. Use generators for large datasets
4. Delete intermediate arrays
5. Use memory-mapped arrays for huge datasets
6. Leverage in-place operations
"""


# Example 1: Chunked Monte Carlo Simulation
def monte_carlo_chunked(S0, K, T, r, sigma, total_sims, chunk_size, num_steps):
    """
    Memory-efficient Monte Carlo using chunks.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    n_chunks = total_sims // chunk_size
    payoff_sum = 0.0

    for _ in range(n_chunks):
        # Process one chunk at a time
        z = np.random.standard_normal((chunk_size, num_steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        S_T = S0 * np.exp(np.sum(log_returns, axis=1))
        payoffs = np.maximum(S_T - K, 0)

        payoff_sum += np.sum(payoffs)

        # Free memory
        del z, log_returns, S_T, payoffs

    return discount * payoff_sum / total_sims


# Example 2: Memory-Efficient Data Type Selection
def optimize_dtype_usage():
    """
    Demonstrate memory savings with appropriate dtypes.
    """
    n = 10_000_000

    # float64 (default) - 8 bytes per element
    arr_float64 = np.random.randn(n)
    memory_float64 = arr_float64.nbytes / 1024**2  # MB

    # float32 - 4 bytes per element
    arr_float32 = np.random.randn(n).astype(np.float32)
    memory_float32 = arr_float32.nbytes / 1024**2  # MB

    print(f"float64: {memory_float64:.2f} MB")
    print(f"float32: {memory_float32:.2f} MB")
    print(f"Savings: {memory_float64 - memory_float32:.2f} MB ({100*(1-memory_float32/memory_float64):.1f}%)")


# Example 3: Generator-Based Simulation
def simulation_generator(S0, K, T, r, sigma, num_steps, batch_size=10000):
    """
    Generator that yields batches of simulated paths.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    while True:
        z = np.random.standard_normal((batch_size, num_steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        S_T = S0 * np.exp(np.sum(log_returns, axis=1))
        payoffs = np.maximum(S_T - K, 0)

        yield payoffs


# Example 4: In-Place Operations
def portfolio_rebalancing_inplace(weights, returns, cov_matrix):
    """
    Demonstrate in-place operations to save memory.
    """
    # BAD: Creates new array
    # weights_new = weights * 2

    # GOOD: In-place operation
    weights *= 2  # or: np.multiply(weights, 2, out=weights)
    weights /= np.sum(weights)  # Normalize in-place

    return weights


# ============================================================================
# 7. PARALLEL PROCESSING COMPARISON
# ============================================================================
"""
Comparison of parallel processing approaches:

1. multiprocessing: Built-in, process-based parallelism
   - Good for: CPU-bound tasks on single machine
   - Limitations: Communication overhead, pickling issues

2. joblib: Simple parallelization with caching
   - Good for: Embarrassingly parallel tasks, NumPy arrays
   - Limitations: Single machine only

3. dask: Distributed computing framework
   - Good for: Large datasets, multi-node clusters
   - Limitations: Higher overhead for small tasks
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Joblib not available. Install with: pip install joblib")

try:
    import dask
    import dask.array as da
    from dask import delayed as dask_delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not available. Install with: pip install dask")


# Shared simulation function
def simulate_single_path(seed, S0, K, T, r, sigma, num_steps):
    """Single path simulation for parallel execution."""
    np.random.seed(seed)
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    z = np.random.standard_normal(num_steps)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
    S_T = S0 * np.exp(np.sum(log_returns))

    return max(S_T - K, 0)


# Example 1: multiprocessing
def monte_carlo_multiprocessing(S0, K, T, r, sigma, num_sims, num_steps, n_jobs=4):
    """
    Monte Carlo using multiprocessing module.
    """
    seeds = np.random.randint(0, 10000000, num_sims)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        payoffs = list(executor.map(
            simulate_single_path,
            seeds,
            [S0]*num_sims, [K]*num_sims, [T]*num_sims,
            [r]*num_sims, [sigma]*num_sims, [num_steps]*num_sims
        ))

    return np.exp(-r * T) * np.mean(payoffs)


# Example 2: joblib
if JOBLIB_AVAILABLE:
    def monte_carlo_joblib(S0, K, T, r, sigma, num_sims, num_steps, n_jobs=4):
        """
        Monte Carlo using joblib (optimized for NumPy).
        """
        seeds = np.random.randint(0, 10000000, num_sims)

        payoffs = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(simulate_single_path)(seed, S0, K, T, r, sigma, num_steps)
            for seed in seeds
        )

        return np.exp(-r * T) * np.mean(payoffs)


# Example 3: dask
if DASK_AVAILABLE:
    def monte_carlo_dask(S0, K, T, r, sigma, num_sims, num_steps):
        """
        Monte Carlo using dask for distributed computing.
        """
        seeds = np.random.randint(0, 10000000, num_sims)

        # Create delayed computations
        delayed_payoffs = [
            dask_delayed(simulate_single_path)(seed, S0, K, T, r, sigma, num_steps)
            for seed in seeds
        ]

        # Compute in parallel
        payoffs = dask.compute(*delayed_payoffs)

        return np.exp(-r * T) * np.mean(payoffs)


    def large_covariance_dask(returns_large):
        """
        Calculate covariance for very large datasets using dask.
        """
        # Convert to dask array
        returns_dask = da.from_array(returns_large, chunks=(100, -1))

        # Calculate covariance (computed lazily)
        cov_dask = da.cov(returns_dask)

        # Trigger computation
        return cov_dask.compute()


# ============================================================================
# PRACTICAL EXAMPLES FOR FINANCIAL APPLICATIONS
# ============================================================================

class FinancialSimulationExamples:
    """
    Comprehensive examples for financial modeling.
    """

    @staticmethod
    def var_calculation_vectorized(returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) using vectorized operations.

        Parameters:
        -----------
        returns : array - Historical returns
        confidence_level : float - Confidence level (e.g., 0.95 for 95%)

        Returns:
        --------
        float - VaR estimate
        """
        return np.percentile(returns, (1 - confidence_level) * 100)


    @staticmethod
    def cvar_calculation_vectorized(returns, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        """
        var = FinancialSimulationExamples.var_calculation_vectorized(
            returns, confidence_level
        )
        return np.mean(returns[returns <= var])


    @staticmethod
    def efficient_frontier_calculation(expected_returns, cov_matrix, n_portfolios=1000):
        """
        Calculate efficient frontier using vectorized operations.
        """
        n_assets = len(expected_returns)

        # Generate random weights (vectorized)
        weights = np.random.random((n_portfolios, n_assets))
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Calculate returns and volatilities (vectorized)
        portfolio_returns = np.dot(weights, expected_returns)

        # Vectorized portfolio variance calculation
        portfolio_variance = np.sum(
            weights @ cov_matrix * weights, axis=1
        )
        portfolio_std = np.sqrt(portfolio_variance)

        return portfolio_returns, portfolio_std, weights


    @staticmethod
    def bollinger_bands_vectorized(prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands using vectorized operations.
        """
        # Rolling mean (using convolution for efficiency)
        weights = np.ones(window) / window
        rolling_mean = np.convolve(prices, weights, mode='valid')

        # Rolling standard deviation
        rolling_std = np.array([
            np.std(prices[i:i+window])
            for i in range(len(prices) - window + 1)
        ])

        upper_band = rolling_mean + num_std * rolling_std
        lower_band = rolling_mean - num_std * rolling_std

        return upper_band, rolling_mean, lower_band


# ============================================================================
# BENCHMARKING AND COMPARISON
# ============================================================================

def benchmark_monte_carlo():
    """
    Compare different Monte Carlo implementations.
    """
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    num_sims = 100000
    num_steps = 252

    print("=" * 70)
    print("Monte Carlo Option Pricing Benchmark")
    print("=" * 70)
    print(f"Simulations: {num_sims:,}, Steps: {num_steps}")
    print()

    # Vectorized NumPy
    start = time.time()
    price_vec = monte_carlo_vectorized(S0, K, T, r, sigma, num_sims, num_steps)
    time_vec = time.time() - start
    print(f"Vectorized NumPy:  {price_vec:.4f}  ({time_vec:.3f}s)")

    # Numba JIT
    if NUMBA_AVAILABLE:
        # Warm-up
        monte_carlo_option_numba(S0, K, T, r, sigma, 1000, num_steps)

        start = time.time()
        price_numba = monte_carlo_option_numba(S0, K, T, r, sigma, num_sims, num_steps)
        time_numba = time.time() - start
        print(f"Numba JIT:         {price_numba:.4f}  ({time_numba:.3f}s)")
        print(f"  Speedup vs NumPy: {time_vec/time_numba:.2f}x")

    # JAX (if available)
    if JAX_AVAILABLE:
        # Warm-up
        key = jax.random.PRNGKey(0)
        monte_carlo_european_jax(key, S0, K, T, r, sigma, 1000, num_steps)

        start = time.time()
        price_jax = monte_carlo_european_jax(
            key, S0, K, T, r, sigma, num_sims, num_steps
        )
        time_jax = time.time() - start
        print(f"JAX:               {float(price_jax):.4f}  ({time_jax:.3f}s)")
        print(f"  Speedup vs NumPy: {time_vec/time_jax:.2f}x")

    print()


def benchmark_parallel_processing():
    """
    Compare parallel processing approaches.
    """
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    num_sims = 50000
    num_steps = 100
    n_jobs = 4

    print("=" * 70)
    print("Parallel Processing Comparison")
    print("=" * 70)
    print(f"Simulations: {num_sims:,}, Workers: {n_jobs}")
    print()

    # Sequential
    start = time.time()
    price_seq = monte_carlo_vectorized(S0, K, T, r, sigma, num_sims, num_steps)
    time_seq = time.time() - start
    print(f"Sequential:      {price_seq:.4f}  ({time_seq:.3f}s)")

    # multiprocessing
    start = time.time()
    price_mp = monte_carlo_multiprocessing(S0, K, T, r, sigma, num_sims, num_steps, n_jobs)
    time_mp = time.time() - start
    print(f"multiprocessing: {price_mp:.4f}  ({time_mp:.3f}s) - Speedup: {time_seq/time_mp:.2f}x")

    # joblib
    if JOBLIB_AVAILABLE:
        start = time.time()
        price_jl = monte_carlo_joblib(S0, K, T, r, sigma, num_sims, num_steps, n_jobs)
        time_jl = time.time() - start
        print(f"joblib:          {price_jl:.4f}  ({time_jl:.3f}s) - Speedup: {time_seq/time_jl:.2f}x")

    # dask
    if DASK_AVAILABLE:
        start = time.time()
        price_dask = monte_carlo_dask(S0, K, T, r, sigma, num_sims, num_steps)
        time_dask = time.time() - start
        print(f"dask:            {price_dask:.4f}  ({time_dask:.3f}s) - Speedup: {time_seq/time_dask:.2f}x")

    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("HIGH-PERFORMANCE NUMERICAL COMPUTING FOR FINANCE")
    print("=" * 70)
    print()

    print("Available Libraries:")
    print(f"  Numba:  {'✓' if NUMBA_AVAILABLE else '✗'}")
    print(f"  JAX:    {'✓' if JAX_AVAILABLE else '✗'}")
    print(f"  CuPy:   {'✓' if CUPY_AVAILABLE else '✗'}")
    print(f"  Joblib: {'✓' if JOBLIB_AVAILABLE else '✗'}")
    print(f"  Dask:   {'✓' if DASK_AVAILABLE else '✗'}")
    print()

    # Run benchmarks
    benchmark_monte_carlo()
    benchmark_parallel_processing()

    # Memory efficiency example
    print("=" * 70)
    print("Memory Optimization Example")
    print("=" * 70)
    optimize_dtype_usage()
    print()

    # Time series example
    print("=" * 70)
    print("GARCH Model Estimation Example")
    print("=" * 70)

    # Generate synthetic returns
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0.001, 0.02, n_days)

    # Add GARCH effects
    omega, alpha, beta = 0.00001, 0.1, 0.85
    variance = np.zeros(n_days)
    variance[0] = 0.02**2

    for t in range(1, n_days):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_normal()

    # Estimate parameters
    estimated_params = estimate_garch_parameters(returns)
    print(f"True parameters:      ω={omega:.6f}, α={alpha:.2f}, β={beta:.2f}")
    print(f"Estimated parameters: ω={estimated_params[0]:.6f}, α={estimated_params[1]:.2f}, β={estimated_params[2]:.2f}")
    print()

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
