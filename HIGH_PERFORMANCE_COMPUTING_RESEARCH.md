# High-Performance Numerical Computing for Financial Applications

## Executive Summary

This research covers seven key areas of high-performance Python computing with specific applications to financial Monte Carlo simulations and time series modeling:

1. **Numba** - JIT compilation for 10-100x speedups
2. **JAX** - Automatic differentiation and GPU acceleration
3. **CuPy** - GPU-accelerated NumPy (50-100x faster)
4. **SciPy** - Robust optimization routines
5. **Vectorization** - Best practices for NumPy
6. **Memory Efficiency** - Strategies for large simulations
7. **Parallel Processing** - multiprocessing vs joblib vs dask

---

## 1. Numba: JIT Compilation for NumPy Code

### Overview
Numba is a just-in-time compiler that translates Python code to machine code using LLVM, enabling high-performance numerical computing with minimal code changes.

### Key Benefits
- **10-100x speedup** for numerical loops
- **GPU acceleration** support via CUDA
- **Automatic parallelization** with `prange`
- **Seamless NumPy integration**
- **Minimal code changes** required

### Financial Applications

#### Monte Carlo Option Pricing
```python
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def monte_carlo_option_numba(S0, K, T, r, sigma, num_sims, num_steps):
    """
    Price European call option using Monte Carlo with Numba JIT.
    Achieves 10-100x speedup over pure Python.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)
    payoffs = np.zeros(num_sims)

    # Parallel loop over simulations
    for i in prange(num_sims):
        S = S0
        for j in range(num_steps):
            z = np.random.standard_normal()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)
        payoffs[i] = max(S - K, 0.0)

    return discount * np.mean(payoffs)
```

#### GARCH(1,1) Volatility Modeling
```python
@njit
def garch_11_loglikelihood_numba(params, returns):
    """
    Calculate log-likelihood for GARCH(1,1) model.
    Used for maximum likelihood estimation of volatility parameters.
    """
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.var(returns)
    log_likelihood = 0.0

    for t in range(n):
        log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) +
                                  returns[t]**2 / sigma2)
        sigma2 = omega + alpha * returns[t]**2 + beta * sigma2

    return -log_likelihood
```

#### Vectorized Black-Scholes
```python
from numba import vectorize

@vectorize(['float64(float64, float64, float64, float64, float64)'],
           target='parallel')
def black_scholes_call_numba(S, K, T, r, sigma):
    """
    Vectorized Black-Scholes formula.
    Automatically parallelized across CPU cores.
    """
    from math import sqrt, log, exp, erf

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
```

### Performance Considerations
- **First call overhead**: JIT compilation happens on first execution
- **Parallel overhead**: Use `prange` only when loops are computationally intensive
- **GPU considerations**: Use `@cuda.jit` for GPU kernels when data is large enough

---

## 2. JAX: Automatic Differentiation and GPU Acceleration

### Overview
JAX provides automatic differentiation, GPU/TPU acceleration, and composable function transformations for high-performance numerical computing.

### Key Benefits
- **Automatic differentiation** (grad, jacobian, hessian)
- **GPU/TPU support** with minimal code changes
- **Functional programming** paradigm for reproducibility
- **vmap** for automatic vectorization
- **Composable transformations** (jit, grad, vmap)

### Financial Applications

#### Portfolio Optimization with Autodiff
```python
import jax
import jax.numpy as jnp
from jax import grad, jit

@jit
def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculate negative Sharpe ratio for optimization.
    JAX automatically computes gradients.
    """
    portfolio_return = jnp.dot(weights, returns)
    portfolio_variance = jnp.dot(weights, jnp.dot(cov_matrix, weights))
    portfolio_std = jnp.sqrt(portfolio_variance)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_std
    return -sharpe  # Negative for minimization

# Automatically compute gradient
grad_sharpe = jit(grad(portfolio_sharpe_ratio))

# Example usage
weights = jnp.array([0.3, 0.3, 0.4])
returns = jnp.array([0.08, 0.12, 0.10])
cov_matrix = jnp.array([[0.04, 0.01, 0.02],
                        [0.01, 0.06, 0.01],
                        [0.02, 0.01, 0.05]])

# Get gradients for gradient-based optimization
gradients = grad_sharpe(weights, returns, cov_matrix)
```

#### GPU-Accelerated Monte Carlo
```python
from jax import vmap
import jax.random

@jit
def monte_carlo_european_jax(key, S0, K, T, r, sigma, num_sims, num_steps):
    """
    GPU-accelerated Monte Carlo option pricing with JAX.
    Automatically runs on GPU if available.
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

# Usage
key = jax.random.PRNGKey(42)
price = monte_carlo_european_jax(key, S0=100, K=100, T=1.0,
                                 r=0.05, sigma=0.2,
                                 num_sims=100000, num_steps=252)
```

#### Implied Volatility with Newton-Raphson
```python
from jax.scipy.stats import norm

@jit
def black_scholes_call_jax(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

@jit
def implied_volatility_objective(sigma, S, K, T, r, market_price):
    """Objective function for implied volatility."""
    model_price = black_scholes_call_jax(S, K, T, r, sigma)
    return (model_price - market_price)**2

# Gradient for Newton-Raphson iteration
grad_iv = jit(grad(implied_volatility_objective))

# Newton-Raphson iteration
def calculate_implied_volatility(S, K, T, r, market_price, initial_guess=0.2):
    sigma = initial_guess
    for _ in range(10):  # Max iterations
        gradient = grad_iv(sigma, S, K, T, r, market_price)
        sigma = sigma - gradient * 0.01  # Update with learning rate
    return sigma
```

### JAX Best Practices
- **Use functional programming**: Avoid side effects
- **JIT compilation**: Use `@jit` for performance
- **Random number generation**: Use `jax.random` with explicit keys
- **GPU utilization**: JAX automatically uses GPU when available

---

## 3. CuPy: GPU-Accelerated NumPy

### Overview
CuPy is a NumPy-compatible library for GPU-accelerated computing that provides a drop-in replacement for NumPy arrays with minimal code changes.

### Key Benefits
- **Drop-in replacement** for NumPy
- **10-100x speedup** for large arrays
- **Custom CUDA kernels** support
- **Seamless CPU-GPU transfer**
- **Memory pooling** for efficiency

### Financial Applications

#### Large-Scale Covariance Calculation
```python
import cupy as cp
import numpy as np

def calculate_covariance_gpu(returns_cpu):
    """
    Calculate covariance matrix on GPU.
    50-100x faster than NumPy for large datasets.

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

# Example with 1000 assets, 5000 time periods
returns = np.random.randn(1000, 5000)
cov_matrix = calculate_covariance_gpu(returns)  # Much faster on GPU
```

#### Custom GPU Kernel for Black-Scholes
```python
# Custom element-wise kernel for Black-Scholes pricing
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

# Price 1 million options simultaneously on GPU
S = cp.random.uniform(90, 110, 1000000)
K = cp.ones(1000000) * 100
T = cp.ones(1000000) * 1.0
r = cp.ones(1000000) * 0.05
sigma = cp.ones(1000000) * 0.2

prices = black_scholes_kernel(S, K, T, r, sigma)
```

#### CPU/GPU Agnostic Code Pattern
```python
def process_data(data):
    """Works with both NumPy and CuPy arrays."""
    # Get the array module (numpy or cupy)
    xp = cp.get_array_module(data)

    # Use xp instead of np/cp
    mean = xp.mean(data)
    std = xp.std(data)
    normalized = (data - mean) / std

    result = xp.sqrt(xp.abs(normalized))
    result = xp.exp(-result ** 2)

    return result

# Works with NumPy (CPU)
cpu_data = np.random.randn(1000, 1000)
cpu_result = process_data(cpu_data)

# Works with CuPy (GPU)
gpu_data = cp.random.randn(1000, 1000)
gpu_result = process_data(gpu_data)
```

### Memory Management
```python
# Free GPU memory when done
cp.get_default_memory_pool().free_all_blocks()

# Handle out of memory errors
try:
    huge_array = cp.zeros((100000, 100000), dtype=cp.float64)
except cp.cuda.memory.OutOfMemoryError:
    print("Out of GPU memory")
    cp.get_default_memory_pool().free_all_blocks()
    huge_array = cp.zeros((10000, 10000), dtype=cp.float64)
```

---

## 4. SciPy Optimization Routines

### Overview
SciPy provides robust optimization algorithms essential for parameter estimation, portfolio optimization, and model calibration in finance.

### Key Methods
- **minimize**: General-purpose optimization
- **least_squares**: Nonlinear least squares
- **curve_fit**: Curve fitting with automatic Jacobian
- **differential_evolution**: Global optimization

### Financial Applications

#### GARCH Model Parameter Estimation
```python
from scipy.optimize import minimize
import numpy as np

def garch_11_negative_loglikelihood(params, returns):
    """
    Negative log-likelihood for GARCH(1,1) model.
    Used for maximum likelihood estimation.
    """
    omega, alpha, beta = params

    # Parameter constraints
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10  # Invalid parameters

    n = len(returns)
    variance = np.zeros(n)
    variance[0] = np.var(returns)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

    # Avoid log(0)
    variance = np.maximum(variance, 1e-10)

    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variance) + returns**2 / variance
    )

    return -log_likelihood

def estimate_garch_parameters(returns):
    """Estimate GARCH(1,1) parameters using MLE."""
    initial_params = [0.01, 0.1, 0.8]

    result = minimize(
        garch_11_negative_loglikelihood,
        initial_params,
        args=(returns,),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (0, 1), (0, 1)]
    )

    return result.x

# Example usage
returns = np.random.normal(0, 0.02, 1000)
omega, alpha, beta = estimate_garch_parameters(returns)
print(f"GARCH(1,1) parameters: ω={omega:.6f}, α={alpha:.3f}, β={beta:.3f}")
```

#### Mean-Variance Portfolio Optimization
```python
from scipy.optimize import minimize

def portfolio_optimization(expected_returns, cov_matrix, target_return=None):
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

# Example
expected_returns = np.array([0.08, 0.12, 0.10, 0.15])
cov_matrix = np.array([
    [0.04, 0.01, 0.02, 0.01],
    [0.01, 0.06, 0.01, 0.02],
    [0.02, 0.01, 0.05, 0.01],
    [0.01, 0.02, 0.01, 0.08]
])

optimal_weights = portfolio_optimization(expected_returns, cov_matrix, target_return=0.11)
print(f"Optimal weights: {optimal_weights}")
```

#### Curve Fitting for Yield Curve
```python
from scipy.optimize import curve_fit

def nelson_siegel(t, beta0, beta1, beta2, tau):
    """
    Nelson-Siegel model for yield curve fitting.
    """
    factor1 = (1 - np.exp(-t/tau)) / (t/tau)
    factor2 = factor1 - np.exp(-t/tau)
    return beta0 + beta1 * factor1 + beta2 * factor2

# Market data: maturities and yields
maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
yields = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.045])

# Fit the model
params, covariance = curve_fit(nelson_siegel, maturities, yields,
                               p0=[0.04, -0.02, 0.01, 2.0])

beta0, beta1, beta2, tau = params
print(f"Nelson-Siegel parameters: β0={beta0:.4f}, β1={beta1:.4f}, β2={beta2:.4f}, τ={tau:.4f}")
```

---

## 5. Vectorization Best Practices

### Key Principles

1. **Avoid explicit loops** - use array operations
2. **Use broadcasting** for operations on different shapes
3. **Generate all random numbers at once**
4. **Use NumPy functions** instead of Python equivalents
5. **Pre-allocate arrays** when possible

### Comparison: Slow vs Fast

#### Monte Carlo Simulation

**BAD: Slow loop-based approach**
```python
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
```

**GOOD: Vectorized approach**
```python
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
```

**Performance**: Vectorized version is typically **10-50x faster**.

### Broadcasting Examples

#### Pairwise Correlation Matrix
```python
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
```

#### Portfolio Metrics for Multiple Scenarios
```python
def calculate_portfolio_metrics_vectorized(weights, returns_scenarios, cov_matrix):
    """
    Calculate portfolio metrics for multiple weight scenarios using broadcasting.

    Parameters:
    -----------
    weights : array (n_scenarios, n_assets)
    returns_scenarios : array (n_assets,)
    cov_matrix : array (n_assets, n_assets)

    Returns:
    --------
    portfolio_returns, portfolio_volatilities
    """
    # Vectorized return calculation: (n_scenarios,)
    portfolio_returns = np.dot(weights, returns_scenarios)

    # Vectorized variance calculation: (n_scenarios,)
    portfolio_variance = np.sum(weights @ cov_matrix * weights, axis=1)
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_returns, portfolio_volatility
```

---

## 6. Memory Efficiency for Large Simulations

### Key Strategies

1. **Use appropriate dtypes** (float32 vs float64)
2. **Process data in chunks**
3. **Use generators** for large datasets
4. **Delete intermediate arrays**
5. **Use memory-mapped arrays** for huge datasets
6. **Leverage in-place operations**

### Data Type Selection

```python
def optimize_dtype_usage():
    """Demonstrate memory savings with appropriate dtypes."""
    n = 10_000_000

    # float64 (default) - 8 bytes per element
    arr_float64 = np.random.randn(n)
    memory_float64 = arr_float64.nbytes / 1024**2  # MB

    # float32 - 4 bytes per element
    arr_float32 = np.random.randn(n).astype(np.float32)
    memory_float32 = arr_float32.nbytes / 1024**2  # MB

    print(f"float64: {memory_float64:.2f} MB")
    print(f"float32: {memory_float32:.2f} MB")
    print(f"Savings: {100*(1-memory_float32/memory_float64):.1f}%")

# Output:
# float64: 76.29 MB
# float32: 38.15 MB
# Savings: 50.0%
```

### Chunked Processing

```python
def monte_carlo_chunked(S0, K, T, r, sigma, total_sims, chunk_size, num_steps):
    """
    Memory-efficient Monte Carlo using chunks.
    Processes large simulations without consuming excessive memory.
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

# Example: 10 million simulations in chunks of 100k
price = monte_carlo_chunked(100, 100, 1.0, 0.05, 0.2,
                           total_sims=10_000_000,
                           chunk_size=100_000,
                           num_steps=252)
```

### Generator-Based Simulation

```python
def simulation_generator(S0, K, T, r, sigma, num_steps, batch_size=10000):
    """
    Generator that yields batches of simulated paths.
    Useful for streaming large-scale simulations.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    while True:
        z = np.random.standard_normal((batch_size, num_steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        S_T = S0 * np.exp(np.sum(log_returns, axis=1))
        payoffs = np.maximum(S_T - K, 0)

        yield payoffs

# Usage
gen = simulation_generator(100, 100, 1.0, 0.05, 0.2, 252)
total_payoff = 0
n_batches = 100

for _ in range(n_batches):
    batch_payoffs = next(gen)
    total_payoff += np.sum(batch_payoffs)

option_price = np.exp(-0.05 * 1.0) * total_payoff / (n_batches * 10000)
```

### In-Place Operations

```python
# BAD: Creates new arrays
def rebalance_bad(weights, returns):
    weights_new = weights * (1 + returns)
    weights_normalized = weights_new / np.sum(weights_new)
    return weights_normalized

# GOOD: In-place operations
def rebalance_good(weights, returns):
    weights *= (1 + returns)  # In-place multiplication
    weights /= np.sum(weights)  # In-place normalization
    return weights
```

### Broadcasting Efficiency

Broadcasting allows NumPy to perform operations on arrays of different shapes without creating large temporary arrays, minimizing memory overhead.

**Source**: [Mastering Memory Efficiency with NumPy Arrays](https://blog.muhammad-ahmed.com/2025/02/25/mastering-memory-efficiency-with-numpy-arrays-in-python/)

---

## 7. Parallel Processing Comparison

### Overview of Approaches

| Library | Best For | Limitations | Speedup |
|---------|----------|-------------|---------|
| **multiprocessing** | CPU-bound tasks on single machine | Communication overhead, pickling | 2-4x |
| **joblib** | Embarrassingly parallel tasks, NumPy | Single machine only | 2-4x |
| **dask** | Large datasets, multi-node clusters | Higher overhead for small tasks | Variable |

### Detailed Comparison

#### 1. multiprocessing (Built-in)

**Pros:**
- Built into Python standard library
- True parallelism (bypasses GIL)
- Good for CPU-bound tasks

**Cons:**
- Communication overhead
- Pickling issues with complex objects
- Higher memory usage

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def simulate_single_path(seed, S0, K, T, r, sigma, num_steps):
    """Single path simulation for parallel execution."""
    np.random.seed(seed)
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    z = np.random.standard_normal(num_steps)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
    S_T = S0 * np.exp(np.sum(log_returns))

    return max(S_T - K, 0)

def monte_carlo_multiprocessing(S0, K, T, r, sigma, num_sims, num_steps, n_jobs=4):
    """Monte Carlo using multiprocessing."""
    seeds = np.random.randint(0, 10000000, num_sims)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        payoffs = list(executor.map(
            simulate_single_path,
            seeds,
            [S0]*num_sims, [K]*num_sims, [T]*num_sims,
            [r]*num_sims, [sigma]*num_sims, [num_steps]*num_sims
        ))

    return np.exp(-r * T) * np.mean(payoffs)
```

#### 2. joblib (Data Science Optimized)

**Pros:**
- Optimized for NumPy arrays
- Memory mapping for large arrays
- Transparent disk caching
- Consistent performance across dataset sizes

**Cons:**
- Single machine only
- Additional dependency

```python
from joblib import Parallel, delayed

def monte_carlo_joblib(S0, K, T, r, sigma, num_sims, num_steps, n_jobs=4):
    """Monte Carlo using joblib (optimized for NumPy)."""
    seeds = np.random.randint(0, 10000000, num_sims)

    payoffs = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(simulate_single_path)(seed, S0, K, T, r, sigma, num_steps)
        for seed in seeds
    )

    return np.exp(-r * T) * np.mean(payoffs)
```

**Performance**: Joblib performs well across all dataset sizes with low and consistent runtime.

#### 3. dask (Distributed Computing)

**Pros:**
- Scales to multi-node clusters
- Handles larger-than-memory computations
- Seamless scaling from laptop to cluster
- Centralized scheduling

**Cons:**
- Higher overhead for small tasks
- More complex setup
- Longer runtime for same dataset sizes compared to joblib

```python
import dask
from dask import delayed as dask_delayed

def monte_carlo_dask(S0, K, T, r, sigma, num_sims, num_steps):
    """Monte Carlo using dask for distributed computing."""
    seeds = np.random.randint(0, 10000000, num_sims)

    # Create delayed computations
    delayed_payoffs = [
        dask_delayed(simulate_single_path)(seed, S0, K, T, r, sigma, num_steps)
        for seed in seeds
    ]

    # Compute in parallel
    payoffs = dask.compute(*delayed_payoffs)

    return np.exp(-r * T) * np.mean(payoffs)
```

#### Dask for Large-Scale Covariance

```python
import dask.array as da

def large_covariance_dask(returns_large):
    """
    Calculate covariance for very large datasets using dask.
    Handles datasets larger than memory.
    """
    # Convert to dask array (lazy evaluation)
    returns_dask = da.from_array(returns_large, chunks=(100, -1))

    # Calculate covariance (computed lazily)
    cov_dask = da.cov(returns_dask)

    # Trigger computation
    return cov_dask.compute()
```

### When to Use Each

- **multiprocessing**: CPU-bound tasks on single machine, simple parallelization needs
- **joblib**: Data science workflows, NumPy-heavy computations, embarrassingly parallel tasks
- **dask**: Large datasets (especially those that don't fit in memory), multi-node clusters, complex task graphs

**Sources**:
- [The best Python libraries for parallel processing | InfoWorld](https://www.infoworld.com/article/2257768/the-best-python-libraries-for-parallel-processing.html)
- [Parallel Processing with Joblib and Dask: A Performance Comparison | Medium](https://medium.com/@kamol.roy08/parallel-processing-with-joblib-and-dask-a-performance-comparison-fe0aed4b5337)
- [How to Process Datasets with Parallel Jobs in Python](https://oneuptime.com/blog/post/2026-01-23-parallel-dataset-processing-python/view)

---

## Financial Time Series Modeling

### GARCH Models for Volatility

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models focus on modeling the variance or volatility of time series, while ARIMA models focus on modeling the mean.

#### Combined ARIMA-GARCH Approach

```python
from scipy.optimize import minimize
import numpy as np

# Step 1: Fit ARIMA model for mean
def fit_arima(returns, p=1, d=0, q=1):
    """
    Simplified ARIMA fitting.
    In practice, use statsmodels.tsa.arima.model.ARIMA
    """
    # Placeholder - use statsmodels in production
    return returns - np.mean(returns)

# Step 2: Fit GARCH model to residuals
def garch_volatility_forecast(returns, omega, alpha, beta, horizon=1):
    """
    Forecast volatility using GARCH(1,1) model.

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
    """
    n = len(returns)
    variance = np.zeros(n + horizon)
    variance[0] = np.var(returns)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

    # Forecast
    for h in range(horizon):
        variance[n + h] = omega + (alpha + beta) * variance[n + h - 1]

    return np.sqrt(variance)

# Example usage
returns = np.random.normal(0, 0.02, 1000)
omega, alpha, beta = 0.00001, 0.1, 0.85

volatility_forecast = garch_volatility_forecast(returns, omega, alpha, beta, horizon=5)
print(f"5-day volatility forecast: {volatility_forecast[-5:]}")
```

#### Libraries for GARCH Modeling

- **arch**: Popular library for GARCH models in Python
- **statsmodels**: ARIMA and statistical models
- **pmdarima**: Auto-ARIMA for automatic parameter selection

```python
# Using arch library (pip install arch)
from arch import arch_model

# Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
results = model.fit()

# Get parameters
print(results.params)

# Forecast volatility
forecast = results.forecast(horizon=5)
print(forecast.variance[-1:])
```

**Sources**:
- [The Volatility Odyssey: A Journey Through Time Series Models with Python | Medium](https://medium.com/@cemalozturk/exploring-financial-volatility-with-garch-model-a-step-by-step-guide-with-python-11c36594bd9a)
- [GARCH Models for Volatility Forecasting: A Python-Based Guide | Medium](https://theaiquant.medium.com/garch-models-for-volatility-forecasting-a-python-based-guide-d48deb5c7d7b)
- [ARIMA-GARCH forecasting with Python | Medium](https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff)

---

## Complete Monte Carlo Example

### Comprehensive Implementation

```python
import numpy as np
import time

class MonteCarloEngine:
    """
    Production-ready Monte Carlo engine with multiple backends.
    """

    def __init__(self, backend='numpy'):
        """
        Initialize engine with specified backend.

        Parameters:
        -----------
        backend : str - 'numpy', 'numba', 'jax', or 'cupy'
        """
        self.backend = backend

    def price_european_option(self, S0, K, T, r, sigma, num_sims, num_steps):
        """
        Price European call option using selected backend.
        """
        if self.backend == 'numpy':
            return self._price_numpy(S0, K, T, r, sigma, num_sims, num_steps)
        elif self.backend == 'numba':
            return self._price_numba(S0, K, T, r, sigma, num_sims, num_steps)
        # Add other backends as needed

    def _price_numpy(self, S0, K, T, r, sigma, num_sims, num_steps):
        """Vectorized NumPy implementation."""
        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)

        z = np.random.standard_normal((num_sims, num_steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        S_T = S0 * np.exp(np.sum(log_returns, axis=1))
        payoffs = np.maximum(S_T - K, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    def calculate_greeks(self, S0, K, T, r, sigma, num_sims, num_steps):
        """
        Calculate option Greeks using finite differences.
        """
        # Price at current spot
        price = self.price_european_option(S0, K, T, r, sigma, num_sims, num_steps)

        # Delta: ∂V/∂S
        dS = 0.01 * S0
        price_up = self.price_european_option(S0 + dS, K, T, r, sigma, num_sims, num_steps)
        price_down = self.price_european_option(S0 - dS, K, T, r, sigma, num_sims, num_steps)
        delta = (price_up - price_down) / (2 * dS)

        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2*price + price_down) / (dS**2)

        # Vega: ∂V/∂σ
        dsigma = 0.01
        price_vol_up = self.price_european_option(S0, K, T, r, sigma + dsigma, num_sims, num_steps)
        vega = (price_vol_up - price) / dsigma

        # Theta: ∂V/∂t
        dT = 1/365
        price_time = self.price_european_option(S0, K, T - dT, r, sigma, num_sims, num_steps)
        theta = (price_time - price) / dT

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }

# Example usage
engine = MonteCarloEngine(backend='numpy')

# Price option
price = engine.price_european_option(
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    num_sims=100000, num_steps=252
)
print(f"Option price: ${price:.4f}")

# Calculate Greeks
greeks = engine.calculate_greeks(
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    num_sims=50000, num_steps=252
)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

---

## Performance Benchmarks

### Expected Speedups

| Method | Typical Speedup | Best For |
|--------|----------------|----------|
| **Pure Python** | Baseline (1x) | Small computations |
| **NumPy Vectorized** | 10-50x | Array operations |
| **Numba JIT** | 10-100x | Numerical loops |
| **JAX (CPU)** | 5-20x | Autodiff needs |
| **JAX (GPU)** | 20-100x | Large parallel tasks |
| **CuPy (GPU)** | 50-100x | Large arrays |
| **Multiprocessing** | 2-4x | CPU-bound tasks |
| **Joblib** | 2-4x | Embarrassingly parallel |
| **Dask** | Variable | Large datasets |

### Benchmark Example

```python
def run_benchmark():
    """Compare different implementations."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    num_sims = 100000
    num_steps = 252

    print("Monte Carlo Option Pricing Benchmark")
    print("=" * 60)

    # NumPy
    start = time.time()
    price_np = monte_carlo_vectorized(S0, K, T, r, sigma, num_sims, num_steps)
    time_np = time.time() - start
    print(f"NumPy:  {price_np:.4f} ({time_np:.3f}s)")

    # Numba (if available)
    if NUMBA_AVAILABLE:
        monte_carlo_option_numba(S0, K, T, r, sigma, 1000, num_steps)  # warm-up
        start = time.time()
        price_numba = monte_carlo_option_numba(S0, K, T, r, sigma, num_sims, num_steps)
        time_numba = time.time() - start
        print(f"Numba:  {price_numba:.4f} ({time_numba:.3f}s) - {time_np/time_numba:.1f}x faster")
```

---

## Recommendations

### For Financial Monte Carlo Simulations

1. **Start with vectorized NumPy** for baseline implementation
2. **Use Numba** for loops that can't be vectorized (e.g., path-dependent options)
3. **Consider JAX** if you need automatic differentiation for Greeks or calibration
4. **Use CuPy** for very large-scale simulations (millions of paths)
5. **Apply chunking** to manage memory for billion-scale simulations

### For Time Series Modeling

1. **Use SciPy optimize** for GARCH/ARIMA parameter estimation
2. **Consider Numba** for custom time series models with complex logic
3. **Use vectorization** for rolling window calculations
4. **Apply JAX** for models requiring gradients (e.g., neural network forecasting)

### For Large-Scale Computations

1. **Start with joblib** for simple parallelization
2. **Use dask** when data exceeds memory
3. **Consider GPU (CuPy/JAX)** for matrix-heavy operations
4. **Profile first** - measure before optimizing

---

## Installation Guide

```bash
# Core libraries
pip install numpy scipy

# JIT compilation
pip install numba

# GPU computing (choose based on CUDA version)
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install jax[cuda12]   # For JAX with CUDA

# Parallel processing
pip install joblib dask[complete]

# Financial time series
pip install arch statsmodels pmdarima

# Optional: visualization
pip install matplotlib seaborn
```

---

## Sources and References

1. [Numba Documentation](https://numba.readthedocs.io/en/stable/)
2. [JAX Documentation](https://github.com/jax-ml/jax)
3. [CuPy Documentation](https://cupy.dev/)
4. [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
5. [The best Python libraries for parallel processing | InfoWorld](https://www.infoworld.com/article/2257768/the-best-python-libraries-for-parallel-processing.html)
6. [Parallel Processing with Joblib and Dask: A Performance Comparison | Medium](https://medium.com/@kamol.roy08/parallel-processing-with-joblib-and-dask-a-performance-comparison-fe0aed4b5337)
7. [How to Process Datasets with Parallel Jobs in Python](https://oneuptime.com/blog/post/2026-01-23-parallel-dataset-processing-python/view)
8. [Monte Carlo Simulation with Python - Practical Business Python](https://pbpython.com/monte-carlo.html)
9. [Mastering Memory Efficiency with NumPy Arrays](https://blog.muhammad-ahmed.com/2025/02/25/mastering-memory-efficiency-with-numpy-arrays-in-python/)
10. [Optimizing Memory Usage with NumPy Arrays - KDnuggets](https://www.kdnuggets.com/optimizing-memory-usage-with-numpy-arrays)
11. [GARCH Models for Volatility Forecasting | Medium](https://theaiquant.medium.com/garch-models-for-volatility-forecasting-a-python-based-guide-d48deb5c7d7b)
12. [ARIMA-GARCH forecasting with Python | Medium](https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff)
13. [GitHub - Vectorization in Monte Carlo](https://github.com/ymh1989/Vectorization)
14. [Master Monte Carlo: Python Simulations with NumPy](https://codepointtech.com/master-monte-carlo-python-simulations-with-numpy/)

---

## Conclusion

High-performance numerical computing in Python offers multiple paths to optimization:

- **Numba** provides the easiest path to significant speedups with minimal code changes
- **JAX** excels when automatic differentiation is needed
- **CuPy** delivers maximum performance for large-scale array operations on GPU
- **SciPy** remains the gold standard for optimization and scientific computing
- **Proper vectorization** is the foundation of all high-performance Python code
- **Memory efficiency** is critical for scaling to production-level simulations
- **Choose the right parallelization** approach based on your specific needs

For financial applications, the combination of vectorized NumPy, Numba for custom loops, and SciPy for optimization provides an excellent balance of performance and maintainability.
