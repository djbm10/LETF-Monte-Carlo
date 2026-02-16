import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from letf import config as cfg
from letf.utils import save_cache, load_cache


class BlockBootstrapReturns:
    """
    Generate realistic returns by sampling blocks from historical data.

    Instead of generating synthetic returns from a normal distribution,
    this class samples contiguous blocks from actual historical returns.
    This preserves:
    - Fat tails (crashes and rallies that actually happened)
    - Volatility clustering (bad days tend to follow bad days)
    - Serial correlation (momentum and mean reversion)

    Usage:
        # Initialize with historical data
        bootstrap = BlockBootstrapReturns(historical_df, block_size=20)

        # Generate returns for a simulation
        returns = bootstrap.sample_returns(n_days=252, regime_path=regime_path)
    """

    def __init__(self, df: pd.DataFrame, block_size: int = 20):
        """
        Initialize the bootstrap sampler with historical data.

        Args:
            df: DataFrame with columns 'SPY_Ret', 'VIX', and optionally
                'QQQ_Ret', 'TLT_Ret', 'Data_Source'
            block_size: Number of consecutive days in each block
        """
        self.block_size = block_size
        self.df = df.copy()

        # Separate returns by regime (inferred from VIX)
        # Low vol: VIX < 25
        # High vol: VIX >= 25
        self.regime_blocks = self._create_regime_blocks()

        print(f"  [INFO] BlockBootstrapReturns initialized:")
        print(f"     Block size: {block_size} days (max, variable length in use)")
        print(f"     Pool A (economy): Low vol={len(self.pool_a[0])}, High vol={len(self.pool_a[1])}")
        print(f"     Pool B (tech):    Low vol={len(self.pool_b[0])}, High vol={len(self.pool_b[1])}")

    def _create_regime_blocks(self) -> Dict[int, List]:
        """
        Build TWO synchronized block pools from historical data.

        Pool A (self.pool_a): 1950-2025
          4 columns: [SPY, TLT, VIX, IRX]
          Represents "The Economy" -- broad equity, bond, vol, and rate history.
          Used for SPY/SSO benchmark strategies.

        Pool B (self.pool_b): 1999-2025
          5 columns: [SPY, QQQ, TLT, VIX, IRX]
          Represents "The Tech Sector" -- real QQQ with all cross-correlations.
          Used for TQQQ and mixed strategies.

        Both pools use overlapping blocks (stride=21 days) for diversity.
        Each entry is a 2-tuple: (block_data, block_return_spy)
        where block_return_spy is the cumulative SPY return over the block,
        used for momentum-based block selection.

        Returns:
            Dict mapping regime_id -> list of pool B entries
            (kept as self.regime_blocks for backward compatibility)
        """
        # Pool A: broad history, no QQQ needed
        self.pool_a = {0: [], 1: []}

        # Pool B: tech era, all assets including real QQQ
        self.pool_b = {0: [], 1: []}

        # Extract raw arrays
        vix = self.df['VIX'].values
        spy_ret = self.df['SPY_Ret'].values
        dates = self.df.index

        if 'QQQ_Ret' in self.df.columns:
            qqq_ret = self.df['QQQ_Ret'].values
        else:
            qqq_ret = spy_ret * 1.25

        if 'TLT_Ret' in self.df.columns:
            tlt_ret = self.df['TLT_Ret'].values
        else:
            tlt_ret = spy_ret * -0.25

        if 'IRX' in self.df.columns:
            irx = self.df['IRX'].values
        else:
            irx = np.full(len(self.df), 4.5)

        # Real QQQ starts March 10, 1999
        REAL_QQQ_CUTOFF = pd.Timestamp('1999-03-10')
        has_real_qqq = dates >= REAL_QQQ_CUTOFF

        n_days = len(self.df)
        block_size = self.block_size

        # Overlapping blocks: stride of 21 days (1 month)
        BLOCK_STRIDE = 21

        pool_a_count = {0: 0, 1: 0}
        pool_b_count = {0: 0, 1: 0}

        for start_idx in range(0, n_days - block_size + 1, BLOCK_STRIDE):
            end_idx = start_idx + block_size

            # Regime from VIX majority
            block_vix = vix[start_idx:end_idx]
            regime = 0 if np.nanmedian(block_vix) < 25 else 1

            # Skip if too many NaN SPY values
            block_spy = spy_ret[start_idx:end_idx]
            if np.isnan(block_spy).sum() > block_size // 4:
                continue

            block_tlt = tlt_ret[start_idx:end_idx]
            block_irx = irx[start_idx:end_idx]

            # SPY cumulative return for momentum selection
            block_return_spy = np.prod(1 + np.nan_to_num(block_spy, nan=0)) - 1

            # ---- Pool A: 4 columns [SPY, TLT, VIX, IRX] ----
            # ALL blocks go into pool A (1950-2025)
            block_a = np.column_stack([
                np.nan_to_num(block_spy, nan=0),
                np.nan_to_num(block_tlt, nan=0),
                np.nan_to_num(block_vix, nan=20),
                np.nan_to_num(block_irx, nan=4.5)
            ])
            self.pool_a[regime].append((block_a, block_return_spy))
            pool_a_count[regime] += 1

            # ---- Pool B: 5 columns [SPY, QQQ, TLT, VIX, IRX] ----
            # Only blocks with REAL QQQ data (1999+) go into pool B
            block_has_real_qqq = has_real_qqq[start_idx:end_idx].all()

            if block_has_real_qqq:
                block_qqq = qqq_ret[start_idx:end_idx]
                block_b = np.column_stack([
                    np.nan_to_num(block_spy, nan=0),
                    np.nan_to_num(block_qqq, nan=0),
                    np.nan_to_num(block_tlt, nan=0),
                    np.nan_to_num(block_vix, nan=20),
                    np.nan_to_num(block_irx, nan=4.5)
                ])
                self.pool_b[regime].append((block_b, block_return_spy))
                pool_b_count[regime] += 1

        print(f"     Pool A (economy): Low vol={pool_a_count[0]}, High vol={pool_a_count[1]}")
        print(f"     Pool B (tech):    Low vol={pool_b_count[0]}, High vol={pool_b_count[1]}")

        # For backward compatibility, regime_blocks points to pool B
        # (sample_block draws from pool B by default for TQQQ strategies)
        return self.pool_b

    def sample_block(self, regime: int, rng: np.random.Generator = None,
                     desired_sign: int = None, momentum_bias: float = 0.0) -> np.ndarray:
        """
        Sample a single block of returns for the given regime.

        Args:
            regime: 0 for low vol, 1 for high vol
            rng: Random number generator (for reproducibility)
            desired_sign: +1 for bull, -1 for bear, None for random
            momentum_bias: probability to pick same-sign block

        Returns:
            Array of shape (block_size, 5) with columns [SPY, QQQ, TLT, VIX, IRX]
        """
        if rng is None:
            rng = np.random.default_rng()

        blocks = self.regime_blocks[regime]

        if len(blocks) == 0:
            return self._generate_synthetic_block(regime, rng)

        if desired_sign is not None and momentum_bias > 0:
            same_sign = [b for b in blocks if (b[1] >= 0 and desired_sign >= 0) or (b[1] < 0 and desired_sign < 0)]
            if same_sign and rng.random() < momentum_bias:
                return same_sign[rng.integers(0, len(same_sign))][0].copy()

        idx = rng.integers(0, len(blocks))
        return blocks[idx][0].copy()

    def _sample_from_pool(self, pool: Dict[int, List], regime: int,
                          rng: np.random.Generator,
                          desired_sign: int = None,
                          momentum_bias: float = 0.0,
                          fallback_cols: int = 5,
                          target_spy_return: float = None) -> np.ndarray:
        """
        Sample a block from a specific pool with momentum bias.
        Same logic as sample_block but works on any pool.

        Args:
            pool: Dict mapping regime_id -> list of (block_data, block_return) tuples
            regime: 0 for low vol, 1 for high vol
            rng: Random number generator
            desired_sign: +1 for bull, -1 for bear, None for random
            momentum_bias: probability to pick same-sign block
            fallback_cols: number of columns if synthetic fallback is needed
            target_spy_return: Optional SPY block return target used to keep
                Pool B (tech blocks) macro-coherent with Pool A (economy blocks)

        Returns:
            block_data array (shape varies by pool: 4 cols for pool A, 5 for pool B)
        """
        blocks = pool[regime]

        if len(blocks) == 0:
            # Fallback: generate synthetic block, trim columns if needed
            synthetic = self._generate_synthetic_block(regime, rng)
            if fallback_cols == 4:
                # Pool A format: drop QQQ column (index 1)
                return np.column_stack([synthetic[:, 0], synthetic[:, 2],
                                        synthetic[:, 3], synthetic[:, 4]])
            return synthetic

        candidate_blocks = blocks
        if desired_sign is not None:
            signed = [b for b in blocks
                      if (b[1] >= 0 and desired_sign >= 0)
                      or (b[1] < 0 and desired_sign < 0)]
            if signed and momentum_bias > 0 and rng.random() < momentum_bias:
                candidate_blocks = signed

        if target_spy_return is not None and len(candidate_blocks) > 5:
            # Keep tech blocks macro-consistent with economy blocks using
            # Gaussian kernel weighting. Blocks with similar SPY returns
            # are much more likely to be picked, but ALL blocks remain
            # eligible -- preserving genuine SPY-QQQ divergence events.
            #
            # How it works:
            #   1. Compute how far each block's SPY return is from the target
            #   2. Convert distances to probabilities using a bell curve
            #   3. Blocks near the target get high probability
            #   4. Distant blocks get low (but nonzero) probability
            #
            # The bandwidth (sigma) controls how "soft" the filter is:
            #   Small sigma -> tight filter (like the old 20% cutoff)
            #   Large sigma -> loose filter (approaches uniform random)
            #   We use the standard deviation of all block returns as sigma,
            #   which means "one sigma away = about 60% as likely as exact match"
            block_returns = np.array([b[1] for b in candidate_blocks])
            distances = block_returns - target_spy_return

            # Bandwidth = standard deviation of block returns in this pool/regime
            sigma = np.std(block_returns)
            if sigma < 1e-8:
                sigma = 0.05  # Fallback: ~5% per block period

            # Gaussian kernel: weight = exp(-0.5 * (distance/sigma)^2)
            weights = np.exp(-0.5 * (distances / sigma) ** 2)

            # Normalize to probabilities that sum to 1
            weights = weights / weights.sum()

            # Pick a block using these weighted probabilities
            chosen_idx = rng.choice(len(candidate_blocks), p=weights)
            return candidate_blocks[chosen_idx][0].copy()

        idx = rng.integers(0, len(candidate_blocks))
        return candidate_blocks[idx][0].copy()

    def _generate_synthetic_block(self, regime: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a synthetic block when no historical data is available.
        Uses Student-t distribution for fat tails.

        CALIBRATED to match historical+synthetic distribution extremes.
        """
        block_size = self.block_size

        # Regime-dependent parameters
        # CALIBRATED: Increased high-vol regime volatility to match historical extremes
        if regime == 0:  # Low vol
            daily_std = 0.011  # ~17.5% annual vol (slightly increased)
            vix_base = 15
            irx_base = 3.5     # Long-run average T-bill rate
        else:  # High vol (crisis periods)
            daily_std = 0.035  # ~55% annual vol (increased from 0.025/40%)
            vix_base = 40      # Higher VIX during crises
            irx_base = 1.5     # Fed typically cuts in crises

        # Generate fat-tailed returns using Student-t
        spy_ret = rng.standard_t(df=cfg.STUDENT_T_DF, size=block_size) * daily_std
        qqq_ret = spy_ret * 1.25  # NASDAQ more volatile
        tlt_ret = -spy_ret * 0.25  # Treasuries inversely correlated
        vix = np.full(block_size, vix_base) + rng.normal(0, 3, block_size)
        irx = np.full(block_size, irx_base) + rng.normal(0, 0.5, block_size)
        irx = np.clip(irx, 0.0, 15.0)  # Rates can't go below 0 or above 15%

        return np.column_stack([spy_ret, qqq_ret, tlt_ret, vix, irx])

    def _draw_block_len(self, remaining: int, rng: np.random.Generator) -> int:
        # Geometric length gives many short blocks, some long blocks
        p = 1.0 / cfg.BOOTSTRAP_BLOCK_MEAN
        length = rng.geometric(p)
        length = int(np.clip(length, cfg.BOOTSTRAP_BLOCK_MIN, cfg.BOOTSTRAP_BLOCK_MAX))
        return min(length, remaining)

    def sample_returns(self, n_days: int, regime_path: np.ndarray,
                       rng: np.random.Generator = None,
                       add_student_t_noise: bool = True,
                       bootstrap_weight: float = 0.85) -> Dict[str, np.ndarray]:
        """
        Generate return series using TWO synchronized block pools.

        Pool A (1950-2025, 4 cols): provides SPY, VIX, IRX
        Pool B (1999-2025, 5 cols): provides QQQ and TLT

        Two independent block-stitching loops run in parallel, each with
        its own momentum tracking. Then one shared Cholesky noise blend
        is applied across all assets to add variation and restore
        cross-asset correlation.

        Returns:
            Dict with keys 'SPY_Ret', 'QQQ_Ret', 'TLT_Ret', 'VIX', 'IRX'
        """
        if rng is None:
            rng = np.random.default_rng()
        # ================================================================
        # SYNCHRONIZED BLOCK LOOP: economy + tech sampled together
        # This keeps macro conditions (SPY/VIX/IRX) aligned with tech outcomes.
        # ================================================================
        spy_returns = np.zeros(n_days)
        vix_series = np.zeros(n_days)
        irx_series = np.zeros(n_days)
        qqq_returns = np.zeros(n_days)
        tlt_returns = np.zeros(n_days)

        current_day = 0
        last_return_a = None
        last_return_b = None

        while current_day < n_days:
            remaining = n_days - current_day
            block_len = self._draw_block_len(remaining, rng)

            block_regime_path = regime_path[current_day:current_day + block_len]
            regime = int(np.median(block_regime_path))
            bias = cfg.BOOTSTRAP_MOMENTUM_BIAS_BY_REGIME.get(regime, 0.52)

            desired_sign_a = 1 if (last_return_a is not None and last_return_a >= 0) else (-1 if last_return_a is not None else None)
            block_a = self._sample_from_pool(
                self.pool_a, regime, rng,
                desired_sign=desired_sign_a,
                momentum_bias=bias,
                fallback_cols=4
            )

            # Random sub-section (not always prefix)
            if block_len < len(block_a):
                max_start = len(block_a) - block_len
                start = rng.integers(0, max_start + 1)
                block_a = block_a[start:start + block_len]

            spy_block_return = np.prod(1 + block_a[:, 0]) - 1

            desired_sign_b = 1 if (last_return_b is not None and last_return_b >= 0) else (-1 if last_return_b is not None else None)
            block_b = self._sample_from_pool(
                self.pool_b, regime, rng,
                desired_sign=desired_sign_b,
                momentum_bias=bias,
                fallback_cols=5,
                target_spy_return=spy_block_return
            )

            if block_len < len(block_b):
                max_start = len(block_b) - block_len
                start = rng.integers(0, max_start + 1)
                block_b = block_b[start:start + block_len]

            # Pool A columns: SPY=0, TLT=1, VIX=2, IRX=3
            spy_returns[current_day:current_day + block_len] = block_a[:, 0]
            vix_series[current_day:current_day + block_len] = block_a[:, 2]
            irx_series[current_day:current_day + block_len] = block_a[:, 3]

            # Pool B columns: SPY=0, QQQ=1, TLT=2, VIX=3, IRX=4
            qqq_returns[current_day:current_day + block_len] = block_b[:, 1]
            tlt_returns[current_day:current_day + block_len] = block_b[:, 2]

            last_return_a = spy_block_return
            last_return_b = np.prod(1 + block_b[:, 1]) - 1
            current_day += block_len

        # SHARED NOISE: Cholesky-correlated Student-t
        # Applied to ALL assets together to add variation and restore
        # cross-asset correlations (especially SPY-QQQ and QQQ-TLT)
        # ================================================================
        if add_student_t_noise and bootstrap_weight < 1.0:
            noise_weight = 1.0 - bootstrap_weight

            noise_scale_spy = np.where(regime_path == 0, 0.007, 0.022)
            noise_scale_qqq = noise_scale_spy * 1.35
            noise_scale_tlt = noise_scale_spy * 0.5

            independent_noise = rng.standard_t(
                df=cfg.STUDENT_T_DF, size=(n_days, 3)
            )

            corr_low_vol = np.array([
                [1.000, 0.835, -0.207],
                [0.835, 1.000, -0.150],
                [-0.207, -0.150, 1.000]
            ])
            corr_high_vol = np.array([
                [1.000, 0.950, -0.447],
                [0.950, 1.000, -0.400],
                [-0.447, -0.400, 1.000]
            ])

            chol_low = np.linalg.cholesky(corr_low_vol)
            chol_high = np.linalg.cholesky(corr_high_vol)

            correlated_noise = np.zeros((n_days, 3))
            for t in range(n_days):
                chol = chol_low if regime_path[t] == 0 else chol_high
                correlated_noise[t] = chol @ independent_noise[t]

            spy_noise = correlated_noise[:, 0] * noise_scale_spy
            qqq_noise = correlated_noise[:, 1] * noise_scale_qqq
            tlt_noise = correlated_noise[:, 2] * noise_scale_tlt

            # Mean-preserving blend: add bootstrap mean to noise so that
            # E[blend] = w_b * mu + w_n * (0 + mu) = mu (drift preserved).
            # Without this, the zero-mean noise dilutes expected returns
            # by (1 - bootstrap_weight), costing ~1.6%/yr on SPY at w_b=0.80.
            #
            # We use the GLOBAL mean (not per-regime) because the daily mean
            # (~0.03%) is tiny vs noise scales (0.7-2.2%), so it doesn't
            # distort the noise distribution. Per-regime correction actually
            # causes worse Sharpe ratio distortion (21% vs 7%) because
            # regime-specific means are large relative to regime-specific
            # noise scales.
            spy_mean = np.mean(spy_returns)
            qqq_mean = np.mean(qqq_returns)
            tlt_mean = np.mean(tlt_returns)

            spy_returns = bootstrap_weight * spy_returns + noise_weight * (spy_noise + spy_mean)
            qqq_returns = bootstrap_weight * qqq_returns + noise_weight * (qqq_noise + qqq_mean)
            tlt_returns = bootstrap_weight * tlt_returns + noise_weight * (tlt_noise + tlt_mean)

        return {
            'SPY_Ret': spy_returns,
            'QQQ_Ret': qqq_returns,
            'TLT_Ret': tlt_returns,
            'VIX': vix_series,
            'IRX': irx_series
        }


def create_bootstrap_sampler(df: pd.DataFrame) -> BlockBootstrapReturns:
    """
    Create and cache the block bootstrap sampler.

    This should be called once with historical data, then the sampler
    can be used for all Monte Carlo simulations.
    """
    cached = load_cache(cfg.BOOTSTRAP_CACHE)
    if cached is not None:
        print("[OK] Using cached bootstrap sampler")
        return cached

    print("  Creating block bootstrap sampler from historical data...")
    sampler = BlockBootstrapReturns(df, block_size=cfg.BOOTSTRAP_BLOCK_SIZE)
    save_cache(sampler, cfg.BOOTSTRAP_CACHE)

    return sampler
