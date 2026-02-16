import numpy as np
import pandas as pd
from typing import Dict, Optional
from letf import config as cfg
from letf.utils import infer_regime_from_vix


def compute_transaction_costs(daily_ret, regime, leverage, trade_size_pct=0.0):
    """
    Enhanced transaction costs with regime-dependent slippage.

    Args:
        daily_ret: Daily return
        regime: Market regime (0=normal, 1=high vol, 2=crisis)
        leverage: Leverage ratio
        trade_size_pct: Trade size as % of portfolio (0-1)

    Returns:
        Total cost as decimal (e.g., 0.001 = 10 bps)
    """
    # Base bid-ask spread
    spread_bps = cfg.BASE_SPREAD_BPS[regime]
    spread_cost = spread_bps / 10000

    # Rebalancing cost (internal fund rebalancing)
    rebalance_cost = cfg.REBALANCE_COST_PER_DOLLAR * leverage * abs(daily_ret)

    # ========================================================================
    # MARKET IMPACT / SLIPPAGE (Regime-Dependent)
    # ========================================================================
    # Large trades in illiquid regimes have significant market impact
    # Uses square-root model with regime multipliers

    if trade_size_pct > 0.01:  # Only apply to trades >1% of portfolio
        # Regime multipliers for slippage
        regime_multiplier = {
            0: 1.0,   # Normal market - standard liquidity
            1: 2.0,   # High vol - wider spreads, less liquidity
            2: 4.0    # Crisis - extreme illiquidity, flash crashes
        }[regime]

        # Square-root scaling for market impact
        # Larger trades have disproportionate impact
        size_multiplier = 1 + np.sqrt(trade_size_pct) * 2

        # Additional slippage
        market_impact = spread_cost * (regime_multiplier - 1) * (size_multiplier - 1)
    else:
        market_impact = 0

    total_cost = spread_cost + rebalance_cost + market_impact

    return total_cost

def run_strategy_fixed(df, strategy_id, regime_path, correlation_matrices,
                       apply_costs=True, trade_journal=None):
    """
    FIX #4: Run strategy with LEVERAGE DRIFT TRACKING for portfolios.
    FIX BUG: Handle regime_path mismatch by inferring from VIX if needed.
    """
    # ========================================================================
    # BUG FIX: Handle regime path mismatch between Sim vs Historical
    # ========================================================================
    if regime_path is None or len(regime_path) != len(df):
        if 'VIX' in df.columns:
            # Infer regime from probabilistic stress model (same logic as calibration)
            realized_vol = df['SPY_Ret'].rolling(20, min_periods=5).std().bfill().fillna(0) * np.sqrt(252)
            term_spread = None
            if 'TNX' in df.columns and 'IRX' in df.columns:
                term_spread = (df['TNX'] - df['IRX']).values
            regime_path = infer_regime_from_vix(
                vix_series=df['VIX'].values,
                realized_vol=realized_vol.values,
                term_spread=term_spread
            )
        else:
            # Fallback if no VIX
            regime_path = np.zeros(len(df), dtype=int)

    config = cfg.STRATEGIES[strategy_id]
    strategy_type = config['type']
    num_trades = 0

    # Benchmark strategies
    if strategy_type == 'benchmark':
        asset = config['asset']
        ret_col = f'{asset}_Ret'

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        returns = df[ret_col].fillna(0)
        equity_curve = cfg.INITIAL_CAPITAL * (1 + returns).cumprod()

        return equity_curve, 0

    # SMA strategies
    if strategy_type == 'sma' or strategy_type == 'sma_band':
        asset = config['asset']
        ret_col = f'{asset}_Ret'

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        sma_period = config.get('sma_period', 200)

        position = pd.Series(0, index=df.index, dtype=int)
        spy_price_prev = df['SPY_Price'].shift(1)
        sma_prev = df['SPY_Price'].rolling(sma_period, min_periods=1).mean().shift(1)

        if strategy_type == 'sma':
            buy_signal = spy_price_prev >= sma_prev
            sell_signal = spy_price_prev < sma_prev
        else:
            band = config.get('band', 0.02)
            buy_signal = spy_price_prev >= sma_prev * (1 - band)
            sell_signal = spy_price_prev < sma_prev * (1 - band)

        buy_signal = buy_signal.fillna(False)
        sell_signal = sell_signal.fillna(False)

        for i in range(1, len(df)):
            if position.iloc[i-1] == 0:
                position.iloc[i] = 1 if buy_signal.iloc[i] else 0
            else:
                position.iloc[i] = 0 if sell_signal.iloc[i] else 1

        position_changes = position.diff().abs()
        num_trades = int(position_changes.sum())

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        target_leverage = cfg.ASSETS[asset]['leverage']

        for i in range(1, len(df)):
            if position.iloc[i] == 1:
                ret = df[ret_col].iloc[i]
            else:
                ret = df['Cash_Ret'].iloc[i]

            if apply_costs and position_changes.iloc[i] > 0:
                # FIX: regime_path is now guaranteed to match len(df)
                regime = regime_path[i]
                cost = compute_transaction_costs(
                    df[ret_col].iloc[i],
                    regime,
                    target_leverage
                )
                ret -= cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # FIX #4: Portfolio strategies with LEVERAGE DRIFT TRACKING
    if strategy_type == 'portfolio':
        assets_weights = config['assets']
        rebalance_freq = config.get('rebalance_freq', 21)

        # Track individual LETF positions AND their embedded leverage
        positions = {asset: cfg.INITIAL_CAPITAL * weight
                    for asset, weight in assets_weights.items()}

        # Track embedded leverage of each position
        # (leverage drifts as underlying moves)
        embedded_leverage = {asset: cfg.ASSETS[asset]['leverage']
                            for asset in assets_weights.keys()}

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        for i in range(1, len(df)):
            # Update each position (value changes, leverage drifts)
            total_value_before = sum(positions.values())

            for asset in assets_weights.keys():
                ret_col = f'{asset}_Ret'
                if ret_col in df.columns:
                    ret = df[ret_col].iloc[i]

                    # Position value changes
                    old_value = positions[asset]
                    new_value = old_value * (1 + ret)
                    positions[asset] = new_value

                    # Embedded leverage drifts
                    # If underlying moves r%, embedded leverage becomes L*(1+r)/(1+L*r)
                    # This is approximate - good enough for simulation
                    target_leverage = cfg.ASSETS[asset]['leverage']
                    if target_leverage > 1.0:
                        # Simplified leverage drift (exact formula is complex)
                        underlying_ret = ret / target_leverage  # Approximate
                        if abs(1 + target_leverage * underlying_ret) > 0.01:
                            embedded_leverage[asset] = target_leverage * (1 + underlying_ret) / (1 + target_leverage * underlying_ret)
                        else:
                            embedded_leverage[asset] = target_leverage
                    else:
                        embedded_leverage[asset] = 1.0

            total_value = sum(positions.values())
            equity_curve.iloc[i] = total_value

            # Rebalance
            if i % rebalance_freq == 0:
                # Current weights
                current_weights = {asset: positions[asset] / total_value
                                 for asset in assets_weights.keys()}

                # Turnover (weight changes)
                weight_turnover = sum(abs(current_weights[asset] - assets_weights[asset])
                                     for asset in assets_weights.keys())

                # ADDITIONAL: Leverage drift turnover
                # If embedded leverage has drifted, we need to trade to bring it back
                leverage_turnover = 0
                for asset in assets_weights.keys():
                    target_leverage = cfg.ASSETS[asset]['leverage']
                    current_leverage = embedded_leverage[asset]
                    leverage_drift = abs(current_leverage - target_leverage) / target_leverage
                    leverage_turnover += leverage_drift * current_weights[asset]

                total_turnover = weight_turnover + leverage_turnover

                # Apply rebalancing costs
                if apply_costs and total_turnover > 0.01:
                    # FIX: regime_path is now guaranteed to match len(df)
                    regime = regime_path[i]

                    # Cost scales with turnover
                    rebal_cost = total_turnover * cfg.REBALANCE_COST_PER_DOLLAR * total_value
                    total_value -= rebal_cost
                    equity_curve.iloc[i] = total_value

                # Reset to target weights AND target leverage
                positions = {asset: total_value * weight
                           for asset, weight in assets_weights.items()}

                embedded_leverage = {asset: cfg.ASSETS[asset]['leverage']
                                   for asset in assets_weights.keys()}

                num_trades += len(assets_weights)

        return equity_curve, num_trades

    # Vol targeting
    if strategy_type == 'vol_targeting':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        target_vol = config['target_vol']
        lookback = config.get('lookback', 20)

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        realized_vol = df[ret_col].rolling(lookback).std() * np.sqrt(252)

        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            current_vol = realized_vol.iloc[i]
            if pd.isna(current_vol) or current_vol < 0.01:
                position_size = 1.0
            else:
                position_size = target_vol / current_vol
                position_size = np.clip(position_size, 0.2, 2.0)

            # Track turnover - count EVERY change as a trade
            turnover = abs(position_size - prev_alloc)
            if turnover > 0.0001:  # Any meaningful change (>0.01%)
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=position_size,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA transaction costs (bid-ask spread only)
            # TQQQ typical spread: ~0.03% (3 bps)
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = position_size

        # Calculate return
            ret = df[ret_col].iloc[i] * position_size

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 8: THE OPTIMIZER (Regime-Based Composite)
    # -----------------------------------------------------------------------------
    if strategy_type == 'composite':
        risky_asset = config['asset']
        safe_asset = config['defensive_asset']

        sma_p = config['sma_period']
        rsi_p = config['rsi_period']
        vix_th = config['vix_threshold']

        # Calculate indicators
        ref_price = df['SPY_Price']
        sma = ref_price.rolling(sma_p).mean()

        # RSI Calculation
        delta = df['SPY_Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        curr_pos = 'CASH' # CASH, SAFE, RISKY

        for i in range(1, len(df)):
            # Get signals from PREVIOUS day (to avoid lookahead bias)
            curr_price = ref_price.iloc[i-1]
            curr_sma = sma.iloc[i-1]
            curr_rsi = rsi.iloc[i-1]
            curr_vix = df['VIX'].iloc[i-1]

            score = 0
            # Signal 1: Trend
            if curr_price > curr_sma: score += 1
            # Signal 2: Momentum (Not overbought, not oversold crash)
            if 40 < curr_rsi < 80: score += 1
            # Signal 3: Volatility Regime
            if curr_vix < vix_th: score += 1

            # Allocation Logic
            ret = 0
            target = 'CASH'

            if score == 3:
                # Full Bull: All in Risky Leveraged
                ret = df[f'{risky_asset}_Ret'].iloc[i]
                target = 'RISKY'
            elif score == 2:
                # Uncertainty: Defensive (SPY or 1x)
                ret = df[f'{safe_asset}_Ret'].iloc[i]
                target = 'SAFE'
            else:
                # Bear/Crash: Cash
                ret = df['Cash_Ret'].iloc[i]
                target = 'CASH'

            if target != curr_pos:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    position_map = {'RISKY': (risky_asset, 1.0), 'SAFE': (safe_asset, 1.0), 'CASH': ('SPY', 0.0)}
                    prev_map = {'RISKY': (risky_asset, 1.0), 'SAFE': (safe_asset, 1.0), 'CASH': ('SPY', 0.0)}

                    trade_asset, new_alloc = position_map.get(target, ('SPY', 0.0))
                    _, prev_alloc_val = prev_map.get(curr_pos, ('SPY', 0.0))

                    asset_price = df[f'{trade_asset}_Price'].iloc[i] if f'{trade_asset}_Price' in df.columns else 100.0
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=trade_asset,
                        prev_allocation=prev_alloc_val,
                        new_allocation=new_alloc,
                        portfolio_value=equity_curve.iloc[i-1],
                        price=asset_price
                    )

                curr_pos = target

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades
    # -----------------------------------------------------------------------------
    # STRATEGY 9: TREND-ADAPTIVE VOL TARGETING (The New Challenger)
    # -----------------------------------------------------------------------------
    if strategy_type == 'adaptive_vol':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        bull_vol = config['bull_target']
        bear_vol = config['bear_target']
        lookback = config['lookback']
        sma_period = config['sma_period']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        # Calculate Realized Volatility (Annualized)
        realized_vol = df[ret_col].rolling(lookback).std().shift(1) * np.sqrt(252)

        # Calculate Trend Signal
        ref_price = df['SPY_Price']
        sma = ref_price.rolling(sma_period).mean().shift(1)

        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            curr_vol = realized_vol.iloc[i]
            curr_price = ref_price.iloc[i-1]
            curr_sma = sma.iloc[i]

            # Skip if data not ready
            if pd.isna(curr_vol) or pd.isna(curr_sma) or curr_vol < 0.001:
                equity_curve.iloc[i] = equity_curve.iloc[i-1]
                continue

            # Determine Regime
            is_bull = curr_price > curr_sma
            target_vol = bull_vol if is_bull else bear_vol

            # Calculate Allocation
            alloc = target_vol / curr_vol
            alloc = np.clip(alloc, 0.0, 1.0)

            # Track turnover - count EVERY change as a trade
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:  # Any meaningful change
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA transaction costs (bid-ask spread)
            # TQQQ typical spread: ~0.03% (3 bps)
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate Return
            r_strat = (alloc * df[ret_col].iloc[i]) + \
                    ((1 - alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                r_strat -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + r_strat)

        return equity_curve, num_trades
    # -----------------------------------------------------------------------------
    # STRATEGY 10: SORTINO-OPTIMIZED TQQQ (Downside Vol Targeting)
    # -----------------------------------------------------------------------------
    if strategy_type == 'downside_vol':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        target = config['target_downside_vol']
        lookback = config['lookback']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

    # Calculate Rolling Downside Volatility
        neg_rets = df[ret_col].where(df[ret_col] < 0, 0)
        downside_vol = neg_rets.rolling(lookback).std().shift(1) * np.sqrt(252)

        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            d_vol = downside_vol.iloc[i]

            if pd.isna(d_vol) or d_vol < 0.001:
                alloc = 1.0
            else:
                alloc = target / d_vol
                alloc = np.clip(alloc, 0.0, 1.5)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs (bid-ask spread)
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            r_strat = (alloc * df[ret_col].iloc[i]) + \
                    ((1 - alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                r_strat -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + r_strat)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 11: HYPER-CONVEX VOL SQUEEZER (Maximizer)
    # -----------------------------------------------------------------------------
    if strategy_type == 'convex_vol':
        asset = config['asset']
        target = config['target_vol']
        p_val = config['power']
        sma_p = config['sma_period']

        real_vol = df[f'{asset}_Ret'].rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)
        sma = df['SPY_Price'].rolling(sma_p, min_periods=1).mean().shift(1)

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            v = real_vol.iloc[i]
            if pd.isna(v) or v < 0.001: v = 0.5

        # Linear Allocation
            alloc = target / v

        # Convex Boost if in uptrend
            if df['SPY_Price'].iloc[i-1] > sma.iloc[i]:
                alloc = pow(alloc, p_val)

            alloc = np.clip(alloc, 0.0, 1.0)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[f'{asset}_Ret'].iloc[i]) + ((1-alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades
    # -----------------------------------------------------------------------------
    # STRATEGY 12: VOL-VELOCITY ENSEMBLE (Optimizer)
    # -----------------------------------------------------------------------------
    if strategy_type == 'vol_velocity':
        asset = config['asset']
        target = config['target_vol']

    # Fast (5d) vs Slow (20d) Volatility
        vol_slow = df[f'{asset}_Ret'].rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_fast = df[f'{asset}_Ret'].rolling(5, min_periods=1).std().shift(1) * np.sqrt(252)

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
        # Use the MAX of the two vols (defensive stance)
            effective_vol = max(vol_slow.iloc[i], vol_fast.iloc[i])

            if pd.isna(effective_vol) or effective_vol < 0.001: effective_vol = 0.5

            alloc = np.clip(target / effective_vol, 0.0, 1.0)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[f'{asset}_Ret'].iloc[i]) + ((1-alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades
    # -----------------------------------------------------------------------------
    # STRATEGY 13: VOL-OF-VOL MOMENTUM (The Anticipator)
    # -----------------------------------------------------------------------------
    if strategy_type == 'vol_mom':
        asset = config['asset']
        target = config['target_vol']
        vol_slow = df[f'{asset}_Ret'].rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_mom = vol_slow.pct_change(5)

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            v = vol_slow.iloc[i]
            vm = vol_mom.iloc[i]
            if pd.isna(v) or v < 0.001: v = 0.5

        # Base alloc
            alloc = target / v

        # Anticipation adjustments
            if pd.notna(vm) and vm < -0.10: alloc *= 1.2
            if pd.notna(vm) and vm > 0.10:  alloc *= 0.7

            alloc = np.clip(alloc, 0.0, 1.0)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[f'{asset}_Ret'].iloc[i]) + ((1-alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 14: SKEWNESS-ADJUSTED CONVEX (The Specialist)
    # -----------------------------------------------------------------------------
    if strategy_type == 'skew_convex':
        asset = config['asset']
        target = config['target_vol']
        skew = df[f'{asset}_Ret'].rolling(60, min_periods=1).skew().shift(1)
        real_vol = df[f'{asset}_Ret'].rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            v = real_vol.iloc[i]
            s = skew.iloc[i]
            if pd.isna(v) or v < 0.001: v = 0.5

            alloc = target / v

        # Skewness adjustments
            if pd.notna(s) and s > 0:
                alloc = pow(alloc, 1.3)
            elif pd.notna(s) and s < -0.5:
                alloc *= 0.5

            alloc = np.clip(alloc, 0.0, 1.0)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[f'{asset}_Ret'].iloc[i]) + ((1-alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 15: THE META-ENSEMBLE (The Final Boss)
    # -----------------------------------------------------------------------------
    if strategy_type == 'meta_ensemble':
        asset = config['asset']
        target = config['target_vol']

    # 1. Downside Vol (Sortino)
        neg_rets = df[f'{asset}_Ret'].where(df[f'{asset}_Ret'] < 0, 0)
        d_vol = neg_rets.rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)

    # 2. Trend (SMA)
        sma = df['SPY_Price'].rolling(200, min_periods=1).mean().shift(1)

    # 3. Velocity (Fast vs Slow Vol)
        v_fast = df[f'{asset}_Ret'].rolling(5, min_periods=1).std().shift(1) * np.sqrt(252)
        v_slow = df[f'{asset}_Ret'].rolling(20, min_periods=1).std().shift(1) * np.sqrt(252)

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0  # Start with no position
        num_trades = 0

        for i in range(1, len(df)):
            dv = d_vol.iloc[i]
            if pd.isna(dv) or dv < 0.001: dv = 0.25

        # Layer 1: Downside Vol Targeting
            alloc = target / dv

        # Layer 2: Trend Convexity
            if df['SPY_Price'].iloc[i-1] > sma.iloc[i]:
                alloc = pow(alloc, 1.2)

        # Layer 3: Velocity Circuit Breaker
            if v_fast.iloc[i] > 1.5 * v_slow.iloc[i]:
                alloc *= 0.5

            alloc = np.clip(alloc, 0.0, 1.0)

        # Track turnover - count EVERY change
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

            # Apply Roth IRA costs
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[f'{asset}_Ret'].iloc[i]) + ((1-alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

# -----------------------------------------------------------------------------
# STRATEGY 16: CRISIS ALPHA (The Asymmetric Hedge)
# -----------------------------------------------------------------------------
    if strategy_type == 'regime_asymmetric':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        base_target = config['base_target_vol']
        crisis_target = config['crisis_target_vol']
        vix_alarm = config['vix_alarm_level']
        vol_threshold = config['vol_expansion_threshold']
        lb_fast = config['lookback_fast']
        lb_slow = config['lookback_slow']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

    # Calculate volatilities
        vol_fast = df[ret_col].rolling(lb_fast, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_slow = df[ret_col].rolling(lb_slow, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_ratio = vol_fast / vol_slow

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)
        prev_alloc = 0.0
        num_trades = 0

        for i in range(1, len(df)):
            current_vix = df['VIX'].iloc[i]
            vr = vol_ratio.iloc[i]
            realized_vol = vol_fast.iloc[i]

        # Regime Detection
        # Crisis Mode: VIX elevated OR volatility expanding rapidly
            crisis_mode = (current_vix > vix_alarm) or (vr > vol_threshold)

        # Choose target based on regime
            target_vol = crisis_target if crisis_mode else base_target

        # Calculate allocation
            if pd.isna(realized_vol) or realized_vol < 0.001:
                alloc = 0.5
            else:
                alloc = target_vol / realized_vol
                alloc = np.clip(alloc, 0.0, 1.2)  # Allow slight overleverage in calm

        # Track turnover
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

        # Calculate return
            ret = (alloc * df[ret_col].iloc[i]) + ((1 - alloc) * df['Cash_Ret'].iloc[i])

        # Apply transaction costs
            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 17: TAIL RISK OPTIMIZER (Skewness-Aware Kelly)
    # -----------------------------------------------------------------------------
    if strategy_type == 'skew_kelly':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        base_target = config['base_target_vol']
        skew_lb = config['skew_lookback']
        vol_lb = config['vol_lookback']
        kelly_frac = config['kelly_fraction']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        # Calculate rolling metrics
        realized_vol = df[ret_col].rolling(vol_lb, min_periods=1).std().shift(1) * np.sqrt(252)
        rolling_skew = df[ret_col].rolling(skew_lb, min_periods=1).skew().shift(1)

        # Downside vol (Sortino denominator)
        neg_rets = df[ret_col].where(df[ret_col] < 0, 0)
        downside_vol = neg_rets.rolling(vol_lb, min_periods=1).std().shift(1) * np.sqrt(252)

        # Rolling mean (for Kelly numerator)
        rolling_mean = df[ret_col].rolling(skew_lb, min_periods=1).mean().shift(1) * 252

        prev_alloc = 0.0
        num_trades = 0

        for i in range(1, len(df)):
            vol = realized_vol.iloc[i]
            d_vol = downside_vol.iloc[i]
            skew = rolling_skew.iloc[i]
            mean_ret = rolling_mean.iloc[i]

            # Safety defaults
            if pd.isna(vol) or vol < 0.001: vol = 0.25
            if pd.isna(d_vol) or d_vol < 0.001: d_vol = vol * 0.6
            if pd.isna(skew): skew = 0.0
            if pd.isna(mean_ret): mean_ret = 0.08

            # Skew adjustment: penalize negative skew
            if skew < -0.5:
                # Negative skew (crashy): use downside vol + reduce target
                effective_vol = d_vol * 1.5
                skew_penalty = 0.6
            elif skew < 0:
                # Mild negative skew: slight penalty
                effective_vol = d_vol * 1.2
                skew_penalty = 0.8
            elif skew > 0.5:
                # Positive skew (smooth grind up): boost leverage
                effective_vol = vol * 0.9
                skew_penalty = 1.2
            else:
                # Neutral skew
                effective_vol = vol
                skew_penalty = 1.0

            # Kelly-style sizing: f = (mu - rf) / sigma^2
            # But fractional and bounded
            if effective_vol > 0.01:
                kelly_size = (mean_ret - 0.03) / (effective_vol ** 2)
                kelly_size = kelly_size * kelly_frac  # Fractional Kelly
                kelly_size = np.clip(kelly_size, 0.2, 2.0)
            else:
                kelly_size = 1.0

            # Combine: Base vol targeting + Skew penalty + Kelly sizing
            raw_alloc = (base_target / effective_vol) * skew_penalty * (kelly_size / 1.5)
            alloc = np.clip(raw_alloc, 0.0, 1.5)

            # Track turnover
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

            # Calculate return
            ret = (alloc * df[ret_col].iloc[i]) + ((1 - alloc) * df['Cash_Ret'].iloc[i])

            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # -----------------------------------------------------------------------------
    # STRATEGY 18: MOMENTUM VOL CONVERGENCE (Dual Alpha)
    # -----------------------------------------------------------------------------
    if strategy_type == 'mom_vol_convergence':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        base_target = config['base_target_vol']
        mom_lb = config['momentum_lookback']
        vol_fast_lb = config['vol_fast']
        vol_slow_lb = config['vol_slow']
        mom_threshold = config['momentum_threshold']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        # Calculate momentum on SPY (cleaner signal than leveraged)
        momentum = df['SPY_Ret'].rolling(mom_lb, min_periods=1).sum().shift(1)

        # Calculate volatilities
        vol_fast = df[ret_col].rolling(vol_fast_lb, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_slow = df[ret_col].rolling(vol_slow_lb, min_periods=1).std().shift(1) * np.sqrt(252)

        prev_alloc = 0.0
        num_trades = 0

        for i in range(1, len(df)):
            mom = momentum.iloc[i]
            v_fast = vol_fast.iloc[i]
            v_slow = vol_slow.iloc[i]

            # Safety defaults
            if pd.isna(mom): mom = 0.0
            if pd.isna(v_fast) or v_fast < 0.001: v_fast = 0.30
            if pd.isna(v_slow) or v_slow < 0.001: v_slow = 0.25

            # Signal 1: Momentum strength
            if mom > mom_threshold:
                mom_multiplier = 1.3  # Strong uptrend: boost leverage
            elif mom > 0:
                mom_multiplier = 1.0  # Weak uptrend: normal
            else:
                mom_multiplier = 0.5  # Downtrend: defensive

            # Signal 2: Volatility regime
            vol_ratio = v_fast / v_slow

            if vol_ratio < 0.8:
                # Vol compressing (calming down): boost leverage
                vol_multiplier = 1.2
                effective_vol = v_fast
            elif vol_ratio > 1.3:
                # Vol expanding (crisis brewing): cut leverage
                vol_multiplier = 0.6
                effective_vol = v_fast  # Use fast vol (more reactive)
            else:
                # Stable vol: normal
                vol_multiplier = 1.0
                effective_vol = v_slow  # Use slow vol (smoother)

            # Combine both signals
            combined_multiplier = mom_multiplier * vol_multiplier
            adjusted_target = base_target * combined_multiplier

            # Calculate allocation
            alloc = adjusted_target / effective_vol
            alloc = np.clip(alloc, 0.0, 1.5)

            # Track turnover
            turnover = abs(alloc - prev_alloc)
            if turnover > 0.0001:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )
                spread_cost = turnover * 0.0003
            else:
                spread_cost = 0.0

            prev_alloc = alloc

            # Calculate return
            ret = (alloc * df[ret_col].iloc[i]) + ((1 - alloc) * df['Cash_Ret'].iloc[i])

            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

# -----------------------------------------------------------------------------
    # STRATEGY 19: CONVICTION COMPOUNDER (Triple Confirmation)
    # -----------------------------------------------------------------------------
    if strategy_type == 'conviction_compounder':
        asset = config['asset']
        ret_col = f'{asset}_Ret'
        base_target = config['base_target_vol']
        mom_lb = config['momentum_lookback']
        vol_lb = config['vol_lookback']
        trend_sma = config['trend_sma']
        rebalance_threshold = config['rebalance_threshold']

        if ret_col not in df.columns:
            return pd.Series(cfg.INITIAL_CAPITAL, index=df.index), 0

        equity_curve = pd.Series(cfg.INITIAL_CAPITAL, index=df.index, dtype=float)

        # Signal 1: Momentum (6-month)
        momentum = df['SPY_Ret'].rolling(mom_lb, min_periods=1).sum().shift(1)

        # Signal 2: Downside Volatility (from Meta-Ensemble)
        neg_rets = df[ret_col].where(df[ret_col] < 0, 0)
        downside_vol = neg_rets.rolling(vol_lb, min_periods=1).std().shift(1) * np.sqrt(252)

        # Signal 3: Volatility Expansion (from Crisis Alpha)
        vol_fast = df[ret_col].rolling(5, min_periods=1).std().shift(1) * np.sqrt(252)
        vol_slow = df[ret_col].rolling(60, min_periods=1).std().shift(1) * np.sqrt(252)

        # Signal 4: Trend Filter
        sma = df['SPY_Price'].rolling(trend_sma, min_periods=1).mean().shift(1)

        prev_alloc = 0.0
        num_trades = 0

        for i in range(1, len(df)):
            mom = momentum.iloc[i]
            d_vol = downside_vol.iloc[i]
            v_fast = vol_fast.iloc[i]
            v_slow = vol_slow.iloc[i]
            price = df['SPY_Price'].iloc[i-1]
            trend_line = sma.iloc[i]

            # Safety defaults
            if pd.isna(mom): mom = 0.0
            if pd.isna(d_vol) or d_vol < 0.001: d_vol = 0.20
            if pd.isna(v_fast) or v_fast < 0.001: v_fast = 0.30
            if pd.isna(v_slow) or v_slow < 0.001: v_slow = 0.25

            # === CONVICTION SCORING (0.0 to 2.0) ===

            # 1. Momentum Score (0.0 to 1.0)
            if mom > 0.15:  # Strong uptrend (>15% over 6mo)
                mom_score = 1.0
            elif mom > 0.05:  # Moderate uptrend
                mom_score = 0.7
            elif mom > 0:  # Weak uptrend
                mom_score = 0.4
            else:  # Downtrend
                mom_score = 0.0

            # 2. Trend Confirmation (0.0 or 0.5)
            trend_score = 0.5 if price > trend_line else 0.0

            # 3. Vol Regime Score (0.0 to 0.5)
            vol_ratio = v_fast / v_slow
            if vol_ratio < 0.9:  # Vol compressing (safe)
                vol_score = 0.5
            elif vol_ratio < 1.2:  # Vol stable
                vol_score = 0.3
            else:  # Vol expanding (danger)
                vol_score = 0.0

            # Total Conviction (0.0 to 2.0)
            conviction = mom_score + trend_score + vol_score

            # === LEVERAGE SCALING ===

            # Base allocation from downside vol
            base_alloc = base_target / d_vol

            # Scale by conviction
            # High conviction (2.0) -> 1.4x multiplier
            # Medium conviction (1.0) -> 1.0x multiplier
            # Low conviction (0.0) -> 0.3x multiplier
            conviction_multiplier = 0.3 + (conviction * 0.55)

            # Final allocation
            alloc = base_alloc * conviction_multiplier
            alloc = np.clip(alloc, 0.0, 1.5)

            # === REBALANCE CONTROL (Reduce trades) ===
            # Only rebalance if allocation changes significantly
            turnover = abs(alloc - prev_alloc)

            if turnover > rebalance_threshold:
                num_trades += 1

                # NEW: Log trade if trade_journal provided
                if trade_journal:
                    asset_price = df[f'{asset}_Price'].iloc[i]
                    portfolio_val = equity_curve.iloc[i-1]
                    trade_journal.log_allocation_change(
                        day=i,
                        asset=asset,
                        prev_allocation=prev_alloc,
                        new_allocation=alloc,
                        portfolio_value=portfolio_val,
                        price=asset_price
                    )

                spread_cost = turnover * 0.0003
                prev_alloc = alloc
            else:
                # Don't rebalance - keep previous allocation
                alloc = prev_alloc
                spread_cost = 0.0

            # Calculate return
            ret = (alloc * df[ret_col].iloc[i]) + ((1 - alloc) * df['Cash_Ret'].iloc[i])

            if apply_costs:
                ret -= spread_cost

            equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + ret)

        return equity_curve, num_trades

    # Default
    returns = df['SPY_Ret'].fillna(0)
    equity_curve = cfg.INITIAL_CAPITAL * (1 + returns).cumprod()

    return equity_curve, 0
