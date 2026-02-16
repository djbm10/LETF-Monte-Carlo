import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict


@dataclass
class WashSaleLot:
    """Represents a single purchase lot for wash sale tracking"""
    day: int
    shares: float
    price: float
    cost_basis: float  # Total cost basis (shares × price + any disallowed losses added)
    original_buy_day: int  # For holding period tacking (may differ from 'day')

    def __post_init__(self):
        # If original_buy_day not set, use the actual buy day
        if self.original_buy_day is None:
            self.original_buy_day = self.day


@dataclass
class WashSaleEvent:
    """Records a wash sale event for audit trail"""
    sale_day: int
    asset: str
    loss_amount: float  # The loss that was disallowed
    replacement_buy_day: int  # The buy that triggered the wash sale
    replacement_shares: float
    basis_adjustment: float  # Amount added to replacement lot basis
    original_lot_buy_day: int  # The buy day of the lot that was sold at a loss (for holding period tacking)
    shares_affected: float  # How many shares have their holding period tacked
    # NEW: Cross-year tracking
    sale_tax_year: int = 0  # Tax year when the loss sale occurred
    replacement_tax_year: int = 0  # Tax year when replacement was bought
    is_cross_year: bool = False  # True if wash sale spans tax years
    chain_id: int = 0  # Links related wash sales in a chain (0 = not part of chain)


class WashSaleTracker:
    """
    Track wash sales per IRC §1091 with BOTH look-back AND look-forward windows.

    This is a complete rewrite that properly handles:
    1. Look-back window (30 days before sale)
    2. Look-forward window (30 days after sale)
    3. Basis adjustment on replacement shares
    4. Holding period tacking
    5. Partial wash sales (when replacement qty < sold qty)

    Usage:
        tracker = WashSaleTracker()

        # First, record ALL trades
        tracker.record_trade('TQQQ', day=100, action='BUY', shares=10, price=50)
        tracker.record_trade('TQQQ', day=150, action='SELL', shares=10, price=40)  # Loss!
        tracker.record_trade('TQQQ', day=160, action='BUY', shares=5, price=42)   # Within 30 days!

        # Then, process wash sales (checks both directions)
        tracker.process_all_wash_sales()

        # Get results
        adjusted_losses = tracker.get_adjusted_losses()
    """

    def __init__(self, days_per_year: int = 252):
        """
        Initialize WashSaleTracker.

        Args:
            days_per_year: Number of trading days per year (default 252).
                          Used for cross-year wash sale tracking.
        """
        self.days_per_year = days_per_year

        # All recorded trades by asset
        self.trades: Dict[str, List[Dict]] = defaultdict(list)

        # Wash sale events (for audit trail)
        self.wash_sale_events: List[WashSaleEvent] = []

        # Disallowed losses by asset
        self.disallowed_losses: Dict[str, float] = defaultdict(float)

        # Losses that ARE allowed (after wash sale processing)
        self.allowed_losses: Dict[str, float] = defaultdict(float)

        # Basis adjustments to apply to lots
        # Key: (asset, buy_day) -> adjustment amount
        self.basis_adjustments: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Holding period adjustments (for tacking)
        # Key: (asset, replacement_buy_day) -> original_lot_buy_day
        self.holding_period_adjustments: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Track how many shares in each replacement lot have tacked holding period
        # Key: (asset, replacement_buy_day) -> shares_with_tacked_period
        self.tacked_shares: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # ====================================================================
        # NEW: Cross-Year Wash Sale Tracking
        # ====================================================================

        # Disallowed losses by tax year (for proper year-by-year reporting)
        # Key: (asset, tax_year) -> disallowed amount
        self.disallowed_by_year: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Allowed losses by tax year
        # Key: (asset, tax_year) -> allowed amount
        self.allowed_by_year: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Track wash sale chains
        # A chain occurs when: loss sale -> wash sale -> that lot sold at loss -> another wash sale
        # Key: chain_id -> list of WashSaleEvent in the chain
        self.wash_sale_chains: Dict[int, List[WashSaleEvent]] = defaultdict(list)
        self._next_chain_id = 1

        # Track which lots are "tainted" by being part of a wash sale chain
        # Key: (asset, buy_day) -> chain_id (0 if not tainted)
        self.tainted_lots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Has process_all_wash_sales() been called?
        self._processed = False

    def _day_to_tax_year(self, day: int) -> int:
        """Convert a day index to a tax year (0-indexed)."""
        return day // self.days_per_year

    def record_trade(self, asset: str, day: int, action: str, shares: float, price: float):
        """
        Record a trade for wash sale analysis.

        Call this for ALL trades before calling process_all_wash_sales().

        Args:
            asset: Stock symbol (e.g., 'TQQQ')
            day: Day index (integer, e.g., 0, 1, 2, ...)
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Price per share
        """
        self.trades[asset].append({
            'day': day,
            'action': action.upper(),
            'shares': shares,
            'price': price,
            'dollar_amount': shares * price
        })

        # Reset processed flag since we have new data
        self._processed = False

    def process_all_wash_sales(self):
        """
        Process all recorded trades to identify wash sales.

        This must be called AFTER recording all trades and BEFORE
        getting adjusted losses.

        The algorithm:
        1. Sort trades by day for each asset
        2. Build a list of all BUY trades
        3. For each SELL at a loss:
           a. Check all BUYs within ±30 days (look-back AND look-forward)
           b. If found, disallow the loss
           c. Add disallowed amount to replacement lot's basis
        """

        for asset, trade_list in self.trades.items():
            if not trade_list:
                continue

            # Sort trades by day
            sorted_trades = sorted(trade_list, key=lambda t: t['day'])

            # Separate buys and sells
            buys = [t for t in sorted_trades if t['action'] == 'BUY']
            sells = [t for t in sorted_trades if t['action'] == 'SELL']

            # Track which buys have been used as replacement shares
            # (A buy can only be used once as replacement)
            used_buys = set()

            # Process sells using FIFO to determine cost basis and gain/loss
            # We need to track lots properly
            buy_lots = []  # List of remaining lots
            for buy in buys:
                buy_lots.append({
                    'day': buy['day'],
                    'shares': buy['shares'],
                    'price': buy['price'],
                    'original_day': buy['day']  # May be updated by wash sale tacking
                })

            buy_lot_idx = 0  # Current lot index for FIFO

            for sell in sells:
                sell_day = sell['day']
                sell_shares = sell['shares']
                sell_price = sell['price']

                # Calculate gain/loss using FIFO
                # Also track which lots were consumed (needed for holding period tacking)
                remaining_to_sell = sell_shares
                total_cost_basis = 0
                lots_consumed = []

                temp_lot_idx = 0
                temp_lots = [lot.copy() for lot in buy_lots]  # Work on copy

                while remaining_to_sell > 0.001 and temp_lot_idx < len(temp_lots):
                    lot = temp_lots[temp_lot_idx]

                    if lot['shares'] <= 0.001:
                        temp_lot_idx += 1
                        continue

                    shares_from_lot = min(remaining_to_sell, lot['shares'])
                    cost_from_lot = shares_from_lot * lot['price']

                    lots_consumed.append({
                        'lot_idx': temp_lot_idx,
                        'shares': shares_from_lot,
                        'cost': cost_from_lot,
                        'buy_day': lot['day'],
                        'original_day': lot['original_day']  # Track original day for holding period
                    })

                    total_cost_basis += cost_from_lot
                    lot['shares'] -= shares_from_lot
                    remaining_to_sell -= shares_from_lot

                    if lot['shares'] <= 0.001:
                        temp_lot_idx += 1

                # Calculate gain/loss
                proceeds = sell_shares * sell_price
                gain_loss = proceeds - total_cost_basis

                # Only check wash sales for LOSSES
                if gain_loss >= 0:
                    continue  # Not a loss, no wash sale possible

                loss_amount = abs(gain_loss)

                # ============================================================
                # WASH SALE CHECK: Look for buys within ±30 days
                # ============================================================

                # Find all buys within the wash sale window
                wash_sale_buys = []
                for i, buy in enumerate(buys):
                    buy_day = buy['day']

                    # Check if within ±30 day window
                    if abs(buy_day - sell_day) <= 30 and buy_day != sell_day:
                        # Don't use the same buy that was just sold (lot matching)
                        # and don't use already-used buys
                        if i not in used_buys:
                            wash_sale_buys.append((i, buy))

                if wash_sale_buys:
                    # WASH SALE TRIGGERED!
                    # Find the closest buy (IRS doesn't specify, so use nearest)
                    wash_sale_buys.sort(key=lambda x: abs(x[1]['day'] - sell_day))
                    replacement_idx, replacement_buy = wash_sale_buys[0]

                    # Calculate how much loss is disallowed
                    # If replacement shares < sold shares, only partial disallowance
                    replacement_shares = replacement_buy['shares']
                    sold_shares = sell_shares

                    if replacement_shares >= sold_shares:
                        # Full wash sale - all loss disallowed
                        disallowed = loss_amount
                        shares_affected = sold_shares  # All sold shares affect the replacement
                    else:
                        # Partial wash sale - proportional disallowance
                        disallowed = loss_amount * (replacement_shares / sold_shares)
                        shares_affected = replacement_shares  # Only replacement shares are affected

                    allowed = loss_amount - disallowed

                    # Record the disallowance
                    self.disallowed_losses[asset] += disallowed
                    self.allowed_losses[asset] += allowed

                    # Record basis adjustment for replacement lot
                    self.basis_adjustments[asset][replacement_buy['day']] += disallowed

                    # ============================================================
                    # HOLDING PERIOD TACKING
                    # ============================================================
                    # The replacement lot inherits the holding period from the
                    # EARLIEST lot that was sold at a loss.
                    #
                    # Per IRC §1223(4): "In determining the period for which the
                    # taxpayer has held stock or securities the acquisition of which
                    # resulted in the nondeductibility of the loss... there shall be
                    # included the period for which he held the stock or securities
                    # the loss from the sale or other disposition of which was not
                    # allowable."

                    # Find the earliest original_day from the lots that were consumed
                    if lots_consumed:
                        earliest_original_day = min(lot['original_day'] for lot in lots_consumed)
                    else:
                        earliest_original_day = sell_day  # Fallback (shouldn't happen)

                    # Store the holding period adjustment
                    # This means: "For the lot bought on replacement_buy['day'],
                    # use earliest_original_day for holding period calculation"
                    self.holding_period_adjustments[asset][replacement_buy['day']] = earliest_original_day

                    # Track how many shares have tacked holding period
                    self.tacked_shares[asset][replacement_buy['day']] += shares_affected

                    # Mark this buy as used
                    used_buys.add(replacement_idx)

                    # ============================================================
                    # CROSS-YEAR WASH SALE TRACKING
                    # ============================================================
                    sale_tax_year = self._day_to_tax_year(sell_day)
                    replacement_tax_year = self._day_to_tax_year(replacement_buy['day'])
                    is_cross_year = (sale_tax_year != replacement_tax_year)

                    # Track disallowed/allowed by year
                    self.disallowed_by_year[asset][sale_tax_year] += disallowed
                    self.allowed_by_year[asset][sale_tax_year] += allowed

                    # Check if this is part of a wash sale chain
                    # A chain exists if the lot we just sold was itself a replacement lot
                    # from a prior wash sale
                    chain_id = self.tainted_lots[asset].get(lots_consumed[0]['buy_day'], 0) if lots_consumed else 0

                    if chain_id == 0 and is_cross_year:
                        # Start a new chain for cross-year wash sales
                        chain_id = self._next_chain_id
                        self._next_chain_id += 1
                    elif chain_id == 0 and len(self.wash_sale_events) > 0:
                        # Check if this connects to an existing wash sale
                        # (the lot we sold was a replacement lot from before)
                        for consumed_lot in lots_consumed:
                            existing_chain = self.tainted_lots[asset].get(consumed_lot['buy_day'], 0)
                            if existing_chain > 0:
                                chain_id = existing_chain
                                break

                    # Mark the replacement lot as tainted (part of a chain)
                    if chain_id > 0:
                        self.tainted_lots[asset][replacement_buy['day']] = chain_id

                    # Record the event for audit trail
                    wash_event = WashSaleEvent(
                        sale_day=sell_day,
                        asset=asset,
                        loss_amount=disallowed,
                        replacement_buy_day=replacement_buy['day'],
                        replacement_shares=min(replacement_shares, sold_shares),
                        basis_adjustment=disallowed,
                        original_lot_buy_day=earliest_original_day,
                        shares_affected=shares_affected,
                        sale_tax_year=sale_tax_year,
                        replacement_tax_year=replacement_tax_year,
                        is_cross_year=is_cross_year,
                        chain_id=chain_id
                    )
                    self.wash_sale_events.append(wash_event)

                    # Add to chain tracking
                    if chain_id > 0:
                        self.wash_sale_chains[chain_id].append(wash_event)

                else:
                    # No wash sale - loss is fully allowed
                    self.allowed_losses[asset] += loss_amount

        self._processed = True

    def check_wash_sale(self, asset: str, sale_day: int, loss_amount: float,
                        all_trades: List[Dict] = None) -> float:
        """
        Check if a specific loss sale triggers wash sale.

        This is a convenience method for checking individual sales.
        For full analysis, use record_trade() + process_all_wash_sales().

        Args:
            asset: Stock symbol
            sale_day: Day of the loss sale
            loss_amount: The loss amount (should be negative or zero)
            all_trades: List of all trades for this asset (needed for look-forward)

        Returns:
            Amount of loss that is ALLOWED (after wash sale disallowance)
        """
        if loss_amount >= 0:
            return loss_amount  # Not a loss

        if all_trades is None:
            # Can't do look-forward without knowing future trades
            # Fall back to look-back only (not recommended!)
            print("WARNING: check_wash_sale called without all_trades - look-forward check disabled!")
            return loss_amount

        # Get all buys for this asset
        buys = [t for t in all_trades if t.get('action', '').upper() == 'BUY'
                and t.get('asset') == asset]

        # Check for any buy within ±30 days
        for buy in buys:
            buy_day = buy.get('day', buy.get('day_index', 0))
            if abs(buy_day - sale_day) <= 30 and buy_day != sale_day:
                # Wash sale triggered!
                self.disallowed_losses[asset] += abs(loss_amount)
                return 0  # Loss fully disallowed

        # No wash sale - loss allowed
        return loss_amount

    def get_total_disallowed(self) -> float:
        """Get total disallowed losses across all assets"""
        return sum(self.disallowed_losses.values())

    def get_total_allowed(self) -> float:
        """Get total allowed losses across all assets"""
        return sum(self.allowed_losses.values())

    def get_basis_adjustment(self, asset: str, buy_day: int) -> float:
        """
        Get the basis adjustment for a specific lot.

        When a wash sale occurs, the disallowed loss is added to the
        replacement lot's cost basis.
        """
        return self.basis_adjustments[asset][buy_day]

    def get_holding_period_adjustment(self, asset: str, buy_day: int) -> int:
        """
        Get the adjusted holding period start day for a specific lot.

        When a wash sale occurs, the replacement lot inherits the holding
        period from the original lot that was sold at a loss.

        Args:
            asset: Stock symbol (e.g., 'TQQQ')
            buy_day: The day the replacement lot was purchased

        Returns:
            The day to use for holding period calculation.
            If no wash sale affected this lot, returns the buy_day itself.
            If a wash sale occurred, returns the original lot's buy day.

        Example:
            # Original lot bought Day 100, sold at loss Day 400
            # Replacement lot bought Day 410 (wash sale!)
            # get_holding_period_adjustment('TQQQ', 410) returns 100
            # So when calculating holding period: sale_day - 100, not sale_day - 410
        """
        adjusted_day = self.holding_period_adjustments[asset].get(buy_day, 0)

        if adjusted_day > 0:
            return adjusted_day
        else:
            return buy_day  # No adjustment - use actual buy day

    def get_tacked_shares(self, asset: str, buy_day: int) -> float:
        """
        Get the number of shares in a lot that have tacked holding period.

        In a partial wash sale, only some shares may have the tacked period.

        Args:
            asset: Stock symbol
            buy_day: The day the lot was purchased

        Returns:
            Number of shares with tacked holding period.
            Returns 0 if no wash sale affected this lot.
        """
        return self.tacked_shares[asset].get(buy_day, 0.0)

    def get_wash_sale_summary(self) -> Dict:
        """Get a summary of all wash sale activity"""
        if not self._processed:
            self.process_all_wash_sales()

        return {
            'total_disallowed': self.get_total_disallowed(),
            'total_allowed': self.get_total_allowed(),
            'events_count': len(self.wash_sale_events),
            'by_asset': {
                asset: {
                    'disallowed': self.disallowed_losses[asset],
                    'allowed': self.allowed_losses[asset]
                }
                for asset in set(list(self.disallowed_losses.keys()) + list(self.allowed_losses.keys()))
            },
            'events': [
                {
                    'sale_day': e.sale_day,
                    'asset': e.asset,
                    'loss_disallowed': e.loss_amount,
                    'replacement_day': e.replacement_buy_day
                }
                for e in self.wash_sale_events
            ]
        }

    def get_disallowed_for_year(self, asset: str, tax_year: int) -> float:
        """
        Get total disallowed losses for a specific asset and tax year.

        This is important for cross-year wash sales where the loss occurs
        in Year 1 but the replacement purchase is in Year 2.
        """
        return self.disallowed_by_year[asset].get(tax_year, 0.0)

    def get_allowed_for_year(self, asset: str, tax_year: int) -> float:
        """Get total allowed losses for a specific asset and tax year."""
        return self.allowed_by_year[asset].get(tax_year, 0.0)

    def get_chain_info(self, chain_id: int) -> Dict:
        """Get information about a wash sale chain."""
        if chain_id not in self.wash_sale_chains:
            return {'chain_id': chain_id, 'events': [], 'total_disallowed': 0}

        events = self.wash_sale_chains[chain_id]
        return {
            'chain_id': chain_id,
            'events': events,
            'total_disallowed': sum(e.loss_amount for e in events),
            'years_spanned': len(set(e.sale_tax_year for e in events)),
            'is_cross_year': any(e.is_cross_year for e in events)
        }

    def get_cross_year_summary(self) -> Dict:
        """Get a summary of all cross-year wash sale activity."""
        cross_year_events = [e for e in self.wash_sale_events if e.is_cross_year]

        by_year_pair = defaultdict(lambda: {'count': 0, 'amount': 0.0})
        for e in cross_year_events:
            key = f"Y{e.sale_tax_year}->Y{e.replacement_tax_year}"
            by_year_pair[key]['count'] += 1
            by_year_pair[key]['amount'] += e.loss_amount

        return {
            'total_cross_year_events': len(cross_year_events),
            'total_cross_year_disallowed': sum(e.loss_amount for e in cross_year_events),
            'chains_count': len(self.wash_sale_chains),
            'by_year_pair': dict(by_year_pair)
        }

    def reset(self):
        """Clear all tracked data"""
        self.trades.clear()
        self.wash_sale_events.clear()
        self.disallowed_losses.clear()
        self.allowed_losses.clear()
        self.basis_adjustments.clear()
        self.holding_period_adjustments.clear()
        self.tacked_shares.clear()
        self.disallowed_by_year.clear()
        self.allowed_by_year.clear()
        self.wash_sale_chains.clear()
        self.tainted_lots.clear()
        self._next_chain_id = 1
        self._processed = False
