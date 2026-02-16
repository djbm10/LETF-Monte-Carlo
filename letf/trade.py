from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import defaultdict


@dataclass
class Trade:
    day_index: int
    asset: str
    action: str
    shares: float  # CRITICAL: Store actual shares traded, not just dollar amount
    price: float
    dollar_amount: float  # Kept for backwards compatibility

class TradeJournal:
    def __init__(self):
        self.trades: List[Trade] = []
        self.positions: Dict[str, float] = defaultdict(float)  # Track actual shares held

    def log_allocation_change(self, day: int, asset: str,
                              prev_allocation: float, new_allocation: float,
                              portfolio_value: float, price: float):
        """
        FIXED v2: Track actual share positions to avoid recalculation errors.

        Previous bug: Recalculating shares from allocation x portfolio_value
        created mismatches because portfolio value changes from price movements.

        New approach: Track the actual shares we own, calculate target shares
        from desired allocation, trade the difference.
        """
        if price <= 0:
            return

        # Calculate target shares based on new allocation
        target_value = new_allocation * portfolio_value
        target_shares = target_value / price

        # Get current actual position
        current_shares = self.positions[asset]

        # Calculate the difference
        share_change = target_shares - current_shares

        # Skip negligible changes
        if abs(share_change) < 0.001:
            return

        # Execute the trade
        if share_change > 0:
            # Buying
            action = 'BUY'
            shares_traded = share_change
        else:
            # Selling
            action = 'SELL'
            shares_traded = abs(share_change)

        dollar_amount = shares_traded * price

        # Record the trade with ACTUAL SHARES
        # This prevents rounding errors when reconstructing shares from dollars
        self.trades.append(Trade(
            day_index=day,
            asset=asset,
            action=action,
            shares=shares_traded,  # CRITICAL: Store exact shares
            price=price,
            dollar_amount=dollar_amount
        ))

        # Update our position tracking
        self.positions[asset] = target_shares

    def get_summary(self) -> dict:
        if not self.trades:
            return {'count': 0, 'volume': 0}
        return {
            'count': len(self.trades),
            'volume': sum(t.dollar_amount for t in self.trades)
        }

    def get_full_trades(self) -> List[Dict]:
        """
        Return complete trade list for precise tax calculation.

        Returns:
            List of trade dicts with keys: day_index, asset, action,
            dollar_amount, price
        """
        return [asdict(trade) for trade in self.trades]

ROTH_IDS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
TAXABLE_IDS = ['S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19']
