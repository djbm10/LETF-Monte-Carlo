from AlgorithmImports import *
from datetime import timedelta
import math


class RealHistoricalLeaps(QCAlgorithm):
    """
    FREE_DISCOVERY LEAPS backtest using QuantConnect/AlgoSeek historical option chains.

    Parameters:
        underlying=SPY|QQQ
        target_delta=0.70|0.80|0.90
        target_dte=365|548|730
        allocation=0.25|0.50|1.00
        slippage_bps=0|5|10

    Execution:
    - daily option-chain / quote data
    - buy calls with actual available historical contracts
    - require positive bid/ask and minimum OI
    - target delta and maturity jointly
    - market orders use quote-side fills through LatestPriceFillModel:
      ask-side for buys, bid-side for sells, plus configured slippage
    - remaining capital is held in BIL as a T-bill collateral proxy
    - roll every six calendar months or when DTE < 180

    This is not OptionMetrics/IvyDB. It is a real historical OPRA-derived
    QuantConnect/AlgoSeek discovery backtest from broadly 2012 onward.
    """

    def initialize(self):
        ticker = (self.get_parameter("underlying") or "SPY").upper()
        if ticker not in ("SPY", "QQQ"):
            raise ValueError("underlying must be SPY or QQQ")

        self.target_delta = float(self.get_parameter("target_delta") or "0.80")
        self.target_dte = int(self.get_parameter("target_dte") or "548")
        self.allocation = float(self.get_parameter("allocation") or "0.50")
        self.min_oi = int(self.get_parameter("min_oi") or "100")
        self.slippage = float(self.get_parameter("slippage_bps") or "0") / 10_000.0

        if self.target_delta not in (0.70, 0.80, 0.90):
            raise ValueError("target_delta must be 0.70, 0.80, or 0.90")
        if self.target_dte not in (365, 548, 730):
            raise ValueError("target_dte must be 365, 548, or 730")
        if self.allocation not in (0.25, 0.50, 1.00):
            raise ValueError("allocation must be 0.25, 0.50, or 1.00")

        self.set_start_date(2012, 1, 3)
        self.set_end_date(2026, 9, 21)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.underlying = self.add_equity(ticker, Resolution.DAILY).symbol
        self.bill = self.add_equity("BIL", Resolution.DAILY).symbol
        self.set_benchmark(self.underlying)

        opt = self.add_option(ticker, Resolution.DAILY)
        self.canonical = opt.symbol
        opt.set_filter(
            lambda u: u.calls_only()
            .delta(0.55, 0.99)
            .expiration(timedelta(days=270), timedelta(days=900))
        )

        self.contract = None
        self.entry_date = None
        self.pending_entry = True
        self.roll_count = 0
        self.entry_count = 0
        self.no_chain_days = 0
        self.no_candidate_days = 0

        # Configure option securities as they are created/subscribed.
        self.set_security_initializer(self._initialize_security)

        self.log(
            f"EVIDENCE_LABEL=FREE_DISCOVERY underlying={ticker} "
            f"delta={self.target_delta} dte={self.target_dte} "
            f"allocation={self.allocation} slippage_bps={self.slippage*10000:.0f}"
        )

    def _initialize_security(self, security):
        # Preserve brokerage fee/buying-power defaults while making spread/slippage
        # treatment explicit for options.
        if security.type == SecurityType.OPTION:
            security.set_fill_model(LatestPriceFillModel())
            if self.slippage > 0:
                security.set_slippage_model(ConstantSlippageModel(self.slippage))
            else:
                security.set_slippage_model(NullSlippageModel.INSTANCE)

    def _needs_roll(self):
        if self.contract is None:
            return True
        if self.entry_date is None:
            return True
        dte = (self.contract.id.date.date() - self.time.date()).days
        six_months = (self.time.date() - self.entry_date).days >= 182
        return dte < 180 or six_months

    def _valid_contracts(self, chain):
        out = []
        for c in chain:
            if c.right != OptionRight.CALL:
                continue
            dte = (c.expiry.date() - self.time.date()).days
            if dte < 270 or dte > 900:
                continue
            if c.open_interest < self.min_oi:
                continue
            if c.bid_price <= 0 or c.ask_price <= 0 or c.ask_price < c.bid_price:
                continue
            delta = abs(float(c.greeks.delta))
            if not math.isfinite(delta) or delta < 0.55 or delta > 0.99:
                continue
            out.append((c, dte, delta))
        return out

    def _select_contract(self, chain):
        candidates = self._valid_contracts(chain)
        if not candidates:
            return None

        # Frozen objective: delta fit dominates, then maturity fit, then spread.
        def score(t):
            c, dte, delta = t
            mid = (c.bid_price + c.ask_price) / 2.0
            spread_pct = (c.ask_price - c.bid_price) / mid if mid > 0 else 1.0
            return (
                abs(delta - self.target_delta)
                + 0.20 * abs(dte - self.target_dte) / 365.0
                + 0.10 * spread_pct
            )

        return min(candidates, key=score)[0]

    def _exit_current(self):
        if self.contract is None:
            return
        if self.portfolio[self.contract].invested:
            self.liquidate(self.contract, tag="LEAPS roll/exit")
        self.contract = None
        self.entry_date = None

    def _enter(self, contract):
        # Explicitly subscribe to selected contract before ordering.
        self.add_option_contract(contract.symbol, Resolution.DAILY)
        security = self.securities[contract.symbol]
        security.set_fill_model(LatestPriceFillModel())
        security.set_slippage_model(
            ConstantSlippageModel(self.slippage)
            if self.slippage > 0
            else NullSlippageModel.INSTANCE
        )

        ask = float(contract.ask_price)
        if ask <= 0:
            return False

        budget = self.portfolio.total_portfolio_value * self.allocation
        quantity = int(budget // (ask * 100.0))
        if quantity < 1:
            return False

        # Leave a small cushion so modeled slippage/fees don't create accidental leverage.
        while quantity > 0 and quantity * ask * 100.0 > budget * 0.995:
            quantity -= 1
        if quantity < 1:
            return False

        self.market_order(contract.symbol, quantity, tag="LEAPS entry")
        self.contract = contract.symbol
        self.entry_date = self.time.date()
        self.entry_count += 1

        # Invest non-premium capital in a short-Treasury proxy.
        if self.allocation < 1.0:
            self.set_holdings(self.bill, 1.0 - self.allocation, tag="cash collateral proxy")
        else:
            if self.portfolio[self.bill].invested:
                self.liquidate(self.bill)

        self.debug(
            f"{self.time.date()} ENTER {contract.symbol} "
            f"dte={(contract.expiry.date()-self.time.date()).days} "
            f"delta={float(contract.greeks.delta):.3f} "
            f"bid={contract.bid_price:.2f} ask={contract.ask_price:.2f} qty={quantity}"
        )
        return True

    def on_data(self, slice: Slice):
        chain = slice.option_chains.get(self.canonical)
        if chain is None:
            self.no_chain_days += 1
            return

        if not self._needs_roll():
            return

        if self.contract is not None:
            self._exit_current()
            self.roll_count += 1

        selected = self._select_contract(chain)
        if selected is None:
            self.no_candidate_days += 1
            return

        self._enter(selected)

    def on_end_of_algorithm(self):
        self.log(
            "LEAPS_AUDIT "
            f"entries={self.entry_count} rolls={self.roll_count} "
            f"no_chain_days={self.no_chain_days} no_candidate_days={self.no_candidate_days} "
            f"EVIDENCE_LABEL=FREE_DISCOVERY"
        )
