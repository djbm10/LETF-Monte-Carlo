from AlgorithmImports import *
from datetime import timedelta
import csv
import io
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
    - select calls only after the daily chain is known
    - require positive bid/ask and minimum OI
    - target delta and maturity jointly
    - submit Market-On-Open orders for the next session, avoiding same-close
      execution on the quote used to select the contract
    - LatestPriceFillModel uses next-session quote-side open fills plus configured slippage
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

        # Preserve brokerage defaults and layer on the frozen option slippage.
        self.add_security_initializer(self._initialize_security)

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
        self.pending_entry_symbol = None
        self.pending_entry_order_id = None
        self.pending_exit_symbol = None
        self.pending_exit_order_id = None
        self.roll_count = 0
        self.entry_count = 0
        self.no_chain_days = 0
        self.no_candidate_days = 0
        self.selection_audit = []
        self.fill_audit = []

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
            security.set_slippage_model(ConstantSlippageModel(self.slippage))

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

    def _queue_exit_current(self):
        if self.contract is None or not self.portfolio[self.contract].invested:
            return None
        qty = -int(self.portfolio[self.contract].quantity)
        if qty == 0:
            return None
        ticket = self.market_on_open_order(
            self.contract, qty, tag="LEAPS roll/exit next open"
        )
        self.pending_exit_symbol = self.contract
        self.pending_exit_order_id = ticket.order_id
        return ticket

    def _queue_entry(self, contract):
        # Subscribe before submitting the next-session MOO order so the fill
        # model can consume the next daily QuoteBar and use its ask open.
        self.add_option_contract(contract.symbol, Resolution.DAILY)
        security = self.securities[contract.symbol]
        security.set_fill_model(LatestPriceFillModel())
        security.set_slippage_model(ConstantSlippageModel(self.slippage))

        ask = float(contract.ask_price)
        if ask <= 0:
            return None

        budget = self.portfolio.total_portfolio_value * self.allocation
        quantity = int(budget // (ask * 100.0))
        if quantity < 1:
            return None

        # Cushion is pre-specified execution headroom for overnight price gaps,
        # slippage, and fees. It is not tuned from backtest outcomes.
        while quantity > 0 and quantity * ask * 100.0 > budget * 0.995:
            quantity -= 1
        if quantity < 1:
            return None

        self.selection_audit.append({
            "decision_time": str(self.time),
            "event": "ENTRY_SELECTION_FOR_NEXT_OPEN",
            "underlying": str(self.underlying),
            "contract": str(contract.symbol),
            "expiry": str(contract.expiry.date()),
            "strike": float(contract.strike),
            "dte": int((contract.expiry.date() - self.time.date()).days),
            "delta": float(contract.greeks.delta),
            "bid": float(contract.bid_price),
            "ask": float(contract.ask_price),
            "open_interest": int(contract.open_interest),
            "quantity": int(quantity),
            "underlying_price": float(self.securities[self.underlying].price),
            "portfolio_value": float(self.portfolio.total_portfolio_value),
            "premium_budget": float(budget),
            "target_delta": self.target_delta,
            "target_dte": self.target_dte,
            "allocation": self.allocation,
            "slippage_bps": self.slippage * 10000.0,
        })

        ticket = self.market_on_open_order(
            contract.symbol, quantity, tag="LEAPS entry next open"
        )
        self.pending_entry_symbol = contract.symbol
        self.pending_entry_order_id = ticket.order_id

        # Re-target the collateral sleeve for the same next-session open.
        target_bill = 1.0 - self.allocation
        bill_qty = self.calculate_order_quantity(self.bill, target_bill)
        if bill_qty:
            self.market_on_open_order(
                self.bill, bill_qty, tag="cash collateral proxy next open"
            )

        self.debug(
            f"{self.time.date()} QUEUE {contract.symbol} FOR NEXT OPEN "
            f"dte={(contract.expiry.date()-self.time.date()).days} "
            f"delta={float(contract.greeks.delta):.3f} "
            f"bid={contract.bid_price:.2f} ask={contract.ask_price:.2f} qty={quantity}"
        )
        return ticket

    def on_data(self, slice: Slice):
        chain = slice.option_chains.get(self.canonical)
        if chain is None:
            self.no_chain_days += 1
            return

        # An entry/exit already scheduled for the next open must resolve before
        # another daily close can create overlapping orders.
        if self.pending_entry_order_id is not None or self.pending_exit_order_id is not None:
            return

        if not self._needs_roll():
            return

        selected = self._select_contract(chain)
        if selected is None:
            self.no_candidate_days += 1
            return

        exit_ticket = None
        if self.contract is not None and self.portfolio[self.contract].invested:
            exit_ticket = self._queue_exit_current()

        entry_ticket = self._queue_entry(selected)
        if entry_ticket is None:
            if exit_ticket is not None:
                exit_ticket.cancel("replacement entry could not be created")
                self.pending_exit_symbol = None
                self.pending_exit_order_id = None
            self.no_candidate_days += 1
            return

    def on_order_event(self, order_event: OrderEvent):
        if order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            if order_event.order_id == self.pending_entry_order_id:
                self.pending_entry_symbol = None
                self.pending_entry_order_id = None
            if order_event.order_id == self.pending_exit_order_id:
                self.pending_exit_symbol = None
                self.pending_exit_order_id = None
            self.debug(
                f"{self.time} ORDER_{order_event.status} "
                f"id={order_event.order_id} symbol={order_event.symbol}"
            )
            return

        if order_event.status != OrderStatus.FILLED:
            return

        fee_amount = float("nan")
        fee_currency = ""
        try:
            fee_amount = float(order_event.order_fee.value.amount)
            fee_currency = str(order_event.order_fee.value.currency)
        except Exception:
            pass
        self.fill_audit.append({
            "fill_time": str(self.time),
            "order_id": int(order_event.order_id),
            "symbol": str(order_event.symbol),
            "direction": str(order_event.direction),
            "fill_quantity": float(order_event.fill_quantity),
            "fill_price": float(order_event.fill_price),
            "fee_amount": fee_amount,
            "fee_currency": fee_currency,
        })

        if order_event.order_id == self.pending_exit_order_id:
            self.pending_exit_symbol = None
            self.pending_exit_order_id = None
            self.roll_count += 1

        if order_event.order_id == self.pending_entry_order_id:
            self.contract = self.pending_entry_symbol
            self.entry_date = self.time.date()
            self.entry_count += 1
            self.pending_entry_symbol = None
            self.pending_entry_order_id = None

    def _save_csv(self, stem, rows):
        if not rows:
            return None, False
        fields = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        key = "%s/free_stage1/%s_%s.csv" % (self.project_id, stem, self.algorithm_id)
        return key, self.object_store.save(key, buf.getvalue())

    def on_end_of_algorithm(self):
        selection_key, selection_saved = self._save_csv("leaps_selection_audit", self.selection_audit)
        fill_key, fill_saved = self._save_csv("leaps_fill_audit", self.fill_audit)
        self.log(
            "LEAPS_AUDIT "
            f"entries={self.entry_count} rolls={self.roll_count} "
            f"no_chain_days={self.no_chain_days} no_candidate_days={self.no_candidate_days} "
            f"selection_rows={len(self.selection_audit)} selection_saved={selection_saved} "
            f"selection_key={selection_key} fill_rows={len(self.fill_audit)} "
            f"fill_saved={fill_saved} fill_key={fill_key} "
            f"EVIDENCE_LABEL=FREE_DISCOVERY"
        )
