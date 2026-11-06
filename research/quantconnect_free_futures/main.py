"""Paste this file into a QuantConnect Cloud free-tier Python backtest.

FROZEN 35/0 on QQQ signals, actual mapped NQ contracts with modeled brokerage
fees, margin and roll orders. Research only: no live trading, no data export.
The explicit one-bar trade delay is conservative and avoids same-bar signals.
Requires validation INSIDE QuantConnect; local Python compilation is not enough.
"""
from AlgorithmImports import *
from collections import deque
from math import sqrt, floor
from statistics import stdev

class FrozenNasdaqThirtyFiveZeroFutures(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 11, 5)
        self.set_cash(100_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.qqq = self.add_equity('QQQ', Resolution.DAILY).symbol
        self.nq = self.add_future('NQ', resolution=Resolution.DAILY,
                                  data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                                  data_normalization_mode=DataNormalizationMode.RAW,
                                  extended_market_hours=True)
        self.nq.set_filter(0, 182)
        self.prices = deque(maxlen=221)
        self.pending_exposure = None
        self.last_contract = None
        self.set_warm_up(260, Resolution.DAILY)

    def on_data(self, slice):
        bar = slice.bars.get(self.qqq)
        if bar is None:
            return
        # Apply only the PREVIOUS session's signal before computing today's.
        if not self.is_warming_up and self.pending_exposure is not None:
            mapped = self.nq.mapped
            if mapped is not None and mapped in self.securities:
                security = self.securities[mapped]
                if security.price > 0:
                    mult = float(security.symbol_properties.contract_multiplier)
                    desired = int(floor(self.pending_exposure * self.portfolio.total_portfolio_value /
                                        (security.price * mult)))
                    if self.last_contract is not None and self.last_contract != mapped:
                        self.liquidate(self.last_contract, tag='NQ expiry/roll; real contract, no synthetic gap')
                    current = self.portfolio[mapped].quantity
                    delta = desired - current
                    if delta != 0 and not self.transactions.get_open_orders():
                        self.market_order(mapped, delta, tag='Frozen 35/0; previous-day signal')
                    self.last_contract = mapped
        self.prices.append(float(bar.close))
        if len(self.prices) < 201 or self.is_warming_up:
            return
        values = list(self.prices)
        average = sum(values[-200:]) / 200
        returns = [values[i]/values[i-1]-1 for i in range(len(values)-20,len(values))]
        vol = stdev(returns) * sqrt(252)
        exposure = min(3., .35 / vol) if vol > 1e-8 and values[-1] > average else 0.
        self.pending_exposure = exposure
        self.plot('Signals','Target Exposure',exposure)

    def on_symbol_changed_events(self, events):
        # Trade-roll logic is done in on_data with mapped (dated) contracts;
        # signal data are QQQ closes, never raw NQ continuous series.
        pass
