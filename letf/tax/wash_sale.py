"""Dated wash-sale analysis facade over the shared portfolio ledger.

The former independent FIFO replay failed to consume gain lots, could use
future lots as sale basis, reused only one replacement purchase, ignored same
-day replacements, and used trading-session offsets for legal calendar rules.
All matching now goes through PortfolioLedger. IRC 1091; Rev. Rul. 2008-5.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from letf.accuracy import UnsupportedModelError


@dataclass
class WashSaleLot:
    asset: str
    buy_day: date
    shares: float
    price: float


@dataclass
class WashSaleEvent:
    sale_day: date
    asset: str
    loss_amount: float
    replacement_buy_day: date
    replacement_shares: float
    basis_adjustment: float
    original_lot_buy_day: date
    shares_affected: float
    sale_tax_year: int
    replacement_tax_year: int
    is_cross_year: bool
    chain_id: int = 0
    permanent: bool = False


class WashSaleTracker:
    def __init__(self, days_per_year=252, *, calendar_dates=None):
        # Retain the old positional parameter for source compatibility; it
        # MUST NOT define legal time windows or tax-year boundaries.
        self.calendar_dates = calendar_dates
        self.days_per_year = days_per_year
        self.reset()

    def _date(self, day):
        from letf.accounts import calendar_date
        if isinstance(day, int):
            if self.calendar_dates is None:
                raise UnsupportedModelError('Wash sales require actual calendar dates, not trading-session ordinals')
            value = self.calendar_dates[day]
            if hasattr(value, 'date'):
                value = value.date()
            return calendar_date(value)
        return calendar_date(day)

    def _day_to_tax_year(self, day):
        return self._date(day).year

    def record_trade(self, asset, day, action, shares, price, *, account_id='main', account_type='taxable'):
        when = self._date(day)
        if account_type not in ('taxable', 'roth'):
            raise ValueError('Unsupported account type')
        self.trades[asset].append({'asset': asset, 'date': when, 'action': action.upper(),
                                  'shares': shares, 'price': price, 'account_id': account_id,
                                  'account_type': account_type, 'sequence': self._sequence})
        self._sequence += 1
        self._processed = False

    def process_all_wash_sales(self):
        from letf.accounts import PortfolioLedger, BrokerProfile, Security
        if self._processed:
            return
        book = PortfolioLedger()
        trades = sorted([t for rows in self.trades.values() for t in rows],
                        key=lambda t: (t['date'], t['sequence']))
        profiles = {}
        for trade in trades:
            key, kind = trade['account_id'], trade['account_type']
            if key in profiles and profiles[key] != kind:
                raise ValueError('An account cannot change type during replay')
            profiles[key] = kind
        for account, kind in profiles.items():
            # This facade analyses already executed trades, not buying power.
            # Execution/buying-power certification belongs to PortfolioLedger.
            cash = sum(t['shares'] * t['price'] for t in trades if t['account_id'] == account and t['action'] == 'BUY')
            book.add_account(account, BrokerProfile('Wash analysis only', kind), cash)
        for asset in self.trades:
            book.add_security(Security(asset))
        for trade in trades:
            book.execute(trade['date'], trade['account_id'], trade['asset'], trade['action'], trade['shares'], trade['price'])
        self._book = book
        self.wash_sale_events = []
        self.disallowed_losses = defaultdict(float)
        self.allowed_losses = defaultdict(float)
        self.basis_adjustments = defaultdict(lambda: defaultdict(float))
        self.disallowed_by_year = defaultdict(lambda: defaultdict(float))
        self.allowed_by_year = defaultdict(lambda: defaultdict(float))
        self.tacked_shares = defaultdict(lambda: defaultdict(float))
        for realization in book.realizations:
            self.disallowed_losses[realization.security] += realization.disallowed
            self.disallowed_by_year[realization.security][realization.sold.year] += realization.disallowed
            allowed = max(0., -realization.gain)
            self.allowed_losses[realization.security] += allowed
            self.allowed_by_year[realization.security][realization.sold.year] += allowed
        lots = {l.id: l for l in book.lots}
        for event in book.events:
            if event['action'] != 'WASH_ADJUSTMENT':
                continue
            lot = lots[event['lot']]
            adjustment = 0. if event['permanent'] else event['disallowed']
            self.basis_adjustments[lot.security][lot.acquired] += adjustment
            self.tacked_shares[lot.security][lot.acquired] += event['shares'] if not event['permanent'] else 0.
            self.wash_sale_events.append(WashSaleEvent(
                event['sale_date'], lot.security, event['disallowed'], lot.acquired,
                event['shares'], adjustment, lot.acquired - timedelta(days=lot.tacked_days),
                event['shares'], event['sale_date'].year, lot.acquired.year,
                event['sale_date'].year != lot.acquired.year, permanent=event['permanent']))
        self._processed = True

    def check_wash_sale(self, asset, sale_day, loss_amount, all_trades=None):
        raise UnsupportedModelError('An aggregate loss cannot determine share matching; record all dated trades first')

    def get_total_disallowed(self):
        self.process_all_wash_sales()
        return sum(self.disallowed_losses.values())

    def get_total_allowed(self):
        self.process_all_wash_sales()
        return sum(self.allowed_losses.values())

    def get_basis_adjustment(self, asset, buy_day):
        self.process_all_wash_sales()
        return self.basis_adjustments[asset][self._date(buy_day)]

    def get_holding_period_adjustment(self, asset, buy_day):
        self.process_all_wash_sales()
        when = self._date(buy_day)
        lots = [l for l in self._book.lots if l.security == asset and l.acquired == when and l.shares > 0]
        offsets = {l.tacked_days for l in lots}
        if len(offsets) > 1:
            raise UnsupportedModelError('Partial replacements have distinct holding periods; inspect individual ledger lots')
        return when - timedelta(days=next(iter(offsets), 0))

    def get_tacked_shares(self, asset, buy_day):
        self.process_all_wash_sales()
        return self.tacked_shares[asset][self._date(buy_day)]

    def get_disallowed_for_year(self, asset, tax_year):
        self.process_all_wash_sales()
        return self.disallowed_by_year[asset][tax_year]

    def get_allowed_for_year(self, asset, tax_year):
        self.process_all_wash_sales()
        return self.allowed_by_year[asset][tax_year]

    def get_wash_sale_summary(self):
        self.process_all_wash_sales()
        return {'total_disallowed': self.get_total_disallowed(), 'total_allowed': self.get_total_allowed(),
                'events_count': len(self.wash_sale_events), 'events': self.wash_sale_events,
                'provisional_until': max((r.sold + timedelta(days=30) for r in self._book.realizations
                                          if r.unmatched > 0), default=None),
                'execution_verified': False}

    def get_cross_year_summary(self):
        self.process_all_wash_sales()
        events = [e for e in self.wash_sale_events if e.is_cross_year]
        return {'total_cross_year_events': len(events), 'total_cross_year_disallowed': sum(e.loss_amount for e in events),
                'chains_count': None, 'note': 'Use ledger lot lineage; legacy heuristic chain IDs are not legal rules'}

    def get_chain_info(self, chain_id):
        raise UnsupportedModelError('Legacy heuristic chain IDs are unsupported; inspect ledger lot events')

    def reset(self):
        self.trades = defaultdict(list)
        self._processed = False
        self._book = None
        self._sequence = 0
        self.wash_sale_events = []
