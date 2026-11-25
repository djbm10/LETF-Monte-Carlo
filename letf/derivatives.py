"""Dated futures settlement and quote-driven long-option mechanics.

Contract mechanics are not evidence that any historical fill was available.
Futures settlement-only runs use an explicit stress label. American-option
assignment and interacting derivative tax provisions fail closed.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isclose, isfinite
from numbers import Integral
from zoneinfo import ZoneInfo

from letf.accounts import positive, calendar_date
from letf.accuracy import Evidence, UnsupportedModelError


@dataclass(frozen=True)
class FuturesContract:
    symbol: str  # e.g. MESZ2025, never MES=F or a back-adjusted continuous chart
    expiry: date
    multiplier: float
    tick_size: float
    initial_margin: float
    maintenance_margin: float
    margin_as_of: date
    source: str
    section1256: bool  # affirmative classification; not inferred from a ticker
    settlement_type: str
    first_notice: date = None
    currency: str = 'USD'

    def __post_init__(self):
        for name in ('multiplier', 'tick_size', 'initial_margin', 'maintenance_margin'):
            positive(getattr(self, name), name)
        if self.maintenance_margin > self.initial_margin:
            raise ValueError('Maintenance margin cannot exceed initial margin')
        if not self.source or not self.symbol or '=' in self.symbol or self.currency != 'USD':
            raise UnsupportedModelError('Dated USD contracts with exchange/broker provenance are required')
        calendar_date(self.expiry); calendar_date(self.margin_as_of)
        if self.settlement_type not in ('cash', 'physical'):
            raise ValueError('Explicit futures settlement type is required')
        if self.settlement_type == 'physical' and self.first_notice is None:
            raise UnsupportedModelError('Physical futures require a first-notice/delivery cutoff')

    def validate_price(self, price):
        if not isfinite(price):
            raise ValueError('Nonfinite futures price')
        # Negative futures prices can occur; do not reject them generically.
        ticks = price / self.tick_size
        if not isclose(ticks, round(ticks), abs_tol=1e-7):
            raise ValueError('Futures price is not on the contract tick grid')


@dataclass
class FuturesPosition:
    contract: FuturesContract
    quantity: int
    settlement: float


class FuturesBook:
    """Uses the same account cash and event list as PortfolioLedger.

    Margin is reserved collateral, not a fee. Settlement cash P&L is counted
    once, including on closure. Margin schedules must be supplied externally
    with each change. No exact intraday margin behavior is inferred from a
    daily close. IRA permissions require an explicit evidenced profile.
    """
    def __init__(self, ledger, account_id, *, commission_per_contract, spread_ticks,
                 ira_margin_multiplier=None, profile_source=None, profile_verified_on=None):
        positive(commission_per_contract, 'commission', True)
        positive(spread_ticks, 'spread ticks', True)
        self.ledger = ledger
        self.account_id = account_id
        self.positions = {}
        self.commission = commission_per_contract
        self.spread_ticks = spread_ticks
        self.ira_margin_multiplier = ira_margin_multiplier
        self.profile_source = profile_source
        self.profile_verified_on = profile_verified_on
        self.last_settlement = {}
        account = ledger.accounts[account_id]
        if 'futures' in account.collateral_reservations:
            raise ValueError('Use one FuturesBook per account to aggregate house margin')
        if 'future' not in account.profile.allowed_kinds:
            raise UnsupportedModelError('Selected brokerage profile does not permit futures')
        if account.profile.account_type == 'roth' and (
            ira_margin_multiplier is None or ira_margin_multiplier < 1 or
            not profile_source or profile_verified_on is None):
            raise UnsupportedModelError('IRA futures require dated broker permission and a supplied house-margin multiplier')
        account.collateral_reservations['futures'] = 0.
        ledger.derivative_books.append(self)

    @property
    def multiplier(self):
        return self.ira_margin_multiplier or 1.

    def reserved(self, maintenance=False):
        return sum(abs(p.quantity) * (p.contract.maintenance_margin if maintenance else p.contract.initial_margin)
                   * self.multiplier for p in self.positions.values())

    def mark_value(self, prices):
        value = 0.
        for symbol, position in self.positions.items():
            price = prices[symbol]
            position.contract.validate_price(price)
            value += position.quantity * position.contract.multiplier * (price - position.settlement)
        return value

    def _book_pnl(self, when, pnl):
        account = self.ledger.accounts[self.account_id]
        account.cash += pnl
        if account.profile.account_type == 'taxable':
            inc = account.income.setdefault(when.year, {})
            inc['section1256'] = inc.get('section1256', 0.) + pnl

    def trade(self, when, contract, quantity, price):
        when = calendar_date(when)
        if (self.ledger.accounts[self.account_id].profile.account_type=='roth' and
                when<calendar_date(self.profile_verified_on)):
            raise UnsupportedModelError('IRA futures permission cannot be backdated before its supplied verification date')
        if type(quantity) is not int or quantity == 0:
            raise ValueError('Integer nonzero futures contracts required')
        if when > contract.expiry or when < contract.margin_as_of:
            raise UnsupportedModelError('Contract expired or margin schedule not yet available')
        if contract.first_notice is not None and when >= contract.first_notice:
            raise UnsupportedModelError('Physical delivery/notice requires an explicit delivery handler')
        if not contract.section1256:
            raise UnsupportedModelError('Non-1256 futures require a separate tax classification model')
        contract.validate_price(price)
        existing = self.positions.get(contract.symbol)
        if existing and existing.contract != contract:
            raise ValueError('Use change_margin to update an open contract schedule')
        old_qty = existing.quantity if existing else 0
        new_qty = old_qty + quantity
        # Spread is expressed as full bid/ask width; one side pays half.
        cost = abs(quantity) * (self.commission + self.spread_ticks / 2 * contract.tick_size * contract.multiplier)
        prior = existing.settlement if existing else price
        # Re-mark existing exposure at the trade price, then start the new
        # exposure there; equal to a full close/reopen without double P&L.
        pnl = old_qty * contract.multiplier * (price - prior)
        new_reserved = (self.reserved() - abs(old_qty) * contract.initial_margin * self.multiplier
                        + abs(new_qty) * contract.initial_margin * self.multiplier)
        when = self.ledger._advance(when)
        account = self.ledger.accounts[self.account_id]
        other_reserves = sum(v for k, v in account.collateral_reservations.items() if k != 'futures')
        opens_risk = abs(new_qty) > abs(old_qty) or old_qty * new_qty < 0
        if new_reserved > account.cash - other_reserves + pnl - cost + 1e-9 and opens_risk:
            raise ValueError('Insufficient initial/house margin')
        self._book_pnl(when, pnl - cost)
        if new_qty:
            self.positions[contract.symbol] = FuturesPosition(contract, new_qty, price)
        else:
            self.positions.pop(contract.symbol, None)
        account.collateral_reservations['futures'] = self.reserved()
        self.ledger.events.append({'date': when, 'account': self.account_id, 'action': 'FUTURES_FILL',
                                   'contract': contract.symbol, 'quantity': quantity, 'price': price,
                                   'cost': cost, 'variation_margin': pnl,
                                   'evidence': Evidence.HYPOTHETICAL.value})
        return self.ledger.events[-1]

    def settle(self, when, symbol, settlement, *, adverse_extreme=None, extreme_source=None):
        when = calendar_date(when)
        position = self.positions[symbol]
        contract = position.contract
        contract.validate_price(settlement)
        if when > contract.expiry:
            raise UnsupportedModelError('Expired contract must be explicitly cash-settled or rolled')
        if contract.first_notice is not None and when >= contract.first_notice:
            raise UnsupportedModelError('Physical delivery/notice requires an explicit delivery handler')
        if self.last_settlement.get(symbol) is not None and when <= self.last_settlement[symbol]:
            raise ValueError('Duplicate/backdated daily settlement')
        if adverse_extreme is not None:
            contract.validate_price(adverse_extreme)
            if not extreme_source:
                raise ValueError('Daily adverse extreme requires a price source')
        when = self.ledger._advance(when)
        account = self.ledger.accounts[self.account_id]
        other_reserves=sum(v for k,v in account.collateral_reservations.items() if k!='futures')
        pnl = position.quantity * contract.multiplier * (settlement - position.settlement)
        if adverse_extreme is None:
            possible_intraday_breach = None  # unknown, not False
        else:
            adverse_pnl = position.quantity * contract.multiplier * (adverse_extreme - position.settlement)
            possible_intraday_breach = account.cash - other_reserves + adverse_pnl < self.reserved(maintenance=True)
        self._book_pnl(when, pnl)
        position.settlement = settlement
        self.last_settlement[symbol] = when
        margin_call = account.cash - other_reserves < self.reserved(maintenance=True)
        event = {'date': when, 'account': self.account_id, 'action': 'DAILY_SETTLEMENT',
                 'contract': symbol, 'variation_margin': pnl, 'margin_call': margin_call,
                 'possible_intraday_breach': possible_intraday_breach,
                 'execution_exact': False, 'evidence': Evidence.HYPOTHETICAL.value}
        self.ledger.events.append(event)
        return event

    def force_liquidate(self, when, prices):
        """Explicit supplied fills, including gaps and commissions; debt may remain."""
        if set(self.positions) - set(prices):
            raise ValueError('A liquidation price is required for each contract')
        for symbol, position in self.positions.items():
            position.contract.validate_price(prices[symbol])
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                raise ValueError('A liquidation price is required for each contract')
            self.trade(when, position.contract, -position.quantity, prices[symbol])
        self.ledger.events.append({'date': calendar_date(when), 'account': self.account_id,
                                   'action': 'FORCED_LIQUIDATION',
                                   'cash': self.ledger.accounts[self.account_id].cash,
                                   'execution_exact': False})

    def change_margin(self, when, symbol, initial, maintenance, *, source):
        from dataclasses import replace
        when = calendar_date(when)
        if not source:
            raise ValueError('Margin change requires source')
        contract = replace(self.positions[symbol].contract, initial_margin=initial,
                           maintenance_margin=maintenance, margin_as_of=when, source=source)
        when = self.ledger._advance(when)
        self.positions[symbol].contract = contract
        self.ledger.accounts[self.account_id].collateral_reservations['futures'] = self.reserved()
        account=self.ledger.accounts[self.account_id]
        other_reserves=sum(v for k,v in account.collateral_reservations.items() if k!='futures')
        breach = account.cash - other_reserves < self.reserved(maintenance=True)
        self.ledger.events.append({'date': when, 'action': 'MARGIN_CHANGE', 'contract': symbol,
                                   'initial': initial, 'maintenance': maintenance, 'margin_call': breach})
        return breach

    def year_end(self, when, settlements):
        """Mark every open qualifying contract to the supplied year-end settlement.

        Daily P&L already booked into that year is not added a second time.
        Caller supplies the exchange's final trading-day date and settlements.
        """
        when = calendar_date(when)
        if set(settlements) != set(self.positions):
            raise ValueError('Year-end settlements required for all open dated contracts')
        for symbol, price in settlements.items():
            if self.last_settlement.get(symbol) == when:
                if self.positions[symbol].settlement != price:
                    raise ValueError('Conflicting year-end and daily settlements')
                continue
            self.settle(when, symbol, price)
        self.ledger.events.append({'date': when, 'action': 'SECTION1256_YEAR_END',
                                   'account': self.account_id, 'basis_reset': True})


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    strike: float
    expiry: date
    right: str
    style: str
    settlement: str
    multiplier: int = 100
    adjusted: bool = False

    def __post_init__(self):
        positive(self.strike, 'strike')
        calendar_date(self.expiry)
        if not self.symbol or not self.underlying:
            raise ValueError('Explicit option and underlying identities required')
        if self.right not in ('call', 'put') or self.style not in ('american', 'european'):
            raise ValueError('Invalid option contract')
        if self.settlement not in ('physical', 'cash') or type(self.multiplier) is not int or self.multiplier <= 0:
            raise ValueError('Invalid option settlement/multiplier')
        if self.adjusted:
            raise UnsupportedModelError('Adjusted options require the OCC deliverable memo')


@dataclass(frozen=True)
class OptionQuote:
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    source: str
    observed: bool = True
    contract_symbol: str = None

    def fill(self, timestamp, side, contracts, *, max_age=timedelta(minutes=1)):
        if not self.observed or not self.source or not self.contract_symbol:
            raise UnsupportedModelError('Theoretical option values cannot be used as historical executable quotes')
        if self.timestamp.tzinfo is None or timestamp.tzinfo is None:
            raise ValueError('Timezone-aware option quote and order timestamps are required')
        if not timedelta(0) <= timestamp - self.timestamp <= max_age:
            raise ValueError('Future/stale option quote')
        if (not isfinite(self.bid) or not isfinite(self.ask) or self.bid < 0 or self.ask <= 0 or self.ask < self.bid):
            raise ValueError('Invalid/crossed option market')
        if type(contracts) is not int or contracts <= 0 or side not in ('BUY', 'SELL'):
            raise ValueError('Integer option contracts and BUY/SELL side required')
        size = self.ask_size if side == 'BUY' else self.bid_size
        if isinstance(size,bool) or not isinstance(size,Integral) or size<0:
            raise ValueError('Displayed option size must be a nonnegative integer, not missing/estimated depth')
        if contracts > size:
            raise UnsupportedModelError('Requested option fill exceeds displayed size; supply explicit partial fills')
        return self.ask if side == 'BUY' else self.bid


def option_collateral(strategy, contracts, *, strike, multiplier=100,
                      premium_debit=0., spread_width=None, stock_shares=0.):
    if type(contracts) is not int or contracts <= 0:
        raise ValueError('Integer positive option contracts required')
    positive(strike, 'strike'); positive(premium_debit, 'premium debit', True)
    if strategy in ('long_call', 'long_put', 'leaps', 'protective_put'):
        return contracts * multiplier * premium_debit
    if strategy == 'cash_secured_put':
        return contracts * multiplier * strike  # Do not assume premium is settled collateral.
    if strategy == 'covered_call':
        if stock_shares < contracts * multiplier:
            raise ValueError('Insufficient unencumbered shares to cover calls')
        return 0.
    if strategy in ('call_spread', 'put_spread'):
        if spread_width is None or spread_width <= 0:
            raise ValueError('Positive spread width required')
        return contracts * multiplier * max(spread_width, premium_debit)
    raise UnsupportedModelError('Unvalidated option collateral combination')


def require_derivative_tax_support(strategy, account_type):
    # Eligibility alone cannot validate straddle loss deferral, qualified
    # covered calls, constructive sales or holding-period suspension.
    if strategy in ('covered_call', 'collar', 'protective_put', 'call_spread', 'put_spread') and account_type == 'taxable':
        raise UnsupportedModelError('Interacting derivative taxes require IRC 1092/1259/1234/qualified-covered-call modeling')


class LongOptionBook:
    """Real-quote long options/LEAPS in an explicitly permitted Roth profile.

    Physical long exercise requires a sourced notice and sufficient cash or
    owned shares. Taxable interactions and short assignment remain blocked.
    Marks use the liquidation bid, not a model. Sale proceeds settle T+1.
    """
    def __init__(self, ledger, account_id, *, commission_per_contract=0.):
        self.ledger = ledger
        self.account_id = account_id
        self.positions = {}
        self.action_cursor = {}
        self.last_timestamp = None
        positive(commission_per_contract, 'option commission', True)
        self.commission = commission_per_contract
        account = ledger.accounts[account_id]
        if 'option' not in account.profile.allowed_kinds:
            raise UnsupportedModelError('Options require brokerage approval in the selected account')
        if account.profile.account_type == 'taxable':
            raise UnsupportedModelError('Taxable option wash/holding-period/exercise treatment is not yet integrated')
        ledger.derivative_books.append(self)

    def _check_actions(self,symbol):
        contract,_=self.positions[symbol]
        if any(e.get('security')==contract.underlying and e['action']=='SPLIT'
               for e in self.ledger.events[self.action_cursor.get(symbol,0):]):
            raise UnsupportedModelError('Held option crosses an underlying split; an OCC adjusted-deliverable handler is required')

    @staticmethod
    def settlement_date(when):
        import exchange_calendars as xcals
        import pandas as pd
        when=calendar_date(when)
        calendar=xcals.get_calendar('XNYS',start=str(when-timedelta(days=10)),end=str(when+timedelta(days=15)))
        if not calendar.is_session(pd.Timestamp(when)):raise ValueError('Option action must occur on an exchange session')
        return calendar.next_session(pd.Timestamp(when)).date()

    def mark_value(self, prices):
        value = 0.
        for symbol, (contract, quantity) in self.positions.items():
            self._check_actions(symbol)
            positive(prices[symbol], 'option liquidation bid', True)
            value += prices[symbol] * quantity * contract.multiplier
        return value

    def trade(self, timestamp, contract, side, contracts, quote):
        if quote.contract_symbol != contract.symbol:
            raise ValueError('Option quote does not match the dated contract')
        price = quote.fill(timestamp, side, contracts)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError('Option executions must be chronological within each session')
        when = timestamp.astimezone(ZoneInfo('America/New_York')).date()
        if when > contract.expiry:
            raise ValueError('Option expired')
        account = self.ledger.accounts[self.account_id]
        old_contract, old_qty = self.positions.get(contract.symbol, (contract, 0))
        if old_contract != contract:
            raise ValueError('Contract adjustment requires OCC action handling')
        if old_qty:self._check_actions(contract.symbol)
        settle=self.settlement_date(when)
        self.ledger._advance(when)
        cash = price * contracts * contract.multiplier
        fee = contracts * self.commission
        if side == 'BUY':
            if account.buying_power < cash + fee:
                raise ValueError('Insufficient option premium cash/collateral')
            new_qty = old_qty + contracts
            delta = -cash - fee
        else:
            if old_qty < contracts:
                raise UnsupportedModelError('Short option writing/assignment is not implemented')
            new_qty = old_qty - contracts
            delta = cash - fee
            if delta<0 and account.buying_power < -delta:
                raise ValueError('Insufficient settled cash to pay option close commission')
        if side=='SELL' and delta>=0:account.unsettled.append((settle,delta))
        else:account.cash += delta
        if new_qty:
            self.positions[contract.symbol] = (contract, new_qty)
            if not old_qty:self.action_cursor[contract.symbol]=len(self.ledger.events)
        else:
            self.positions.pop(contract.symbol, None)
            self.action_cursor.pop(contract.symbol,None)
        self.ledger.events.append({'date': when, 'timestamp': timestamp, 'account': self.account_id,
                                   'action': 'OPTION_FILL', 'contract': contract.symbol,
                                   'side': side, 'contracts': contracts, 'price': price,
                                   'commission': fee, 'source': quote.source,
                                   'settlement_date':settle,
                                   'evidence': Evidence.OBSERVED.value,
                                   'execution_exact': False})
        self.last_timestamp=timestamp
        return self.ledger.events[-1]

    def exercise(self,when,symbol,contracts,*,notice_source,exercise_fee=0.):
        """Explicit exercise of a standard long option; never inferred optimal.

        OCC equity delivery is T+1 from May 28, 2024. Earlier delivery calendars
        and adjusted contracts need their own effective-date evidence.
        """
        when=calendar_date(when);positive(exercise_fee,'exercise fee',True)
        contract,quantity=self.positions[symbol];self._check_actions(symbol)
        if not notice_source:raise ValueError('Explicit exercise notice/election evidence is required')
        if type(contracts)is not int or not 0<contracts<=quantity:raise ValueError('Invalid integer exercise quantity')
        if contract.settlement!='physical':raise UnsupportedModelError('Cash-settled early exercise needs product-specific settlement rules')
        if when>contract.expiry or (contract.style=='european' and when!=contract.expiry):raise ValueError('Exercise outside the contract exercise window')
        if when<date(2024,5,28):raise UnsupportedModelError('Historical physical exercise settlement version is not verified')
        event=self.ledger.execute(when,self.account_id,contract.underlying,
            'BUY' if contract.right=='call' else 'SELL',contracts*contract.multiplier,contract.strike,
            settlement_date=self.settlement_date(when),execution_kind='exercise',
            execution_source=notice_source,commission_override=exercise_fee)
        remaining=quantity-contracts
        if remaining:self.positions[symbol]=(contract,remaining)
        else:self.positions.pop(symbol);self.action_cursor.pop(symbol,None)
        self.ledger.events.append({'date':when,'account':self.account_id,'action':'OPTION_EXERCISE',
            'contract':symbol,'contracts':contracts,'source':notice_source,
            'settlement_date':event['settlement_date'],'automatic_exercise_assumed':False})
        return self.ledger.events[-1]

    def expire(self, when, symbol, settlement_price,*,exercise_notice_source=None):
        contract, quantity = self.positions[symbol]
        self._check_actions(symbol)
        when = calendar_date(when)
        positive(settlement_price, 'underlying official settlement', True)
        if when != contract.expiry:
            raise ValueError('Expiry settlement must be on the contract expiration date')
        intrinsic = max(0., settlement_price - contract.strike) if contract.right == 'call' else max(0., contract.strike - settlement_price)
        if intrinsic > 0 and contract.settlement == 'physical':
            if not exercise_notice_source:
                raise UnsupportedModelError('In-the-money physical options require an explicit exercise notice, not assumed assignment')
            return self.exercise(when,symbol,quantity,notice_source=exercise_notice_source)
        settle=self.settlement_date(when)
        self.ledger._advance(when)
        proceeds = intrinsic * quantity * contract.multiplier
        if proceeds:self.ledger.accounts[self.account_id].unsettled.append((settle,proceeds))
        del self.positions[symbol]
        self.action_cursor.pop(symbol,None)
        self.ledger.events.append({'date': when, 'account': self.account_id, 'action': 'OPTION_EXPIRY',
                                   'contract': symbol, 'cash': proceeds,'settlement_date':settle})

    def mark(self, timestamp, quotes):
        if timestamp.tzinfo is None:raise ValueError('Option valuation timestamp must be timezone-aware')
        if self.last_timestamp is not None and timestamp<self.last_timestamp:
            raise ValueError('Cannot value positions before their latest option execution')
        value = 0.
        for symbol, (contract, quantity) in self.positions.items():
            self._check_actions(symbol)
            # A valuation does not assume enough displayed depth to liquidate
            # the full position. Fill size is checked only on actual orders.
            quote = quotes[symbol]
            if quote.contract_symbol != symbol:
                raise ValueError('Valuation quote does not match the held option contract')
            if timestamp.astimezone(ZoneInfo('America/New_York')).date() > contract.expiry:
                raise UnsupportedModelError('Expired position requires explicit expiry/exercise settlement')
            quote.fill(timestamp, 'SELL', 1)
            value += quote.bid * quantity * contract.multiplier
        return value
