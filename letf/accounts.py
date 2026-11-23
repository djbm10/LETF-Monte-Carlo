"""Shared dated cash/position/tax ledger for supported cash-account instruments.

This module deliberately rejects borrowing and unimplemented derivative taxes.
Both replay and new simulations use the same execution methods. Tax reporting
is a separate close process: a December loss remains provisional through its
January wash-sale window. Rules and exclusions: docs/accuracy-audit/TAX_RULES.md.
"""
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from math import isfinite
from typing import Optional

from letf.accuracy import UnsupportedModelError
from letf.tax.engine import TaxpayerElections, compute_capital_gains
from letf.tax.marginal import calculate_comprehensive_tax_v6
from letf.tax.years import federal_parameters


def positive(value, name, allow_zero=False):
    if not isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')


def calendar_date(value):
    if isinstance(value, str):
        return date.fromisoformat(value)
    if type(value) is not date:
        raise ValueError('A calendar date is required; trading-session ordinals are not dates')
    return value


@dataclass(frozen=True)
class BrokerProfile:
    name: str
    account_type: str
    allowed_kinds: frozenset = frozenset({'stock', 'etf'})
    fractional_shares: bool = True
    minimum_equity: float = 0.
    commission: float = 0.
    spread_bps: float = 0.
    slippage_bps: float = 0.
    # No profile is represented as a verified actual broker by default.
    source: Optional[str] = None
    verified_on: Optional[date] = None

    def __post_init__(self):
        if self.account_type not in ('taxable', 'roth'):
            raise ValueError('Account type must be taxable or roth')
        for name in ('minimum_equity', 'commission', 'spread_bps', 'slippage_bps'):
            positive(getattr(self, name), name, True)


TAXABLE_CASH = BrokerProfile('Generic taxable cash account', 'taxable')
CONSERVATIVE_ROTH = BrokerProfile('Generic conservative Roth cash account', 'roth')


@dataclass(frozen=True)
class Security:
    symbol: str
    kind: str = 'etf'
    identity: Optional[str] = None  # caller-reviewed substantially identical group
    inception: Optional[date] = None
    closure: Optional[date] = None
    tax_structure: str = 'corporation'


@dataclass
class Lot:
    id: str
    account: str
    security: str
    acquired: date
    shares: float
    basis_per_share: float
    tacked_days: int = 0
    replacement_used: bool = False


@dataclass
class Realization:
    account: str
    security: str
    sold: date
    lot_id: str
    shares: float
    proceeds: float
    basis: float
    long_term: bool
    held_days: int
    disallowed: float = 0.
    permanent_disallowed: float = 0.
    unmatched: float = 0.

    @property
    def gain(self):
        return self.proceeds - self.basis + self.disallowed


@dataclass
class Account:
    id: str
    profile: BrokerProfile
    cash: float
    taxpayer: str = 'owner'
    income: dict = field(default_factory=dict)
    unsettled: list = field(default_factory=list)
    collateral_reservations: dict = field(default_factory=dict)

    @property
    def buying_power(self):
        return self.cash - sum(self.collateral_reservations.values())


class PortfolioLedger:
    """Chronological event replay, across accounts belonging to the same owner.

    Quotes are execution inputs, not fabricated from total-return adjusted
    prices. Commissions enter basis/proceeds, and each fill is separately
    recorded. No order is partially filled unless a fill quantity is supplied.
    """
    def __init__(self):
        self.accounts = {}
        self.securities = {}
        self.lots = []
        self.realizations = []
        self.events = []
        self.last_date = None
        self._next_id = 1
        self._sealed_through = None
        self.derivative_books = []

    def add_account(self, account_id, profile, cash=0., taxpayer='owner'):
        positive(cash, 'cash', True)
        if account_id in self.accounts:
            raise ValueError('Duplicate account id')
        self.accounts[account_id] = Account(account_id, profile, cash, taxpayer)

    def add_security(self, security):
        if security.symbol in self.securities:
            raise ValueError('Duplicate security')
        if security.kind not in ('stock', 'etf') or security.tax_structure != 'corporation':
            raise UnsupportedModelError('Use the derivative engine or supply a validated partnership/UBTI model')
        self.securities[security.symbol] = security

    def _advance(self, when):
        when = calendar_date(when)
        if self._sealed_through is not None and when <= self._sealed_through:
            raise ValueError('Cannot append an event inside a sealed tax observation window')
        if self.last_date is not None and when < self.last_date:
            raise ValueError('Events must be chronological across all linked accounts')
        self.last_date = when
        for account in self.accounts.values():
            matured = [x for x in account.unsettled if x[0] <= when]
            account.cash += sum(x[1] for x in matured)
            account.unsettled = [x for x in account.unsettled if x[0] > when]
        return when

    def _id(self):
        result = f'L{self._next_id}'
        self._next_id += 1
        return result

    def quantity(self, account, symbol):
        return sum(l.shares for l in self.lots if l.account == account and l.security == symbol)

    def _identity(self, symbol):
        security = self.securities[symbol]
        return security.identity or symbol

    def _match(self, realization, lot, when):
        if realization.unmatched <= 1e-10 or lot.replacement_used or lot.shares <= 1e-10:
            return
        qty = min(realization.unmatched, lot.shares)
        loss_per_share = (realization.basis - realization.proceeds) / realization.shares
        # Split partial replacements so basis and holding periods do not leak
        # into shares that did not replace the sold shares.
        if qty < lot.shares - 1e-10:
            remainder = replace(lot, id=self._id(), shares=lot.shares - qty)
            self.lots.append(remainder)
            lot.shares = qty
        disallowed = qty * loss_per_share
        lot.replacement_used = True
        realization.disallowed += disallowed
        realization.unmatched -= qty
        permanent = self.accounts[lot.account].profile.account_type == 'roth'
        if permanent:
            realization.permanent_disallowed += disallowed
        else:
            lot.basis_per_share += loss_per_share
            lot.tacked_days += realization.held_days
        self.events.append({'date': when, 'action': 'WASH_ADJUSTMENT',
                            'sale_date': realization.sold, 'lot': lot.id,
                            'shares': qty, 'disallowed': disallowed, 'permanent': permanent})

    def _match_new_purchase(self, lot, when):
        for realization in self.realizations:
            if (self.accounts[realization.account].taxpayer == self.accounts[lot.account].taxpayer
                and self._identity(realization.security) == self._identity(lot.security)
                and timedelta(0) <= when - realization.sold <= timedelta(days=30)):
                # The initial split can leave an unused sibling for other losses.
                available = sorted([l for l in self.lots if l.account == lot.account
                                    and l.security == lot.security and l.acquired == when
                                    and not l.replacement_used], key=lambda l: int(l.id[1:]))
                for candidate in available:
                    self._match(realization, candidate, when)

    def execute(self, when, account_id, symbol, side, shares, price, *,
                method='FIFO', lot_ids=None, fill_shares=None, settlement_date=None):
        when = calendar_date(when)
        positive(shares, 'shares'); positive(price, 'price')
        if side not in ('BUY', 'SELL'):
            raise ValueError('Side must be BUY or SELL')
        qty = shares if fill_shares is None else fill_shares
        positive(qty, 'fill shares')
        if qty > shares:
            raise ValueError('Fill exceeds order quantity')
        account = self.accounts[account_id]
        security = self.securities[symbol]
        profile = account.profile
        if security.kind not in profile.allowed_kinds:
            raise UnsupportedModelError('Instrument is not permitted by the brokerage profile')
        if not profile.fractional_shares and not float(qty).is_integer():
            raise ValueError('Broker requires integer shares')
        if security.inception and when < security.inception:
            raise UnsupportedModelError('Product did not yet exist; use a labelled proxy instrument')
        if security.closure and when >= security.closure:
            raise UnsupportedModelError('Product is closed; provide liquidation corporate action')
        slip = (profile.spread_bps / 2 + profile.slippage_bps) / 10000
        execution_price = price * (1 + slip if side == 'BUY' else 1 - slip)
        positive(execution_price, 'execution price')
        settle = calendar_date(settlement_date) if settlement_date else when
        if settle < when:
            raise ValueError('Settlement cannot precede execution')
        # Advance only after elementary validation. Settle earlier proceeds
        # before evaluating today's buying power.
        self._advance(when)
        if side == 'BUY':
            cost = qty * execution_price + profile.commission
            if account.buying_power + 1e-9 < cost or account.cash < profile.minimum_equity:
                raise ValueError('Insufficient settled cash or brokerage account minimum')
            account.cash -= cost
            lot = Lot(self._id(), account_id, symbol, when, qty, cost / qty)
            self.lots.append(lot)
            self._match_new_purchase(lot, when)
        else:
            lots = [l for l in self.lots if l.account == account_id and l.security == symbol and l.shares > 1e-10]
            if method == 'FIFO':
                lots.sort(key=lambda l: (l.acquired, int(l.id[1:])))
            elif method == 'HIFO':
                lots.sort(key=lambda l: (-l.basis_per_share, l.acquired, int(l.id[1:])))
            elif method == 'SPEC_ID':
                if not lot_ids or len(lot_ids) != len(set(lot_ids)):
                    raise ValueError('Specific identification requires unique explicit lot ids')
                by_id = {l.id: l for l in lots}
                if any(k not in by_id for k in lot_ids):
                    raise ValueError('Specific lot does not exist in this account/security')
                lots = [by_id[k] for k in lot_ids]
            else:
                raise UnsupportedModelError('Audited lot methods: FIFO, HIFO, SPEC_ID')
            if sum(l.shares for l in lots) < qty - 1e-9:
                raise ValueError('Insufficient shares; naked short sales are unsupported')
            proceeds = qty * execution_price - profile.commission
            if proceeds < 0:
                raise ValueError('Commission exceeds sale proceeds')
            remaining = qty
            new_realizations = []
            for lot in lots:
                n = min(remaining, lot.shares)
                if n <= 1e-10:
                    continue
                effective_acquired = lot.acquired - timedelta(days=lot.tacked_days)
                try:
                    anniversary = effective_acquired.replace(year=effective_acquired.year + 1)
                except ValueError:  # February 29
                    anniversary = effective_acquired.replace(year=effective_acquired.year + 1, day=28)
                realization = Realization(account_id, symbol, when, lot.id, n,
                                          proceeds * n / qty, n * lot.basis_per_share,
                                          when > anniversary, (when - lot.acquired).days + lot.tacked_days)
                lot.shares -= n
                remaining -= n
                if profile.account_type == 'taxable':
                    if realization.gain < 0:
                        realization.unmatched = n
                    new_realizations.append(realization)
                if remaining <= 1e-10:
                    break
            # Determine pre-sale replacements AFTER reducing every sold lot.
            # This prevents treating the very shares sold as replacements.
            for realization in new_realizations:
                self.realizations.append(realization)
                candidates = sorted([l for l in self.lots if l.shares > 1e-10
                                     and self.accounts[l.account].taxpayer == account.taxpayer
                                     and self._identity(l.security) == self._identity(symbol)
                                     and timedelta(0) <= when - l.acquired <= timedelta(days=30)],
                                    key=lambda l: (l.acquired, int(l.id[1:])))
                for candidate in candidates:
                    self._match(realization, candidate, when)
            if settle == when:
                account.cash += proceeds
            else:
                account.unsettled.append((settle, proceeds))
        self.events.append({'date': when, 'account': account_id, 'security': symbol,
                            'action': side, 'shares': qty, 'price': execution_price,
                            'commission': profile.commission,
                            'spread_slippage_cost': qty * abs(execution_price - price),
                            'settlement_date': settle})
        return self.events[-1]

    def income(self, when, account_id, amount, category, *, qualified_verified=False):
        positive(amount, 'income', True)
        if category not in ('interest', 'ordinary_dividend', 'qualified_dividend', 'capital_gain_distribution'):
            raise UnsupportedModelError('Unsupported distribution classification')
        if category == 'qualified_dividend' and not qualified_verified:
            raise UnsupportedModelError('Qualified-dividend issuer and holding-period tests must be verified')
        when = self._advance(when)
        account = self.accounts[account_id]
        account.cash += amount
        bucket = account.income.setdefault(when.year, {})
        bucket[category] = bucket.get(category, 0.) + amount
        self.events.append({'date': when, 'account': account_id, 'action': 'INCOME',
                            'category': category, 'amount': amount})

    def split(self, when, symbol, ratio):
        positive(ratio, 'split ratio')
        when = self._advance(when)
        if any(r.unmatched > 1e-10 and abs((when - r.sold).days) <= 30
               and self._identity(r.security) == self._identity(symbol) for r in self.realizations):
            raise UnsupportedModelError('Split during pending wash matching requires adjusted share identities')
        for lot in self.lots:
            if lot.security == symbol:
                lot.shares *= ratio
                lot.basis_per_share /= ratio
        self.events.append({'date': when, 'action': 'SPLIT', 'security': symbol, 'ratio': ratio})

    def wealth(self, account_id, prices):
        account = self.accounts[account_id]
        value = account.cash + sum(x[1] for x in account.unsettled)
        for lot in self.lots:
            if lot.account == account_id and lot.shares > 1e-10:
                positive(prices[lot.security], 'mark', True)
                value += lot.shares * prices[lot.security]
        for book in self.derivative_books:
            if book.account_id == account_id:
                value += book.mark_value(prices)
        return value

    def annual_tax(self, account_id, year, ordinary_income, *, state_code=None,
                   federal_only=False, filing_status='single', st_cf=0., lt_cf=0.,
                   as_of=None, freeze_tax_year=None):
        account = self.accounts[account_id]
        if account.profile.account_type == 'roth':
            return {'tax': 0., 'st_cf': 0., 'lt_cf': 0., 'scope': 'Roth internal trades only'}
        as_of = calendar_date(as_of) if as_of else self.last_date
        pending = [r for r in self.realizations if r.account == account_id and r.sold.year == year
                   and r.unmatched > 1e-10 and r.sold + timedelta(days=30) > as_of]
        if pending:
            raise ValueError('Tax year is provisional until the wash-sale observation window closes')
        table_year = freeze_tax_year if freeze_tax_year is not None else year
        _, _, standard, _ = federal_parameters(table_year, filing_status)
        activity = {k: 0. for k in ('st_gains', 'st_losses', 'lt_gains', 'lt_losses')}
        for realization in self.realizations:
            if realization.account == account_id and realization.sold.year == year:
                character = 'lt' if realization.long_term else 'st'
                activity[character + ('_gains' if realization.gain >= 0 else '_losses')] += abs(realization.gain)
        inc = account.income.get(year, {})
        activity['lt_gains'] += inc.get('capital_gain_distribution', 0.)
        section1256 = inc.get('section1256', 0.)
        for character, fraction in (('st', .4), ('lt', .6)):
            activity[character + ('_gains' if section1256 >= 0 else '_losses')] += abs(section1256) * fraction
        result = compute_capital_gains(**activity, st_loss_cf_in=st_cf, lt_loss_cf_in=lt_cf,
                                      elections=TaxpayerElections(), filing_status=filing_status,
                                      taxable_income_before_loss=max(0., ordinary_income +
                                          inc.get('interest', 0.) + inc.get('ordinary_dividend', 0.) +
                                          inc.get('qualified_dividend', 0.) - standard))
        kwargs = dict(ordinary_income=ordinary_income, include_state=not federal_only,
                      state_code=state_code, filing_status=filing_status, tax_year=table_year)
        baseline = calculate_comprehensive_tax_v6(0., 0., 0., **kwargs)
        total = calculate_comprehensive_tax_v6(
            result.taxable_st, result.taxable_lt, result.capital_loss_deduction,
            qualified_dividends=inc.get('qualified_dividend', 0.),
            ordinary_dividends=inc.get('ordinary_dividend', 0.),
            interest_income=inc.get('interest', 0.), **kwargs)
        # Loss deductions may reduce household taxes: preserve the negative
        # incremental liability; cash refunds require a separate payment event.
        return {'tax': total['total_tax'] - baseline['total_tax'],
                'st_cf': result.st_loss_cf_out, 'lt_cf': result.lt_loss_cf_out,
                'taxable_st': result.taxable_st, 'taxable_lt': result.taxable_lt,
                'capital_loss_deduction': result.capital_loss_deduction,
                'calculation': total, 'baseline': baseline,
                'evidence': 'HYPOTHETICAL STRESS MODEL' if freeze_tax_year is not None else 'OBSERVED',
                'scope': 'regular federal tax/NIIT, supplied income; excludes AMT/credits'}

    def pay_tax(self, when, account_id, amount):
        if not isfinite(amount):
            raise ValueError('Nonfinite tax payment')
        when = self._advance(when)
        account = self.accounts[account_id]
        if account.profile.account_type != 'taxable':
            raise UnsupportedModelError('Personal tax cannot be paid from Roth cash as an internal trade')
        if account.buying_power < amount - 1e-9:
            raise ValueError('Insufficient tax cash: execute and tax the necessary sales first')
        account.cash -= amount
        self.events.append({'date': when, 'account': account_id, 'action': 'TAX_PAYMENT', 'amount': amount})

    def harvest_losses(self, when, account_id, prices, *, minimum_loss, replacements=None):
        """Realize selected losing lots; no deduction or future-wash guarantee.

        Replacements must be caller-reviewed distinct identities. This is a
        trading action, not an elective change to Schedule D ordering. Existing
        prior-window purchases and subsequent linked-account purchases still
        pass through ordinary wash matching.
        """
        positive(minimum_loss, 'minimum harvest loss')
        if self.accounts[account_id].profile.account_type != 'taxable':
            raise UnsupportedModelError('Roth internal losses cannot be tax-loss harvested')
        for symbol, replacement in (replacements or {}).items():
            if self._identity(symbol) == self._identity(replacement):
                raise UnsupportedModelError('Harvest replacement must have a reviewed distinct identity')
        candidates = [l for l in self.lots if l.account == account_id and l.shares > 1e-10
                      and (l.basis_per_share - prices[l.security]) * l.shares >= minimum_loss]
        start = len(self.realizations)
        for lot in candidates:
            quantity, symbol = lot.shares, lot.security
            before = self.accounts[account_id].cash
            self.execute(when, account_id, symbol, 'SELL', quantity, prices[symbol],
                         method='SPEC_ID', lot_ids=[lot.id])
            replacement = (replacements or {}).get(symbol)
            if replacement:
                proceeds = self.accounts[account_id].cash - before
                profile = self.accounts[account_id].profile
                slip = (profile.spread_bps / 2 + profile.slippage_bps) / 10000
                shares = max(0., (proceeds - profile.commission) / (prices[replacement] * (1 + slip)))
                if not profile.fractional_shares:
                    shares = float(int(shares))
                if shares > 1e-10:
                    self.execute(when, account_id, replacement, 'BUY', shares, prices[replacement])
        return self.realizations[start:]

    def seal_observations(self, through):
        """Caller attests all linked-account events through this date were supplied."""
        through = calendar_date(through)
        if self.last_date and through < self.last_date:
            raise ValueError('Cannot seal before the last recorded event')
        self._sealed_through = through
