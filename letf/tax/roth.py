"""Aggregate owner-level Roth cash flows; distinct from investment taxation.

IRC 408A; IRS Publications 590-A/590-B (2025), Notice 2025-67.
Only ordinary owner distributions and explicit age/disability/death exceptions
are covered. Inherited IRAs, first-home exceptions, excess corrections, and
designated-Roth-plan rollover clocks require their own rules and are rejected.
"""
from dataclasses import dataclass
from datetime import date
from math import ceil, isfinite
from dateutil.relativedelta import relativedelta

from letf.accounts import calendar_date, positive
from letf.accuracy import UnsupportedModelError

LIMITS = {2024: (7000, 1000), 2025: (7000, 1000), 2026: (7500, 1100)}
PHASEOUT = {2024: {'single': (146000, 161000), 'married': (230000, 240000)},
            2025: {'single': (150000, 165000), 'married': (236000, 246000)},
            2026: {'single': (153000, 168000), 'married': (242000, 252000)}}


def contribution_limit(year, age_at_year_end, magi, compensation, *,
                       filing_status='single', other_ira_contributions=0.,
                       lived_with_spouse=True):
    """590-A Worksheet 2-2; full-precision phaseout then ceiling to $10."""
    for value in (age_at_year_end, magi, compensation, other_ira_contributions):
        if not isfinite(value):
            raise ValueError('Nonfinite Roth eligibility input')
    positive(compensation, 'taxable compensation', True)
    positive(other_ira_contributions, 'other IRA contributions', True)
    if age_at_year_end < 0:
        raise ValueError('Age cannot be negative')
    if year not in LIMITS:
        raise UnsupportedModelError(f'No verified IRA contribution table for {year}')
    status = {'married_filing_jointly': 'married', 'head_of_household': 'single'}.get(filing_status, filing_status)
    if status == 'married_filing_separately':
        low, high = (0, 10000) if lived_with_spouse else PHASEOUT[year]['single']
    elif status in PHASEOUT[year]:
        low, high = PHASEOUT[year][status]
    else:
        raise UnsupportedModelError('Unsupported Roth filing status')
    annual, catchup = LIMITS[year]
    base = min(compensation, annual + (catchup if age_at_year_end >= 50 else 0))
    if magi >= high or base == 0:
        return 0.
    if magi <= low:
        reduced = base
    else:
        reduced = min(base, max(200., ceil(base * (high - magi) / (high - low) / 10) * 10.))
    return max(0., min(reduced, base - other_ira_contributions))


@dataclass
class Conversion:
    year: int
    taxable_remaining: float
    nontaxable_remaining: float


class RothHistory:
    """One instance per owner, shared by all of that owner's Roth accounts."""
    def __init__(self, ledger, taxpayer, birth_date, *, first_tax_year=None,
                 regular_basis=0., conversions=()):
        positive(regular_basis, 'regular contribution basis', True)
        self.ledger = ledger
        self.taxpayer = taxpayer
        self.birth_date = calendar_date(birth_date)
        self.first_tax_year = first_tax_year
        self.regular_basis = regular_basis
        self.contributions = {}
        self.conversions = sorted([Conversion(c.year, c.taxable_remaining, c.nontaxable_remaining)
                                   for c in conversions], key=lambda c: c.year)
        for conversion in self.conversions:
            positive(conversion.taxable_remaining, 'conversion taxable basis', True)
            positive(conversion.nontaxable_remaining, 'conversion nontaxable basis', True)

    def _account(self, account_id):
        account = self.ledger.accounts[account_id]
        if account.profile.account_type != 'roth' or account.taxpayer != self.taxpayer:
            raise ValueError('Roth history must be shared only by this owner’s Roth accounts')
        return account

    def contribute(self, when, account_id, amount, *, tax_year, magi, compensation,
                   filing_status='single', other_ira_contributions=0.,
                   lived_with_spouse=True, contribution_deadline=None, eligibility_table_year=None):
        when = calendar_date(when)
        positive(amount, 'contribution')
        deadline = calendar_date(contribution_deadline) if contribution_deadline else date(tax_year, 12, 31)
        if when < date(tax_year, 1, 1) or when > deadline or when.year > tax_year + 1:
            raise ValueError('Outside the supplied contribution-year deadline')
        account = self._account(account_id)
        age = relativedelta(date(tax_year, 12, 31), self.birth_date).years
        table_year=eligibility_table_year if eligibility_table_year is not None else tax_year
        limit = contribution_limit(table_year, age, magi, compensation,
                                   filing_status=filing_status,
                                   other_ira_contributions=other_ira_contributions,
                                   lived_with_spouse=lived_with_spouse)
        if self.contributions.get(tax_year, 0.) + amount > limit + 1e-9:
            raise ValueError('Roth contribution exceeds owner-wide annual/eligibility limit')
        when = self.ledger._advance(when)
        self.contributions[tax_year] = self.contributions.get(tax_year, 0.) + amount
        self.regular_basis += amount
        account.cash += amount
        self.first_tax_year = min(self.first_tax_year or tax_year, tax_year)
        self.ledger.events.append({'date': when, 'action': 'ROTH_CONTRIBUTION',
                                   'account': account_id, 'amount': amount, 'tax_year': tax_year,
                                   'eligibility_table_year':table_year,
                                   'eligibility_evidence':'HYPOTHETICAL_STRESS' if table_year!=tax_year else 'OBSERVED'})

    def convert(self, when, account_id, amount, taxable_amount, *, kind='conversion'):
        """External traditional-IRA conversion or non-Roth qualified-plan rollover.

        Taxable amount must come from the owner's Form 8606/pro-rata analysis.
        Conversion income is an external personal-tax input, never an internal
        investment gain. No automatic deduction of conversion taxes from Roth.
        """
        positive(amount, 'conversion')
        positive(taxable_amount, 'taxable conversion', True)
        if taxable_amount > amount:
            raise ValueError('Taxable conversion exceeds total amount')
        if kind not in ('conversion', 'non_roth_plan_rollover'):
            raise UnsupportedModelError('Supply separate basis/clock history for other rollover types')
        when = calendar_date(when)
        account = self._account(account_id)
        when = self.ledger._advance(when)
        self.conversions.append(Conversion(when.year, taxable_amount, amount - taxable_amount))
        self.conversions.sort(key=lambda c: c.year)
        self.first_tax_year = min(self.first_tax_year or when.year, when.year)
        account.cash += amount
        self.ledger.events.append({'date': when, 'action': kind.upper(), 'account': account_id,
                                   'amount': amount, 'personal_ordinary_income': taxable_amount})

    def transfer(self, when, from_account, to_account, amount):
        """Trustee-to-trustee Roth transfer within this aggregate owner history."""
        positive(amount, 'transfer')
        source = self._account(from_account)
        target = self._account(to_account)
        if source.buying_power < amount:
            raise ValueError('Insufficient settled transfer cash')
        when = self.ledger._advance(when)
        source.cash -= amount
        target.cash += amount
        self.ledger.events.append({'date': when, 'action': 'ROTH_TRANSFER',
                                   'from': from_account, 'to': to_account, 'amount': amount})

    def withdraw(self, when, account_id, amount, *, exception=None):
        """Return taxable earnings and additional tax separately from cash paid."""
        positive(amount, 'withdrawal')
        when = calendar_date(when)
        account = self._account(account_id)
        if account.buying_power < amount:
            raise ValueError('Sell assets to provide settled withdrawal cash first')
        if exception not in (None, 'disability', 'death'):
            raise UnsupportedModelError('This distribution exception needs a dedicated eligibility model')
        if self.first_tax_year is None:
            raise ValueError('Complete Roth basis and first-contribution-year history is required')
        mature_age = when >= self.birth_date + relativedelta(years=59, months=6)
        mature_clock = when >= date(self.first_tax_year + 5, 1, 1)
        qualified = mature_clock and (mature_age or exception in ('disability', 'death'))
        when = self.ledger._advance(when)
        # Track ordering even for qualified distributions to retain usable
        # history if a future distribution is nonqualified.
        remaining = amount
        regular = min(remaining, self.regular_basis)
        self.regular_basis -= regular
        remaining -= regular
        conversion_distributed = 0.
        penalty_base = 0.
        # Aggregate taxable portions BEFORE nontaxable portions for each year,
        # including multiple conversions within the same tax year.
        for year in sorted(set(c.year for c in self.conversions)):
            for attribute in ('taxable_remaining', 'nontaxable_remaining'):
                for conversion in [c for c in self.conversions if c.year == year]:
                    n = min(remaining, getattr(conversion, attribute))
                    setattr(conversion, attribute, getattr(conversion, attribute) - n)
                    remaining -= n
                    conversion_distributed += n
                    if attribute == 'taxable_remaining' and when < date(year + 5, 1, 1):
                        penalty_base += n
        earnings = remaining
        taxable = 0. if qualified else earnings
        if qualified or mature_age or exception in ('disability', 'death'):
            penalty = 0.
        else:
            penalty = .10 * (penalty_base + taxable)
        account.cash -= amount
        result = {'date': when, 'account': account_id, 'action': 'ROTH_WITHDRAWAL',
                  'amount': amount, 'qualified': qualified, 'regular_basis': regular,
                  'conversion_principal': conversion_distributed, 'earnings': earnings,
                  'personal_taxable_income': taxable, 'additional_tax': penalty}
        self.ledger.events.append(result)
        return result
