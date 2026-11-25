"""Actual common-share dividend holding days, excluding acquisition day.

IRS Publication 550 (2025), Qualified dividends / Holding period:
https://www.irs.gov/publications/p550
Common shares require >60 days in the 121-day ex-date window. This bounded
tracker handles unhedged common stock/corporate ETF lots only. Preferred stock,
hedged/risk-reduced days and derivative entitlements require separate evidence.
"""
from dataclasses import dataclass
from datetime import timedelta

from letf.accuracy import UnsupportedModelError


@dataclass
class HoldingGroup:
    acquired:object
    remaining:float
    qualifies_on:object


class DividendHolding:
    def __init__(self,distribution,book,account):
        if distribution.category!='qualified_dividend' or not distribution.qualified_verified:
            raise UnsupportedModelError('Tracked common-share QDI requires verified issuer eligibility')
        security=book.securities[distribution.security]
        if security.kind not in ('stock','etf') or security.tax_structure!='corporation':
            raise UnsupportedModelError('Unhedged common corporate shares only')
        self.distribution=distribution;self.account=account;self.groups=[];self.split_factor=1.
        self.qualified_shares=0.;self.unqualified_shares=0.;self.booked_qualified=0.;self.booked_year=None
        by={}
        for lot in book.lots:
            if lot.account==account and lot.security==distribution.security and lot.shares>1e-10:
                if lot.tacked_days:raise UnsupportedModelError('QDI after tacked wash-sale holding periods requires separate review')
                by[lot.acquired]=by.get(lot.acquired,0.)+lot.shares
        for acquired,shares in by.items():
            start=max(acquired+timedelta(days=1),distribution.ex_date-timedelta(days=60))
            self.groups.append(HoldingGroup(acquired,shares,start+timedelta(days=60)))
        self.total_shares=sum(by.values())

    @property
    def qualified_amount(self):return self.qualified_shares*self.distribution.amount_per_share

    @property
    def unresolved(self):return sum(g.remaining for g in self.groups)>1e-9

    def update(self,when,book):
        held={}
        for lot in book.lots:
            if lot.account==self.account and lot.security==self.distribution.security:
                held[lot.acquired]=held.get(lot.acquired,0.)+lot.shares/self.split_factor
        for group in self.groups:
            available=min(group.remaining,max(0.,held.get(group.acquired,0.)))
            sold=group.remaining-available
            if when>=group.qualifies_on:
                # Sale day counts, including shares sold on their 61st day.
                self.qualified_shares+=group.remaining;group.remaining=0.
            else:
                self.unqualified_shares+=sold;group.remaining=available
        if abs(self.qualified_shares+self.unqualified_shares+sum(g.remaining for g in self.groups)-self.total_shares)>1e-7:
            raise AssertionError('Dividend entitlement share reconciliation failed')
