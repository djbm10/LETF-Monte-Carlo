"""Actual common-share dividend holding days, excluding acquisition day.

IRS Publication 550 (2025), Qualified dividends / Holding period:
https://www.irs.gov/publications/p550
Common shares require >60 days in the 121-day ex-date window. This bounded
tracker handles unhedged common stock/corporate ETF lots only. Preferred stock,
hedged/risk-reduced days and derivative entitlements require separate evidence.
IRC 1(h)(11)(B)(iii) incorporates 246(c); 246(c)(3)(B) excludes 1223(3)
wash-sale tacking for this test. Count actual acquisition dates, NOT tax tacks:
https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section246

Verified RIC Section199A dividends instead require >45 days in the91-day
window:26CFR1.199A-3(d)(4)(ii),2025 annual edition, incorporating246(c)(3)/(4).
https://www.govinfo.gov/content/pkg/CFR-2025-title26-vol4/xhtml/CFR-2025-title26-vol4-sec1-199A-3.html
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
        if distribution.category=='qualified_dividend' and distribution.qualified_verified is True:
            days=60
        elif distribution.category=='section199a_dividend' and distribution.section199a_verified is True:
            tax_date=distribution.tax_date or distribution.pay_date
            if tax_date.year<2021:
                raise UnsupportedModelError('Pre2021 RIC Section199A transition/election scope is not reviewed')
            days=45
        else:
            raise UnsupportedModelError('Tracked preferential dividends require verified issuer eligibility')
        security=book.securities[distribution.security]
        if security.kind not in ('stock','etf') or security.tax_structure!='corporation':
            raise UnsupportedModelError('Unhedged common corporate shares only')
        self.distribution=distribution;self.account=account;self.groups=[];self.split_factor=1.
        self.qualified_shares=0.;self.unqualified_shares=0.;self.booked_qualified=0.;self.booked_year=None
        by={}
        for lot in book.lots:
            if lot.account==account and lot.security==distribution.security and lot.shares>1e-10:
                by[lot.acquired]=by.get(lot.acquired,0.)+lot.shares
        for acquired,shares in by.items():
            start=max(acquired+timedelta(days=1),distribution.ex_date-timedelta(days=days))
            self.groups.append(HoldingGroup(acquired,shares,start+timedelta(days=days)))
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
                # Sale day counts, including the QDI61st / Section199A46th day.
                self.qualified_shares+=group.remaining;group.remaining=0.
            else:
                self.unqualified_shares+=sold;group.remaining=available
        if abs(self.qualified_shares+self.unqualified_shares+sum(g.remaining for g in self.groups)-self.total_shares)>1e-7:
            raise AssertionError('Dividend entitlement share reconciliation failed')
