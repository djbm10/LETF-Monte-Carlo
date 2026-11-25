"""Exact, bounded canonical-ledger lookup for a very narrow future scenario.

Initial cash-only buy, no intermediate trades/distributions/cash yield/flows,
long-term terminal sale; single federal-only investor, 100k ordinary income,
explicit frozen 2026 law. Other scenarios MUST use run_account_backtest.
No separate tax or wealth formula is implemented: every affine-segment endpoint
is an actual PortfolioLedger replay, with independent interior reconciliation.
"""
from dataclasses import replace,dataclass
from datetime import date,timedelta
from functools import lru_cache
import numpy as np
from letf.accounts import PortfolioLedger,Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.tax.years import federal_parameters
from letf.accuracy import UnsupportedModelError


@lru_cache(maxsize=5)
def projected_dates(horizon):
    import exchange_calendars as xcals
    import pandas as pd
    calendar=xcals.get_calendar('XNYS',start='2027-01-01',end=f'{2027+horizon}-02-01')
    first=calendar.date_to_session(pd.Timestamp('2027-01-04'),direction='next')
    last=calendar.date_to_session(pd.Timestamp(f'{2026+horizon}-12-31'),direction='previous')
    return first.date(),last.date(),calendar.session_offset(first,1).date(),calendar.session_offset(last,1).date()


def terminal_replay(account,nav,*,horizon=40,initial=10000.,distributions=0.,contributions=0.,state='FEDERAL_ONLY',tax_year=2026):
    if account not in ('taxable','roth') or horizon not in (5,10,20,30,40) or initial!=10000. or state!='FEDERAL_ONLY' or tax_year!=2026:
        raise UnsupportedModelError('Terminal lookup supports only its registered initial-10k/federal/single/frozen-2026 profile')
    if distributions or contributions:raise UnsupportedModelError('Distributions/contributions require full canonical daily account replay')
    if not np.isfinite(nav) or nav<=0:raise UnsupportedModelError('Zero/extinguished or nonfinite fund needs actual dated liquidation event')
    profile=replace(TAXABLE_CASH if account=='taxable' else CONSERVATIVE_ROTH,spread_bps=2.,slippage_bps=1.)
    first,last,buy_settlement,sale_settlement=projected_dates(horizon)
    book=PortfolioLedger();book.add_account('main',profile,initial);book.add_security(Security('US_CF'))
    slip=(profile.spread_bps/2+profile.slippage_bps)/10000
    book.execute(first,'main','US_CF','BUY',initial/(1+slip),1.,settlement_date=buy_settlement)
    before=book.accounts['main'].cash
    book.execute(last,'main','US_CF','SELL',book.quantity('main','US_CF'),float(nav),settlement_date=sale_settlement)
    tax=book.annual_tax('main',last.year,100000.,federal_only=True,filing_status='single',
        freeze_tax_year=tax_year,as_of=last+timedelta(days=31))['tax']
    gross=book.wealth('main',{'US_CF':float(nav)})
    return {'net':gross-tax,'tax':tax,'gross':gross,'buy_cash_residual':before,
        'realized_gain':sum(r.gain for r in book.realizations),'positions':book.quantity('main','US_CF'),
        'cash':book.accounts['main'].cash,'unsettled':sum(v for _,v in book.accounts['main'].unsettled),
        'events':book.events,'terminal_tax_paid':False,'scope':'Assets net of tax payable; Roth internal sale is not a personal withdrawal'}


@dataclass
class CanonicalTerminalMap:
    account:str
    horizon:int
    knots:np.ndarray
    net:np.ndarray
    tax:np.ndarray
    maximum_independent_residual:float

    @classmethod
    def build(cls,account,minimum_nav,maximum_nav,*,horizon=40):
        low=terminal_replay(account,minimum_nav,horizon=horizon);high=terminal_replay(account,maximum_nav,horizon=horizon)
        if not minimum_nav<maximum_nav:raise ValueError('Nonempty observed NAV range required')
        # Realized gains are affine in the terminal price for this exact trade
        # profile. Capital-loss cap, preferential brackets and NIIT are the only
        # tax kinks. Ordinary income stays in the SAME bracket over the 3k cap.
        _,pref,ded,niit=federal_parameters(2026,'single');ordinary=100000.-ded
        gains=[-3000.,0.,niit-100000.,*[edge-ordinary for edge,rate in pref if np.isfinite(edge)]]
        if account=='taxable':
            slope=(high['realized_gain']-low['realized_gain'])/(maximum_nav-minimum_nav)
        else:
            # Roth does not record taxable realizations. Its terminal net is
            # still exactly affine under the same cash/execution profile.
            slope=None
        knots=[minimum_nav,maximum_nav]
        if slope is not None:
            knots.extend(minimum_nav+(g-low['realized_gain'])/slope for g in gains if low['realized_gain']<g<high['realized_gain'])
        knots=np.unique(knots);values=[terminal_replay(account,float(v),horizon=horizon) for v in knots]
        net=np.array([v['net'] for v in values]);tax=np.array([v['tax'] for v in values])
        if (np.diff(net)<0).any():raise AssertionError('Nonmonotone endpoint map cannot preserve quantile ranks')
        # Independent ledger checks at every interior, near kink, and random
        # log NAV. This is a conformance check, not curve fitting.
        check=list(np.exp(np.random.default_rng(901).uniform(np.log(minimum_nav),np.log(maximum_nav),256)))
        for a,b in zip(knots[:-1],knots[1:]):check.extend([a+(b-a)*v for v in (.000001,.25,.5,.75,.999999)])
        residual=0.
        for nav in check:
            actual=terminal_replay(account,float(nav),horizon=horizon)
            prediction=float(np.interp(nav,knots,net));error=abs(prediction-actual['net'])
            if error>max(1e-7,abs(actual['net'])*2e-13):raise AssertionError('Canonical lookup segment failed ledger conformance')
            residual=max(residual,error)
        return cls(account,horizon,knots,net,tax,residual)

    def evaluate(self,nav):
        nav=np.asarray(nav,float)
        if not np.isfinite(nav).all() or (nav<self.knots[0]).any() or (nav>self.knots[-1]).any():
            raise UnsupportedModelError('NAV outside independently reconciled canonical lookup; no extrapolation or path deletion')
        return np.interp(nav,self.knots,self.net),np.interp(nav,self.knots,self.tax)
