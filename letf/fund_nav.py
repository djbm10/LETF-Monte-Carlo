"""Issuer split-restated NAV total returns, distinct from executable prices."""
import numpy as np
import pandas as pd
from letf.accuracy import UnsupportedModelError


def issuer_nav_returns(nav_frame,distributions,*,start,end):
    """Cash reinvested at ex-date NAV; all values must share split convention.

    No fees/funding are charged: NAV already embeds fund internal costs. This
    valuation series cannot be used as an exchange fill price or tax-lot basis.
    Distribution completeness and common split restatement need source review.
    """
    nav=nav_frame['NAV']
    if not isinstance(nav.index,pd.DatetimeIndex) or not nav.index.is_unique or not nav.index.is_monotonic_increasing:
        raise ValueError('Issuer NAV chronology must be unique and increasing')
    if not np.isfinite(nav).all() or (nav<=0).any():raise ValueError('Positive finite observed NAV required')
    cash=pd.Series(0.,index=nav.index)
    for event in distributions:
        ex=pd.Timestamp(event['ExDate'])
        if ex<pd.Timestamp(start) or ex>pd.Timestamp(end):raise ValueError('Cash event outside reviewed distribution interval')
        if ex not in nav.index:raise ValueError('Issuer cash ex-date absent from NAV observations')
        if any(float(event.get(k) or 0) for k in ('OtherPerShare','SpecialPerShare')):
            raise UnsupportedModelError('Special issuer cash components need separate review')
        amount=sum(float(event.get(k) or 0) for k in ('CashDividendPerShare','LongTermCapGains','ShortTermCapGains','ReturnOfCapital'))
        if not np.isfinite(amount) or amount<0:raise ValueError('Invalid issuer cash amount')
        cash.loc[ex]+=amount
    # Shift over authentic consecutive issuer sessions, never over a filtered
    # evaluation sample. A fixed-date boundary retains its prior close.
    returns=(nav+cash)/nav.shift(1)-1
    result=returns.loc[start:end]
    if not np.isfinite(result).all():raise UnsupportedModelError('NAV interval needs a genuine previous close')
    return result,cash.loc[start:end]
