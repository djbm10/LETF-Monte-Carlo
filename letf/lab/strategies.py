"""Versioned strategy adapters, preserving original research definitions.

The original vectorized wealth engines remain archived, not a second account
ledger. Versioned cash adapters execute through run_account_backtest only.
"""
import copy
from hashlib import sha256
from pathlib import Path

import numpy as np

from letf import config
from letf.account_backtest import strategy_allocator
from letf.strategy_allocators import LegacyAllocator, SUPPORTED
from letf.accuracy import UnsupportedModelError


def registry():
    original = Path(__file__).parents[1] / 'strategy.py'
    rows = []
    for key, value in config.STRATEGIES.items():
        rows.append({'id': key, 'original_definition': copy.deepcopy(value),
            'original_source_sha256': sha256(original.read_bytes()).hexdigest(),
            'versions': {'legacy_vector_v0': 'PRESERVED_RESEARCH_ONLY; not canonical cash accounting',
                         'legacy_rules_v1': 'Causal canonical adapter; allocations above1 reject; S16 same-bar variant blocked',
                         'causal_cash_v2': 'Canonical cash cap1; prior-close TR signals; hypothetical fills'},
            'historical_performance_certification': 'BLOCKED',
            'required_evidence': 'Reviewed daily instruments/distributions, causal signal warmup; VIX for S8/S16; taxpayer/broker inputs'})
    rows.extend([
        {'id': 'fixed_weights', 'versions': {'cash_v1': 'User-declared stock/ETF weights, sum<=1; remainder cash'}},
        {'id': 'HFEA', 'versions': {'cash_v1': 'Explicit 55%UPRO/45%TMF monthly21-session cash wrapper, not a reconstruction'}},
        {'id': '35_0', 'versions': {'frozen_ETF_wrapper_v1': 'Frozen 35% target,200SMA,20-session excess-vol,3x cap; TQQQ weight exposure/3'},
         'required_evidence': 'QQQ TR signal, decision-time RF_DAILY and actual TQQQ quotes/distributions; futures remain separately gated',
         'changes': 'ETF vehicle/internal costs and canonical settled cash replace fractional proxy futures accounting'},
        {'id': 'sector_rotation', 'status': 'BLOCKED', 'reason': 'Point-in-time membership and all instrument actions/decision releases required'},
        {'id': 'managed_futures', 'status': 'BLOCKED', 'reason': 'KMLM pre-inception index/constituent contract/fee history not captured'},
        {'id': 'options_futures', 'status': 'BLOCKED', 'reason': 'Use dated contract replay; no continuous-chart or theoretical-quote historical rankings'},
    ])
    return rows


def make_allocator(spec):
    allowed = {'id', 'version', 'weights'}
    if set(spec) - allowed or not {'id', 'version'} <= set(spec):
        raise ValueError('Explicit registered strategy id/version required; unknown fields rejected')
    sid, version = spec['id'], spec['version']
    if sid in ('fixed_weights', 'HFEA'):
        if version != 'cash_v1': raise UnsupportedModelError('Unknown fixed-allocation version')
        weights = {'UPRO': .55, 'TMF': .45} if sid == 'HFEA' else dict(spec.get('weights', {}))
        if sid == 'HFEA' and 'weights' in spec: raise ValueError('Changed HFEA weights require a separately named fixed_weights profile')
        if (not weights or any(not isinstance(k, str) or not np.isfinite(v) or v < 0 for k,v in weights.items())
                or sum(weights.values()) > 1 + 1e-12): raise ValueError('Cash portfolio weights must be nonnegative and sum<=1')
        return lambda history,current: dict(weights), set(weights), {'id':sid,'version':version,'weights':weights}, 21
    if sid == '35_0':
        if version != 'frozen_ETF_wrapper_v1': raise UnsupportedModelError('Do not silently redefine frozen35/0')
        def allocate(history, current):
            if not {'QQQ_TR_Close','RF_DAILY'} <= set(history):
                raise UnsupportedModelError('Frozen35/0 requires QQQ TR and reviewed dated RF_DAILY')
            if len(history) < 200: return {}
            q=history.QQQ_TR_Close;rf=history.RF_DAILY
            excess=q.pct_change(fill_method=None)-rf
            vol=excess.iloc[-20:].std(ddof=1)*np.sqrt(252)
            exposure=min(3., .35/vol) if q.iloc[-1]>q.iloc[-200:].mean() and np.isfinite(vol) and vol>0 else 0.
            return {'TQQQ':exposure/3}
        return allocate, {'QQQ','TQQQ'}, {'id':sid,'version':version,'target_vol':.35,'sma':200,'vol_sessions':20,
            'exposure_cap':3.,'signal':'Prior-close QQQ TR less supplied RF_DAILY',
            'original_source_sha256':sha256((Path(__file__).parents[2]/'research/frozen_35_0_actual_futures_validation_20261004.py').read_bytes()).hexdigest(),
            'vehicle_change':'TQQQ ETF cash wrapper; not claimed equal to original fractional proxy or futures P&L'}, 1
    if sid not in config.STRATEGIES or version not in ('legacy_rules_v1','causal_cash_v2'):
        raise UnsupportedModelError('Unknown/unsupported strategy version; original vectorized cash is not a production ledger')
    c=copy.deepcopy(config.STRATEGIES[sid]);required=set(c.get('assets',{})) or {c['asset']}
    if sid in ('S4','S5','S8','S9','S11','S15','S18','S19'): required.add('SPY')
    if sid in SUPPORTED:
        allocator=LegacyAllocator(sid,version=version)
        metadata=allocator.metadata
    else:
        base=strategy_allocator(sid)
        def allocator(history, current):
            if version=='causal_cash_v2':
                history=history.copy()
                for symbol in required:
                    history[f'{symbol}_Close']=history[f'{symbol}_TR_Close']
            return base(history,current)
        metadata={'strategy_id':sid,'version':version,'config':c,
                  'changes':[] if version=='legacy_rules_v1' else ['TR signals including ex-date cash; cash cap<=1'],
                  'original_source_sha256':sha256((Path(__file__).parents[1]/'strategy.py').read_bytes()).hexdigest()}
    return allocator,required,metadata,c.get('rebalance_freq',1)
