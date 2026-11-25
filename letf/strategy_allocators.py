"""Versioned migrations of legacy allocation rules into canonical execution.

The original strategy.py is preserved. legacy_rules_v1 retains causal signal
formulas and original caps; an allocation >1 is rejected by cash accounts.
causal_cash_v2 explicitly lags S16's VIX and caps all personal allocations at1.
No legacy vectorized wealth accounting or borrowing is reused.
"""
from hashlib import sha256
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from letf.config import STRATEGIES
from letf.accuracy import UnsupportedModelError

SUPPORTED=('S8','S11','S12','S13','S14','S15','S16','S17','S18','S19')


class LegacyAllocator:
    def __init__(self,strategy_id,*,version='legacy_rules_v1',asset_map=None):
        if strategy_id not in SUPPORTED:raise UnsupportedModelError('Not a migrated advanced strategy')
        if version not in ('legacy_rules_v1','causal_cash_v2'):raise UnsupportedModelError('Explicit registered strategy version required')
        if strategy_id=='S16' and version=='legacy_rules_v1':
            raise UnsupportedModelError('S16 legacy uses current-close VIX for same-bar return; use explicit causal_cash_v2')
        self.strategy_id=strategy_id;self.version=version;self.config=copy.deepcopy(STRATEGIES[strategy_id])
        self.asset_map=asset_map or {};self.reset()
        self.metadata={'strategy_id':strategy_id,'version':version,'asset_map':self.asset_map,
            'legacy_source_sha256':sha256(Path(__file__).with_name('strategy.py').read_bytes()).hexdigest(),
            'config':self.config,'changes':[] if version=='legacy_rules_v1' else ['S16 VIX uses prior close','All personal allocation caps limited to 1'],
            'execution':'Canonical nominal cash ledger; exact fills/costs/settlement replace vectorized assumed allocation returns'}

    def reset(self):self.previous_target=0.

    def __call__(self,history,weights):
        if not len(history):return {}
        c=self.config;sid=self.strategy_id;asset=self.asset_map.get(c['asset'],c['asset'])
        def price(name):
            name=self.asset_map.get(name,name);column=f'{name}_TR_Close'
            if column not in history:raise UnsupportedModelError(f'Missing total-return signal reference: {column}')
            return history[column]
        r=history[f'{asset}_Signal_Ret'] if f'{asset}_Signal_Ret' in history else price(c['asset']).pct_change(fill_method=None).fillna(0.)
        def vol(n,default):
            value=r.iloc[-n:].std(ddof=1)*np.sqrt(252)
            return default if pd.isna(value) or value<.001 else float(value)
        def downside(n,default):
            value=r.clip(upper=0).iloc[-n:].std(ddof=1)*np.sqrt(252)
            return default if pd.isna(value) or value<.001 else float(value)
        def trend(n):
            p=price('SPY');return p.iloc[-1]>p.iloc[-n:].mean()
        def spy_returns():
            name=self.asset_map.get('SPY','SPY');column=f'{name}_Signal_Ret'
            return history[column] if column in history else price('SPY').pct_change(fill_method=None).fillna(0.)
        if sid=='S8':
            if 'VIX' not in history:raise UnsupportedModelError('S8 requires observed decision-time VIX')
            p=price('SPY');delta=p.diff();gain=delta.where(delta>0,0).rolling(c['rsi_period']).mean().iloc[-1]
            loss=(-delta.where(delta<0,0)).rolling(c['rsi_period']).mean().iloc[-1]
            rsi=100. if loss==0 and gain>0 else 100-100/(1+gain/loss) if loss!=0 else np.nan
            sma=p.rolling(c['sma_period']).mean().iloc[-1]
            score=int(p.iloc[-1]>sma)+int(40<rsi<80)+int(history.VIX.iloc[-1]<c['vix_threshold'])
            return {asset:1.} if score==3 else {self.asset_map.get(c['defensive_asset'],c['defensive_asset']):1.} if score==2 else {}
        if sid=='S11':
            allocation=c['target_vol']/vol(20,.5)
            if trend(c['sma_period']):allocation=allocation**c['power']
            cap=1.
        elif sid=='S12':
            slow=r.iloc[-20:].std()*np.sqrt(252);fast=r.iloc[-5:].std()*np.sqrt(252)
            v=max(slow,fast);v=.5 if pd.isna(v) or v<.001 else v
            allocation=c['target_vol']/v;cap=1.
        elif sid=='S13':
            v=vol(20,.5);slow=r.rolling(20,min_periods=1).std()*np.sqrt(252)
            momentum=slow.ffill().pct_change(5,fill_method=None).iloc[-1]
            allocation=c['target_vol']/v
            if pd.notna(momentum) and momentum<-.10:allocation*=1.2
            if pd.notna(momentum) and momentum>.10:allocation*=.7
            cap=1.
        elif sid=='S14':
            allocation=c['target_vol']/vol(20,.5);skew=r.iloc[-60:].skew()
            if pd.notna(skew) and skew>0:allocation=allocation**1.3
            elif pd.notna(skew) and skew<-.5:allocation*=.5
            cap=1.
        elif sid=='S15':
            allocation=c['target_vol']/downside(20,.25)
            if trend(200):allocation=allocation**1.2
            fast=r.iloc[-5:].std()*np.sqrt(252);slow=r.iloc[-20:].std()*np.sqrt(252)
            if fast>1.5*slow:allocation*=.5
            cap=1.
        elif sid=='S16':
            if 'VIX' not in history:raise UnsupportedModelError('S16 requires decision-time VIX')
            fast=r.iloc[-c['lookback_fast']:].std()*np.sqrt(252);slow=r.iloc[-c['lookback_slow']:].std()*np.sqrt(252)
            crisis=history.VIX.iloc[-1]>c['vix_alarm_level'] or fast/slow>c['vol_expansion_threshold']
            target=c['crisis_target_vol'] if crisis else c['base_target_vol']
            allocation=.5 if pd.isna(fast) or fast<.001 else target/fast;cap=1.2
        elif sid=='S17':
            v=vol(c['vol_lookback'],.25);dv=downside(c['vol_lookback'],v*.6)
            sk=r.iloc[-c['skew_lookback']:].skew();sk=0. if pd.isna(sk) else sk
            mean=r.iloc[-c['skew_lookback']:].mean()*252;mean=.08 if pd.isna(mean) else mean
            if sk<-.5:effective,penalty=dv*1.5,.6
            elif sk<0:effective,penalty=dv*1.2,.8
            elif sk>.5:effective,penalty=v*.9,1.2
            else:effective,penalty=v,1.
            kelly=np.clip((mean-.03)/(effective**2)*c['kelly_fraction'],.2,2.) if effective>.01 else 1.
            allocation=c['base_target_vol']/effective*penalty*kelly/1.5;cap=1.5
        elif sid=='S18':
            p=spy_returns()
            mom=p.iloc[-c['momentum_lookback']:].sum();fast=vol(c['vol_fast'],.30);slow=vol(c['vol_slow'],.25)
            mm=1.3 if mom>c['momentum_threshold'] else 1. if mom>0 else .5
            ratio=fast/slow
            vm,effective=(1.2,fast) if ratio<.8 else (.6,fast) if ratio>1.3 else (1.,slow)
            allocation=c['base_target_vol']*mm*vm/effective;cap=1.5
        else: # S19
            p=price('SPY');mom=spy_returns().iloc[-c['momentum_lookback']:].sum()
            dv=downside(c['vol_lookback'],.20);fast=vol(5,.30);slow=vol(60,.25)
            ms=1. if mom>.15 else .7 if mom>.05 else .4 if mom>0 else 0.
            ts=.5 if trend(c['trend_sma']) else 0.;ratio=fast/slow
            vs=.5 if ratio<.9 else .3 if ratio<1.2 else 0.
            allocation=c['base_target_vol']/dv*(.3+(ms+ts+vs)*.55);cap=1.5
        if self.version=='causal_cash_v2':cap=min(cap,1.)
        allocation=float(np.clip(allocation,0.,cap))
        if sid=='S19':
            if abs(allocation-self.previous_target)>c['rebalance_threshold']:self.previous_target=allocation
            else:allocation=self.previous_target
        return {asset:allocation}
