from __future__ import annotations
import math, json
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yfinance as yf

OUT=Path('results/account_location_study'); OUT.mkdir(parents=True,exist_ok=True)
TD=252; TC=.0003

def dl(t):
    x=yf.download(t,start='2010-02-01',end='2026-08-22',auto_adjust=True,progress=False)
    if x.empty: raise RuntimeError(f'no data {t}')
    s=x['Close']; s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s
    s.index=pd.to_datetime(s.index).tz_localize(None); return pd.to_numeric(s,errors='coerce').dropna()

def met(r):
    r=pd.Series(r).dropna(); e=(1+r).cumprod(); y=len(r)/TD; dd=e/e.cummax()-1
    return {'years':y,'cagr':e.iloc[-1]**(1/y)-1,'max_dd':dd.min(),'vol':r.std()*math.sqrt(TD),'terminal_multiple':e.iloc[-1]}

def rsi(px,w=14):
    d=px.diff(); g=d.where(d>0,0).rolling(w).mean(); l=(-d.where(d<0,0)).rolling(w).mean(); rs=g/l.replace(0,np.nan); return 100-100/(1+rs)

def s8_targets(spy,vix):
    ma=spy.rolling(200).mean(); rr=rsi(spy,14); out=pd.Series('CASH',index=spy.index,dtype=object)
    for i in range(1,len(spy)):
        p, m, q = spy.iloc[i-1],ma.iloc[i-1],rr.iloc[i-1]
        vv=vix.reindex(spy.index).ffill().iloc[i-1]
        score=int(pd.notna(m) and p>m)+int(pd.notna(q) and 40<q<80)+int(pd.notna(vv) and vv<25)
        out.iloc[i]='TQQQ' if score==3 else ('SPY' if score==2 else 'CASH')
    return out

def s9_band_targets(spy,tq):
    sr=spy.pct_change(); tr=tq.pct_change(); ma=spy.rolling(200).mean(); vol=tr.rolling(20).std()*math.sqrt(TD)
    want=pd.Series(0.0,index=spy.index)
    for i in range(1,len(spy)):
        if pd.isna(ma.iloc[i-1]) or pd.isna(vol.iloc[i-1]) or vol.iloc[i-1]<=0: continue
        tv=.35 if spy.iloc[i-1]>ma.iloc[i-1] else .12
        want.iloc[i]=float(np.clip(tv/vol.iloc[i-1],0,1))
    a=pd.Series(0.0,index=spy.index)
    for i in range(1,len(spy)):
        a.iloc[i]=want.iloc[i] if abs(want.iloc[i]-a.iloc[i-1])>=.05 else a.iloc[i-1]
    return a

def strategy_returns(spy,tq,s8,s9):
    d=pd.DataFrame({'spy':spy.pct_change(),'tq':tq.pct_change()}).dropna(); out={}
    # S8 full switches; 5bp switch cost as tournament
    st=s8.reindex(d.index); prev=st.shift(1); cost=(st!=prev).astype(float)*.0005
    out['S8']=np.where(st.eq('TQQQ'),d.tq,np.where(st.eq('SPY'),d.spy,0))-cost
    a=s9.reindex(d.index).fillna(0); out['S9_band5']=a*d.tq-TC*a.diff().abs().fillna(a.abs())
    out['SPY_BH']=d.spy; out['TQQQ_BH']=d.tq
    return {k:pd.Series(v,index=d.index) for k,v in out.items()}

def turnover_s8(state):
    ch=(state!=state.shift(1)); n=max(int(ch.sum())-1,0); years=len(state)/TD
    # A switch between two invested assets is 200% one-way notional; cash<->asset 100%.
    one=0.0
    for i in range(1,len(state)):
        if state.iloc[i]==state.iloc[i-1]: continue
        one += 2.0 if state.iloc[i]!='CASH' and state.iloc[i-1]!='CASH' else 1.0
    return {'change_days_per_year':n/years,'one_way_turnover_per_year':one/years}

def turnover_s9(a):
    d=a.diff().abs().fillna(a.abs()); years=len(a)/TD
    return {'change_days_per_year':float((d>1e-12).sum()/years),'one_way_turnover_per_year':float(d.sum()/years),'median_trade':float(d[d>1e-12].median())}

# Lot book is for characterization only: pre-tax strategy wealth determines trade sizes; taxes are not removed from account.
class Lots:
    def __init__(self): self.lots=[]
    def buy(self,date,shares,px):
        if shares>1e-12:self.lots.append({'date':date,'shares':shares,'basis':px})
    def value(self,px): return sum(x['shares'] for x in self.lots)*px
    def sell(self,date,shares,px):
        # Schwab-like tax-aware order: ST losses, LT losses, LT gains, ST gains; within gain classes smallest gain first.
        cand=[]
        for j,x in enumerate(self.lots):
            age=(date-x['date']).days; gain=px-x['basis']; long=age>365
            if gain<0: cls=0 if not long else 1
            elif long: cls=2
            else: cls=3
            cand.append((cls, abs(gain) if gain<0 else gain,j))
        cand.sort(); rem=shares; ev=[]
        for _,_,j in cand:
            if rem<=1e-12: break
            x=self.lots[j]; q=min(rem,x['shares']); g=(px-x['basis'])*q; long=(date-x['date']).days>365
            ev.append((date,g,long,q)); x['shares']-=q; rem-=q
        self.lots=[x for x in self.lots if x['shares']>1e-10]
        if rem>1e-6: raise RuntimeError('oversell')
        return ev

def realized_characterization(spy,tq,state_or_alloc,kind):
    cash=1.0; books={'SPY':Lots(),'TQQQ':Lots()}; events=[]; vals=[]; prev_target=None
    dates=spy.index.intersection(tq.index)
    for i,date in enumerate(dates):
        ps=float(spy.loc[date]); pt=float(tq.loc[date])
        wealth=cash+books['SPY'].value(ps)+books['TQQQ'].value(pt)
        if kind=='S8':
            st=state_or_alloc.reindex(dates).iloc[i]; target={'SPY':0.0,'TQQQ':0.0}
            if st in target: target[st]=1.0
        else:
            a=float(state_or_alloc.reindex(dates).iloc[i]); target={'SPY':0.0,'TQQQ':a}
        # sell excess first
        for asset,px in [('SPY',ps),('TQQQ',pt)]:
            cur=books[asset].value(px); want=wealth*target[asset]
            if cur>want+1e-10:
                q=(cur-want)/px; events.extend(books[asset].sell(date,q,px)); cash+=q*px
        # buy shortages
        for asset,px in [('SPY',ps),('TQQQ',pt)]:
            cur=books[asset].value(px); want=wealth*target[asset]
            if want>cur+1e-10:
                spend=min(want-cur,cash); q=spend/px; books[asset].buy(date,q,px); cash-=spend
        vals.append(wealth)
    ev=pd.DataFrame(events,columns=['date','gain','long','shares']) if events else pd.DataFrame(columns=['date','gain','long','shares'])
    if len(ev): ev['year']=pd.to_datetime(ev.date).dt.year
    avg_wealth=float(np.mean(vals)); yrs=len(dates)/TD
    stg=float(ev.loc[(ev.gain>0)&(~ev.long),'gain'].sum()) if len(ev) else 0
    ltg=float(ev.loc[(ev.gain>0)&(ev.long),'gain'].sum()) if len(ev) else 0
    stl=float(-ev.loc[(ev.gain<0)&(~ev.long),'gain'].sum()) if len(ev) else 0
    ltl=float(-ev.loc[(ev.gain<0)&(ev.long),'gain'].sum()) if len(ev) else 0
    return {'realized_ST_gains_per_year_pct_avg_wealth':stg/yrs/avg_wealth,'realized_LT_gains_per_year_pct_avg_wealth':ltg/yrs/avg_wealth,'realized_ST_losses_per_year_pct_avg_wealth':stl/yrs/avg_wealth,'realized_LT_losses_per_year_pct_avg_wealth':ltl/yrs/avg_wealth,'gain_share_short_term':stg/max(stg+ltg,1e-12),'n_sale_lot_events':len(ev)}

def tax_drag(char,ordinary,lt):
    # two bounds: (1) gains taxed with no loss benefit (wash-sale-heavy conservative), (2) same-character losses net immediately (optimistic).
    stg=char['realized_ST_gains_per_year_pct_avg_wealth']; ltg=char['realized_LT_gains_per_year_pct_avg_wealth']; stl=char['realized_ST_losses_per_year_pct_avg_wealth']; ltl=char['realized_LT_losses_per_year_pct_avg_wealth']
    upper=stg*ordinary+ltg*lt
    lower=max(stg-stl,0)*ordinary+max(ltg-ltl,0)*lt
    return lower,upper

def main():
    spy=dl('SPY'); tq=dl('TQQQ'); vix=dl('^VIX'); idx=spy.index.intersection(tq.index);spy=spy.reindex(idx);tq=tq.reindex(idx);vix=vix.reindex(idx).ffill()
    s8=s8_targets(spy,vix); s9=s9_band_targets(spy,tq); rets=strategy_returns(spy,tq,s8,s9)
    perf=[]
    for k,r in rets.items(): perf.append({'strategy':k,**met(r)})
    turn=[{'strategy':'S8',**turnover_s8(s8)},{'strategy':'S9_band5',**turnover_s9(s9)}]
    chars={'S8':realized_characterization(spy,tq,s8,'S8'),'S9_band5':realized_characterization(spy,tq,s9,'S9')}
    tax=[]
    for name,c in chars.items():
        for ordinary,lt,label in [(0.24,0.15,'24_15'),(0.32,0.15,'32_15'),(0.37,0.20,'37_20')]:
            lo,hi=tax_drag(c,ordinary,lt); tax.append({'strategy':name,'federal_rate_case':label,'optimistic_annual_tax_drag_pct_wealth':lo,'conservative_annual_tax_drag_pct_wealth':hi,'approx_after_tax_cagr_lower_bound':next(x['cagr'] for x in perf if x['strategy']==name)-hi,'approx_after_tax_cagr_upper_bound':next(x['cagr'] for x in perf if x['strategy']==name)-lo})
    # Section 1256 blended rate examples for a futures implementation.
    fut=[{'ordinary_rate':o,'ltcg_rate':l,'section1256_blended_rate':.4*o+.6*l} for o,l in [(0.24,.15),(.32,.15),(.37,.20)]]
    pd.DataFrame(perf).to_csv(OUT/'real_etf_performance.csv',index=False);pd.DataFrame(turn).to_csv(OUT/'turnover.csv',index=False);pd.DataFrame([{'strategy':k,**v} for k,v in chars.items()]).to_csv(OUT/'realized_tax_character.csv',index=False);pd.DataFrame(tax).to_csv(OUT/'tax_drag_sensitivity.csv',index=False);pd.DataFrame(fut).to_csv(OUT/'section1256_rates.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'period':[str(idx.min().date()),str(idx.max().date())],'source':'Yahoo Finance adjusted closes','tax_note':'Tax results are sensitivity bounds, not return-filed tax accounting. Pre-tax strategy drives lots; optimistic bound nets same-character losses immediately; conservative bound gives losses no current benefit, approximating frequent wash-sale deferral. State/local tax, NIIT, dividends and final liquidation tax excluded.','lot_method':'Schwab-like tax-aware ordering approximating Tax Lot Optimizer; gains prefer LT before ST.'},indent=2))
    print(pd.DataFrame(perf).to_string(index=False));print(pd.DataFrame(turn).to_string(index=False));print(pd.DataFrame([{'strategy':k,**v} for k,v in chars.items()]).to_string(index=False));print(pd.DataFrame(tax).to_string(index=False));print(pd.DataFrame(fut).to_string(index=False))
if __name__=='__main__': main()
