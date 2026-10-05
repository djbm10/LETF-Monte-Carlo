from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import research.hostile_strategy_tournament_fast as fast
base=fast.base

@njit(cache=True)
def two_sleeve(r_core,r_tac,tac_weight,rebalance_days,tc):
    b,n=r_core.shape; out=np.zeros_like(r_core)
    c=np.full(b,1.0-tac_weight); t=np.full(b,tac_weight)
    for j in range(n):
        total=c+t
        wc=c/np.maximum(total,1e-15); wt=t/np.maximum(total,1e-15)
        out[:,j]=wc*r_core[:,j]+wt*r_tac[:,j]
        c*=1.0+r_core[:,j]; t*=1.0+r_tac[:,j]
        if rebalance_days>0 and (j+1)%rebalance_days==0:
            total=c+t; target_t=total*tac_weight; target_c=total-target_t
            turnover=np.abs(target_t-t)/np.maximum(total,1e-15)+np.abs(target_c-c)/np.maximum(total,1e-15)
            total*=1.0-tc*turnover
            c=total*(1.0-tac_weight); t=total*tac_weight
    return out

orig=base.build_strategies
def build(bm,br):
    o=orig(bm,br); core=o['1x_buy_hold']; s8=o['repo_s8_composite']
    r1=base.levret(bm,br,1); r3=base.levret(bm,br,3); px=np.cumprod(np.maximum(1+r1,1e-12),axis=1);a=base.desired_s9(px,r3,.35,.12,200,0);s9=base.alloc_returns(fast.hysteresis_fast(a,.10),r3,br)
    o['s9_band10']=s9
    for w in (.25,.50,.75):
        o[f'core_s8_qtr_tac{int(w*100)}']=two_sleeve(core,s8,w,63,.0003)
    o['core_s8_monthly_50']=two_sleeve(core,s8,.5,21,.0003)
    o['core_s8_annual_50']=two_sleeve(core,s8,.5,252,.0003)
    o['core_s8_norebal_50']=two_sleeve(core,s8,.5,0,.0003)
    o['core_s9_qtr_50']=two_sleeve(core,s9,.5,63,.0003)
    return o
base.build_strategies=build
base.STRATEGIES=('1x_buy_hold','repo_s8_composite','s9_band10','core_s8_qtr_tac25','core_s8_qtr_tac50','core_s8_qtr_tac75','core_s8_monthly_50','core_s8_annual_50','core_s8_norebal_50','core_s9_qtr_50')

def actual_blend():
    import yfinance as yf
    def dl(t):
        x=yf.download(t,start='2010-02-01',end='2026-08-22',auto_adjust=True,progress=False);s=x['Close'];s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s;s.index=pd.to_datetime(s.index).tz_localize(None);return pd.to_numeric(s,errors='coerce').dropna()
    spy=dl('SPY');tq=dl('TQQQ');vix=dl('^VIX');idx=spy.index.intersection(tq.index);spy=spy.reindex(idx);tq=tq.reindex(idx);vix=vix.reindex(idx).ffill();sr=spy.pct_change().fillna(0);tr=tq.pct_change().fillna(0)
    ma=spy.rolling(200).mean();d=spy.diff();g=d.where(d>0,0).rolling(14).mean();l=(-d.where(d<0,0)).rolling(14).mean();rsi=100-100/(1+g/l.replace(0,np.nan));s8=pd.Series(0.,index=idx)
    for i in range(1,len(idx)):
        score=int(pd.notna(ma.iloc[i-1]) and spy.iloc[i-1]>ma.iloc[i-1])+int(pd.notna(rsi.iloc[i-1]) and 40<rsi.iloc[i-1]<80)+int(pd.notna(vix.iloc[i-1]) and vix.iloc[i-1]<25)
        s8.iloc[i]=tr.iloc[i] if score==3 else (sr.iloc[i] if score==2 else 0.)
        if i>1:
            pscore=int(pd.notna(ma.iloc[i-2]) and spy.iloc[i-2]>ma.iloc[i-2])+int(pd.notna(rsi.iloc[i-2]) and 40<rsi.iloc[i-2]<80)+int(pd.notna(vix.iloc[i-2]) and vix.iloc[i-2]<25)
            if score!=pscore:s8.iloc[i]-=.0005
    def rb(a,b,w,step):
        c=1-w;t=w;out=[]
        for i,(x,y) in enumerate(zip(a,b)):
            z=c+t;wc=c/z;wt=t/z;out.append(wc*x+wt*y);c*=1+x;t*=1+y
            if step and (i+1)%step==0:
                z=c+t;to=abs(t-z*w)/z+abs(c-z*(1-w))/z;z*=1-.0003*to;c=z*(1-w);t=z*w
        return pd.Series(out,index=a.index)
    rows=[]
    for name,r in [('SPY',sr),('S8',s8),('50_50_monthly',rb(sr,s8,.5,21)),('50_50_quarterly',rb(sr,s8,.5,63)),('50_50_annual',rb(sr,s8,.5,252)),('25pct_S8_quarterly',rb(sr,s8,.25,63)),('75pct_S8_quarterly',rb(sr,s8,.75,63))]:
        e=(1+r).cumprod();yrs=len(r)/252;dd=e/e.cummax()-1;rows.append({'strategy':name,'cagr':e.iloc[-1]**(1/yrs)-1,'max_dd':dd.min(),'vol':r.std()*math.sqrt(252)})
    return pd.DataFrame(rows)

def main():
    out=Path('results/robust_portfolio_final');out.mkdir(parents=True,exist_ok=True);ff,meta=base.load_ff();parts=[]
    for i,(n,s) in enumerate(base.SCENARIOS.items()):parts.append(base.run_scenario(ff,n,s,10_000,base.SEED+10_000*i,batch_size=128))
    d=pd.concat(parts,ignore_index=True);agg=base.aggregate(d);real=actual_blend();d.to_csv(out/'scenario_results.csv',index=False);agg.to_csv(out/'robust_ranking.csv',index=False);real.to_csv(out/'actual_2010_2026.csv',index=False);(out/'manifest.json').write_text(json.dumps({'data':meta,'paths':10000,'strategies':list(base.STRATEGIES),'scenarios':base.SCENARIOS,'sleeve_rebalance_cost':.0003},indent=2));print(agg.to_string(index=False));print(real.to_string(index=False))
if __name__=='__main__':main()
