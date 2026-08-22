from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import research.hostile_strategy_tournament as base

TD=252
OUT=Path('results/futures_exhaustive')
OUT.mkdir(parents=True,exist_ok=True)

# Continuous futures approximation: fully collateralized index futures earn
# cash RF plus exposure times equity excess return. This avoids pretending
# futures have LETF expense/borrowing drag. Trading-cost sensitivity is applied
# per 1.0x change in notional exposure.
FUT_TURNOVER_COSTS=(0.00005,0.00010,0.00020,0.00040)  # 0.5/1/2/4 bp per 1x change
STRATS=(
 'equity_1x',
 'fut_2x_buyhold','fut_3x_buyhold',
 'fut_2x_sma200','fut_3x_sma200',
 'fut_s9_daily','fut_s9_band15','fut_s9_band30','fut_s9_band45','fut_s9_weekly','fut_s9_lag1',
 'fut_s9_175_band30','fut_s9_225_band30','fut_s9_30_10_band30','fut_s9_35_0_band30',
 'fut_tsmom12m_3x','fut_vol25','fut_vol30',
 'blend50_equity_s9band30',
)

def desired_exposure(px, r1, bull=.35, bear=.12, sma=200, extra_lag=0):
    ma=base.rm2(px,sma); vol=base.rs2(r1,20)*math.sqrt(TD); e=np.zeros_like(px)
    shift=1+extra_lag
    p=px[:,:-shift]; mm=ma[:,:-shift]; vv=vol[:,:-shift]
    valid=np.isfinite(mm)&np.isfinite(vv)&(vv>0)
    tv=np.where(p>mm,bull,bear)
    e[:,shift:]=np.where(valid,np.clip(tv/np.maximum(vv,1e-12),0,3),0)
    return e

def band(e,thresh):
    a=np.zeros_like(e)
    for t in range(1,e.shape[1]):
        prev=a[:,t-1]; want=e[:,t]; move=np.abs(want-prev)>=thresh
        a[:,t]=np.where(move,want,prev)
    return a

def weekly(e):
    a=np.zeros_like(e)
    for t in range(1,e.shape[1]): a[:,t]=np.where(t%5==0,e[:,t],a[:,t-1])
    return a

def futures_ret(exposure,mkt,rf,tc):
    turn=np.abs(np.diff(exposure,axis=1,prepend=np.zeros((exposure.shape[0],1))))
    return rf + exposure*(mkt-rf) - tc*turn

def trend_exp(px,L,w=200):
    ma=base.rm2(px,w); e=np.zeros_like(px); e[:,1:]=np.where(np.isfinite(ma[:,:-1])&(px[:,:-1]>ma[:,:-1]),L,0); return e

def tsmom_exp(px,L=3,w=252):
    e=np.zeros_like(px); e[:,w+1:]=np.where(px[:,w:-1]>px[:,:-w-1],L,0); return e

def vol_exp(r1,target):
    v=base.rs2(r1,20)*math.sqrt(TD); e=np.zeros_like(r1); vv=v[:,:-1]
    e[:,1:]=np.where(np.isfinite(vv)&(vv>0),np.clip(target/np.maximum(vv,1e-12),0,3),0); return e

def build(m,rf,tc):
    # m is total equity return from Fama-French market; use low-cost 1x benchmark.
    r1=base.levret(m,rf,1); px=np.cumprod(np.maximum(1+r1,1e-12),axis=1)
    e=desired_exposure(px,r1,.35,.12,200,0)
    eb30=band(e,.30)
    out={'equity_1x':r1,
         'fut_2x_buyhold':futures_ret(np.full_like(m,2.0),m,rf,tc),
         'fut_3x_buyhold':futures_ret(np.full_like(m,3.0),m,rf,tc),
         'fut_2x_sma200':futures_ret(trend_exp(px,2,200),m,rf,tc),
         'fut_3x_sma200':futures_ret(trend_exp(px,3,200),m,rf,tc),
         'fut_s9_daily':futures_ret(e,m,rf,tc),
         'fut_s9_band15':futures_ret(band(e,.15),m,rf,tc),
         'fut_s9_band30':futures_ret(eb30,m,rf,tc),
         'fut_s9_band45':futures_ret(band(e,.45),m,rf,tc),
         'fut_s9_weekly':futures_ret(weekly(e),m,rf,tc),
         'fut_s9_lag1':futures_ret(desired_exposure(px,r1,.35,.12,200,1),m,rf,tc),
         'fut_s9_175_band30':futures_ret(band(desired_exposure(px,r1,.35,.12,175,0),.30),m,rf,tc),
         'fut_s9_225_band30':futures_ret(band(desired_exposure(px,r1,.35,.12,225,0),.30),m,rf,tc),
         'fut_s9_30_10_band30':futures_ret(band(desired_exposure(px,r1,.30,.10,200,0),.30),m,rf,tc),
         'fut_s9_35_0_band30':futures_ret(band(desired_exposure(px,r1,.35,0,200,0),.30),m,rf,tc),
         'fut_tsmom12m_3x':futures_ret(tsmom_exp(px,3,252),m,rf,tc),
         'fut_vol25':futures_ret(vol_exp(r1,.25),m,rf,tc),
         'fut_vol30':futures_ret(vol_exp(r1,.30),m,rf,tc),
    }
    out['blend50_equity_s9band30']=.5*r1+.5*out['fut_s9_band30']
    return out

def summarize_scenario(ff,name,sc,npaths,seed,tc,batch=64):
    rng=np.random.default_rng(seed); maxd=max(base.H)*TD; total=maxd+base.WARM; hds=np.array(base.H)*TD
    m=ff.mkt.to_numpy(); rf=ff.rf.to_numpy(); dates=ff.index
    store={s:[[],[]] for s in STRATS}; beat={s:[[] for _ in base.H] for s in STRATS if s!='equity_1x'}
    done=0
    while done<npaths:
        b=min(batch,npaths-done); bm,br=base.sample_blocks(m,rf,dates,b,total,rng,sc); R=build(bm,br,tc); bt={}
        for s in STRATS:
            x=R[s][:,base.WARM:base.WARM+maxd]; t,d,_=base.pathstats(x,hds); bt[s]=t; store[s][0].append(t); store[s][1].append(d)
        for s in beat:
            for j in range(len(base.H)): beat[s][j].append(bt[s][:,j]>bt['equity_1x'][:,j])
        done+=b
        if done%1000==0: print(name,tc,done,flush=True)
    rows=[]
    for s in STRATS:
        T=np.concatenate(store[s][0]); D=np.concatenate(store[s][1])
        for j,h in enumerate(base.H):
            c=np.maximum(T[:,j],1e-300)**(1/h)-1
            rows.append(dict(cost_per_1x_turnover=tc,scenario=name,strategy=s,horizon_years=h,
                             median_cagr=np.median(c),p10_cagr=np.quantile(c,.1),
                             prob_beat_1x=np.nan if s=='equity_1x' else np.mean(np.concatenate(beat[s][j])),
                             prob_dd_gt_90=np.mean(D[:,j]<=-.9),median_drawdown=np.median(D[:,j])))
    return pd.DataFrame(rows)

def aggregate(d):
    g=d.groupby(['cost_per_1x_turnover','strategy','horizon_years'])
    return g.agg(worst_median_cagr=('median_cagr','min'),median_scenario_cagr=('median_cagr','median'),
                 worst_p10_cagr=('p10_cagr','min'),min_prob_beat_1x=('prob_beat_1x','min'),
                 max_prob_dd_gt_90=('prob_dd_gt_90','max'),worst_median_drawdown=('median_drawdown','min')).reset_index()

def discrete_mes_actual():
    # Actual-period MES feasibility from Micro E-mini launch. The P/L engine uses
    # SPY total-return excess return and ^GSPC only to size the $5*index contract.
    tickers=['SPY','^GSPC','^IRX']
    x=yf.download(tickers,start='2019-05-06',end='2026-08-22',auto_adjust=True,progress=False,threads=False)
    def close(t):
        s=x[('Close',t)] if isinstance(x.columns,pd.MultiIndex) else x['Close']; return pd.to_numeric(s,errors='coerce')
    spy=close('SPY').dropna(); spx=close('^GSPC').reindex(spy.index).ffill(); irx=close('^IRX').reindex(spy.index).ffill()/100/TD
    r=spy.pct_change().fillna(0); excess=(r-irx).fillna(0); ma=spy.rolling(200).mean(); vol=r.rolling(20).std()*math.sqrt(TD)
    rows=[]
    for capital in (25_000,50_000,100_000,250_000):
      for fee in (2.50,3.00,4.00,5.00):
       for cash_mult in (0.0,.5,1.0):
        wealth=float(capital); nprev=0; target_prev=0.; peak=wealth; mdd=0.; trades=0; rolls=0
        for i in range(1,len(spy)):
            if pd.isna(ma.iloc[i-1]) or pd.isna(vol.iloc[i-1]) or vol.iloc[i-1]<=0: desired=0.
            else:
                tv=.35 if spy.iloc[i-1]>ma.iloc[i-1] else .12; desired=float(np.clip(tv/vol.iloc[i-1],0,3))
            if abs(desired-target_prev)>=.30: target_prev=desired
            notional=float(spx.iloc[i-1])*5
            n=int(np.round(target_prev*wealth/max(notional,1)))
            # Conservative 25% initial-margin stress *125% IRA factor: if exceeded, reduce contracts.
            while n>0 and n*notional*.25*1.25>wealth: n-=1
            if n!=nprev:
                wealth-=abs(n-nprev)*fee; trades+=abs(n-nprev); nprev=n
            # quarterly roll approximation every 63 sessions: close + reopen, 2 sides per live contract
            if i%63==0 and nprev:
                wealth-=2*abs(nprev)*fee; rolls+=abs(nprev)
            exposure=(nprev*notional/max(wealth,1e-12))
            dayret=cash_mult*float(irx.iloc[i])+exposure*float(excess.iloc[i])
            wealth*=max(1+dayret,0); peak=max(peak,wealth); mdd=min(mdd,wealth/peak-1)
        yrs=(len(spy)-1)/TD; rows.append(dict(starting_capital=capital,fee_per_side=fee,cash_yield_fraction=cash_mult,
             cagr=(wealth/capital)**(1/yrs)-1,max_dd=mdd,terminal=wealth,contract_sides_per_year=(trades+2*rolls)/yrs))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--paths',type=int,default=10_000); a=ap.parse_args()
    if a.paths<10_000: raise SystemExit('requires >=10000 paths')
    ff,meta=base.load_ff(); parts=[]
    for ci,tc in enumerate(FUT_TURNOVER_COSTS):
        for si,(name,sc) in enumerate(base.SCENARIOS.items()): parts.append(summarize_scenario(ff,name,sc,a.paths,20260822+100000*ci+10000*si,tc))
    d=pd.concat(parts,ignore_index=True); agg=aggregate(d); disc=discrete_mes_actual()
    d.to_csv(OUT/'scenario_results.csv',index=False); agg.to_csv(OUT/'robust_ranking.csv',index=False); disc.to_csv(OUT/'mes_actual_discrete.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'paths_per_scenario':a.paths,'scenarios':base.SCENARIOS,'horizons':base.H,'data':meta,
      'continuous_formula':'rf + exposure*(equity_total_return-rf) - turnover_cost; exposure capped 0..3x',
      'band30':'equivalent to 10 percentage points of a 3x ETF allocation',
      'MES_actual':'2019-05-06 onward; $5*SPX notional; Schwab fee sensitivities; quarterly roll; 25% stressed initial margin and 125% IRA multiplier; cash yield 0/50/100% of T-bill proxy'},indent=2))
    print('\nTOP, 1bp turnover cost\n',agg[agg.cost_per_1x_turnover==.0001].sort_values(['horizon_years','worst_p10_cagr'],ascending=[True,False]).groupby('horizon_years').head(8).to_string(index=False))
    print('\nMES actual discrete\n',disc.to_string(index=False))
if __name__=='__main__': main()
