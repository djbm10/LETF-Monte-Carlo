"""Frozen 80% Nasdaq 35/0 + 20% risk-managed x2 sector rotation.
Long before actual 2x fund inception use synthetic DAILY RESET 2x constructed
from real sector ETF price histories and explicit borrow/funding costs.
No historical parameter optimization or claim of untouched holdout.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('/mnt/data/long_history_80_20_20261106')
C=pd.read_csv(ROOT/'adjusted_close_1985_2026.csv',index_col=0,parse_dates=True).sort_index()
O=pd.read_csv(ROOT/'adjusted_open_1985_2026.csv',index_col=0,parse_dates=True).sort_index().reindex(C.index)
I=C.index
P10={'XLK':'ROM','SOXX':'USD','XLF':'UYG','XLV':'RXL','XLE':'DIG','XLY':'UCC','XLB':'UYM','XLI':'UXI','XLP':'UGE','VNQ':'URE'}
P9={k:P10.get(k, 'none') for k in ['XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLY','XLB']}
ALL_UNDER=sorted(set(P10)|set(P9))
# Macro day x -> x+1; returns indexed at the starting open. All signals from previous CLOSE.
RF=(C['^IRX'].ffill().clip(lower=0)/100/252).clip(upper=.001).fillna(.025/252).to_numpy()
QQQ=C['QQQ'].to_numpy(dtype=float); QQO=O['QQQ'].to_numpy(dtype=float)
SPY=C['SPY'].to_numpy(dtype=float)
VQQ=C['QQQ'].pct_change(fill_method=None).sub(pd.Series(RF,index=I)).rolling(20).std().to_numpy(dtype=float)*np.sqrt(252)
QMA=C['QQQ'].rolling(200).mean().to_numpy(dtype=float)
SMA=C['SPY'].rolling(200).mean().to_numpy(dtype=float)
# Underlying open-to-open gross daily returns; NOT actual sector ETF leveraged NAV returns.
UR={k:O[k].shift(-1).div(O[k]).sub(1).to_numpy(dtype=float) for k in ALL_UNDER}
actual={k:O[lev].shift(-1).div(O[lev]).sub(1).to_numpy(dtype=float) if lev in O else np.full(len(I),np.nan) for k,lev in P10.items()}
UNDERC={k:C[k].to_numpy(dtype=float) for k in ALL_UNDER}

def create_2x(mode, keys, funding_spread=.01):
    """Construct per-interval sector 2x ETF return; missing history remains nan."""
    out={}
    for k in keys:
        r=2*UR[k] - RF - (.0095+funding_spread)/252
        r=np.where(r<=-1,np.nan,r)
        if mode=='actual':r=actual.get(k,np.full(len(I),np.nan))
        elif mode=='stitched':
            a=actual.get(k,np.full(len(I),np.nan))
            r=np.where(np.isfinite(a),a,r)
        elif mode!='synthetic':raise ValueError(mode)
        out[k]=r
    return out

def base_returns(extra_spread=0.):
    x=np.full(len(I),np.nan);exposure=np.zeros(len(I))
    prev=0.
    for t in range(202,len(I)-1):
        if not(np.isfinite(QQO[t]) and np.isfinite(QQO[t+1]) and np.isfinite(QQQ[t-1]) and np.isfinite(QMA[t-1]) and np.isfinite(VQQ[t-1])):continue
        v=min(3.,.35/max(float(VQQ[t-1]),.04)) if QQQ[t-1]>QMA[t-1] else 0.
        qret=QQO[t+1]/QQO[t]-1
        # Cash earns T-bill, financing on Nasdaq exposure is at T-bill rate.
        ret=RF[t]+v*(qret-RF[t])-0.0002*abs(v-prev)-extra_spread/252*max(v-1,0)
        if ret<=-1:raise RuntimeError(f'Benchmark negative wealth {I[t]}')
        x[t+1]=ret;exposure[t+1]=v
        prev=v
    return pd.Series(x,index=I),pd.Series(exposure,index=I)

def sector_returns(mode='synthetic',universe='P10',spread=.01,lookback=63,cost=.0007,extra_delay=0,exclude=None):
    keys=list(P10 if universe=='P10' else P9)
    rets=create_2x(mode,keys,spread)
    # Keep sector-ETF 20-day realized volatility from open returns; <= t-1 open is known at open t.
    vv={k:pd.Series(rets[k]).shift(1).rolling(20,min_periods=20).std().to_numpy()*np.sqrt(252) for k in keys}
    scores={k:C[k].div(C[k].shift(lookback)).sub(1).to_numpy(dtype=float) for k in keys}
    fullma=I.to_period('M');n=len(I)
    ret=np.full(n,np.nan);wealth=1.;cash=1.;pos={k:0. for k in keys};chosen=[];trade_count=0;turnover=0.;cash_days=0
    prev_index=None
    for t in range(275,n-1):
        # Do not manufacture P&L before QQQ exists, but sector sleeve independently has earlier prices.
        if t<276:continue
        if I[t]<pd.Timestamp('1999-03-11'):continue
        before=cash+sum(pos.values());
        if not np.isfinite(before) or before<=0: raise RuntimeError('Sector bankrupt')
        if prev_index is None or fullma[t]!=fullma[prev_index]:
            prev=t-1-extra_delay
            chosen_sector=None
            if prev>lookback and np.isfinite(SPY[prev]) and np.isfinite(SMA[prev]) and SPY[prev]>SMA[prev]:
                eligible=[]
                for k in keys:
                    # Candidate must have been in available universe at prior close.
                    if exclude==k:continue
                    # True OHLC cannot check next return at selection time.
                    if not(np.isfinite(scores[k][prev]) and np.isfinite(vv[k][prev]) and vv[k][prev]>.001 and
                           np.isfinite(O.at[I[t],k]) and np.isfinite(UNDERC[k][prev])):continue
                    if mode=='actual':
                        lev=P10.get(k)
                        if lev not in O or not np.isfinite(O.at[I[t],lev]):continue
                    eligible.append(k)
                if eligible:chosen_sector=max(eligible,key=lambda k:scores[k][prev])
            target={k:0. for k in keys}
            if chosen_sector is not None:
                target[chosen_sector]=min(1.,.35/max(float(vv[chosen_sector][prev]),.08))
            old={k:pos[k]/before for k in keys}
            one_way=sum(abs(target[k]-old[k]) for k in keys)
            turncost=before*one_way*cost
            after=before-turncost
            if after<=0:raise RuntimeError('Sector execution cost bankruptcy')
            for k in keys:pos[k]=after*target[k]
            cash=after*(1-sum(target.values()))
            trade_count+=1;turnover+=one_way
            chosen.append(dict(date=I[t].date().isoformat(),selected=chosen_sector,investment_weight=target.get(chosen_sector,0) if chosen_sector else 0,turnover=one_way,signal_date=I[prev].date().isoformat()))
        # At open[t], all prices are observed. Trading was possible using prior close signals.
        cash*=1+RF[t]
        for k in keys:
            if pos[k]:
                rr=rets[k][t]
                if not np.isfinite(rr):
                    # No zero-filling in portfolio holdings. Fail closed for affected model.
                    raise RuntimeError(f'Unobserved held sector return for {k} {I[t]} {mode} {universe}')
                pos[k]*=(1+rr)
        after=cash+sum(pos.values())
        if after<=0: raise RuntimeError(f'Sector sleeve zero {I[t]}')
        ret[t+1]=after/before-1
        if sum(pos.values())==0:cash_days+=1
        prev_index=t
    return pd.Series(ret,index=I),pd.DataFrame(chosen),dict(trades=trade_count,turnover=turnover,cash_days=cash_days)

def mix_portfolios(base,sector,weight=.20,freq='M',cost=.0005):
    xx=pd.concat([base,sector],axis=1,keys=['base','sector']).dropna()
    periods=xx.index.to_period(freq);a=1-weight;b=weight;returns=[];totalturn=0
    for t,(dt,x) in enumerate(xx.iterrows()):
        before=a+b
        if t and periods[t]!=periods[t-1]:
            wcur=b/before
            turnover=2*abs(wcur-weight)
            before=before*(1-cost*turnover)
            a=before*(1-weight);b=before*weight;totalturn+=turnover
        a*=1+x['base'];b*=1+x['sector']
        returns.append((a+b)/before-1)
    return pd.Series(returns,index=xx.index),totalturn

def metric(r):
    a=pd.Series(r).dropna()
    if len(a)<252:return None
    eq=np.exp(np.log1p(a).cumsum());dd=eq/eq.cummax()-1
    yr=len(a)/252;rol=np.expm1(np.log1p(a).rolling(252).sum()).dropna()
    return {'start':str(a.index[0].date()),'end':str(a.index[-1].date()),'years':float(yr),'n':len(a),'cagr':float(eq.iloc[-1]**(1/yr)-1),
        'maxdd':float(dd.min()),'vol':float(a.std()*np.sqrt(252)),'logcagr':float(np.log1p(a).mean()*252),
        'sharpe':float(np.sqrt(252)*a.mean()/a.std()),'worst_rolling_1y':float(rol.min()) if len(rol) else None,
        'p10_rolling_1y':float(rol.quantile(.1)) if len(rol) else None,'terminal':float(eq.iloc[-1])}

BASE,EXPOSURE=base_returns()
WINDOWS=[('longest','1999-01-01','2026-11-05'),('2000plus','2000-01-01','2026-11-05'),('2007plus','2007-06-01','2026-11-05'),('2010plus','2010-01-01','2026-11-05'),('2015plus','2015-01-01','2026-11-05'),('2019plus','2019-01-01','2026-11-05'),('dotcom_bear','2000-01-01','2002-12-31'),('GFC','2007-06-01','2009-12-31'),('COVID','2020-01-01','2020-12-31'),('inflation_2022','2022-01-01','2022-12-31'),('post2023','2023-01-01','2026-11-05')]
DAILY=pd.DataFrame({'base35':BASE,'rf':pd.Series(RF,index=I)})
rows=[];calibrations=[];selections=[]
# Exact 2019 construction, and original-nine (stable 1998 membership) as an alternative-universe stress.
scenarios=[]
for universe,mode,spread,delay,cost in [('P10','synthetic',.01,0,.0007),('P10','stitched',.01,0,.0007),('P10','actual',.01,0,.0007),('P9','synthetic',.01,0,.0007),('P10','synthetic',.03,0,.0007),('P10','synthetic',0,0,.0007),('P10','synthetic',.01,1,.0007),('P10','synthetic',.01,0,.003)]:
    sid=f'{universe}_{mode}_spread{int(spread*100)}_delay{delay}_tc{int(cost*10000)}'
    try:
        sect,sel,detail=sector_returns(mode,universe,spread,63,cost,delay)
    except Exception as e:
        rows.append(dict(scenario=sid,window='ERROR',error=str(e)));continue
    DAILY[f'sector_{sid}']=sect
    blend,turn=mix_portfolios(BASE,sect)
    DAILY[f'blend_{sid}']=blend
    scenarios.append(sid)
    sel['scenario']=sid;selections.append(sel)
    for name,series in [('base35',BASE),('sector',sect),('blend80_20',blend),('cash80_20',.8*BASE+.2*pd.Series(RF,index=I))]:
        for label,start,end in WINDOWS:
            x=series.loc[start:end]
            # only compare when both base AND sleeve have valid history on every common trading day
            valid=BASE.loc[start:end].notna()&sect.loc[start:end].notna()
            if valid.sum()<252:continue
            if name=='blend80_20':z=x.loc[valid.index[valid]]
            else:z=x.loc[valid.index[valid]]
            m=metric(z)
            if m:rows.append({'scenario':sid,'window':label,'strategy':name,**m})
    print('SCENARIO',sid,detail,'first',sect.first_valid_index(),flush=True)
res=pd.DataFrame(rows);res.to_csv(ROOT/'long_historical_metrics.csv',index=False)
DAILY.to_csv(ROOT/'long_daily_strategy_returns.csv')
pd.concat(selections,ignore_index=True).to_csv(ROOT/'sector_selection_audit.csv',index=False)
# Independently check synthetic 2x tracking against *actual* ETF histories in overlapping dates.
for s in P10:
    raw=UR[s];real=actual.get(s)
    if real is None:continue
    synth=2*raw-RF-(.0095+.01)/252
    finite=np.isfinite(real)&np.isfinite(synth)
    if finite.sum()<200:continue
    a=real[finite];b=synth[finite]
    calibrations.append(dict(underlying=s,actual_ticker=P10[s],n=int(finite.sum()),first=str(I[np.where(finite)[0][0]].date()),
        daily_correlation=float(np.corrcoef(a,b)[0,1]),mean_tracking_diff_ann=float((a-b).mean()*252),
        tracking_error_vol_ann=float((a-b).std()*np.sqrt(252)),actual_cagr=float(np.expm1(np.log1p(a).mean()*252)),synthetic_cagr=float(np.expm1(np.log1p(b).mean()*252))))
pd.DataFrame(calibrations).to_csv(ROOT/'synthetic_2x_vs_real_tracking.csv',index=False)
manifest={'status':'EXPLORE, not untouched holdout; must not promote from this sample',
 'investable_comparison_start':'1999 QQQ inception, next-open execution after indicators warm up',
 'original_sector_data':'1998-12-22 onwards, real sector ETFs; 2x fund P10 symbols 2007-02-01 onwards',
 'benchmark':'Nasdaq QQQ 35/0 using prior close > 200SMA; prior 20 day realized close vol; 35pct exposure vol target, 0 below; 3x cap. Return QQQ auto-adjusted open[t]->open[t+1] plus T-bill cash and 2bp per notional exposure turn. Not dated CME futures. ',
 'sector':'P10 originally selected sector ETF underlying ranks prior close 63d, above prior-day SPY 200SMA, 20-day realized vol on synthetic/actual sector leveraged return, target .35 fund volatility, max fund weight 100%, monthly at open, 7bp one-way traded notional',
 '2x_synthetic':'ETF adjusted opening return reset daily at OPEN instead of actual NAV close reset: 2*under_open_return - ^IRX_daily - 0.95pct/yr expense - variable 0/1/3 pct/yr financing spread. NOT available before inception.',
 '2x_actual':'Actual available 2007+ adjusted opening prices of ROM,USD,UYG,RXL,DIG,UCC,UYM,UXI,UGE,URE. Fees/distributions in adjusted returns.',
 'sleeve_mix':'Monthly 80/20 drifting weights; 5bps on sleeve 2-way turnover.',
 'prospective_control':'Prior trend family approved as research; actual dated futures execution still MONITOR.',
 'limitations':['Entire past examined; no fresh holdout','Daily-reset at opening is not actual fund reset at close','Risk free from 13-week Tbill ^IRX, funding differs from 2x ETF swaps','Survivorship/product-selection bias in choosing present sector ETFs and labels','Sector universe sparse before 2004, reconstructed leverage until 2007','No personal tax or margin/liquidity stresses','Yahoo adjusted historical opens can have corporate action errors','Need separate limit if 2x synthetic daily loss near 100pct']}
(ROOT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
print('\nPRIMARY RESULTS')
print(res[(res.scenario=='P10_synthetic_spread1_delay0_tc7')&(res.window.isin(['longest','2000plus','2007plus','2019plus','dotcom_bear','GFC']))][['window','strategy','cagr','maxdd','vol']].to_string(index=False))
print('\nCALIBRATION')
print(pd.DataFrame(calibrations).to_string(index=False))
