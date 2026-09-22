from __future__ import annotations

import argparse, hashlib, io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

TD=252
FF_URL='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
H=(5,10,20,30,40,50)
PATHS=10_000
SEED=20260821
WARM=300
EXP_1X=.0003
EXP_LEV=.0091
FIN_SPREAD=.005
TC_ALLOC=.0003
TC_SWITCH=.0005

SCENARIOS={
    'baseline': dict(block=63, drift_cut=0.0, rf_floor=0.0, stress_prob=0.0, crash_rate=0.0, crash_lo=0.0, crash_hi=0.0),
    'low_equity_premium': dict(block=63, drift_cut=.04, rf_floor=0.0, stress_prob=0.0, crash_rate=0.0, crash_lo=0.0, crash_hi=0.0),
    'permanent_5pct_rates': dict(block=63, drift_cut=0.0, rf_floor=.05, stress_prob=0.0, crash_rate=0.0, crash_lo=0.0, crash_hi=0.0),
    'stagflation_heavy': dict(block=126, drift_cut=.01, rf_floor=.04, stress_prob=.60, stress_kind='stagflation', crash_rate=.05, crash_lo=.06, crash_hi=.12),
    'long_high_vol_clusters': dict(block=126, drift_cut=0.0, rf_floor=0.0, stress_prob=.65, stress_kind='highvol', crash_rate=.10, crash_lo=.06, crash_hi=.14),
    'larger_surprise_crashes': dict(block=63, drift_cut=0.0, rf_floor=0.0, stress_prob=0.0, crash_rate=.20, crash_lo=.08, crash_hi=.18),
    'combined_hostile': dict(block=126, drift_cut=.03, rf_floor=.05, stress_prob=.65, stress_kind='combined', crash_rate=.25, crash_lo=.08, crash_hi=.18),
}

STRATEGIES=(
    '1x_buy_hold','2x_buy_hold','3x_buy_hold',
    '2x_sma200_cash','3x_sma200_cash','3x_sma225_cash',
    '2x_sma200_1x_below','3x_sma200_1x_below',
    'tsmom_6m_3x_cash','tsmom_12m_3x_cash',
    'vol20_tqqq_cash','vol30_tqqq_cash',
    's9_200_35_12','s9_200_35_12_lag1','s9_200_35_12_band5','s9_200_35_12_weekly',
    's9_200_35_0','s9_200_30_10','s9_225_40_0',
    's9_discrete_0_1_2_3x','s9_35_bull_spy_bear',
)

def load_ff():
    req=urllib.request.Request(FF_URL,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=60) as r: blob=r.read()
    sha=hashlib.sha256(blob).hexdigest(); rows=[]
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw=z.read([n for n in z.namelist() if n.lower().endswith('.csv')][0]).decode('latin-1')
    for line in raw.splitlines():
        q=[x.strip() for x in line.split(',')]
        if len(q)>=5 and len(q[0])==8 and q[0].isdigit():
            try: rows.append((pd.to_datetime(q[0],format='%Y%m%d'),(float(q[1])+float(q[4]))/100,float(q[4])/100))
            except ValueError: pass
    d=pd.DataFrame(rows,columns=['date','mkt','rf']).set_index('date').sort_index().loc['1926-07-01':]
    if len(d)<20_000: raise RuntimeError(f'history too short: {len(d)}')
    return d,{'source':FF_URL,'sha256':sha,'start':str(d.index.min().date()),'end':str(d.index.max().date()),'rows':len(d)}

def levret(m,rf,L):
    exp=EXP_1X if L==1 else EXP_LEV
    x=L*m-max(L-1,0)*(rf+FIN_SPREAD/TD)-exp/TD
    return np.maximum(x,-1.)

def rm2(x,w):
    out=np.full_like(x,np.nan,float); cs=np.cumsum(x,1); s=cs[:,w-1:].copy(); s[:,1:]-=cs[:,:-w]; out[:,w-1:]=s/w; return out

def rs2(x,w=20):
    out=np.full_like(x,np.nan,float); c=np.cumsum(x,1); c2=np.cumsum(x*x,1); s=c[:,w-1:].copy(); s2=c2[:,w-1:].copy(); s[:,1:]-=c[:,:-w]; s2[:,1:]-=c2[:,:-w]; mu=s/w; out[:,w-1:]=np.sqrt(np.maximum((s2-w*mu*mu)/(w-1),0)); return out

def desired_s9(px,r3,bull=.35,bear=.12,sma=200,extra_lag=0):
    ma=rm2(px,sma); vol=rs2(r3,20)*math.sqrt(TD); a=np.zeros_like(px)
    shift=1+extra_lag
    if shift>=px.shape[1]: return a
    p=px[:,:-shift]; mm=ma[:,:-shift]; vv=vol[:,:-shift]
    valid=np.isfinite(mm)&np.isfinite(vv)&(vv>0)
    target=np.where(p>mm,bull,bear)
    a[:,shift:]=np.where(valid,np.clip(target/np.maximum(vv,1e-12),0,1),0)
    return a

def alloc_returns(a,r3,rf,tc=TC_ALLOC):
    turnover=np.abs(np.diff(a,axis=1,prepend=np.zeros((a.shape[0],1))))
    return a*r3+(1-a)*rf-tc*turnover

def hysteresis_alloc(desired,threshold=.05):
    a=np.zeros_like(desired)
    for t in range(1,desired.shape[1]):
        prev=a[:,t-1]; want=desired[:,t]
        move=np.abs(want-prev)>=threshold
        a[:,t]=np.where(move,want,prev)
    return a

def weekly_alloc(desired,step=5):
    a=np.zeros_like(desired)
    for t in range(1,desired.shape[1]):
        a[:,t]=np.where(t%step==0,desired[:,t],a[:,t-1])
    return a

def trend_returns(px,risk,defensive,window=200):
    ma=rm2(px,window); sig=np.zeros_like(px,dtype=bool); sig[:,1:]=np.isfinite(ma[:,:-1])&(px[:,:-1]>ma[:,:-1])
    sw=np.abs(np.diff(sig.astype(float),axis=1,prepend=np.zeros((sig.shape[0],1))))
    return np.where(sig,risk,defensive)-TC_SWITCH*sw

def tsmom_returns(px,r3,rf,lookback):
    sig=np.zeros_like(px,dtype=bool)
    # Day t uses price known at t-1 versus t-1-lookback.
    sig[:,lookback+1:]=px[:,lookback:-1]>px[:,:-lookback-1]
    sw=np.abs(np.diff(sig.astype(float),axis=1,prepend=np.zeros((sig.shape[0],1))))
    return np.where(sig,r3,rf)-TC_SWITCH*sw

def discrete_s9(a,r1,r2,r3,rf):
    exposure=3*a
    state=np.zeros_like(exposure,dtype=np.int8)
    state[(exposure>=.5)&(exposure<1.5)]=1
    state[(exposure>=1.5)&(exposure<2.5)]=2
    state[exposure>=2.5]=3
    ret=np.where(state==0,rf,np.where(state==1,r1,np.where(state==2,r2,r3)))
    sw=np.abs(np.diff(state,axis=1,prepend=np.zeros((state.shape[0],1))))>0
    return ret-TC_SWITCH*sw

def s9_spy_bear(px,r3,r1,rf):
    ma=rm2(px,200); vol=rs2(r3,20)*math.sqrt(TD); bull=np.zeros_like(px,dtype=bool); bull[:,1:]=np.isfinite(ma[:,:-1])&(px[:,:-1]>ma[:,:-1])
    a=np.zeros_like(px); vv=vol[:,:-1]; a[:,1:]=np.where(np.isfinite(vv)&(vv>0),np.clip(.35/np.maximum(vv,1e-12),0,1),0)
    bullret=alloc_returns(a,r3,rf)
    return np.where(bull,bullret,r1)

def sample_blocks(m,rf,dates,b,n,rng,sc):
    block=sc['block']; nb=math.ceil(n/block); max_start=len(m)-block-1
    allstarts=np.arange(max_start+1)
    year=np.asarray(pd.DatetimeIndex(dates).year)
    stag=allstarts[(year[allstarts]>=1966)&(year[allstarts]<=1982)]
    rv=pd.Series(m).rolling(20).std().to_numpy(); cutoff=np.nanquantile(rv,.85)
    high=allstarts[np.nan_to_num(rv[allstarts],nan=0)>=cutoff]
    if sc.get('stress_kind')=='combined': special=np.unique(np.concatenate([stag,high]))
    elif sc.get('stress_kind')=='stagflation': special=stag
    elif sc.get('stress_kind')=='highvol': special=high
    else: special=np.array([],dtype=int)
    starts=rng.integers(0,max_start+1,size=(b,nb))
    if len(special) and sc.get('stress_prob',0)>0:
        mask=rng.random((b,nb))<sc['stress_prob']; repl=rng.choice(special,size=(b,nb)); starts=np.where(mask,repl,starts)
    idx=(starts[:,:,None]+np.arange(block)).reshape(b,-1)[:,:n]
    bm=m[idx].copy(); br=rf[idx].copy()
    if sc['drift_cut']:
        bm=(1+bm)*math.exp(-sc['drift_cut']/TD)-1
    if sc['rf_floor']:
        br=np.maximum(br,sc['rf_floor']/TD)
    rate=sc.get('crash_rate',0)
    if rate>0:
        years=n/TD
        for i in range(b):
            k=rng.poisson(rate*years)
            if k:
                pos=rng.integers(WARM,n,size=k)
                shock=rng.uniform(sc['crash_lo'],sc['crash_hi'],size=k)
                bm[i,pos]=(1+bm[i,pos])*(1-shock)-1
    return bm,br

def build_strategies(bm,br):
    r1=levret(bm,br,1); r2=levret(bm,br,2); r3=levret(bm,br,3); px=np.cumprod(np.maximum(1+r1,1e-12),axis=1)
    a=desired_s9(px,r3,.35,.12,200,0)
    out={
        '1x_buy_hold':r1,'2x_buy_hold':r2,'3x_buy_hold':r3,
        '2x_sma200_cash':trend_returns(px,r2,br,200),
        '3x_sma200_cash':trend_returns(px,r3,br,200),
        '3x_sma225_cash':trend_returns(px,r3,br,225),
        '2x_sma200_1x_below':trend_returns(px,r2,r1,200),
        '3x_sma200_1x_below':trend_returns(px,r3,r1,200),
        'tsmom_6m_3x_cash':tsmom_returns(px,r3,br,126),
        'tsmom_12m_3x_cash':tsmom_returns(px,r3,br,252),
        'vol20_tqqq_cash':alloc_returns(desired_s9(px,r3,.20,.20,10_000,0),r3,br),
        'vol30_tqqq_cash':alloc_returns(desired_s9(px,r3,.30,.30,10_000,0),r3,br),
        's9_200_35_12':alloc_returns(a,r3,br),
        's9_200_35_12_lag1':alloc_returns(desired_s9(px,r3,.35,.12,200,1),r3,br),
        's9_200_35_12_band5':alloc_returns(hysteresis_alloc(a,.05),r3,br),
        's9_200_35_12_weekly':alloc_returns(weekly_alloc(a,5),r3,br),
        's9_200_35_0':alloc_returns(desired_s9(px,r3,.35,0,200,0),r3,br),
        's9_200_30_10':alloc_returns(desired_s9(px,r3,.30,.10,200,0),r3,br),
        's9_225_40_0':alloc_returns(desired_s9(px,r3,.40,0,225,0),r3,br),
        's9_discrete_0_1_2_3x':discrete_s9(a,r1,r2,r3,br),
        's9_35_bull_spy_bear':s9_spy_bear(px,r3,r1,br),
    }
    # The huge-SMA trick above makes constant-vol targeting ignore trend after warm-up,
    # but its first 10k days would remain uninitialized. Replace with direct lagged vol allocation.
    v=rs2(r3,20)*math.sqrt(TD)
    for name,target in [('vol20_tqqq_cash',.20),('vol30_tqqq_cash',.30)]:
        aa=np.zeros_like(px); vv=v[:,:-1]; aa[:,1:]=np.where(np.isfinite(vv)&(vv>0),np.clip(target/np.maximum(vv,1e-12),0,1),0)
        out[name]=alloc_returns(aa,r3,br)
    return out

def pathstats(r,hds):
    b,n=r.shape; term=np.empty((b,len(hds))); mdd=np.empty_like(term); rec=np.empty_like(term)
    wealth=np.ones(b); peak=np.ones(b); ddmin=np.zeros(b); cur=np.zeros(b,int); mx=np.zeros(b,int); j=0
    for t in range(n):
        wealth*=np.maximum(1+r[:,t],0); new=wealth>=peak; peak=np.maximum(peak,wealth); ddmin=np.minimum(ddmin,wealth/peak-1); cur=np.where(new,0,cur+1); mx=np.maximum(mx,cur)
        if j<len(hds) and t+1==hds[j]: term[:,j]=wealth; mdd[:,j]=ddmin; rec[:,j]=mx/TD; j+=1
    return term,mdd,rec

def run_scenario(ff,name,sc,npaths,seed,batch_size=64):
    rng=np.random.default_rng(seed); maxd=max(H)*TD; total=maxd+WARM; hds=np.array(H)*TD
    store={s:[[],[],[]] for s in STRATEGIES}; beat={s:[[] for _ in H] for s in STRATEGIES if s!='1x_buy_hold'}
    m=ff.mkt.to_numpy(); rf=ff.rf.to_numpy(); dates=ff.index; done=0
    while done<npaths:
        b=min(batch_size,npaths-done); bm,br=sample_blocks(m,rf,dates,b,total,rng,sc); R=build_strategies(bm,br); batchterm={}
        for s in STRATEGIES:
            x=R[s][:,WARM:WARM+maxd]; t,d,rc=pathstats(x,hds); batchterm[s]=t; store[s][0].append(t);store[s][1].append(d);store[s][2].append(rc)
        for s in beat:
            for j in range(len(H)): beat[s][j].append(batchterm[s][:,j]>batchterm['1x_buy_hold'][:,j])
        done+=b
        if done%1000==0 or done==npaths: print(f'{name}: {done}/{npaths}',flush=True)
    rows=[]
    for s in STRATEGIES:
        T=np.concatenate(store[s][0]); D=np.concatenate(store[s][1]); RC=np.concatenate(store[s][2])
        for j,y in enumerate(H):
            c=np.where(T[:,j]>0,T[:,j]**(1/y)-1,-1)
            rows.append({'scenario':name,'strategy':s,'horizon_years':y,'n_paths':npaths,
                         'median_cagr':float(np.median(c)),'p10_cagr':float(np.quantile(c,.10)),
                         'p25_cagr':float(np.quantile(c,.25)),'prob_negative_cagr':float(np.mean(c<0)),
                         'prob_beat_1x':0.0 if s=='1x_buy_hold' else float(np.mean(np.concatenate(beat[s][j]))),
                         'prob_dd_gt_90':float(np.mean(D[:,j]<=-.90)),'prob_dd_gt_70':float(np.mean(D[:,j]<=-.70)),
                         'median_max_drawdown':float(np.median(D[:,j])),'p10_max_drawdown':float(np.quantile(D[:,j],.10)),
                         'median_underwater_years':float(np.median(RC[:,j])),'p90_underwater_years':float(np.quantile(RC[:,j],.90))})
    return pd.DataFrame(rows)

def aggregate(df):
    rows=[]
    for (s,h),g in df.groupby(['strategy','horizon_years']):
        rows.append({'strategy':s,'horizon_years':h,
                     'worst_median_cagr':g.median_cagr.min(),'median_of_scenario_medians':g.median_cagr.median(),
                     'worst_p10_cagr':g.p10_cagr.min(),'median_p10_cagr':g.p10_cagr.median(),
                     'min_prob_beat_1x':g.prob_beat_1x.min() if s!='1x_buy_hold' else 0,
                     'mean_prob_beat_1x':g.prob_beat_1x.mean() if s!='1x_buy_hold' else 0,
                     'max_prob_dd_gt_90':g.prob_dd_gt_90.max(),'max_prob_dd_gt_70':g.prob_dd_gt_70.max(),
                     'worst_median_drawdown':g.median_max_drawdown.min(),'worst_p90_underwater_years':g.p90_underwater_years.max(),
                     'scenario_median_wins':0})
    out=pd.DataFrame(rows)
    for h in H:
        sub=df[df.horizon_years==h]
        wins={s:0 for s in STRATEGIES}
        for sc,g in sub.groupby('scenario'):
            best=g.loc[g.median_cagr.idxmax(),'strategy']; wins[best]+=1
        for s,n in wins.items(): out.loc[(out.horizon_years==h)&(out.strategy==s),'scenario_median_wins']=n
    out['tail_safe']=out.max_prob_dd_gt_90<=.05
    out['robust_rank']=out.groupby('horizon_years')['worst_p10_cagr'].rank(ascending=False,method='min')
    return out.sort_values(['horizon_years','robust_rank','worst_median_cagr'],ascending=[True,True,False])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--paths',type=int,default=PATHS); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--output',type=Path,default=Path('results/hostile_strategy_tournament')); a=ap.parse_args()
    if a.paths<10_000: raise SystemExit('serious tournament requires >=10,000 paths per scenario')
    a.output.mkdir(parents=True,exist_ok=True); ff,meta=load_ff(); parts=[]
    for i,(name,sc) in enumerate(SCENARIOS.items()): parts.append(run_scenario(ff,name,sc,a.paths,a.seed+10_000*i))
    d=pd.concat(parts,ignore_index=True); agg=aggregate(d); d.to_csv(a.output/'scenario_results.csv',index=False); agg.to_csv(a.output/'robust_ranking.csv',index=False)
    manifest={'data':meta,'paths_per_scenario':a.paths,'seed':a.seed,'horizons':H,'scenarios':SCENARIOS,'strategies':STRATEGIES,
              'costs':{'1x_expense':EXP_1X,'lev_expense':EXP_LEV,'financing_spread':FIN_SPREAD,'allocation_turnover':TC_ALLOC,'switch_cost':TC_SWITCH},
              'selection_rule':'Prefer highest worst-scenario p10 CAGR subject to max P(>90% DD)<=5%; also report raw median-CAGR winner per horizon.',
              'execution_tests':['extra full-day signal lag','5 percentage point rebalance band','weekly rebalance','discrete 0/1/2/3x ETF state']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nTOP ROBUST BY HORIZON')
    for h in H: print('\n',h,'years\n',agg[(agg.horizon_years==h)&agg.tail_safe].head(8).to_string(index=False))
if __name__=='__main__': main()
