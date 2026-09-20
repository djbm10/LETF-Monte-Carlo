from __future__ import annotations

import json, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit

TD=252
OUT=Path('results/nasdaq1985_final_tournament'); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
H=(5,10,20,30,40,50)
N_PATHS=25000
TC_FUT=0.00010  # 1 bp per 1.0x notional change, long-horizon base case
ETF_TC=0.0003   # 3 bp per traded dollar sensitivity proxy

@dataclass(frozen=True)
class Cand:
    name:str; kind:str; p:tuple

def read_series(name,col='Close'):
    d=pd.read_csv(BASE+name,sep='\t')
    d['Date']=pd.to_datetime(d['Date']).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

def load_data(div_yield_pre1999=0.007):
    q=read_series('synthetic-qqq.tsv'); rate=read_series('short-rates.tsv','Rate')
    q=q[q.index>=pd.Timestamp('1985-01-31')]
    rate=rate.reindex(q.index).ffill().bfill()/100/TD
    qr=q.pct_change().fillna(0)
    # synthetic-qqq subtracts ~20bp QQQ expense. Add it back to approximate NDX total return.
    # Pre-1999 NDX source is price-only, so add an explicit dividend-yield sensitivity.
    ndx=qr + 0.002/TD
    ndx.loc[ndx.index<pd.Timestamp('1999-03-10')] += div_yield_pre1999/TD
    return pd.DataFrame({'ndx':ndx,'rf':rate,'qqq_like':qr},index=q.index)

def synth_letf(ndx,rf,L,expense,spread):
    return L*ndx-(L-1)*rf-((L-1)*spread+expense)/TD

def roll_mean(x,w): return pd.Series(x).rolling(w).mean().to_numpy()
def roll_vol(r,w=20): return pd.Series(r).rolling(w).std().to_numpy()*math.sqrt(TD)

def exposure_sma(px,w,L,bear=0.0,buffer=0.0):
    ma=roll_mean(px,w); e=np.zeros(len(px)); state=0.0
    for i in range(1,len(px)):
        if not np.isfinite(ma[i-1]): continue
        ratio=px[i-1]/ma[i-1]-1
        if state<=bear+1e-12:
            if ratio>buffer: state=L
            else: state=bear
        else:
            if ratio<-buffer: state=bear
            else: state=L
        e[i]=state
    return e

def exposure_vol(r,target,cap=3.0):
    v=roll_vol(r,20); e=np.zeros(len(r))
    for i in range(1,len(r)):
        if np.isfinite(v[i-1]) and v[i-1]>0: e[i]=min(cap,target/v[i-1])
    return e

def exposure_tv(px,r,w,bull,bear,bandv,cap=3.0):
    ma=roll_mean(px,w); v=roll_vol(r,20); e=np.zeros(len(r)); hold=0.0
    for i in range(1,len(r)):
        if not (np.isfinite(ma[i-1]) and np.isfinite(v[i-1]) and v[i-1]>0): continue
        tv=bull if px[i-1]>ma[i-1] else bear
        want=min(cap,tv/v[i-1])
        if abs(want-hold)>=bandv: hold=want
        e[i]=hold
    return e

def exposure_mom(px,look,L):
    e=np.zeros(len(px))
    for i in range(look+1,len(px)):
        e[i]=L if px[i-1]>px[i-look-1] else 0.0
    return e

def fut_ret(ndx,rf,e,tc=TC_FUT):
    turn=np.abs(np.diff(e,prepend=0.0)); return rf+e*(ndx-rf)-tc*turn

def alloc_ret(asset,rf,a,tc=ETF_TC,cash_mult=0.0):
    turn=np.abs(np.diff(a,prepend=0.0)); return a*asset+(1-a)*(cash_mult*rf)-tc*turn

def metric(r):
    r=np.nan_to_num(np.asarray(r,float)); g=np.maximum(1+r,1e-12); eq=np.cumprod(g); yrs=len(r)/TD
    dd=eq/np.maximum.accumulate(eq)-1
    return dict(cagr=eq[-1]**(1/yrs)-1,max_dd=dd.min(),vol=np.std(r)*math.sqrt(TD),terminal=eq[-1])

def candidates():
    out=[Cand('ndx_1x','fixed',(1.0,))]
    for L in (1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0): out.append(Cand(f'fut_fixed_{L:g}x','fixed',(L,)))
    for w in (100,125,150,175,200,225,250,300):
      for L in (1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0):
       for bear in (0.0,0.5,1.0):
        for buf in (0.0,0.01,0.02,0.03): out.append(Cand(f'fut_sma{w}_{L:g}x_b{bear:g}_buf{int(buf*100)}','sma',(w,L,bear,buf)))
    for t in np.arange(.15,.425,.025): out.append(Cand(f'fut_vol{int(round(t*100))}','vol',(float(t),)))
    for w in (150,175,200,225,250):
      for bull in (.25,.30,.35,.40):
       for bear in (0.0,.05,.10,.12,.15):
        for bandv in (.10,.20,.30,.40): out.append(Cand(f'fut_tv_w{w}_u{int(bull*100)}_d{int(bear*100)}_b{int(bandv*100)}','tv',(w,bull,bear,bandv)))
    for look in (63,126,189,252):
      for L in (1.5,2.0,2.5,3.0): out.append(Cand(f'fut_mom{look}_{L:g}x','mom',(look,L)))
    # ETF implementations / known challengers. ETF allocation is 0..100%; TQQQ is internally 3x.
    for veh in ('qld','tqqq'):
      for w in (175,200,225):
       for bull in (.30,.35,.40):
        for bear in (0.0,.12):
         for bandp in (.05,.10,.15): out.append(Cand(f'{veh}_s9_w{w}_u{int(bull*100)}_d{int(bear*100)}_b{int(bandp*100)}','etf_tv',(veh,w,bull,bear,bandp)))
    return out

def evaluate(c,ndx,rf,px,qld,tqqq):
    if c.kind=='fixed':
        L,=c.p; e=np.full(len(ndx),L); e[0]=0; return fut_ret(ndx,rf,e)
    if c.kind=='sma': return fut_ret(ndx,rf,exposure_sma(px,*c.p))
    if c.kind=='vol': return fut_ret(ndx,rf,exposure_vol(ndx,c.p[0]))
    if c.kind=='tv': return fut_ret(ndx,rf,exposure_tv(px,ndx,*c.p))
    if c.kind=='mom': return fut_ret(ndx,rf,exposure_mom(px,*c.p))
    if c.kind=='etf_tv':
        veh,w,bull,bear,bandp=c.p; ar=qld if veh=='qld' else tqqq
        # Convert target portfolio volatility to ETF weight; ETF itself is the risky asset.
        ma=roll_mean(px,w); v=roll_vol(ar,20); a=np.zeros(len(ar)); hold=0.0
        for i in range(1,len(ar)):
            if not (np.isfinite(ma[i-1]) and np.isfinite(v[i-1]) and v[i-1]>0): continue
            tv=bull if px[i-1]>ma[i-1] else bear; want=min(1.0,tv/v[i-1])
            if abs(want-hold)>=bandp: hold=want
            a[i]=hold
        return alloc_ret(ar,rf,a,ETF_TC,0.0)
    raise ValueError(c)

def historical_screen(df):
    ndx=df.ndx.to_numpy(); rf=df.rf.to_numpy(); px=np.cumprod(np.maximum(1+ndx,1e-12))
    qld=synth_letf(ndx,rf,2,.0095,.0050); tqqq=synth_letf(ndx,rf,3,.0088,.0065)
    rows=[]; foldrows=[]; cs=candidates(); dates=df.index
    holdout_start=pd.Timestamp('2010-02-11')
    for j,c in enumerate(cs):
        r=evaluate(c,ndx,rf,px,qld,tqqq)
        full=metric(r)
        train=r[dates<holdout_start]; test=r[dates>=holdout_start]
        mt=metric(train) if len(train)>TD else {}; mh=metric(test) if len(test)>TD else {}
        # fixed 5-year forward blocks after a 10-year warm-up; no parameter refit inside blocks.
        fcs=[]; fdds=[]
        start=dates.min()+pd.DateOffset(years=10)
        while start < dates.max()-pd.DateOffset(years=4):
            end=start+pd.DateOffset(years=5); mask=(dates>=start)&(dates<end)
            if mask.sum()>TD*3:
                mm=metric(r[mask]); fcs.append(mm['cagr']); fdds.append(mm['max_dd'])
                foldrows.append({'strategy':c.name,'start':str(start.date()),'end':str(end.date()),**mm})
            start=end
        rows.append({'strategy':c.name,'kind':c.kind,**full,
                     'train_cagr':mt.get('cagr',np.nan),'train_dd':mt.get('max_dd',np.nan),
                     'holdout_cagr':mh.get('cagr',np.nan),'holdout_dd':mh.get('max_dd',np.nan),
                     'median_5y_cagr':np.median(fcs) if fcs else np.nan,'p10_5y_cagr':np.quantile(fcs,.1) if fcs else np.nan,
                     'worst_5y_cagr':np.min(fcs) if fcs else np.nan,'worst_5y_dd':np.min(fdds) if fdds else np.nan})
        if (j+1)%250==0: print('screen',j+1,'/',len(cs),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(foldrows),cs

# ---------- Monte Carlo ----------
@njit(cache=True)
def pathstats(r,hds):
    b,n=r.shape; k=len(hds); term=np.empty((b,k)); mdd=np.empty((b,k)); rec=np.empty((b,k))
    for i in range(b):
        wealth=1.; peak=1.; ddmin=0.; cur=0; mx=0; j=0
        for t in range(n):
            g=1.+r[i,t]
            if g<1e-12:g=1e-12
            wealth*=g
            if wealth>=peak: peak=wealth;cur=0
            else:
                cur+=1; mx=max(mx,cur)
            dd=wealth/peak-1.
            if dd<ddmin:ddmin=dd
            if j<k and t+1==hds[j]:term[i,j]=wealth;mdd[i,j]=ddmin;rec[i,j]=mx/TD;j+=1
    return term,mdd,rec

def bootstrap(ndx,rf,dates,b,n,rng,scenario):
    L=scenario.get('block',63); N=len(ndx); stress=np.zeros(N,dtype=bool)
    yrs=pd.DatetimeIndex(dates).year
    stress |= np.isin(yrs,[1987,2000,2001,2002,2008,2009,2020,2022])
    starts=np.arange(0,max(1,N-L))
    if scenario.get('stress_weight',1)>1:
        w=np.ones(len(starts)); w[stress[:len(starts)]]=scenario['stress_weight']; w/=w.sum()
    else:w=None
    M=np.empty((b,n)); R=np.empty((b,n))
    for i in range(b):
        pos=0
        while pos<n:
            s=int(rng.choice(starts,p=w)) if w is not None else int(rng.integers(0,len(starts)))
            take=min(L,n-pos); M[i,pos:pos+take]=ndx[s:s+take];R[i,pos:pos+take]=rf[s:s+take];pos+=take
    if scenario.get('drift_cut',0): M-=scenario['drift_cut']/TD
    if scenario.get('rate_floor',0): R=np.maximum(R,scenario['rate_floor']/TD)
    crash_rate=scenario.get('crashes_per_year',0)
    if crash_rate:
        p=crash_rate/TD; mask=rng.random(M.shape)<p; shocks=rng.uniform(.08,.18,size=M.shape); M=np.where(mask,M-shocks,M)
    return M,R

SCEN={
 'baseline':{'block':63},
 'low_premium':{'block':63,'drift_cut':.04},
 'high_rates':{'block':63,'rate_floor':.05},
 'stress_regimes':{'block':126,'stress_weight':4},
 'long_clusters':{'block':252,'stress_weight':2},
 'crash_heavy':{'block':63,'crashes_per_year':.20},
 'combined_hostile':{'block':126,'stress_weight':4,'drift_cut':.04,'rate_floor':.05,'crashes_per_year':.25},
}

def eval_path_candidate(c,M,R):
    b,n=M.shape; PX=np.cumprod(np.maximum(1+M,1e-12),axis=1)
    # selected finalists intentionally limited to forms that vectorize reliably.
    if c.kind=='fixed':
        L=c.p[0]; E=np.full_like(M,L); E[:,0]=0
    elif c.kind=='sma':
        w,L,bear,buf=c.p; E=np.zeros_like(M)
        # simple no-hysteresis equivalent for MC if buffer=0; finalists with buffers handled statefully below
        for i in range(b): E[i]=exposure_sma(PX[i],w,L,bear,buf)
    elif c.kind=='vol':
        t=c.p[0]; E=np.zeros_like(M)
        for i in range(b): E[i]=exposure_vol(M[i],t)
    elif c.kind=='tv':
        E=np.zeros_like(M)
        for i in range(b): E[i]=exposure_tv(PX[i],M[i],*c.p)
    elif c.kind=='mom':
        E=np.zeros_like(M)
        for i in range(b): E[i]=exposure_mom(PX[i],*c.p)
    else:
        # ETF finalist: synthesize vehicle then size 0..1 to it.
        veh,w,bull,bear,bandp=c.p
        AR=synth_letf(M,R,2,.0095,.0050) if veh=='qld' else synth_letf(M,R,3,.0088,.0065)
        A=np.zeros_like(M)
        for i in range(b):
            ma=roll_mean(PX[i],w); v=roll_vol(AR[i],20); hold=0.
            for t in range(1,n):
                if np.isfinite(ma[t-1]) and np.isfinite(v[t-1]) and v[t-1]>0:
                    tv=bull if PX[i,t-1]>ma[t-1] else bear; want=min(1.,tv/v[t-1])
                    if abs(want-hold)>=bandp: hold=want
                A[i,t]=hold
        turn=np.abs(np.diff(A,axis=1,prepend=np.zeros((b,1))))
        return A*AR-ETF_TC*turn
    turn=np.abs(np.diff(E,axis=1,prepend=np.zeros((b,1))))
    return R+E*(M-R)-TC_FUT*turn

def choose_finalists(screen,cs):
    by={c.name:c for c in cs}
    pre=screen[(screen.train_dd>-0.90)&np.isfinite(screen.train_cagr)].copy()
    pre['score']=pre.train_cagr + .5*pre.p10_5y_cagr + .15*pre.worst_5y_cagr
    chosen=list(pre.sort_values('score',ascending=False).strategy.head(8))
    frozen=['ndx_1x','fut_fixed_2x','fut_sma200_2x_b0_buf0','fut_sma200_3x_b0_buf0',
            'fut_tv_w200_u35_d12_b30','fut_tv_w200_u35_d0_b30','fut_vol25','fut_mom252_2x',
            'qld_s9_w200_u35_d12_b10','tqqq_s9_w200_u35_d12_b10']
    names=[]
    for x in chosen+frozen:
        if x in by and x not in names:names.append(x)
    return [by[x] for x in names],chosen

def monte_carlo(df,finalists):
    ndx=df.ndx.to_numpy(); rf=df.rf.to_numpy(); dates=df.index; maxd=max(H)*TD; warm=300; total=maxd+warm; hds=np.array(H)*TD
    rng=np.random.default_rng(20260822); store=[]; batch=64
    for sn,sc in SCEN.items():
        acc={c.name:[[],[],[]] for c in finalists}; acc['benchmark_1x']=[[],[],[]]; done=0
        while done<N_PATHS:
            b=min(batch,N_PATHS-done); M,R=bootstrap(ndx,rf,dates,b,total,rng,sc)
            B=R+(M-R); tb,db,rb=pathstats(B[:,warm:],hds); acc['benchmark_1x'][0].append(tb);acc['benchmark_1x'][1].append(db);acc['benchmark_1x'][2].append(rb)
            for c in finalists:
                X=eval_path_candidate(c,M,R)[:,warm:]; t,d,rr=pathstats(X,hds);acc[c.name][0].append(t);acc[c.name][1].append(d);acc[c.name][2].append(rr)
            done+=b
            if done%2500==0: print('mc',sn,done,flush=True)
        BT=np.concatenate(acc['benchmark_1x'][0])
        for name,(aa,dd,rr) in acc.items():
            T=np.concatenate(aa);D=np.concatenate(dd);REC=np.concatenate(rr)
            for j,h in enumerate(H):
                cagr=np.maximum(T[:,j],1e-300)**(1/h)-1
                beat=np.nan if name=='benchmark_1x' else np.mean(T[:,j]>BT[:,j])
                store.append({'scenario':sn,'strategy':name,'horizon_years':h,'median_cagr':np.median(cagr),'p10_cagr':np.quantile(cagr,.1),
                              'p01_cagr':np.quantile(cagr,.01),'prob_beat_1x':beat,'prob_dd_gt_90':np.mean(D[:,j]<=-.9),
                              'median_drawdown':np.median(D[:,j]),'p90_recovery_years':np.quantile(REC[:,j],.9)})
    d=pd.DataFrame(store)
    agg=d.groupby(['strategy','horizon_years']).agg(worst_median_cagr=('median_cagr','min'),worst_p10_cagr=('p10_cagr','min'),worst_p01_cagr=('p01_cagr','min'),
        min_prob_beat_1x=('prob_beat_1x','min'),max_prob_dd_gt_90=('prob_dd_gt_90','max'),worst_median_drawdown=('median_drawdown','min'),worst_p90_recovery=('p90_recovery_years','max')).reset_index()
    return d,agg

def annual_after_tax_futures(r,dates,ordinary,lt):
    rate=.4*ordinary+.6*lt; s=pd.Series(r,index=dates); wealth=1.; carry=0.
    for _,g in s.groupby(s.index.year):
        gr=float(np.prod(1+g)-1); taxable=max(gr-carry,0.); carry=max(carry-gr,0.) if gr<carry else 0.
        wealth*=max(1+gr-rate*taxable,1e-12)
    yrs=len(s)/TD; return wealth**(1/yrs)-1

def taxable_history(df,screen,cs):
    by={c.name:c for c in cs}; ndx=df.ndx.to_numpy();rf=df.rf.to_numpy();px=np.cumprod(np.maximum(1+ndx,1e-12));qld=synth_letf(ndx,rf,2,.0095,.005);tq=synth_letf(ndx,rf,3,.0088,.0065)
    # Taxable focus: direct futures candidates + passive QQQ-like benchmark. ETF active taxes already characterized elsewhere; here use turnover penalty sensitivity separately.
    names=['ndx_1x','fut_sma200_2x_b0_buf0','fut_tv_w200_u35_d0_b30','fut_tv_w200_u35_d12_b30','fut_mom252_2x']
    rows=[]
    for n in names:
        c=by[n];r=evaluate(c,ndx,rf,px,qld,tq)
        for o,l,label in [(.24,.15,'24_15'),(.32,.15,'32_15'),(.37,.20,'37_20')]:
            rows.append({'strategy':n,'rate_case':label,'pretax_cagr':metric(r)['cagr'],'after_tax_cagr_section1256_model':annual_after_tax_futures(r,df.index,o,l),'blended_rate':.4*o+.6*l})
    # Passive QQQ-like: tax deferred until liquidation; report pre-liquidation CAGR and simple terminal-liquidation sensitivity.
    qr=df.qqq_like.to_numpy(); yrs=len(qr)/TD; terminal=np.prod(1+qr)
    for o,l,label in [(.24,.15,'24_15'),(.32,.15,'32_15'),(.37,.20,'37_20')]:
        aft=1+(terminal-1)*(1-l); rows.append({'strategy':'QQQ_like_buyhold','rate_case':label,'pretax_cagr':terminal**(1/yrs)-1,'after_tax_cagr_section1256_model':aft**(1/yrs)-1,'blended_rate':np.nan})
    return pd.DataFrame(rows)

def dividend_sensitivity():
    rows=[]
    for y in (0.0,.007,.010):
        d=load_data(y); ndx=d.ndx.to_numpy();rf=d.rf.to_numpy();px=np.cumprod(np.maximum(1+ndx,1e-12));
        for name,e in [('2x_sma200',exposure_sma(px,200,2,0,0)),('3x_sma200',exposure_sma(px,200,3,0,0)),('s9_35_12',exposure_tv(px,ndx,200,.35,.12,.30))]:
            rows.append({'pre1999_dividend_assumption':y,'strategy':name,**metric(fut_ret(ndx,rf,e))})
    return pd.DataFrame(rows)

def cross_vehicle():
    files={'QQQ':'synthetic-qqq.tsv','QLD':'synthetic-qld.tsv','TQQQ':'synthetic-tqqq.tsv','SPY':'spy.tsv','SSO':'synthetic-sso.tsv','SPXL':'synthetic-spxl.tsv'}
    s={k:read_series(v) for k,v in files.items()}; start=pd.Timestamp('1988-01-04'); end=min(x.index.max() for x in s.values()); idx=s['QQQ'].index[(s['QQQ'].index>=start)&(s['QQQ'].index<=end)]
    rows=[]
    for k,x in s.items():
        r=x.reindex(idx).ffill().pct_change().fillna(0); rows.append({'vehicle':k,**metric(r.to_numpy())})
    return pd.DataFrame(rows)

def main():
    df=load_data(.007); screen,folds,cs=historical_screen(df); finalists,train_selected=choose_finalists(screen,cs)
    print('candidates',len(cs),'finalists',[c.name for c in finalists],flush=True)
    mc,agg=monte_carlo(df,finalists);tax=taxable_history(df,screen,cs);div=dividend_sensitivity();veh=cross_vehicle()
    screen.to_csv(OUT/'historical_screen.csv',index=False);folds.to_csv(OUT/'five_year_blocks.csv',index=False);mc.to_csv(OUT/'mc_scenarios.csv',index=False);agg.to_csv(OUT/'robust_ranking.csv',index=False);tax.to_csv(OUT/'taxable_futures.csv',index=False);div.to_csv(OUT/'pre1999_dividend_sensitivity.csv',index=False);veh.to_csv(OUT/'cross_vehicle_1988plus.csv',index=False)
    manifest={'paths_per_scenario':N_PATHS,'scenarios':SCEN,'horizons':H,'candidate_count':len(cs),'finalists':[c.name for c in finalists],'pre2010_selected':train_selected,
              'data_start':str(df.index.min().date()),'data_end':str(df.index.max().date()),'data_source':BASE,'ndx_method':'synthetic QQQ reverse 20bp expense; pre-1999 add 0.7% dividend base sensitivity; 0 and 1.0% also scored',
              'tax_note':'Federal-only sensitivity; Section 1256 modeled 60/40 annual mark-to-market with loss carryforward; state tax/NIIT omitted; QQQ buyhold taxed only at terminal liquidation for comparison.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nPRE2010 TOP\n',screen.sort_values('train_cagr',ascending=False).head(20).to_string(index=False))
    print('\nROBUST TOP BY HORIZON\n',agg.sort_values(['horizon_years','worst_p10_cagr'],ascending=[True,False]).groupby('horizon_years').head(8).to_string(index=False))
    print('\nTAXABLE\n',tax.to_string(index=False)); print('\nDIV SENS\n',div.to_string(index=False)); print('\nVEHICLES\n',veh.to_string(index=False))

if __name__=='__main__': main()
