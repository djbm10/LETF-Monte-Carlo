from __future__ import annotations
import argparse, hashlib, io, json, math, urllib.request, zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

TD=252; FF_URL='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
H=(5,10,20,30,40,50); LEVS=(1.,1.5,2.,2.5,3.); SMAS=(175,200,225); BULL=(.25,.30,.35,.40); BEAR=(0,.05,.10,.15)
FIXED=(200,.35,.12); PATHS=10_000; SEED=20260821; BLOCK=63; WARM=252; EXP1=.0003; EXPL=.0091; SPREAD=.005; TC=.0003
TQQQ_START=pd.Timestamp('1985-01-31')

@dataclass(frozen=True)
class P:
    sma:int; bull:float; bear:float
    @property
    def key(self): return f'sma{self.sma}_bull{self.bull:.2f}_bear{self.bear:.2f}'

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
    if len(d)<20_000: raise RuntimeError(f'Fama-French history too short: {len(d)}')
    return d,{'source':FF_URL,'sha256':sha,'start':str(d.index.min().date()),'end':str(d.index.max().date()),'rows':len(d)}

def levret(m,rf,L,expense=None,spread=SPREAD):
    if expense is None: expense=EXP1 if L==1 else EXPL
    x=L*np.asarray(m,float)-max(L-1,0)*(np.asarray(rf,float)+spread/TD)-expense/TD
    return np.maximum(x,-1.)

def universe(ff):
    d=ff.copy()
    for L in LEVS:d[f'L{L:g}']=levret(d.mkt,d.rf,L)
    d['px']=np.cumprod(1+d.L1.to_numpy()); return d

def rollmean(x,w):return pd.Series(x).rolling(w,min_periods=w).mean().to_numpy()
def rollvol(x,w=20):return pd.Series(x).rolling(w,min_periods=w).std().to_numpy()*math.sqrt(TD)

def s9(px,r3,rf,p):
    ma=rollmean(px,p.sma); vol=rollvol(r3); a=np.zeros(len(px))
    for i in range(1,len(px)):
        if np.isfinite(ma[i-1]) and np.isfinite(vol[i-1]) and vol[i-1]>0:
            a[i]=np.clip((p.bull if px[i-1]>ma[i-1] else p.bear)/vol[i-1],0,1)
    return a*r3+(1-a)*rf-TC*np.abs(np.diff(a,prepend=0))

def trend(px,r3,rf,w=200):
    ma=rollmean(px,w); a=np.zeros(len(px)); valid=np.isfinite(ma[:-1]); a[1:]=np.where(valid,px[:-1]>ma[:-1],0)
    return a*r3+(1-a)*rf-.0005*np.abs(np.diff(a,prepend=0))

def metrics(r):
    r=np.asarray(r,float); r=r[np.isfinite(r)]; eq=np.cumprod(np.maximum(1+r,0)); yrs=len(r)/TD
    peak=np.maximum.accumulate(eq); dd=eq/peak-1; cur=mx=0; hi=-np.inf
    for v in eq:
        if v>=hi: hi=v;cur=0
        else: cur+=1;mx=max(mx,cur)
    return {'cagr':eq[-1]**(1/yrs)-1 if eq[-1]>0 else -1.,'max_drawdown':dd.min(),'max_recovery_years':mx/TD,'terminal':eq[-1]}

def grid(): return [P(s,b,d) for s in SMAS for b in BULL for d in BEAR]

def walkforward(u):
    pars=grid(); fixed=P(*FIXED); allret={p:pd.Series(s9(u.px.values,u.L3.values,u.rf.values,p),index=u.index) for p in pars}
    allret[fixed]=pd.Series(s9(u.px.values,u.L3.values,u.rf.values,fixed),index=u.index)
    start=u.index.min()+pd.DateOffset(years=20); rows=[]; parts=[]
    while start<=u.index.max():
        tr0=start-pd.DateOffset(years=20); tr1=start-pd.Timedelta(days=1); te1=min(start+pd.DateOffset(years=5)-pd.Timedelta(days=1),u.index.max())
        tr=(u.index>=tr0)&(u.index<=tr1); te=(u.index>=start)&(u.index<=te1)
        if tr.sum()>=10*TD and te.sum()>=TD:
            chosen=max(pars,key=lambda p:metrics(allret[p].loc[tr].values)['cagr']); parts.append(allret[chosen].loc[te])
            rows.append({'train_start':str(tr0.date()),'train_end':str(tr1.date()),'test_start':str(start.date()),'test_end':str(te1.date()),'chosen':chosen.key,'sma':chosen.sma,'bull':chosen.bull,'bear':chosen.bear,'test_cagr':metrics(allret[chosen].loc[te].values)['cagr'],'fixed_cagr':metrics(allret[fixed].loc[te].values)['cagr'],'one_x_cagr':metrics(u.L1.loc[te].values)['cagr']})
        start+=pd.DateOffset(years=5)
    f=pd.DataFrame(rows); modal_key=Counter(f.chosen).most_common(1)[0][0]; modal=next(p for p in pars if p.key==modal_key)
    oos=pd.DataFrame({'walk_forward':pd.concat(parts).sort_index()}); oos['fixed']=allret[fixed].reindex(oos.index);oos['one_x']=u.L1.reindex(oos.index)
    return f,oos,modal

def rm2(x,w):
    out=np.full_like(x,np.nan,dtype=float);cs=np.cumsum(x,axis=1);s=cs[:,w-1:].copy();s[:,1:]-=cs[:,:-w];out[:,w-1:]=s/w;return out

def rs2(x,w=20):
    out=np.full_like(x,np.nan,dtype=float);c=np.cumsum(x,1);c2=np.cumsum(x*x,1);s=c[:,w-1:].copy();s2=c2[:,w-1:].copy();s[:,1:]-=c[:,:-w];s2[:,1:]-=c2[:,:-w];mu=s/w;out[:,w-1:]=np.sqrt(np.maximum((s2-w*mu*mu)/(w-1),0));return out

def s9m(px,r3,rf,p):
    ma=rm2(px,p.sma);vol=rs2(r3)*math.sqrt(TD);a=np.zeros_like(px);pv=vol[:,:-1];valid=np.isfinite(ma[:,:-1])&np.isfinite(pv)&(pv>0)
    target=np.where(px[:,:-1]>ma[:,:-1],p.bull,p.bear);a[:,1:]=np.where(valid,np.clip(target/np.maximum(pv,1e-12),0,1),0)
    return a*r3+(1-a)*rf-TC*np.abs(np.diff(a,axis=1,prepend=np.zeros((len(a),1))))

def trendm(px,r3,rf):
    ma=rm2(px,200);a=np.zeros_like(px);v=np.isfinite(ma[:,:-1]);a[:,1:]=np.where(v,px[:,:-1]>ma[:,:-1],0);return a*r3+(1-a)*rf-.0005*np.abs(np.diff(a,axis=1,prepend=np.zeros((len(a),1))))

def pathstats(r,hds):
    b,n=r.shape;term=np.empty((b,len(hds)));mdd=np.empty_like(term);rec=np.empty_like(term)
    wealth=np.ones(b);peak=np.ones(b);ddmin=np.zeros(b);cur=np.zeros(b,int);mx=np.zeros(b,int);j=0
    for t in range(n):
        wealth*=np.maximum(1+r[:,t],0);new=wealth>=peak;peak=np.maximum(peak,wealth);ddmin=np.minimum(ddmin,wealth/peak-1);cur=np.where(new,0,cur+1);mx=np.maximum(mx,cur)
        if j<len(hds) and t+1==hds[j]:term[:,j]=wealth;mdd[:,j]=ddmin;rec[:,j]=mx/TD;j+=1
    return term,mdd,rec

def boot(m,rf,b,n,rng):
    nb=math.ceil(n/BLOCK);st=rng.integers(0,len(m),(b,nb));idx=(st[:,:,None]+np.arange(BLOCK))%len(m);idx=idx.reshape(b,-1)[:,:n];return m[idx],rf[idx]

def mc(u,modal,npaths,seed):
    rng=np.random.default_rng(seed);mx=max(H)*TD;tot=mx+WARM;hds=np.array(H)*TD;fixed=P(*FIXED);names=['1x','1.5x','2x','2.5x','3x','3x_200SMA_cash','S9_fixed',f'S9_modal_{modal.key}']
    S={n:[[],[],[]] for n in names};beats={n:[[] for _ in H] for n in names if n!='1x'};done=0;m=u.mkt.values;rfh=u.rf.values
    while done<npaths:
        b=min(128,npaths-done);bm,br=boot(m,rfh,b,tot,rng);R={f'{L:g}x':levret(bm,br,L) for L in LEVS};px=np.cumprod(np.maximum(1+R['1x'],1e-12),1)
        R['3x_200SMA_cash']=trendm(px,R['3x'],br);R['S9_fixed']=s9m(px,R['3x'],br,fixed);R[f'S9_modal_{modal.key}']=s9m(px,R['3x'],br,modal)
        batch={}
        for n in names:
            t,d,rc=pathstats(R[n][:,WARM:WARM+mx],hds);batch[n]=t;S[n][0].append(t);S[n][1].append(d);S[n][2].append(rc)
        for n in beats:
            for j in range(len(H)):beats[n][j].append(batch[n][:,j]>batch['1x'][:,j])
        done+=b;print(f'MC {done}/{npaths}',flush=True)
    rows=[]
    for n in names:
        T=np.concatenate(S[n][0]);D=np.concatenate(S[n][1]);RC=np.concatenate(S[n][2])
        for j,y in enumerate(H):
            c=np.where(T[:,j]>0,T[:,j]**(1/y)-1,-1);rows.append({'strategy':n,'horizon_years':y,'n_paths':npaths,'median_cagr':np.median(c),'p10_cagr':np.quantile(c,.1),'prob_beat_1x':0 if n=='1x' else np.mean(np.concatenate(beats[n][j])),'prob_dd_gt_90':np.mean(D[:,j]<=-.9),'median_max_drawdown':np.median(D[:,j]),'p10_max_drawdown':np.quantile(D[:,j],.1),'median_max_recovery_years':np.median(RC[:,j]),'p90_max_recovery_years':np.quantile(RC[:,j],.9)})
    return pd.DataFrame(rows)

def tqqq_check(ff):
    try:
        import yfinance as yf
        x=yf.download('^NDX',start='1985-01-31',progress=False,auto_adjust=True);c=x['Close'];c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c;c.index=pd.to_datetime(c.index).tz_localize(None);r=c.pct_change().dropna();a=pd.DataFrame({'ndx':r}).join(ff.rf,how='inner').dropna();a=a.loc[a.index>=TQQQ_START]
        return {'scored':False,'status':'ok','start':str(a.index.min().date()),'end':str(a.index.max().date()),'rows':len(a),'rule':'never use pre-1985 TQQQ proxy'}
    except Exception as e:return {'scored':False,'status':'unavailable','minimum_date':'1985-01-31','error':repr(e)}

def selftest():
    rng=np.random.default_rng(1);idx=pd.bdate_range('1990-01-01',periods=8000);u=universe(pd.DataFrame({'mkt':rng.normal(.0003,.01,len(idx)),'rf':np.full(len(idx),.02/TD)},index=idx));f,o,p=walkforward(u);assert len(f)>0 and len(o)>0;z=mc(u,p,24,7);assert (z.n_paths==24).all();print('SELF_TEST_OK')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paths',type=int,default=PATHS);ap.add_argument('--seed',type=int,default=SEED);ap.add_argument('--output',type=Path,default=Path('results/clean_1926_walkforward'));ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:return selftest()
    if a.paths<10_000:raise SystemExit('serious runs require >=10000 paths')
    a.output.mkdir(parents=True,exist_ok=True);ff,meta=load_ff();u=universe(ff);folds,oos,modal=walkforward(u);fixed=P(*FIXED)
    hist=[]
    for L in LEVS:hist.append({'strategy':f'{L:g}x_buy_hold',**metrics(u[f'L{L:g}'].values)})
    hist += [{'strategy':'3x_200SMA_cash',**metrics(trend(u.px.values,u.L3.values,u.rf.values))},{'strategy':'S9_fixed_200_35_12',**metrics(s9(u.px.values,u.L3.values,u.rf.values,fixed))},{'strategy':f'S9_modal_{modal.key}',**metrics(s9(u.px.values,u.L3.values,u.rf.values,modal))},{'strategy':'OOS_walk_forward',**metrics(oos.walk_forward.values)},{'strategy':'OOS_fixed',**metrics(oos.fixed.values)},{'strategy':'OOS_1x',**metrics(oos.one_x.values)}]
    hist=pd.DataFrame(hist);M=mc(u,modal,a.paths,a.seed);folds.to_csv(a.output/'walk_forward_folds.csv',index=False);oos.to_csv(a.output/'walk_forward_oos_returns.csv');hist.to_csv(a.output/'historical_summary.csv',index=False);M.to_csv(a.output/'monte_carlo_summary.csv',index=False)
    manifest={'data':meta,'rules':{'pre_1950_qqq_tlt_scored':False,'pre_1985_tqqq_scored':False,'horizons':H,'leverage_universe':LEVS,'mc_paths':a.paths,'seed':a.seed,'block_days':BLOCK,'walk_forward':'20y train / 5y test'},'grid':{'sma':SMAS,'bull':BULL,'bear':BEAR},'fixed':asdict(fixed),'modal':asdict(modal),'selection_counts':dict(Counter(folds.chosen)),'tqqq_1985plus':tqqq_check(ff)}
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print(folds.chosen.value_counts());print(hist.to_string(index=False));print(M.to_string(index=False))
if __name__=='__main__':main()
