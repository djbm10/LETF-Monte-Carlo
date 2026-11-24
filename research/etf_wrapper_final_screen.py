from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np, pandas as pd
import research.nasdaq1985_final_tournament as n

OUT=Path('results/etf_wrapper_final'); OUT.mkdir(parents=True,exist_ok=True); TD=252; TC=.0003

def met(r):
    r=pd.Series(r).dropna(); eq=(1+r).cumprod(); y=len(r)/TD; dd=eq/eq.cummax()-1
    return {'cagr':eq.iloc[-1]**(1/y)-1,'max_dd':dd.min(),'vol':r.std()*math.sqrt(TD),'terminal':eq.iloc[-1]}

def target_s9(under,asset,w,bull,bear,band):
    ma=under.rolling(w).mean(); v=asset.pct_change().rolling(20).std()*math.sqrt(TD); a=pd.Series(0.,index=under.index); hold=0.
    for i in range(1,len(a)):
        if pd.isna(ma.iloc[i-1]) or pd.isna(v.iloc[i-1]) or v.iloc[i-1]<=0: a.iloc[i]=hold; continue
        tv=bull if under.iloc[i-1]>ma.iloc[i-1] else bear; want=min(1.,max(0.,tv/v.iloc[i-1]))
        if abs(want-hold)>=band: hold=want
        a.iloc[i]=hold
    return a

def target_binary(under,w,buf):
    ma=under.rolling(w).mean(); a=pd.Series(0.,index=under.index);hold=0.
    for i in range(1,len(a)):
        if pd.isna(ma.iloc[i-1]): a.iloc[i]=hold;continue
        ratio=under.iloc[i-1]/ma.iloc[i-1]-1
        if hold<.5:
            if ratio>buf:hold=1.
        else:
            if ratio<-buf:hold=0.
        a.iloc[i]=hold
    return a

def strat(asset,a,rf=None,cash_mult=0.):
    r=asset.pct_change().fillna(0); turn=a.diff().abs().fillna(a.abs()); cash=0 if rf is None else (1-a)*cash_mult*rf
    return a*r+cash-TC*turn

def main():
    files={'QQQ':'synthetic-qqq.tsv','QLD':'synthetic-qld.tsv','TQQQ':'synthetic-tqqq.tsv','SPY':'spy.tsv','SSO':'synthetic-sso.tsv','SPXL':'synthetic-spxl.tsv'}
    s={k:n.read_series(v) for k,v in files.items()}; rf=n.read_series('short-rates.tsv','Rate')/100/TD
    start=pd.Timestamp('1988-01-04'); end=min(x.index.max() for x in s.values()); idx=s['QQQ'].index[(s['QQQ'].index>=start)&(s['QQQ'].index<=end)]
    s={k:v.reindex(idx).ffill() for k,v in s.items()}; rf=rf.reindex(idx).ffill().bfill()
    rows=[]
    for under,vehicles in [('QQQ',['QLD','TQQQ']),('SPY',['SSO','SPXL'])]:
      for veh in vehicles:
        # buyhold
        r=s[veh].pct_change().fillna(0); full=met(r); tr=met(r[idx<pd.Timestamp('2010-02-11')]); ho=met(r[idx>=pd.Timestamp('2010-02-11')])
        rows.append({'under':under,'vehicle':veh,'strategy':f'{veh}_BH',**full,'train_cagr':tr['cagr'],'train_dd':tr['max_dd'],'holdout_cagr':ho['cagr'],'holdout_dd':ho['max_dd'],'trades_per_year':0})
        for w in (100,125,150,175,200,225,250,300):
          for buf in (0.,.01,.02,.03):
            a=target_binary(s[under],w,buf); rr=strat(s[veh],a,rf,0); full=met(rr); tr=met(rr[idx<pd.Timestamp('2010-02-11')]);ho=met(rr[idx>=pd.Timestamp('2010-02-11')]); changes=(a.diff().abs()>0).sum()/(len(a)/TD)
            rows.append({'under':under,'vehicle':veh,'strategy':f'{veh}_bin_w{w}_buf{int(buf*100)}',**full,'train_cagr':tr['cagr'],'train_dd':tr['max_dd'],'holdout_cagr':ho['cagr'],'holdout_dd':ho['max_dd'],'trades_per_year':changes})
        for w in (150,175,200,225,250):
          for bull in (.25,.30,.35,.40):
           for bear in (0.,.05,.10,.12,.15):
            for band in (.05,.10,.15):
             a=target_s9(s[under],s[veh],w,bull,bear,band); rr=strat(s[veh],a,rf,0); full=met(rr);tr=met(rr[idx<pd.Timestamp('2010-02-11')]);ho=met(rr[idx>=pd.Timestamp('2010-02-11')]);changes=(a.diff().abs()>1e-12).sum()/(len(a)/TD)
             rows.append({'under':under,'vehicle':veh,'strategy':f'{veh}_s9_w{w}_u{int(bull*100)}_d{int(bear*100)}_b{int(band*100)}',**full,'train_cagr':tr['cagr'],'train_dd':tr['max_dd'],'holdout_cagr':ho['cagr'],'holdout_dd':ho['max_dd'],'trades_per_year':changes})
    d=pd.DataFrame(rows); d.to_csv(OUT/'screen.csv',index=False)
    # Strict pre-2010 selection per wrapper, require <85% DD, mild DD penalty. Holdout not used.
    picks=[]
    for veh,g in d.groupby('vehicle'):
        q=g[(g.train_dd>-0.85)&g.train_cagr.notna()].copy();q['score']=q.train_cagr+.10*q.train_dd
        picks.append(q.sort_values('score',ascending=False).head(5))
    p=pd.concat(picks).sort_values(['vehicle','score'],ascending=[True,False]);p.to_csv(OUT/'pre2010_picks.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'period':[str(idx.min().date()),str(idx.max().date())],'selection':'pre-2010 only; 2010+ untouched','transaction_cost_per_traded_dollar':TC,'cash_yield':'0% conservative','pre1999_nasdaq_caveat':'QQQ-family synthetic source is price-only NDX before 1999; missing dividends biases Nasdaq family low.'},indent=2))
    print('TOP STRICT PICKS\n',p[['vehicle','strategy','train_cagr','train_dd','holdout_cagr','holdout_dd','cagr','max_dd','trades_per_year','score']].to_string(index=False))
    print('\nBEST HOLDOUT AMONG PRESELECTED ONLY\n',p.sort_values('holdout_cagr',ascending=False)[['vehicle','strategy','train_cagr','holdout_cagr','holdout_dd','trades_per_year']].head(20).to_string(index=False))
if __name__=='__main__':main()
