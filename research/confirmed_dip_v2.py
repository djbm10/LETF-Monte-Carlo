from __future__ import annotations
import math
import numpy as np
import pandas as pd

BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data/'
TD=252; TC=0.0003

def read(name):
    d=pd.read_csv(BASE+name,sep='\t')
    d['Date']=pd.to_datetime(d.Date).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(d.Close,errors='coerce').to_numpy(),index=d.Date).dropna().sort_index()

def metrics(r):
    e=(1+r).cumprod(); yrs=len(r)/TD; dd=e/e.cummax()-1
    roll=(1+r).rolling(5*TD).apply(np.prod,raw=True)**(1/5)-1
    return dict(cagr=e.iloc[-1]**(1/yrs)-1,max_dd=dd.min(),min_5y_cagr=roll.min(),vol=r.std()*math.sqrt(TD),terminal=e.iloc[-1])

def portfolio(rtq, targets, band):
    wealth=1.; wtq=0.; out=[]
    for i in range(len(targets)):
        dtq=float(targets.iloc[i])
        if abs(dtq-wtq)>=band:
            wealth*=max(1-TC*abs(dtq-wtq),0); wtq=dtq
        nt=wealth*wtq*(1+rtq.iloc[i]); nc=wealth*(1-wtq)
        nw=nt+nc; out.append(nw/wealth-1); wealth=nw
        wtq=nt/max(wealth,1e-15)
    return pd.Series(out,index=targets.index)

def confirmed_overlay(idx, base, bull_s, q, ddq_s, thr, boost, confirm, hold, band):
    ma10=q.rolling(10).mean().shift(1); ma20=q.rolling(20).mean().shift(1)
    low20=q.rolling(20).min().shift(1)
    mom5=(q.shift(1)/q.shift(6)-1)
    cross10=(q.shift(1)>ma10)&(q.shift(2)<=q.rolling(10).mean().shift(2))
    cross20=(q.shift(1)>ma20)&(q.shift(2)<=q.rolling(20).mean().shift(2))
    rebound3=(q.shift(1)/low20-1)>=.03
    rebound5=(q.shift(1)/low20-1)>=.05
    if confirm=='cross10': conf=cross10
    elif confirm=='cross20': conf=cross20
    elif confirm=='rebound3': conf=rebound3 & (mom5>0)
    elif confirm=='rebound5': conf=rebound5 & (mom5>0)
    else: conf=(mom5>0)&(q.shift(1)>ma10)

    out=base.copy(); armed=False; active=0
    for i in range(len(idx)):
        if pd.notna(ddq_s.iloc[i]) and ddq_s.iloc[i] <= -thr:
            armed=True
        if active>0:
            active-=1
            # exit early if drawdown recovered by half, or trend fails
            if (pd.notna(ddq_s.iloc[i]) and ddq_s.iloc[i] > -thr/2) or (not bool(bull_s.iloc[i])):
                active=0
        if armed and bool(bull_s.iloc[i]) and bool(conf.iloc[i]):
            active=hold; armed=False
        if active>0:
            out.iloc[i]=min(1.0, float(base.iloc[i])+boost)
    return out

def main():
    q=read('synthetic-qqq.tsv'); tq=read('synthetic-tqqq.tsv'); spy=read('spy.tsv')
    idx=q.index.intersection(tq.index).intersection(spy.index)
    idx=idx[idx>=pd.Timestamp('1985-01-31')]
    q=q.reindex(idx); tq=tq.reindex(idx); spy=spy.reindex(idx)
    rt=tq.pct_change().fillna(0)
    spma=spy.rolling(200).mean(); bull=(spy>spma).shift(1).fillna(False)
    vt=(rt.rolling(20).std()*math.sqrt(TD)).shift(1)
    base=(pd.Series(np.where(bull,.35,.12),index=idx)/vt.replace(0,np.nan)).clip(0,1).fillna(0)
    ddq=(q/q.rolling(252,min_periods=63).max()-1).shift(1)

    rows=[]
    # baseline
    for label, target, band in [('S9_10',base,.10)]:
        r=portfolio(rt,target,band)
        for period,mask in [('full',idx>=idx.min()),('pre2010',idx<pd.Timestamp('2010-01-01')),('holdout2010',idx>=pd.Timestamp('2010-01-01'))]:
            rows.append({'strategy':label,'period':period,'thr':np.nan,'boost':0,'confirm':'base','hold':0,'band':band,**metrics(r.loc[mask])})

    thrs=[.05,.075,.10,.125,.15,.20,.25]
    boosts=[.025,.05,.075,.10,.125,.15,.20]
    confirms=['cross10','cross20','rebound3','rebound5','mom5ma10']
    holds=[10,20,40,60]
    bands=[.05,.10]
    for thr in thrs:
        for boost in boosts:
            for confirm in confirms:
                for hold in holds:
                    for band in bands:
                        targ=confirmed_overlay(idx,base,bull,q,ddq,thr,boost,confirm,hold,band)
                        name=f'CONF_t{int(thr*1000):03d}_b{int(boost*1000):03d}_{confirm}_h{hold}_rb{int(band*100):02d}'
                        r=portfolio(rt,targ,band)
                        for period,mask in [('full',idx>=idx.min()),('pre2010',idx<pd.Timestamp('2010-01-01')),('holdout2010',idx>=pd.Timestamp('2010-01-01'))]:
                            rows.append({'strategy':name,'period':period,'thr':thr,'boost':boost,'confirm':confirm,'hold':hold,'band':band,**metrics(r.loc[mask])})
    z=pd.DataFrame(rows)
    z.to_csv('results/confirmed_dip_v2_all.csv',index=False)

    pre=z[z.period=='pre2010'].copy()
    # Aggressive-but-not-reckless frozen selection rule: require max DD no worse than -75% and worst 5y > -20%, then rank CAGR.
    eligible=pre[(pre.max_dd>=-.75)&(pre.min_5y_cagr>=-.20)].sort_values('cagr',ascending=False)
    shortlist=eligible.head(25)
    final=z[z.strategy.isin(shortlist.strategy.tolist()+['S9_10'])].copy()
    final.to_csv('results/confirmed_dip_v2_shortlist.csv',index=False)
    print('PRE-2010 ELIGIBLE WINNERS')
    print(shortlist[['strategy','cagr','max_dd','min_5y_cagr','vol']].head(15).to_string(index=False))
    print('\nUNTOUCHED 2010+ FOR FROZEN TOP 15')
    h=final[(final.period=='holdout2010') & final.strategy.isin(shortlist.head(15).strategy)]
    print(h[['strategy','cagr','max_dd','min_5y_cagr','vol']].sort_values('cagr',ascending=False).to_string(index=False))
    print('\nBASELINE')
    print(final[final.strategy=='S9_10'][['period','cagr','max_dd','min_5y_cagr','vol']].to_string(index=False))

if __name__=='__main__': main()
