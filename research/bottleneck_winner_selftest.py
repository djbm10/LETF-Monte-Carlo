from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bottleneck_winner_tournament as b

def make():
    dates=pd.date_range("1989-01-31","2020-12-31",freq="ME")
    cr=[]
    for perm,mu in [(10001,.012),(10002,.008),(10003,.006)]:
        for i,d in enumerate(dates):
            ret=mu + .02*np.sin(i/11+perm)
            cr.append((perm,d,ret,100+perm%10+i*.1))
    crsp=pd.DataFrame(cr,columns=["permno","date","ret","me"])
    ccm=pd.DataFrame({
      "permno":[10001,10002,10003],"gvkey":["000001","000002","000003"],
      "linkdt":[pd.Timestamp("1980-01-01")]*3,"linkenddt":[pd.Timestamp("2099-12-31")]*3})
    fund=[]
    for gv,mult in [("000001",1.15),("000002",1.05),("000003",.98)]:
      for d in pd.date_range("1988-03-31","2020-12-31",freq="QE"):
        # information becomes available 45 days after quarter end
        av=d+pd.Timedelta(days=45)
        yr=(d.year-1988)*4+d.quarter
        sale=100*(mult**(yr/4))
        fund.append((gv,av,sale,sale*.55,sale*.20,sale*.06,sale*.12,200+yr,70+yr*.5,10,30+yr*.1))
    fund=pd.DataFrame(fund,columns=["gvkey","available_date","sale","cogs","oibdp","capx","oancf","at","lt","csho","prcc"])
    t=[]
    for gv,hhi,sim in [("000001",.30,.08),("000002",.20,.12),("000003",.10,.18)]:
      for y in range(1988,2021):t.append((gv,y,hhi,sim))
    tnic=pd.DataFrame(t,columns=["gvkey","year","tnic3hhi","tnic3tsimm"])
    return crsp,ccm,fund,tnic

def main():
    crsp,ccm,fund,tnic=make()
    p=b.score_models(b.build_panel(crsp,ccm,fund,tnic,None))
    assert len(p)>100, len(p)
    # PIT test: a quarter ending 1999-12-31 and available 2000-02-14 cannot appear in Jan 2000.
    g=p[(p.gvkey=="000001")&(p.date.between("2000-01-01","2000-03-31"))].sort_values("date")
    jan=g[g.date.dt.month==1].iloc[0]; feb=g[g.date.dt.month==2].iloc[0]
    assert pd.Timestamp(jan.available_date)<=jan.date
    assert pd.Timestamp(feb.available_date)<=feb.date
    # TNIC year Y is deliberately unavailable until June Y+1.
    pre=p[(p.gvkey=="000001")&(p.date==pd.Timestamp("2000-05-31"))].iloc[0]
    post=p[(p.gvkey=="000001")&(p.date==pd.Timestamp("2000-06-30"))].iloc[0]
    assert int(pre.year)<=1998, pre.year
    assert int(post.year)<=1999, post.year
    # Full ranking/backtest pipeline executes.
    scored=p[p.me>=p.groupby("date").me.transform(lambda s:s.quantile(.0))]
    bt=b.backtest_one(scored,"bottleneck_full_score",topn=2,cost_bps=15)
    assert len(bt)>100 and np.isfinite(bt.net_ret).all()
    print({"rows":len(p),"jan_fund_available":str(jan.available_date),"jan_date":str(jan.date),
           "pre_tnic_year":int(pre.year),"post_tnic_year":int(post.year),
           "backtest_months":len(bt),"terminal":float((1+bt.net_ret).prod())})

if __name__=="__main__":main()
