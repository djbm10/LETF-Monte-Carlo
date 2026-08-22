from __future__ import annotations
import numpy as np, pandas as pd
import research.nasdaq1985_final_tournament as n
TD=252
EVENTS={'1987 crash':('1987-08-01','1989-07-31'),'dot-com':('2000-03-01','2007-06-30'),'GFC':('2007-10-01','2013-03-31'),'COVID':('2020-02-01','2020-08-31'),'2022 inflation':('2022-01-01','2024-07-31')}
def stats(r):
 g=np.maximum(1+np.asarray(r,float),1e-12);eq=np.cumprod(g);yrs=len(r)/TD;peak=np.maximum.accumulate(eq);dd=eq/peak-1;cur=mx=0
 for x in dd:
  if x<0:cur+=1;mx=max(mx,cur)
  else:cur=0
 return {'ann_return':eq[-1]**(1/yrs)-1,'total_return':eq[-1]-1,'max_dd':dd.min(),'max_underwater_years':mx/TD}
def main():
 df=n.load_data(.007);m=df.ndx.to_numpy();rf=df.rf.to_numpy();px=np.cumprod(np.maximum(1+m,1e-12));qld=n.synth_letf(m,rf,2,.0095,.005);tq=n.synth_letf(m,rf,3,.0088,.0065)
 cs=[n.Cand('NDX_1x','fixed',(1.0,)),n.Cand('VOL15','vol',(.15,)),n.Cand('VOL20','vol',(.20,)),n.Cand('VOL25','vol',(.25,)),n.Cand('TV15_0','tv',(200,.15,0.,.30)),n.Cand('TV20_0','tv',(200,.20,0.,.30)),n.Cand('TV25_0','tv',(200,.25,0.,.30)),n.Cand('TV30_0','tv',(200,.30,0.,.30)),n.Cand('TV35_0','tv',(200,.35,0.,.30))]
 rows=[]
 for c in cs:
  r=pd.Series(n.evaluate(c,m,rf,px,qld,tq),index=df.index)
  for ev,(a,b) in EVENTS.items():
   rr=r.loc[(r.index>=a)&(r.index<=b)];rows.append({'strategy':c.name,'event':ev,**stats(rr)})
 z=pd.DataFrame(rows);z.to_csv('results/friend_safe_crisis_frontier.csv',index=False);print(z.to_string(index=False))
if __name__=='__main__':main()
