from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
OUT=Path('results/taxable_s9_variants');OUT.mkdir(parents=True,exist_ok=True);TD=252;TC=.0003

def dl(t):
 x=yf.download(t,start='2010-02-01',end='2026-08-22',auto_adjust=True,progress=False);s=x['Close'];s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s;s.index=pd.to_datetime(s.index).tz_localize(None);return pd.to_numeric(s,errors='coerce').dropna()
def met(r):
 e=(1+pd.Series(r).dropna()).cumprod();y=len(e)/TD;dd=e/e.cummax()-1;return {'cagr':e.iloc[-1]**(1/y)-1,'max_dd':dd.min(),'vol':pd.Series(r).std()*math.sqrt(TD)}
def desired(spy,tq):
 ma=spy.rolling(200).mean();v=tq.pct_change().rolling(20).std()*math.sqrt(TD);w=pd.Series(0.,index=spy.index)
 for i in range(1,len(w)):
  if pd.notna(ma.iloc[i-1]) and pd.notna(v.iloc[i-1]) and v.iloc[i-1]>0:w.iloc[i]=np.clip((.35 if spy.iloc[i-1]>ma.iloc[i-1] else .12)/v.iloc[i-1],0,1)
 return w
def variant(w,mode):
 a=pd.Series(0.,index=w.index)
 for i in range(1,len(w)):
  if mode=='band5':a.iloc[i]=w.iloc[i] if abs(w.iloc[i]-a.iloc[i-1])>=.05 else a.iloc[i-1]
  elif mode=='band10':a.iloc[i]=w.iloc[i] if abs(w.iloc[i]-a.iloc[i-1])>=.10 else a.iloc[i-1]
  elif mode=='weekly':a.iloc[i]=w.iloc[i] if i%5==0 else a.iloc[i-1]
  else:a.iloc[i]=w.iloc[i]
 return a
class Lots:
 def __init__(self):self.l=[]
 def buy(self,d,q,p):
  if q>1e-12:self.l.append([d,q,p])
 def val(self,p):return sum(x[1] for x in self.l)*p
 def sell(self,d,q,p):
  c=[]
  for j,x in enumerate(self.l):
   g=p-x[2];long=(d-x[0]).days>365;cls=0 if g<0 and not long else 1 if g<0 else 2 if long else 3;c.append((cls,abs(g) if g<0 else g,j))
  c.sort();ev=[]
  for _,_,j in c:
   if q<=1e-12:break
   x=self.l[j];z=min(q,x[1]);ev.append(((p-x[2])*z,(d-x[0]).days>365));x[1]-=z;q-=z
  self.l=[x for x in self.l if x[1]>1e-10];return ev
def char(px,a):
 b=Lots();cash=1.;ev=[];vals=[]
 for i,d in enumerate(px.index):
  p=float(px.iloc[i]);wealth=cash+b.val(p);want=wealth*float(a.iloc[i]);cur=b.val(p)
  if cur>want:q=(cur-want)/p;ev+=b.sell(d,q,p);cash+=q*p
  elif want>cur:sp=min(want-cur,cash);b.buy(d,sp/p,p);cash-=sp
  vals.append(wealth)
 yrs=len(px)/TD;avg=np.mean(vals);stg=sum(g for g,l in ev if g>0 and not l);ltg=sum(g for g,l in ev if g>0 and l);stl=-sum(g for g,l in ev if g<0 and not l);ltl=-sum(g for g,l in ev if g<0 and l)
 return {'stg':stg/yrs/avg,'ltg':ltg/yrs/avg,'stl':stl/yrs/avg,'ltl':ltl/yrs/avg,'short_gain_share':stg/max(stg+ltg,1e-12)}
def main():
 spy=dl('SPY');tq=dl('TQQQ');qqq=dl('QQQ');ix=spy.index.intersection(tq.index).intersection(qqq.index);spy=spy.reindex(ix);tq=tq.reindex(ix);qqq=qqq.reindex(ix);w=desired(spy,tq);rows=[];tax=[]
 for mode in ['daily','band5','band10','weekly']:
  a=variant(w,mode);r=a*tq.pct_change()-TC*a.diff().abs().fillna(a.abs());m=met(r);d=a.diff().abs().fillna(a.abs());c=char(tq,a);rows.append({'strategy':mode,**m,'change_days_per_year':(d>1e-12).sum()/(len(a)/TD),'turnover_per_year':d.sum()/(len(a)/TD),**c})
  for o,l,label in [(.24,.15,'24_15'),(.32,.15,'32_15'),(.37,.20,'37_20')]:
   low=max(c['stg']-c['stl'],0)*o+max(c['ltg']-c['ltl'],0)*l;high=c['stg']*o+c['ltg']*l;tax.append({'strategy':mode,'case':label,'tax_drag_optimistic':low,'tax_drag_conservative':high,'after_tax_cagr_low':m['cagr']-high,'after_tax_cagr_high':m['cagr']-low})
 bh=[]
 for name,s in [('SPY_BH',spy),('QQQ_BH',qqq),('TQQQ_BH',tq)]:bh.append({'strategy':name,**met(s.pct_change())})
 pd.DataFrame(rows).to_csv(OUT/'variants.csv',index=False);pd.DataFrame(tax).to_csv(OUT/'tax.csv',index=False);pd.DataFrame(bh).to_csv(OUT/'buy_hold.csv',index=False);(OUT/'manifest.json').write_text(json.dumps({'note':'Tax drag bounds as in account-location study; final liquidation/dividends/state/NIIT excluded.'},indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(tax).to_string(index=False));print(pd.DataFrame(bh).to_string(index=False))
if __name__=='__main__':main()
