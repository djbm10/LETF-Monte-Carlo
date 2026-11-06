from pathlib import Path
import pandas as pd, yfinance as yf
P=Path("results/leveraged_sector_prices_20261106");P.mkdir(parents=True,exist_ok=True)
F3={"XLK":"TECL","SOXX":"SOXL","XLF":"FAS","XLV":"CURE","XBI":"LABU","VNQ":"DRN","XHB":"NAIL","XLU":"UTSL","XLY":"WANT","XRT":"RETL","XLI":"DUSL","IWM":"TNA"}
F2={"XLK":"ROM","SOXX":"USD","XLF":"UYG","XLV":"RXL","XLE":"DIG","XLY":"UCC","XLB":"UYM","XLI":"UXI","XLP":"UGE","VNQ":"URE"}
T=sorted(set(["SPY","VOO","QQQ","^IRX"]+list(F3)+list(F3.values())+list(F2)+list(F2.values())))
d={}
for i in range(0,len(T),10):
 q=T[i:i+10]
 try:
  x=yf.download(q,start="2005-01-01",end="2026-11-06",auto_adjust=True,progress=False,threads=False,group_by="ticker",timeout=35)
  for t in q:
   try:
    y=x[t]["Close"];y=y.iloc[:,0] if isinstance(y,pd.DataFrame) else y
    y=pd.to_numeric(y,errors="coerce").dropna()
    if len(y)>250:d[t]=y
   except Exception as e:print("TICKERFAIL",t,str(e)[:90],flush=True)
 except Exception as e:print("CHUNKFAIL",q,str(e)[:120],flush=True)
 print("batch",i+len(q),"of",len(T),"success",len(d),flush=True)
if "SPY" not in d:raise RuntimeError("No SPY")
p=pd.DataFrame(d).loc[:'2026-11-05'].sort_index()
p.to_csv(P/"adjusted_closes.csv")
print("FINISHED",len(p.columns),"symbols",len(p),"rows",flush=True)
