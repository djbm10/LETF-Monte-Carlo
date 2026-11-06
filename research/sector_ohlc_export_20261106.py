from pathlib import Path
import pandas as pd,yfinance as yf
P=Path('results/sector_ohlc_20261106');P.mkdir(parents=True,exist_ok=True)
F={'XLK':'ROM','SOXX':'USD','XLF':'UYG','XLV':'RXL','XLE':'DIG','XLY':'UCC','XLB':'UYM','XLI':'UXI','XLP':'UGE','VNQ':'URE','XBI':'LABU','XLU':'UTSL','XHB':'NAIL','XRT':'RETL','IWM':'TNA','XLK':'TECL'}
T=sorted(set(['SPY','QQQ','VOO','^IRX','ROM','USD','UYG','RXL','DIG','UCC','UYM','UXI','UGE','URE','TECL','SOXL','FAS','CURE','LABU','DRN','NAIL','UTSL','WANT','RETL','DUSL','TNA','XLK','SOXX','XLF','XLV','XBI','VNQ','XHB','XLU','XLY','XRT','XLI','IWM','XLE','XLB','XLP']))
opens={};closes={}
for i in range(0,len(T),12):
 b=T[i:i+12]
 try:
  x=yf.download(b,start='2015-01-01',end='2026-11-06',auto_adjust=True,progress=False,threads=False,group_by='ticker',timeout=35)
  for t in b:
   try:
    y=x[t]
    for col,acc in [('Open',opens),('Close',closes)]:
     z=pd.to_numeric(y[col],errors='coerce').dropna()
     if len(z)>200:acc[t]=z
   except Exception as e: print('FAIL',t,str(e)[:100],flush=True)
 except Exception as e:print('BATCH_FAIL',str(e)[:120],flush=True)
 print('PROGRESS',i+len(b),len(T),'OHLC symbols',len(opens),flush=True)
if not {'SPY','VOO','ROM','USD'}.issubset(closes):raise RuntimeError('Missing required anchor history')
pd.DataFrame(opens).sort_index().loc[:'2026-11-05'].to_csv(P/'adjusted_open.csv')
pd.DataFrame(closes).sort_index().loc[:'2026-11-05'].to_csv(P/'adjusted_close.csv')
print('DONE',len(opens),len(closes))
