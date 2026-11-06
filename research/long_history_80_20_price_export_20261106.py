"""Read-only daily OHLC extraction; no signals are fit or optimized."""
from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf, json, io, zipfile, requests
out=Path('results/long_history_80_20_20261106');out.mkdir(parents=True,exist_ok=True)
pairs={'XLK':'ROM','SOXX':'USD','XLF':'UYG','XLV':'RXL','XLE':'DIG','XLY':'UCC','XLB':'UYM','XLI':'UXI','XLP':'UGE','VNQ':'URE'}
allsector=['XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLY','XLB']
syms=sorted(set(['QQQ','SPY','^IRX','^NDX']+list(pairs)+list(pairs.values())+allsector))
opens={};closes={};fails=[]
for k in range(0,len(syms),8):
    names=syms[k:k+8]
    try:
        z=yf.download(names,start='1985-01-01',end='2026-11-07',auto_adjust=True,progress=False,threads=False,group_by='ticker',timeout=40)
        for ticker in names:
            try:
                sub=z[ticker] if isinstance(z.columns,pd.MultiIndex) else z
                for field,dest in [('Open',opens),('Close',closes)]:
                    x=sub[field]
                    if isinstance(x,pd.DataFrame):x=x.iloc[:,0]
                    x=pd.to_numeric(x,errors='coerce').dropna()
                    if len(x)>150:
                        x.index=pd.to_datetime(x.index).tz_localize(None)
                        dest[ticker]=x[~x.index.duplicated()].sort_index()
            except Exception as ex:fails.append([ticker,str(ex)[:120]])
    except Exception as ex:fails.append([names,str(ex)[:160]])
    print('BATCH',k+len(names),len(syms),'OPEN',len(opens),'CLOSE',len(closes),flush=True)
a=pd.DataFrame(opens).sort_index().loc[:'2026-11-05'];c=pd.DataFrame(closes).sort_index().loc[:'2026-11-05']
if not {'QQQ','SPY','XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLY','XLB'}.issubset(c):raise RuntimeError('Fail closed: QQQ or original nine sectors incomplete')
a.to_csv(out/'adjusted_open_1985_2026.csv');c.to_csv(out/'adjusted_close_1985_2026.csv')
coverage=[dict(ticker=t,first_open=str(a[t].first_valid_index().date()) if t in a else None,first_close=str(c[t].first_valid_index().date()) if t in c else None,valid_close=int(c[t].count()) if t in c else 0) for t in syms]
pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
(out/'manifest.json').write_text(json.dumps({'retrieval_asof':'2026-11-06','last_fully_closed_session':'2026-11-05','source':'Yahoo Finance auto-adjusted OHLC via yfinance','survivorship':'Known original nine SPDR sector instruments plus sector-ETF 2x symbols; no synthetic data in this step','warnings':fails},indent=2))
print('FINISHED',a.shape,c.shape,flush=True)
