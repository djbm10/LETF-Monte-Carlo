from __future__ import annotations
import io, json, math, time, zipfile, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings('ignore')
ROOT=Path('results/expanded_tournament_20261106'); ROOT.mkdir(parents=True,exist_ok=True)
END='2026-11-06'  # exclusive, 2026-11-05 last fully closed session
SECTORS_3={'XLK':'TECL','SOXX':'SOXL','XLF':'FAS','XLV':'CURE','XBI':'LABU','VNQ':'DRN','XHB':'NAIL','XLU':'UTSL','XLY':'WANT','XRT':'RETL','XLI':'DUSL','IWM':'TNA'}
SECTORS_2={'XLK':'ROM','SOXX':'USD','XLF':'UYG','XLV':'RXL','XLE':'DIG','XLY':'UCC','XLB':'UYM','XLI':'UXI','XLP':'UGE','VNQ':'URE'}
MACRO={'equity':['SPY','IWM','EFA','EEM'],'rates':['IEF','TLT','SHY'],'currency':['UUP','FXE','FXY','FXB','FXA'],'commodities':['GLD','SLV','DBC','DBA','USO','CPER']}
FUNDS=['QYLD','XYLD','RYLD','JEPI','JEPQ','DIVO','PBP','PUTW','QYLG','XYLG','TAIL']
OTHER=['SPY','QQQ','VOO','TQQQ','^IRX','^VIX','^VXN']
RET_FILE=ROOT/'daily_returns.csv'; OUT_FILE=ROOT/'metrics.csv'

def gather_pt_universe():
    src='https://github.com/pierrebrunelle/sp500-historical-constituents/archive/refs/heads/main.zip'
    import requests
    r=requests.get(src,timeout=90);r.raise_for_status();z=zipfile.ZipFile(io.BytesIO(r.content))
    dates={}; allsyms=set()
    for n in z.namelist():
        base=Path(n).name
        if not base.startswith('spy') or len(base)!=9:continue
        ym=base[3:]
        if not ym.isdigit() or ym<'201601' or ym>'202604':continue
        symbols=[]
        for line in z.read(n).decode('utf-8-sig').splitlines():
            parts=line.split('\t')
            if parts and parts[0].strip():
                t=parts[0].strip().replace('/','-')
                symbols.append(t);allsyms.add(t)
        dates[ym]=sorted(set(symbols))
    if len(dates)<120:raise RuntimeError(f'Historical membership insufficient: {len(dates)} months')
    (ROOT/'historical_membership.json').write_text(json.dumps(dates))
    return dates,sorted(allsyms)

def yf_batch(tickers,start='2005-01-01',chunk=18):
    import yfinance as yf
    out={};fail=[]
    for j in range(0,len(tickers),chunk):
        batch=tickers[j:j+chunk]
        try:
            p=yf.download(batch,start=start,end=END,auto_adjust=True,progress=False,threads=False,group_by='ticker',timeout=35)
            for t in batch:
                try:
                    raw=p[t]['Close'] if isinstance(p.columns,pd.MultiIndex) else p['Close']
                    if isinstance(raw,pd.DataFrame):raw=raw.iloc[:,0]
                    s=pd.to_numeric(raw,errors='coerce').dropna()
                    if len(s)>=200:
                        s.index=pd.to_datetime(s.index).tz_localize(None)
                        out[t]=s[~s.index.duplicated(keep='last')]
                    else:fail.append(t)
                except Exception:fail.append(t)
        except Exception as e:
            fail+=batch
        if j%180==0:print('download_progress',j,len(tickers),'success',len(out),flush=True)
    return out,sorted(set(fail))

def load_data():
    print('HISTORICAL UNIVERSE',flush=True)
    dates, tickers=gather_pt_universe()
    baseline=list(set(OTHER+FUNDS+list(SECTORS_3)+list(SECTORS_3.values())+list(SECTORS_2)+list(SECTORS_2.values())+sum(MACRO.values(),[])))
    base,fail=yf_batch(sorted(baseline),start='2000-01-01')
    if not {'SPY','QQQ','^VIX'}.issubset(base):raise RuntimeError('Required SPY/QQQ/VIX are unavailable: fail closed')
    growth,failed_growth=yf_batch(sorted(set(tickers)-set(base)),start='2015-01-01',chunk=20)
    # Historical membership data are not assumed to carry delisting prices.
    allpx=pd.DataFrame({**base,**growth}).sort_index();allpx=allpx.loc[allpx.index<='2026-11-05']
    allpx.to_pickle(ROOT/'downloaded_prices.pkl')
    pd.DataFrame({'ticker':sorted(set(tickers)),'downloadable':[int(t in allpx.columns) for t in sorted(set(tickers))]}).to_csv(ROOT/'historical_symbol_download_coverage.csv',index=False)
    print('DOWNLOAD',len(base),len(growth),'historical symbols',len(tickers),'missing',len(failed_growth),flush=True)
    return allpx,dates,tickers

def retmetrics(s):
    s=s.dropna()
    if len(s)<252:return None
    wealth=(1+s).cumprod(); mdd=(wealth/wealth.cummax()-1).min(); yrs=len(s)/252
    if wealth.iloc[-1]<=0:return None
    roll1=(1+s).rolling(252).apply(np.prod,raw=True)-1
    return dict(start=str(s.index[0].date()),end=str(s.index[-1].date()),n=len(s),years=yrs,
        cagr=float(wealth.iloc[-1]**(1/yrs)-1),maxdd=float(mdd),vol=float(s.std()*np.sqrt(252)),
        sharpe=float(s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else None,
        worst_day=float(s.min()),worst_1y=float(roll1.min()),terminal=float(wealth.iloc[-1]))

def sector_runs(px,rf):
    results={};notes=[]
    ix=px.loc['2014-01-01':].index
    spy=px['SPY'].reindex(ix);spy_ma=spy.rolling(200).mean()
    for leverage,pairs in [('3x',SECTORS_3),('2x',SECTORS_2)]:
        available={sig:lev for sig,lev in pairs.items() if sig in px.columns and lev in px.columns}
        sigpx=px[list(available)].reindex(ix)
        trpx=px[list(available.values())].reindex(ix)
        sr=sigpx.pct_change(fill_method=None); lr=trpx.pct_change(fill_method=None)
        for start in ('2016-01-01','2019-01-01'):
            mask=ix>=pd.Timestamp(start)
            months=ix.to_period('M')
            for lookback in (63,126,252):
                scores=sigpx/sigpx.shift(lookback)-1
                for n in (1,2,3,5):
                    if n>len(available):continue
                    for gate in ('none','SPY_200','sector_200'):
                        for sizing in ('full','vol25','vol35'):
                            # Position weights set on first trading day of each month using last session's data.
                            w=np.zeros((len(ix),len(available)),dtype=float)
                            vl=lr.rolling(20).std()*math.sqrt(252)
                            for i in range(253,len(ix)):
                                if i>0 and months[i]==months[i-1]:
                                    w[i]=w[i-1];continue
                                row=scores.iloc[i-1]
                                valid=row[(~row.isna())].index.tolist()
                                valid=[t for t in valid if pd.notna(lr[available[t]].iloc[i-1]) and
                                       pd.notna(vl[available[t]].iloc[i-1]) and
                                       pd.notna(trpx[available[t]].iloc[i-40])]
                                if gate=='SPY_200' and not (spy.iloc[i-1]>spy_ma.iloc[i-1]):valid=[]
                                if gate=='sector_200':
                                    valid=[t for t in valid if sigpx[t].iloc[i-1]>sigpx[t].iloc[i-200:i].mean()]
                                chosen=sorted(valid,key=lambda t:row[t],reverse=True)[:n]
                                if len(chosen)<n:continue # remaining in cash if fewer than n
                                weights=np.ones(n)/n
                                if sizing!='full':
                                    target=.25 if sizing=='vol25' else .35
                                    risk=np.array([float(vl.at[ix[i-1],available[t]]) for t in chosen])
                                    mult=np.clip(target/np.maximum(risk,.08),0,1.0)
                                    weights=weights*mult
                                for t,v in zip(chosen,weights):w[i,list(available).index(t)]=v
                            wdf=pd.DataFrame(w,index=ix,columns=list(available))
                            # Exposure applies next session after prior-session signal; trading at next open approximated by close-to-close return.
                            rr=(wdf*lr.rename(columns={v:k for k,v in available.items()})).sum(axis=1,min_count=1).fillna(0)+rf.reindex(ix).fillna(0)*(1-wdf.sum(axis=1))
                            tc=wdf.diff().abs().sum(axis=1).fillna(0)*.0007
                            rr=(rr-tc).loc[mask]
                            # Fail-close: unknown/absent leveraged ETF daily return while holding => exclude strategy days, not reclassify as zero gains.
                            missing=((wdf>0)&lr.rename(columns={v:k for k,v in available.items()}).isna()).any(axis=1)
                            if missing.loc[mask].any():continue
                            nm=f'sector_{leverage}_{start[:4]}_L{lookback}_N{n}_{gate}_{sizing}'
                            results[nm]=rr
                            notes.append(dict(name=nm,coverage=float((wdf.loc[mask].sum(axis=1)>0).mean()),avg_cash_weight=float((1-wdf.loc[mask].sum(axis=1)).mean()),turnover=float(tc.loc[mask].sum()/.0007/(len(rr)/252))))
    return results,pd.DataFrame(notes)

def growth_runs(px,members,rf):
    # Source symbols PIT; Yahoo prices of delisted firms can be unavailable. We do NOT label outputs survivor-bias-free.
    idx=px.loc['2016-01-01':'2026-04-30'].index
    vals=pd.DataFrame(index=idx)
    syms=sorted(set(sum(members.values(),[])))
    px0=px.reindex(columns=syms).reindex(idx)
    hist=px.reindex(columns=syms)
    daily=hist.pct_change(fill_method=None)
    # Prior-close signals, next-session returns; value factor not available point-in-time here.
    mom126=hist/hist.shift(126)-1
    mom252=hist.shift(21)/hist.shift(252)-1
    r=hist.pct_change(fill_method=None)
    vl=r.rolling(63).std()*np.sqrt(252)
    high=hist/hist.rolling(252).max()
    ma=hist.rolling(200).mean()
    # Coverage includes historical members *without* prices; don't silently remove them from universe denominator.
    coverage=[];out={}
    months=idx.to_period('M')
    for cadence in ('M','Q'):
        for name in ('mom6','mom12_1','risk_adjusted_mom','near_high','momentum_plus_trend'):
            for n in (20,50):
                weights=np.zeros((len(idx),len(syms)),dtype=np.float32)
                valid_dates=0;good_months=0; selected_covered=[]
                prev=np.zeros(len(syms),dtype=np.float32)
                for i,date in enumerate(idx):
                    if i and (months[i]==months[i-1] if cadence=='M' else (date.to_period('Q')==idx[i-1].to_period('Q'))):
                        weights[i]=prev;continue
                    if i==0: prior=hist.index[hist.index<date][-1] if len(hist.index[hist.index<date]) else None
                    else: prior=idx[i-1]
                    if prior is None:continue
                    ym=str((date-pd.offsets.MonthEnd(1)).strftime('%Y%m'))
                    if ym not in members:ym=max((k for k in members if k<date.strftime('%Y%m')),default='')
                    if not ym:continue
                    universe=members[ym]
                    if not len(universe):continue
                    # Universe data at 'prior month' is conservative. Availability of price data measured independently.
                    known=[s for s in universe if s in hist and pd.notna(hist.at[prior,s])]
                    fraction=len(known)/len(universe)
                    coverage.append((date,ym,len(universe),len(known),fraction,name,cadence,n))
                    if fraction<.85:continue
                    if name=='mom6':score=mom126.loc[prior]
                    elif name=='mom12_1':score=mom252.loc[prior]
                    elif name=='risk_adjusted_mom':score=mom252.loc[prior]/vl.loc[prior].clip(lower=.08)
                    elif name=='near_high':score=high.loc[prior]
                    else:score=mom252.loc[prior].where(hist.loc[prior]>ma.loc[prior])
                    ranked=score.reindex(known).dropna().sort_values(ascending=False)
                    # To prevent missing-price zero-return substitution, fail if any selected stock has nonfinite current-session price return.
                    chosen=ranked.head(n).index.tolist()
                    if len(chosen)<n:continue
                    next_ret=daily.loc[date,chosen]
                    if next_ret.isna().any():continue
                    prev=np.zeros(len(syms),dtype=np.float32)
                    for t in chosen:prev[syms.index(t)]=1/n
                    weights[i]=prev;good_months+=1
                w=pd.DataFrame(weights,index=idx,columns=syms)
                rr=(w*daily.reindex(idx)).sum(axis=1,min_count=1).fillna(0)
                # If a held stock lacks next-day return, do not assume zero loss: mark incomplete, exclude that model entirely.
                gaps=((w>0)&daily.reindex(idx).isna()).any(axis=1)
                # Upper bound values missing-day stock returns at 0%; lower bound assumes total capital loss on missing-day position.
                # Neither bound replaces delisting prices: estimates are NOT decision-grade.
                missing_count=int(gaps.sum());missing_weights=(w.where(daily.reindex(idx).isna(),0)).sum(axis=1)
                rr_lower=rr-missing_weights
                rr+=rf.reindex(idx).fillna(0)*(1-w.sum(axis=1))
                turnover=w.diff().abs().sum(axis=1).fillna(0)
                rr-=.001*turnover
                key=f'growth_{cadence}_{name}_N{n}'
                out[key+'_missing0']=rr
                out[key+'_missing100']=rr_lower+rf.reindex(idx).fillna(0)*(1-w.sum(axis=1))-.001*turnover
                selected_covered.append(dict(strategy=key,n_rebalances=good_months,n_universe_months=len(members),fraction_invested=float((w.sum(axis=1)>0).mean()),turnover_oneway_year=float(turnover.sum()/(len(idx)/252)),missing_price_gaps=missing_count))
    return out,pd.DataFrame(coverage,columns=['date','membership_month','members','priced','coverage','screen','rebalance','n']),pd.DataFrame(selected_covered)

def macro_runs(px,rf):
    ix=px.loc['2007-01-01':].index;rr=px.reindex(ix).pct_change(fill_method=None);months=ix.to_period('M')
    out={};details=[]
    for lookback in (63,126,252):
        for side in ('longflat','longshort'):
            for cadence in ('M','W'):
                for target in (.10,.15,.20):
                    used=sorted({s for group in MACRO.values() for s in group if s in px})
                    mom=px[used].reindex(ix)/px[used].reindex(ix).shift(lookback)-1
                    vols=rr[used].rolling(63).std()*np.sqrt(252)
                    w=np.zeros((len(ix),len(used)),dtype=np.float32)
                    prev=np.zeros(len(used),dtype=np.float32)
                    for i in range(253,len(ix)):
                        if i and ((months[i]==months[i-1]) if cadence=='M' else (ix[i].isocalendar().week==ix[i-1].isocalendar().week and ix[i].year==ix[i-1].year)):
                            w[i]=prev;continue
                        ww=np.zeros(len(used))
                        for group,li in MACRO.items():
                            candidates=[t for t in li if t in used and pd.notna(mom.at[ix[i-1],t]) and pd.notna(vols.at[ix[i-1],t]) and vols.at[ix[i-1],t]>.0001]
                            if not candidates:continue
                            # Equal risk budget per asset class and then per available instrument.
                            for t in candidates:
                                trend=math.copysign(1,float(mom.at[ix[i-1],t]))
                                if trend<0 and side=='longflat':continue
                                ww[used.index(t)]=(target/4/len(candidates))/max(float(vols.at[ix[i-1],t]),.04)*trend
                        if np.sum(abs(ww))>2:ww*=2/np.sum(abs(ww))
                        prev=ww.astype('float32');w[i]=prev
                    alloc=pd.DataFrame(w,index=ix,columns=used)
                    gaps=((alloc!=0)&rr[used].isna()).any(axis=1)
                    if gaps.any():continue
                    turn=alloc.diff().abs().sum(axis=1).fillna(0)
                    # Flat positions earn risk-free; short ETF exposure modeled with 2% borrow charge annually.
                    pnl=(alloc*rr[used]).sum(axis=1)+rf.reindex(ix).fillna(0)*(1-alloc.sum(axis=1))-.0005*turn-.02/252*alloc.clip(upper=0).abs().sum(axis=1)
                    nm=f'macro_{lookback}_{side}_{cadence}_{int(target*100)}'
                    out[nm]=pnl.loc['2011-01-01':]
                    details.append(dict(strategy=nm,avg_gross_exposure=float(alloc.abs().sum(axis=1).mean()),turnover_year=float(turn.sum()/(len(ix)/252))))
    return out,pd.DataFrame(details)

def bs_put(s,k,t,v,rate):
    if t<=1e-8:return max(k-s,0)
    v=max(v,.03);d1=(math.log(s/k)+(rate+v*v/2)*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
    return k*math.exp(-rate*t)*norm.cdf(-d2)-s*norm.cdf(-d1)

def bs_call(s,k,t,v,rate):
    if t<=1e-8:return max(s-k,0)
    v=max(v,.03);d1=(math.log(s/k)+(rate+v*v/2)*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
    return s*norm.cdf(d1)-k*math.exp(-rate*t)*norm.cdf(d2)

def option_synthetic(px,rf):
    # Monthly cash-secured short puts, collars, and covered calls, marked daily with contemporaneous VIX.
    # Model, NOT observed options and NOT comparable to true options execution.
    out={}
    for underlying,volname in [('SPY','^VIX'),('QQQ','^VXN')]:
        if underlying not in px or volname not in px:continue
        idx=px.loc['2012-01-01':].index
        spot=px[underlying].reindex(idx);iv=px[volname].reindex(idx)/100
        if iv.dropna().empty:continue
        annual=(rf.reindex(idx)*252).clip(0,.1).fillna(.025)
        for strategy in ('cash_put','covered_call','collar'):
          for delta in (.15,.30):
           for regime in ('always','trend'):
            wealth=1.0;cash=1.0;units=0.;strike=0.;days=0;current=None;values=[]
            ma=spot.rolling(200).mean();dtlast=None
            for i,date in enumerate(idx):
                s=float(spot.iloc[i]) if np.isfinite(spot.iloc[i]) else np.nan
                v=float(iv.iloc[i]) if np.isfinite(iv.iloc[i]) else np.nan
                rate=float(annual.iloc[i]);
                if not np.isfinite(s) or not np.isfinite(v):values.append((date,np.nan));continue
                if i>0:cash*=1+rate/252
                if current is not None:
                    days=max(days-1,0)
                    optionprice=(bs_call(s,strike,days/252,v,rate) if current=='call' else bs_put(s,strike,days/252,v+.03,rate))
                    value=cash+units*s-(contract_count*optionprice if current in ('call','put') else 0.)
                    # For covered-call, units holds underlying, short call liability.
                    # For short put, units is zero, short put liability.
                    if days==0:
                        if current=='call':
                            cash-=contract_count*max(s-strike,0)
                        else:cash-=contract_count*max(strike-s,0)
                        current=None;contract_count=0.;value=cash+units*s
                else:value=cash+units*s
                if i>=205 and current is None and (dtlast is None or (date-dtlast).days>=29):
                    allow=regime=='always' or bool(s>ma.iloc[i-1])
                    if allow:
                        dtlast=date;days=21;t=days/252
                        # K such that Black-Scholes delta magnitude near parameter.
                        put_v=v+.03
                        if strategy=='cash_put':
                            d1=-norm.ppf(delta)
                            strike=s*math.exp((rate+put_v**2/2)*t-d1*put_v*math.sqrt(t))
                            premium=bs_put(s,strike,t,put_v,rate)
                            contract_count=max(cash,0)/s
                            cash+=contract_count*premium*.975
                            current='put';units=0
                        else:
                            # Turn invested equity from cash, sell calls with chosen delta.
                            old=cash+units*s
                            units=old/s;cash=0.
                            d1=norm.ppf(delta)
                            strike=s*math.exp((rate+v*v/2)*t-d1*v*math.sqrt(t))
                            contract_count=units
                            premium=bs_call(s,strike,t,v,rate)
                            cash+=contract_count*premium*.975
                            current='call'
                            if strategy=='collar':
                                # put premium discounted cash, buying 10pct OTM protective put; fixed cost deducted with MTM not represented.
                                # A full mark-to-market collar cannot be modeled faithfully in this simplistic engine.
                                pass
                    else:
                        dtlast=date
                values.append((date,max(cash+units*s-(contract_count*(bs_call(s,strike,days/252,v,rate) if current=='call' else bs_put(s,strike,days/252,v+.03,rate)) if current else 0),1e-9)))
            if strategy=='collar':continue # fail closed: no correct simultaneous put+call book
            eq=pd.Series(dict(values)).dropna()
            if len(eq)>1000:
                key=f'options_MODEL_{underlying}_{strategy}_d{int(delta*100)}_{regime}'
                out[key]=eq.pct_change(fill_method=None).dropna()
    return out

def run():
    import os
    if os.environ.get('LOCAL_INPUT'):
        px=pd.read_csv(os.environ['LOCAL_INPUT'],index_col=0,parse_dates=True)
        members={};syms=[]
        # Local smoke test uses original supplied adjusted series; stock PIT history unavailable locally.
    else:px,members,syms=load_data()
    px=px.loc[px.index<='2026-11-05'].copy()
    rf=(px['^IRX']/100/252 if '^IRX' in px else pd.Series(.025/252,index=px.index)).reindex(px.index).ffill().fillna(.025/252)
    base_px=px['QQQ'];qret=base_px.pct_change(fill_method=None);excess=qret-rf
    vol=excess.rolling(20).std()*np.sqrt(252)
    expo=(base_px>base_px.rolling(200).mean()).shift(1).fillna(False).astype(float)*(.35/vol.clip(lower=.04)).clip(upper=3).shift(1).fillna(0)
    benchmark=rf+expo*(qret-rf)-expo.diff().abs().fillna(expo.abs())*.0002
    allret={'35_0_QQQ_proxy':benchmark,'SPY':px['SPY'].pct_change(fill_method=None),'QQQ':qret}
    if 'VOO' in px:allret['VOO']=px['VOO'].pct_change(fill_method=None)
    print('SECTOR TESTS',flush=True)
    sector,sectorinfo=sector_runs(px,rf);allret.update(sector)
    if members:
        print('GROWTH TESTS',flush=True)
        growth,cov,gi=growth_runs(px,members,rf);allret.update(growth)
        cov.to_csv(ROOT/'pit_membership_price_coverage.csv',index=False)
        gi.to_csv(ROOT/'growth_execution_coverage.csv',index=False)
    print('MACRO TESTS',flush=True)
    macro,macinfo=macro_runs(px,rf);allret.update(macro)
    print('OPTIONS MODEL TESTS',flush=True)
    options=option_synthetic(px,rf);allret.update(options)
    for fund in FUNDS:
        if fund in px:allret[f'options_FUND_{fund}']=px[fund].pct_change(fill_method=None)
    # Actual Cboe total-return OPTION BENCHMARK indices, where provider publishes history.
    # Not equivalent to individual brokerage fills; no look-ahead strategy selection from these indices.
    try:
        import requests
        for symbol in ('BXM','BXY','PUT','WPUT','PUTY','PPUT','CLL','CLLZ','CNDR','RXM'):
            try:
                url=f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv'
                response=requests.get(url,timeout=18);response.raise_for_status()
                raw=pd.read_csv(io.BytesIO(response.content),on_bad_lines='skip')
                if raw.shape[1]<2:continue
                dates=pd.to_datetime(raw.iloc[:,0],errors='coerce')
                cols=[]
                for col in raw.columns[1:]:
                    nums=pd.to_numeric(raw[col].astype(str).str.replace(',',''),errors='coerce')
                    if (nums.gt(0)&dates.notna()).sum()>250:cols.append((col,nums))
                if not cols:continue
                a=pd.Series(cols[-1][1].values,index=dates).dropna().sort_index()
                a=a[~a.index.duplicated(keep='last')];a=a.loc[:'2026-11-05']
                if len(a)>500:
                    allret['CBOE_'+symbol]=a.pct_change(fill_method=None)
                    print('CBOE_OPTION_INDEX',symbol,len(a),str(a.index.min()),str(a.index.max()),flush=True)
            except Exception as e:print('CBOE_UNAVAILABLE',symbol,str(e)[:180],flush=True)
    except Exception as e:print('CBOE_BENCHMARKS_UNAVAILABLE',str(e)[:200],flush=True)
    # All historical comparisons use original actual dates; returns reported from different start dates marked.
    series=pd.DataFrame(allret).loc[:'2026-11-05']
    series.to_pickle(ROOT/'daily_returns.pkl')
    windowlist=[('2016_to_Apr2026','2016-05-01','2026-04-30'),('2019_to_Apr2026','2019-01-01','2026-04-30'),('2019_to_Nov2026','2019-01-01','2026-11-05'),('2021_to_Nov2026','2021-01-01','2026-11-05')]
    rows=[]
    for period,start,end in windowlist:
        for name in series:
            raw=series.loc[start:end,name]
            # Available data only: no zero-filling missing histories, no terminal fill.
            clean=raw.dropna()
            if len(clean)<252 or float(len(clean))/len(raw)<.90:continue
            m=retmetrics(clean)
            if m:rows.append(dict(period=period,strategy=name,**m))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT_FILE,index=False)
    sectorinfo.to_csv(ROOT/'sector_allocation_summary.csv',index=False)
    macinfo.to_csv(ROOT/'crossasset_allocation_summary.csv',index=False)
    details={'asof':'2026-11-05','revision':'nov6 expanded tournament', 'variants_total':len(allret),
        'data_source':'Yahoo adjusted closes and historical S&P 500 membership snapshots',
        'growth':'Historical membership Jan2016-Apr2026 but daily yahoo prices exclude delisted; price momentum is NOT fundamental revenue growth. Two explicit unverified boundary estimates: 0% and -100% missing stock-day returns, not real delisting returns. Both are biased and not decision-grade',
        'leverage':'Actual leveraged sector ETF returns where available (never synthetic pre-inception), equal weights, 70bps per 1000% annual turnover, etc.',
        'options':'CBOE_BXM/PUT/etc observed benchmark index returns when obtained from official Cboe; QYLD/XYLD etc observed ETF returns; options_MODEL synthetic Black-Scholes based on VIX/VXN, NOT actual executable options chains',
        'crossasset':'ETF total-return proxies, not dated futures; borrow and financing approximations',
        'holdout':'Data inspected in earlier rounds; not untouched. No PROMOTE status possible. Original sector train/validation killed, not resurrected by variations.',
        'implementation':'Previous close signal uses close-to-close return, does not reconstruct market-on-open trades. Taxes omitted.',
        'parameters':{'sector':{'leverage':['2x','3x'],'lookbacks':[63,126,252],'topn':[1,2,3,5],'gate':['none','SPY_200','sector_200'],'sizing':['full','vol25','vol35'],'rebal':'monthly'}, 'growth':{'20':[.05],'50':[.02],'screens':['mom6','mom12_1','risk_adjusted_mom','near_high','momentum_plus_trend'],'frequency':['M','Q']},'macro':{'lookbacks':[63,126,252],'longshort':['longflat','longshort'],'cadence':['M','W'],'target_vols':[.10,.15,.20]},'options':{'strats':['cash_put','covered_call'],'delta':[.15,.30],'regime':['always','trend']}}}
    (ROOT/'manifest.json').write_text(json.dumps(details,indent=2))
    print('SUCCESS VARIANTS',len(allret),'metrics',len(metrics),flush=True)
    for period in ('2019_to_Apr2026','2019_to_Nov2026'):
        z=metrics[metrics.period==period]
        for prefix in ('sector','growth','macro','options_'):
            q=z[z.strategy.str.startswith(prefix)].sort_values('cagr',ascending=False).head(8)
            print(period,prefix,'top candidates',q[['strategy','cagr','maxdd','years']].to_string(index=False),flush=True)
    print('FILES:',[p.name for p in ROOT.iterdir()],flush=True)
if __name__=='__main__':run()
