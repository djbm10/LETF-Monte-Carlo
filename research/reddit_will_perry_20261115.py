"""Research-only, independently written reconstruction of Reddit Will Perry's TQQQ gates.

Source: BestFolio rejection log, Nov 9 2026. Original whitepaper unavailable.
Real fund window only: TQQQ Feb 2010 onward. No synthetic pre-inception returns.
No live trading, no parameter search, no claim of untouched holdout.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

TICKERS = ("QQQ", "SPY", "TQQQ", "BIL")
RULES = {"vol20_cap": 0.35, "vol_ratio_cap": 1.10, "dd_exit": -0.20,
         "dd_reentry": -0.10, "confirm_sessions": 2, "spy_entry_buffer": 0.04,
         "spy_exit_buffer": -0.03, "cost_per_side": 0.001, "days_per_year": 252}

def prices_from_yahoo(end='2026-11-14'):
    import yfinance as yf
    raw=yf.download(list(TICKERS), start='2008-01-01', end=end,
                    auto_adjust=True, threads=False, progress=False, group_by='ticker')
    if raw.empty:
        raise RuntimeError('No Yahoo returns; no results asserted')
    cols={}
    for t in TICKERS:
        if isinstance(raw.columns,pd.MultiIndex):
            series=raw[t]['Close'] if t in raw.columns.get_level_values(0) else raw['Close'][t]
        else:
            raise RuntimeError('Unexpected provider shape')
        cols[t]=pd.to_numeric(series,errors='coerce')
    all_prices=pd.DataFrame(cols).sort_index()
    all_prices.index=pd.to_datetime(all_prices.index).tz_localize(None).normalize()
    start=all_prices.TQQQ.first_valid_index()
    if start is None or start < pd.Timestamp('2010-01-01') or start > pd.Timestamp('2011-01-01'):
        raise RuntimeError('Unexpected TQQQ inception, not using data')
    data=all_prices.loc[start:].dropna(subset=list(TICKERS))
    if len(data)<3000 or data.index.max()<pd.Timestamp('2026-11-01'):
        raise RuntimeError('Insufficient verified overlapping real-fund history')
    if (data<=0).any().any() or data.index.has_duplicates:
        raise RuntimeError('Invalid adjusted-close data')
    return all_prices.loc[:data.index.max()],data

def indicators(prices):
    q=prices.QQQ
    qret=q.pct_change()
    v20=qret.rolling(20).std(ddof=1)*np.sqrt(252)
    v250=qret.rolling(250).std(ddof=1)*np.sqrt(252)
    ratio=v20/v250
    peak=q.rolling(250).max()
    dd=q/peak-1
    sma=prices.SPY.rolling(200).mean()
    return pd.DataFrame({'vol20':v20,'ratio':ratio,'dd':dd,'spy_rel':prices.SPY/sma-1},index=prices.index)

def signals(ind):
    perry=[]; spy=[]; drawdown_gate=False; pos=False; good=bad=0; band=False
    for row in ind.itertuples():
        ready=np.isfinite(row.vol20) and np.isfinite(row.ratio) and np.isfinite(row.dd)
        if ready:
            if drawdown_gate and row.dd < RULES['dd_exit']: drawdown_gate=False
            elif not drawdown_gate and row.dd >= RULES['dd_reentry']: drawdown_gate=True
            gate = drawdown_gate and row.vol20 < RULES['vol20_cap'] and row.ratio < RULES['vol_ratio_cap']
            good=good+1 if gate else 0
            bad=bad+1 if not gate else 0
            if good>=2:pos=True
            if bad>=2:pos=False
        else:
            pos=False;drawdown_gate=False;good=bad=0
        perry.append(float(pos))
        if np.isfinite(row.spy_rel):
            if row.spy_rel >= RULES['spy_entry_buffer']: band=True
            elif row.spy_rel <= RULES['spy_exit_buffer']:band=False
        spy.append(float(band))
    return pd.DataFrame({'perry':perry,'spy_band':spy},index=ind.index)

def equity_curve(closes,signal,delay=2,cost=0.001):
    """delay=1 same-close optimistic, delay=2 next-session-close executable approximation.
    Daily returns d-1 to d are earned by the position set at close d-1.
    Signal at close t cannot trade that close after observation; next-close t+1
    executes for returns t+1 -> t+2, hence exposure for return date d = signal[d-2].
    """
    r=closes[['TQQQ','BIL']].pct_change().fillna(0)
    w=signal.shift(delay).fillna(0).clip(0,1)
    transition=w.diff().abs().fillna(w.abs())
    gross=w*r.TQQQ+(1-w)*r.BIL
    # Applying fee at the return timestamp is a daily approximation;
    # excludes spreads beyond cost parameter and fund market-impact details.
    net=(1+gross)*(1-cost*transition)-1
    return (1+net).cumprod(),w,net

def summary(curve,ret,exposure):
    n=len(curve);years=(curve.index[-1]-curve.index[0]).days/365.25
    if n<250 or years<=1:raise ValueError('Insufficient years')
    peak=curve.cummax()
    vol=ret.std(ddof=1)*np.sqrt(252)
    return {'cagr':float(curve.iloc[-1]**(1/years)-1),
            'max_drawdown':float((curve/peak-1).min()),
            'volatility':float(vol),
            'sharpe_vs_zero':float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)) if ret.std(ddof=1)>0 else None,
            'fraction_in_tqqq':float(exposure.mean()),
            'roundtrips_per_year':float(exposure.diff().abs().sum()/2/years),
            'terminal_multiple':float(curve.iloc[-1])}

def compare(closes,indicator_panel=None):
    ind=indicators(closes) if indicator_panel is None else indicator_panel
    sig=signals(ind)
    start=max(pd.Timestamp('2010-02-11'),closes.TQQQ.first_valid_index())
    p=closes.loc[start:].dropna(subset=list(TICKERS)).copy()
    if len(p)<1000:raise ValueError('Data coverage insufficient')
    sig=sig.reindex(p.index).fillna(0)
    output={};curves={}
    for name,s,delay in [
        ('perry_next_close',sig.perry,2),('perry_same_close_optimistic',sig.perry,1),
        ('spy_band_next_close',sig.spy_band,2),('spy_band_same_close_optimistic',sig.spy_band,1),
        ('tqqq_buy_hold',pd.Series(1.,index=p.index),1),
        ('bil_cash',pd.Series(0.,index=p.index),1)]:
        curve,w,ret=equity_curve(p,s,delay=delay)
        output[name]=summary(curve,ret,w)
        curves[name]=curve
    return output,pd.DataFrame(curves),sig

def bootstrap(real_prices, reps=150, path_days=2520, blocks=(21,63), seed=20261115):
    """Paired joint block resampling; sensitivity only, NOT independent holdout or objective forecast."""
    rng=np.random.default_rng(seed)
    daily=real_prices[list(TICKERS)].pct_change().dropna().to_numpy()
    rows=[];m=len(daily)
    for b in blocks:
        for i in range(reps):
            starts=rng.integers(0,m-b,size=int(np.ceil((path_days+280)/b)))
            ix=np.concatenate([np.arange(k,k+b) for k in starts])[:path_days+280]
            ret=daily[ix]
            synthetic=np.vstack([np.ones((1,4)),np.cumprod(1+ret,axis=0)])*100
            dates=pd.bdate_range('2000-01-03',periods=len(synthetic))
            q=pd.DataFrame(synthetic,index=dates,columns=TICKERS)
            # No actual period start, we exclude 280-day feature warmup.
            q=q.iloc[280:].copy()
            # Indicators need 250 warmup. Keep complete then slice outcome.
            # Rebuild from all synthetic prices and take last path_days steps.
            allq=pd.DataFrame(synthetic,index=dates,columns=TICKERS)
            inds=indicators(allq);ss=signals(inds)
            q=allq.iloc[280:]
            ss=ss.iloc[280:]
            for name,s in [('perry',ss.perry),('spy_band',ss.spy_band),('buy_hold',pd.Series(1.,index=q.index))]:
                ec,w,r=equity_curve(q,s,delay=2)
                stats=summary(ec,r,w)
                rows.append({'block':b,'rep':i,'strategy':name,'cagr':stats['cagr'],'max_drawdown':stats['max_drawdown'],
                             'terminal_multiple':stats['terminal_multiple']})
    frame=pd.DataFrame(rows)
    out={}
    for b in blocks:
        f=frame.loc[frame.block==b]
        base=f.loc[f.strategy=='spy_band'].set_index('rep')
        test=f.loc[f.strategy=='perry'].set_index('rep')
        out[str(b)]={'paths':reps,
            'perry_cagr_p05_p50_p95':[float(v) for v in np.quantile(test.cagr,[.05,.5,.95])],
            'perry_dd_p05_p50_p95':[float(v) for v in np.quantile(test.max_drawdown,[.05,.5,.95])],
            'perry_beat_spy_band_frequency':float((test.terminal_multiple>base.terminal_multiple).mean())}
    return out

def run(output, reps=150):
    allp,real=prices_from_yahoo()
    stats,curves,sigs=compare(allp)
    result={'description':'Will Perry 3-gate reconstruction; BestFolio description, not verified original author code',
            'sample':'real TQQQ, QQQ, SPY, BIL adjusted daily prices; no pre-inception synthetic LETF',
            'start':str(curves.index[0].date()),'end':str(curves.index[-1].date()),
            'source':'yfinance auto-adjust daily Yahoo prices, fetched at research run',
            'prices_hash':hashlib.sha256(real.to_csv(float_format='%.10f').encode()).hexdigest(),
            'gate_rules':RULES,'method':'next-close delay=2; optimistic same-close delay=1',
            'data_caveats':['2010+ is not an untouched out-of-sample period for published 1985-2026 strategy',
              'QQQ is a Nasdaq100 ETF signal proxy, not original NDX index series',
              'BIL is cash-bill ETF proxy; rates and taxes vary over time',
              'Yahoo adjusted ETF prices reflect a particular data vendor and revisions',
              'Trades estimated at 10 bps per equity transition; no real fills or bid/ask',
              'Two-day confirmation and drawdown hystereses reconstructed from BestFolio',
              'Monte Carlo resamples known sample and cannot validate strategy edge'],
            'historical_comparison':stats,
            'paired_block_bootstrap':bootstrap(real,reps=reps),
            'last_signal_date':str(sigs.index[-1].date()),
            'last_signal_raw':{k:float(v) for k,v in sigs.iloc[-1].to_dict().items()}}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    (out/'historical_equity_curves.csv').write_text(curves.to_csv(float_format='%.8f'))
    print('REAL_DATA',result['start'],result['end'],'prices_hash',result['prices_hash'])
    for k,v in stats.items():print('RESULT',k,'CAGR',round(100*v['cagr'],2),'MDD',round(100*v['max_drawdown'],2),'Sharpe0',None if v['sharpe_vs_zero'] is None else round(v['sharpe_vs_zero'],3),'Invested',round(v['fraction_in_tqqq'],3))
    print('BOOTSTRAP',json.dumps(result['paired_block_bootstrap']))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/reddit_will_perry_20261115')
    p.add_argument('--reps',type=int,default=150)
    args=p.parse_args()
    run(args.output,reps=args.reps)
