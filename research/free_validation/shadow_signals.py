"""Append-only forward signal evidence, NOT a live brokerage/paper trading account.

Each dated record is created only from closing data available at that close.
No past record is edited. All forward fill/P&L/execution must be evaluated later.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from hashlib import sha256
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

SECTOR_PAIRS = {"XLK":"ROM", "SOXX":"USD", "XLF":"UYG", "XLV":"RXL",
                "XLE":"DIG", "XLY":"UCC", "XLB":"UYM", "XLI":"UXI",
                "XLP":"UGE", "VNQ":"URE"}


def is_nyse_session(asof: str) -> bool:
    """Exchange calendar, including Good Friday and unscheduled closures when known.

    This is an availability gate, not a guarantee of Yahoo settlement freshness.
    """
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS")
    return bool(cal.is_session(pd.Timestamp(asof)))


def validate_prospective_run(asof: str, observed_at: datetime) -> bool:
    """Only contemporaneous, fully closed trading days may enter the forward ledger.

    A historical replay is allowed only via the explicit --dry-run path.
    Returns False for holidays/weekends without suggesting a missing signal.
    """
    if not is_nyse_session(asof):
        return False
    ny = observed_at.astimezone(ZoneInfo("America/New_York"))
    if asof != ny.date().isoformat():
        raise ValueError("Prospective ledger refuses backdated or future session; use --dry-run to replay history")
    if ny.hour < 19:
        raise ValueError("Prospective ledger refuses to record before 7pm New York time")
    return True


def compute_frozen_signals(close: pd.DataFrame, asof: str) -> dict:
    required = {"QQQ", "SPY", *SECTOR_PAIRS, *SECTOR_PAIRS.values()}
    if not required.issubset(close.columns):
        raise ValueError(f"Missing price columns {required-set(close.columns)}")
    frame = close.copy().sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    dt = pd.Timestamp(asof)
    if dt not in frame.index:
        raise ValueError("Missing as-of session: refusing to use incomplete/previous data")
    frame = frame.loc[:dt]
    if len(frame) < 270 or frame.iloc[-270:][list(required)].isnull().any().any():
        raise ValueError("Need 270 complete prior daily closes for every instrument")
    q = frame.QQQ
    qv = q.pct_change().iloc[-20:].std(ddof=1) * np.sqrt(252)
    if not (np.isfinite(qv) and qv > 0):
        raise ValueError("Invalid volatility")
    exposure = min(3., .35/qv) if q.iloc[-1] > q.iloc[-200:].mean() else 0.
    selected = None
    sec_weight = 0.
    if frame.SPY.iloc[-1] > frame.SPY.iloc[-200:].mean():
        momentum = {sector: float(frame[sector].iloc[-1]/frame[sector].iloc[-64]-1)
                    for sector in SECTOR_PAIRS}
        selected = max(momentum, key=momentum.get)
        leveraged = SECTOR_PAIRS[selected]
        sv = frame[leveraged].pct_change().iloc[-20:].std(ddof=1) * np.sqrt(252)
        if not (np.isfinite(sv) and sv > 0):
            raise ValueError("Invalid leveraged sector volatility")
        sec_weight = min(1., .35/max(sv,.08))
    digest=sha256(frame.iloc[-270:][sorted(required)].to_csv(float_format='%.12f').encode()).hexdigest()
    return {"asof":str(dt.date()), "entry":"NEXT_TRADING_OPEN_NOT_AN_EXECUTION",
            "qqq_35_0_exposure":round(float(exposure),8),
            "qqq_above_200sma":bool(q.iloc[-1]>q.iloc[-200:].mean()),
            "spy_above_200sma":bool(frame.SPY.iloc[-1]>frame.SPY.iloc[-200:].mean()),
            "candidate_sector":selected or "CASH", "candidate_etf":SECTOR_PAIRS[selected] if selected else "CASH",
            "candidate_sector_risk_weight":round(float(sec_weight),8),
            "source_hash":digest, "source":"Yahoo adjusted ETF close (indicative only)",
            "status":"SIGNAL_ONLY_NO_PNL_NO_ORDERS"}


def append_immutable(record: dict, ledger_file: str | Path) -> bool:
    p=Path(ledger_file); p.parent.mkdir(parents=True,exist_ok=True)
    existing = p.read_text().splitlines() if p.exists() else []
    for raw in existing:
        obj=json.loads(raw)
        if obj['asof']==record['asof']:
            if obj!=record:
                raise ValueError("An existing frozen signal changed: would overwrite historical evidence")
            return False
        if obj['asof'] > record['asof']:
            raise ValueError("Cannot backdate prospective signal ledger")
    with p.open('a') as f:
        f.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n')
    return True


def yahoo_closes(asof: str) -> pd.DataFrame:
    import yfinance as yf
    names=sorted({"QQQ","SPY",*SECTOR_PAIRS,*SECTOR_PAIRS.values()})
    cutoff = pd.Timestamp(asof) + pd.Timedelta(days=1)
    d={}
    for start in range(0,len(names),7):
        chunk=names[start:start+7]
        result=yf.download(chunk, start=(pd.Timestamp(asof)-pd.Timedelta(days=500)).date().isoformat(),
                           end=cutoff.date().isoformat(),auto_adjust=True,
                           group_by="ticker",threads=False,progress=False,timeout=40)
        for ticker in chunk:
            try:
                x=result[ticker]['Close']
                if isinstance(x,pd.DataFrame):x=x.iloc[:,0]
                d[ticker]=pd.to_numeric(x,errors='coerce')
            except Exception as e:
                raise RuntimeError(f"Missing provider result for {ticker}") from e
    return pd.DataFrame(d).sort_index()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asof')
    parser.add_argument('--close-file')
    parser.add_argument('--ledger',default='research/forward_signal_ledger.jsonl')
    parser.add_argument('--strict',action='store_true',help='Fail the GitHub job on missing data during a valid trading session')
    parser.add_argument('--dry-run',action='store_true',help='Print historical or current indicative signal without writing the protected ledger')
    args=parser.parse_args()
    ny=datetime.now(ZoneInfo('America/New_York'))
    asof=args.asof or str(ny.date())
    if not args.dry_run:
        if not is_nyse_session(asof):
            print('SKIP: exchange closed on',asof)
            return
        if not args.asof and ny.hour < 19:
            if args.strict:
                raise ValueError('Trading day is not complete; forward journal not ready')
            print('SKIP: not yet after 7pm New York time')
            return
        validate_prospective_run(asof,ny)
    prices=pd.read_csv(args.close_file,index_col=0,parse_dates=True) if args.close_file else yahoo_closes(asof)
    try:
        record=compute_frozen_signals(prices,asof)
    except ValueError as e:
        if args.strict:raise
        print('SKIP: no complete market session for',asof,'reason',str(e))
        return
    if args.dry_run:
        print(json.dumps({'mode':'HISTORICAL_OR_CURRENT_DRY_RUN_NOT_FORWARD_EVIDENCE','record':record},sort_keys=True))
        return
    changed=append_immutable(record,args.ledger)
    print(json.dumps(dict(record_written=changed,record=record),sort_keys=True))

if __name__=='__main__':main()
