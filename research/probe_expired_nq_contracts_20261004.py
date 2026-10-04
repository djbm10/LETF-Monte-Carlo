from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

OUT = Path("results/actual_futures_contract_probe_20261004")
OUT.mkdir(parents=True, exist_ok=True)

YF_SYMBOLS = [
    ("NQZ24.CME", "2024-06-01", "2025-01-15"),
    ("NQH20.CME", "2019-09-01", "2020-04-15"),
    ("NQZ10.CME", "2010-06-01", "2011-01-15"),
    ("NQ=F", "2010-06-01", "2011-01-15"),
]

def probe_yf(symbol: str, start: str, end: str) -> dict:
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result = {
            "symbol": symbol,
            "source": "yfinance",
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "start": None if df.empty else str(pd.Timestamp(df.index.min()).date()),
            "end": None if df.empty else str(pd.Timestamp(df.index.max()).date()),
        }
        if not df.empty:
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    result[f"{col}_first"] = None if s.empty else float(s.iloc[0])
                    result[f"{col}_last"] = None if s.empty else float(s.iloc[-1])
        return result
    except Exception as exc:
        return {"symbol": symbol, "source": "yfinance", "error": repr(exc)}

def probe_stooq() -> dict:
    url = "https://stooq.com/q/d/l/?s=nq.f&i=d&d1=20240101&d2=20241231"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        txt = r.text
        return {
            "source": "stooq",
            "url": url,
            "status_code": r.status_code,
            "chars": len(txt),
            "head": txt[:300],
        }
    except Exception as exc:
        return {"source": "stooq", "url": url, "error": repr(exc)}

rows = [probe_yf(*x) for x in YF_SYMBOLS]
rows.append(probe_stooq())
(OUT / "probe.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
