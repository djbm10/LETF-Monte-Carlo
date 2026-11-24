"""Point-in-time SEC XBRL fundamental snapshots. Free, but NOT Compustat.

SEC companyfacts include late-filed amendments; never look ahead by period end.
The SEC requires a declared User-Agent; configure SEC_USER_AGENT in secrets/env.
"""
from __future__ import annotations
import os
import time
import requests
import pandas as pd


def get_companyfacts(cik: int, user_agent: str | None = None, session=None) -> dict:
    agent = user_agent or os.environ.get("SEC_USER_AGENT")
    if not agent or "@" not in agent or "example.com" in agent:
        raise ValueError("Provide a real SEC_USER_AGENT contact email; do not use an anonymous bot")
    if not 0 < int(cik) <= 9999999999:
        raise ValueError("Invalid CIK")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    sess = session or requests.Session()
    response = sess.get(url, headers={"User-Agent":agent, "Accept-Encoding":"gzip, deflate"}, timeout=30)
    response.raise_for_status()
    return response.json()


def facts_pit(facts: dict, tag: str, asof: str, taxonomy: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """Use filed date as conservative daily availability; no period-end lookahead.

    Multiple values can share a fiscal end and accession. For amendments or
    revisions we prefer the *latest filing known by asof*, never the final
    value reported after asof. Next session is the earliest tradable day.
    """
    try:
        entries = facts["facts"][taxonomy][tag]["units"][unit]
    except KeyError:
        return pd.DataFrame(columns=["end", "filed", "accn", "form", "value"])
    frame = pd.DataFrame(entries)
    required = {"end", "filed", "accn", "val", "form"}
    if not required.issubset(frame.columns):
        raise ValueError("Malformed SEC companyfacts tag")
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["value"] = pd.to_numeric(frame["val"], errors="coerce")
    cutoff = pd.Timestamp(asof)
    frame = frame.loc[(frame["filed"] <= cutoff) & (frame["end"] <= cutoff)].dropna(subset=["value"])
    frame = frame.sort_values(["end", "filed", "accn"]).drop_duplicates(["end"], keep="last")
    return frame[["end", "filed", "accn", "form", "value"]].reset_index(drop=True)


def latest_pit_metric(facts: dict, tag: str, asof: str, **kwargs) -> dict | None:
    table = facts_pit(facts, tag, asof, **kwargs)
    if table.empty:
        return None
    return table.iloc[-1].to_dict()
