from __future__ import annotations
"""
WRDS extraction helper for the Bottleneck Winner / LEAPS project.

This file does NOT contain credentials. Run it only in an environment where the
user has authorized WRDS access. It inspects available schemas first so we do
not hallucinate PIT table names that may differ by subscription/version.
"""
import argparse, json
from pathlib import Path
import pandas as pd

def connect():
    import wrds
    return wrds.Connection()

def inspect(db,out):
    libs=set(db.list_libraries())
    wanted=["crsp","comp","comp_pit","optionm","optionm_all","optionmsamp_us","ibes","wrdsapps"]
    report={"libraries_present":[x for x in wanted if x in libs],"tables":{}}
    for lib in report["libraries_present"]:
        try:report["tables"][lib]=db.list_tables(library=lib)
        except Exception as e:report["tables"][lib]={"error":repr(e)}
    Path(out).write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))

def extract_crsp_monthly(db,start,end,out):
    q=f"""
    select a.permno, a.date, a.ret, a.dlret,
           abs(a.prc)*a.shrout as me, a.prc, a.shrout,
           n.shrcd, n.exchcd
    from crsp.msf a
    left join crsp.msenames n
      on a.permno=n.permno
     and a.date between n.namedt and coalesce(n.nameendt, '2099-12-31')
    where a.date between '{start}' and '{end}'
      and n.shrcd in (10,11)
      and n.exchcd in (1,2,3)
    """
    x=db.raw_sql(q,date_cols=["date"]);x.to_parquet(out,index=False);return x

def extract_ccm(db,out):
    q="""
    select gvkey, lpermno as permno, linkdt,
           coalesce(linkenddt,'2099-12-31') as linkenddt, linktype, linkprim
    from crsp.ccmxpf_lnkhist
    where lpermno is not null
      and linktype in ('LC','LU')
      and linkprim in ('P','C')
    """
    x=db.raw_sql(q,date_cols=["linkdt","linkenddt"]);x.to_parquet(out,index=False);return x

def extract_optionmetrics_flat(db,tickers,start_year,end_year,out):
    # Discover SECIDs from name history. If your subscription exposes a different
    # table name, inspect first and adjust explicitly rather than guessing.
    names=db.raw_sql("""
      select secid, ticker, effect_date
      from optionm.secnmd
      where ticker is not null
      order by secid,effect_date
    """,date_cols=["effect_date"])
    tickers=[t.upper() for t in tickers]
    ids=names[names.ticker.str.upper().isin(tickers)].secid.dropna().unique().tolist()
    if not ids:raise RuntimeError(f"No OptionMetrics SECIDs for {tickers}")
    parts=[]
    idcsv=",".join(str(int(x)) for x in ids)
    for y in range(start_year,end_year+1):
        # Raw Option Price File. Strike is stored in OptionMetrics raw units;
        # backtest engine normalizes the common *1000 convention.
        q=f"""
        select o.secid,o.date,o.exdate,o.cp_flag,o.strike_price,
               o.best_bid,o.best_offer,o.volume,o.open_interest,o.optionid,
               o.impl_volatility,o.delta,o.ss_flag,
               p.close as underlying_price
        from optionm.opprcd{y} o
        left join optionm.secprd{y} p
          on o.secid=p.secid and o.date=p.date
        where o.secid in ({idcsv})
          and o.cp_flag='C'
          and o.best_offer>0
          and o.date between '{y}-01-01' and '{y}-12-31'
        """
        try:
            z=db.raw_sql(q,date_cols=["date","exdate"])
            if len(z):parts.append(z)
        except Exception as e:
            print(f"year {y} skipped/error: {e}")
    if not parts:raise RuntimeError("No OptionMetrics rows extracted.")
    x=pd.concat(parts,ignore_index=True)
    # Attach contemporaneous ticker by last name record <= date.
    map_parts=[]
    for secid,g in x.groupby("secid"):
        nm=names[names.secid==secid].sort_values("effect_date")
        if nm.empty:continue
        q=pd.merge_asof(g.sort_values("date"),nm[["effect_date","ticker"]],left_on="date",right_on="effect_date",direction="backward")
        map_parts.append(q)
    x=pd.concat(map_parts,ignore_index=True)
    x.to_parquet(out,index=False);return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--inspect",action="store_true")
    ap.add_argument("--start",default="1989-01-01");ap.add_argument("--end",default="2026-09-21")
    ap.add_argument("--outdir",default="licensed_inputs");ap.add_argument("--option-tickers",nargs="*",default=["SPY","QQQ"])
    ap.add_argument("--extract-crsp",action="store_true");ap.add_argument("--extract-ccm",action="store_true");ap.add_argument("--extract-options",action="store_true")
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);db=connect()
    if a.inspect:inspect(db,out/"wrds_schema_inventory.json")
    if a.extract_crsp:extract_crsp_monthly(db,a.start,a.end,out/"crsp_monthly.parquet")
    if a.extract_ccm:extract_ccm(db,out/"ccm_links.parquet")
    if a.extract_options:extract_optionmetrics_flat(db,a.option_tickers,int(a.start[:4]),int(a.end[:4]),out/"optionmetrics_leaps_flat.parquet")
if __name__=="__main__":main()
