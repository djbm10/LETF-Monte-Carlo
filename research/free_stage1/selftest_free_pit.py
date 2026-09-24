from __future__ import annotations
import tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sec_item1_network as net
import sec_fsd_pit as fsd
import sec_formation_features as formfeat



def test_item1_table_heading():
    # Regression: modern inline-XBRL filings can put the real Item 1 heading
    # inside a layout table. Table text must survive HTML normalization.
    body = " ".join(["cloud retail logistics infrastructure services competition"] * 80)
    html = f"""
    <html><body>
      <table><tr><td>Item 1.</td><td>Business</td></tr></table>
      <p>{body}</p>
      <p>Item 1A. Risk Factors</p>
      <p>risk text</p>
    </body></html>
    """
    item1 = net.extract_item1(f"<DOCUMENT><TYPE>10-K<TEXT>{html}</TEXT></DOCUMENT>")
    assert item1 is not None
    assert len(item1.split()) >= 250
    assert "cloud retail logistics" in item1

def test_network():
    docs=[]
    themes=[
        "semiconductor memory data center high bandwidth compute accelerator chip fabrication capacity",
        "cloud software recurring subscription cybersecurity identity endpoint platform enterprise",
        "industrial power cooling electrical data center infrastructure capacity transformer",
        "consumer retail apparel stores ecommerce merchandise brand inventory",
    ]
    for i in range(40):
        theme=themes[i%len(themes)]
        text=("Item 1 Business "+theme+" ")*120
        docs.append({
            "cik":str(100000+i).zfill(10),
            "filing_date":pd.Timestamp("2022-03-01")+pd.Timedelta(days=i%20),
            "filename":f"fake{i}.txt",
            "item1":text,
            "word_count":len(text.split()),
            "error":None,
        })
    f=pd.DataFrame(docs)
    sample=net.latest_asof(f,pd.Timestamp("2022-06-30"))
    m,e,manifest=net.build_network(sample,pd.Timestamp("2022-06-30"),pair_density=.10,max_features=5000)
    assert len(m)==40
    assert manifest["n_firms"]==40
    assert 0 < len(e) < 40*39
    assert m["peer_count"].max()>0
    assert (m["source_filing_date"]<=pd.Timestamp("2022-06-30")).all()





def test_sec_amendment_pit():
    sub = pd.DataFrame([
        {
            "adsh": "orig", "cik": "0000000003", "name": "Example A",
            "form": "10-K", "filed": pd.Timestamp("2023-02-15"),
            "period": pd.Timestamp("2022-12-31"), "fy": 2022, "fp": "FY", "sic": 3571,
        },
        {
            "adsh": "amnd", "cik": "0000000003", "name": "Example A",
            "form": "10-K/A", "filed": pd.Timestamp("2023-04-15"),
            "period": pd.Timestamp("2022-12-31"), "fy": 2022, "fp": "FY", "sic": 3571,
        },
    ])
    num = pd.DataFrame([
        {
            "adsh": "orig", "tag": "Revenues", "ddate": pd.Timestamp("2022-12-31"),
            "qtrs": 4, "uom": "USD", "value": 100.0, "coreg": np.nan, "segments": 0,
        },
        {
            "adsh": "orig", "tag": "Assets", "ddate": pd.Timestamp("2022-12-31"),
            "qtrs": 0, "uom": "USD", "value": 500.0, "coreg": np.nan, "segments": 0,
        },
        {
            "adsh": "amnd", "tag": "Revenues", "ddate": pd.Timestamp("2022-12-31"),
            "qtrs": 4, "uom": "USD", "value": 105.0, "coreg": np.nan, "segments": 0,
        },
    ])
    panel = fsd.apply_amendment_carryforward(fsd.canonicalize_quarter(sub, num))
    panel["information_date"] = panel["filed"]
    before = formfeat.build_formation_panel(panel, [pd.Timestamp("2023-03-31")])
    after = formfeat.build_formation_panel(panel, [pd.Timestamp("2023-06-30")])
    assert before.iloc[0].revenue == 100.0
    assert before.iloc[0].form_original == "10-K"
    assert after.iloc[0].revenue == 105.0
    assert after.iloc[0].assets == 500.0
    assert after.iloc[0].form_original == "10-K/A"
    assert bool(after.iloc[0].amended)

def test_sec_flow_period_strictness():
    sub = pd.DataFrame([{
        "adsh": "q1", "cik": "0000000002", "name": "Example Q", "form": "10-Q",
        "filed": pd.Timestamp("2023-05-01"), "period": pd.Timestamp("2023-03-31"),
        "fy": 2023, "fp": "Q1", "sic": 3571,
    }])
    # A qtrs=2 YTD revenue must not masquerade as a single-quarter 10-Q flow.
    num = pd.DataFrame([
        {
            "adsh": "q1",
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "ddate": pd.Timestamp("2023-03-31"),
            "qtrs": 2,
            "uom": "USD",
            "value": 200.0,
            "coreg": np.nan,
            "segments": 0,
        },
        {
            "adsh": "q1",
            "tag": "Assets",
            "ddate": pd.Timestamp("2023-03-31"),
            "qtrs": 0,
            "uom": "USD",
            "value": 500.0,
            "coreg": np.nan,
            "segments": 0,
        },
    ])
    out = fsd.canonicalize_quarter(sub, num)
    assert len(out) == 1
    assert pd.isna(out.iloc[0].revenue)
    assert pd.isna(out.iloc[0].revenue_qtrs)
    assert out.iloc[0].assets == 500.0

def test_sec_formation_panel():
    panel = pd.DataFrame([
        {
            "cik": "0000000001",
            "information_date": pd.Timestamp("2022-02-15"),
            "period": pd.Timestamp("2021-12-31"),
            "form": "10-K",
            "revenue": 100.0,
            "gross_margin": 0.40,
            "operating_margin": 0.20,
            "fcf_margin": 0.15,
            "leverage": 0.30,
            "shares": 10.0,
            "revenue_growth_yoy": 0.10,
            "gross_margin_change_yoy": 0.01,
            "operating_margin_change_yoy": 0.02,
            "share_growth_yoy": 0.00,
            "asset_growth_yoy": 0.05,
        },
        {
            "cik": "0000000001",
            "information_date": pd.Timestamp("2022-05-10"),
            "period": pd.Timestamp("2022-03-31"),
            "form": "10-Q",
            "revenue": 110.0,
            "gross_margin": 0.42,
            "operating_margin": 0.21,
            "fcf_margin": 0.16,
            "leverage": 0.29,
            "shares": 10.1,
            "revenue_growth_yoy": 0.12,
            "gross_margin_change_yoy": 0.02,
            "operating_margin_change_yoy": 0.01,
            "share_growth_yoy": 0.01,
            "asset_growth_yoy": 0.06,
        },
    ])
    out = formfeat.build_formation_panel(
        panel,
        [pd.Timestamp("2022-03-31"), pd.Timestamp("2022-06-30")],
    )
    q1 = out[out.formation_date == pd.Timestamp("2022-03-31")].iloc[0]
    q2 = out[out.formation_date == pd.Timestamp("2022-06-30")].iloc[0]
    assert q1.information_date == pd.Timestamp("2022-02-15")
    assert q1.revenue == 100.0
    assert q2.information_date == pd.Timestamp("2022-05-10")
    assert q2.revenue == 110.0
    assert (out.information_date <= out.formation_date).all()


def test_frozen_source_invariants():
    bottleneck = (HERE.parent / "quantconnect" / "BottleneckWinnerFreePIT" / "main.py").read_text()
    leaps = (HERE.parent / "quantconnect" / "RealHistoricalLeaps" / "main.py").read_text()

    # Stage-1 accounting factors must be sourced from the SEC formation panel.
    assert "fundamentals_url" in bottleneck
    assert "_sec_fundamental_asof" in bottleneck
    assert "self.set_warm_up(252, Resolution.DAILY)" in bottleneck
    assert "revenue_qtrs" in bottleneck
    assert ".financial_statements" not in bottleneck
    assert "income_statement" not in bottleneck

    # Daily-chain selections must execute on the following session's open,
    # never as an immediate same-close market order.
    assert "market_on_open_order" in leaps
    assert "self.market_order(" not in leaps
    assert "ENTRY_SELECTION_FOR_NEXT_OPEN" in leaps

def test_sec_fsd():
    sub=pd.DataFrame([{
        "adsh":"0001","cik":"0000320193","name":"Example","form":"10-K",
        "filed":pd.Timestamp("2022-10-28"),"period":pd.Timestamp("2022-09-24"),
        "fy":2022,"fp":"FY","sic":3571
    }])
    rows=[]
    vals={
      "RevenueFromContractWithCustomerExcludingAssessedTax":1000,
      "GrossProfit":430,
      "OperatingIncomeLoss":280,
      "NetIncomeLoss":220,
      "Assets":3500,
      "Liabilities":1200,
      "NetCashProvidedByUsedInOperatingActivities":310,
      "PaymentsToAcquirePropertyPlantAndEquipment":-90,
      "CommonStockSharesOutstanding":160,
    }
    for tag,value in vals.items():
        rows.append({
            "adsh":"0001","tag":tag,"ddate":pd.Timestamp("2022-09-24"),
            "qtrs":4 if tag not in ("Assets","Liabilities","CommonStockSharesOutstanding") else 0,
            "uom":"USD","value":value,"coreg":np.nan,"segments":0
        })
    num=pd.DataFrame(rows)
    c=fsd.canonicalize_quarter(sub,num)
    assert len(c)==1
    r=c.iloc[0]
    assert r.revenue==1000
    assert r.gross_profit==430
    assert r.assets==3500
    assert r.filed==pd.Timestamp("2022-10-28")

    # Regression: CIK must survive YoY derivation across pandas versions.
    older=c.copy()
    older["adsh"]="0000"
    older["filed"]=pd.Timestamp("2021-10-29")
    older["period"]=pd.Timestamp("2021-09-25")
    older["revenue"]=900.0
    older["gross_profit"]=360.0
    older["shares"]=150.0
    panel=pd.concat([older,c],ignore_index=True)
    d=fsd.derive_features(panel)
    assert "cik" in d.columns
    assert d.cik.nunique()==1
    latest=d.sort_values("filed").iloc[-1]
    assert abs(latest.revenue_growth_yoy-(1000/900-1)) < 1e-12


if __name__=="__main__":
    test_item1_table_heading()
    test_network()
    test_sec_amendment_pit()
    test_sec_flow_period_strictness()
    test_sec_formation_panel()
    test_frozen_source_invariants()
    test_sec_fsd()
    print("FREE PIT SELFTEST PASS")
