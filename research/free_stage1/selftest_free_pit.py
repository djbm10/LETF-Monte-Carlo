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
    test_sec_formation_panel()
    test_sec_fsd()
    print("FREE PIT SELFTEST PASS")
