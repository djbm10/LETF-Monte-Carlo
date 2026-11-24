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
import local_sp500_bottleneck_backtest as local_bottleneck
import direct_sec_cache_completion as direct_sec_completion



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
    stale_text=("Item 1 Business obsolete legacy dormant filer text ")*120
    docs.append({
        "cik":"0000999999",
        "filing_date":pd.Timestamp("2019-03-01"),
        "filename":"stale.txt",
        "item1":stale_text,
        "word_count":len(stale_text.split()),
        "error":None,
    })
    f=pd.DataFrame(docs)
    sample=net.latest_asof(f,pd.Timestamp("2022-06-30"))
    assert len(sample)==40
    assert "0000999999" not in set(sample.cik)
    m,e,manifest=net.build_network(sample,pd.Timestamp("2022-06-30"),pair_density=.10,max_features=5000)
    assert len(m)==40
    assert manifest["n_firms"]==40
    assert 0 < len(e) < 40*39
    assert m["peer_count"].max()>0
    assert (m["source_filing_date"]<=pd.Timestamp("2022-06-30")).all()





def test_local_bottleneck_exact_snapshot_and_period_metrics():
    frame = pd.DataFrame([
        {"cik":"0000000001","formation_date":pd.Timestamp("2014-12-31"),"value":1.0},
        {"cik":"0000000002","formation_date":pd.Timestamp("2014-12-31"),"value":2.0},
        {"cik":"0000000002","formation_date":pd.Timestamp("2015-03-31"),"value":3.0},
    ])
    snap = local_bottleneck.exact_snapshot_by_cik(frame, pd.Timestamp("2015-03-31"))
    # CIK 1 must not be carried forward from a prior network snapshot.
    assert set(snap.cik) == {"0000000002"}
    assert snap.iloc[0].value == 3.0
    assert local_bottleneck.prior_quarter_snapshot_date(pd.Timestamp("2015-04-01")) == pd.Timestamp("2015-03-31")
    assert local_bottleneck.prior_quarter_snapshot_date(pd.Timestamp("2015-01-02")) == pd.Timestamp("2014-12-31")

    curve = pd.Series(
        [9_985_000.0, 10_050_000.0, 10_100_000.0, 10_200_000.0],
        index=pd.to_datetime(["2014-12-31","2015-01-02","2015-06-30","2015-12-31"]),
    )
    full = local_bottleneck.metrics_from_curve(
        curve, baseline_value=10_000_000.0, baseline_date=pd.Timestamp("2014-12-31")
    )
    assert abs(full["total_return"] - 0.02) < 1e-12
    val = local_bottleneck.period_metrics(curve, "2015-01-01", "2015-12-31")
    # Validation must use the last 2014 NAV as its baseline, not discard the
    # first 2015 return observation.
    assert abs(val["total_return"] - (10_200_000.0 / 9_985_000.0 - 1.0)) < 1e-12

    # A recycled ticker that changes CIK is a different security. A rebalance
    # from old-issuer AGN to new-issuer AGN must incur an exit and an entry,
    # rather than netting the two identities as if the holding were unchanged.
    old_key = local_bottleneck.security_identity_key("AGN", "0000850693")
    new_key = local_bottleneck.security_identity_key("AGN", "0001578845")
    assert old_key != new_key
    source = (HERE / "local_sp500_bottleneck_backtest.py").read_text()
    assert "if len(selected) < top_n:" in source
    assert "min(top_n, MIN_CROSS_SECTION)" not in source
    nav_after, targets, gross, turnover = local_bottleneck.solve_rebalance(
        100.0, {old_key: 100.0}, [new_key], 0.0015
    )
    assert gross > 199.0
    assert turnover > 1.99
    assert new_key in targets
    assert nav_after < 100.0


def test_direct_sec_cache_completion_uses_snapshot_gaps():
    membership = pd.DataFrame([
        {"symbol":"AAA","cik":"0000000001","date_added":pd.Timestamp("2009-01-01"),"date_removed":pd.NaT},
        {"symbol":"BBB","cik":"0000000002","date_added":pd.Timestamp("2009-01-01"),"date_removed":pd.NaT},
    ])
    cached = pd.DataFrame([
        {
            "cik":"0000000001","filing_date":pd.Timestamp("2009-02-15"),
            "filename":"a.txt","item1":"business text "*300,"word_count":600,
            "error":None,"source":"SEC_DIRECT_CACHE",
        },
        {
            # CIK 2 exists in the cache, but this filing is too old for the
            # later formation. Presence-by-CIK alone must not mark it complete.
            "cik":"0000000002","filing_date":pd.Timestamp("2008-01-15"),
            "filename":"b.txt","item1":"business text "*300,"word_count":600,
            "error":None,"source":"SEC_DIRECT_CACHE",
        },
    ])
    missing, audit = direct_sec_completion.missing_snapshot_ciks(
        membership,
        cached,
        [pd.Timestamp("2009-03-31"), pd.Timestamp("2010-09-30")],
        max_item1_age_days=550,
    )
    assert "0000000002" in missing
    assert "0000000001" in missing  # valid in 2009, >550 days old by 2010-09-30
    assert len(audit) == 2
    assert audit.iloc[-1].missing_snapshot_ciks == 2


def test_curated_reused_ticker_identity_provenance():
    # Curated/date-resolved aliases may reuse a ticker across non-overlapping
    # issuer eras; legacy membership without that provenance must still exclude.
    rows = pd.DataFrame([
        {
            "symbol": "AGN", "cik": "0000850693",
            "date_added": "2009-01-01", "date_removed": "2015-03-17",
            "alias_source": "CURATED_ALIAS_RESOLVED",
        },
        {
            "symbol": "AGN", "cik": "0001578845",
            "date_added": "2015-06-15", "date_removed": "2019-01-01",
            "alias_source": "LAWCAL_SAME_CIK_IDENTITY_WINDOW",
        },
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "membership.csv"
        rows.to_csv(p, index=False)
        kept = local_bottleneck.load_membership(p)
        assert len(kept) == 2
        assert kept.attrs.get("reused_symbol_exclusions") == []

        legacy = rows.drop(columns=["alias_source"])
        legacy.to_csv(p, index=False)
        excluded = local_bottleneck.load_membership(p)
        assert excluded.empty
        assert excluded.attrs.get("reused_symbol_exclusions") == ["AGN"]


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
    amended_row = panel[panel["amended"].fillna(False).astype(bool)].iloc[0]
    assert amended_row.assets == 500.0
    panel["information_date"] = panel["filed"]
    before = formfeat.build_formation_panel(panel, [pd.Timestamp("2023-03-31")])
    after = formfeat.build_formation_panel(panel, [pd.Timestamp("2023-06-30")])
    assert before.iloc[0].revenue == 100.0
    assert before.iloc[0].form_original == "10-K"
    assert after.iloc[0].revenue == 105.0
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

def test_direct_sec_targeted_gap_filings():
    membership = pd.DataFrame([
        {"symbol":"AAA","cik":"0000000001","date_added":pd.Timestamp("2008-01-01"),"date_removed":pd.NaT},
        {"symbol":"BBB","cik":"0000000002","date_added":pd.Timestamp("2008-01-01"),"date_removed":pd.NaT},
    ])
    cached = pd.DataFrame([
        {
            "cik":"0000000001","filing_date":pd.Timestamp("2010-02-15"),
            "filename":"cached-a.txt","item1":"business "*300,
            "word_count":300,"error":None,
        }
    ])
    index = pd.DataFrame([
        {"cik":"0000000001","date_filed":pd.Timestamp("2008-02-15"),"filename":"old-a.txt"},
        {"cik":"0000000001","date_filed":pd.Timestamp("2010-02-15"),"filename":"same-a.txt"},
        {"cik":"0000000002","date_filed":pd.Timestamp("2008-02-20"),"filename":"old-b.txt"},
        {"cik":"0000000002","date_filed":pd.Timestamp("2009-02-20"),"filename":"b09.txt"},
        {"cik":"0000000002","date_filed":pd.Timestamp("2010-02-20"),"filename":"b10.txt"},
    ])
    work, audit = direct_sec_completion.required_gap_filings(
        membership, cached, index,
        [pd.Timestamp("2010-03-31")],
        max_item1_age_days=550,
    )
    # AAA is covered by cache and must not be fetched. BBB is missing; only
    # its 2009/2010 filings inside the 550-day snapshot window are relevant.
    assert set(work.cik) == {"0000000002"}
    assert set(work.filename) == {"b09.txt","b10.txt"}
    assert "old-b.txt" not in set(work.filename)
    assert int(audit.iloc[0].missing_snapshot_ciks) == 1


def test_sec_structural_empty_source_quarter():
    assert fsd.is_structural_empty_source_quarter(2009, 1)
    assert not fsd.is_structural_empty_source_quarter(2009, 2)
    assert not fsd.is_structural_empty_source_quarter(2010, 1)


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
    test_local_bottleneck_exact_snapshot_and_period_metrics()
    test_direct_sec_cache_completion_uses_snapshot_gaps()
    test_curated_reused_ticker_identity_provenance()
    test_sec_amendment_pit()
    test_sec_flow_period_strictness()
    test_sec_formation_panel()
    test_frozen_source_invariants()
    test_direct_sec_targeted_gap_filings()
    test_sec_structural_empty_source_quarter()
    test_sec_fsd()
    print("FREE PIT SELFTEST PASS")
