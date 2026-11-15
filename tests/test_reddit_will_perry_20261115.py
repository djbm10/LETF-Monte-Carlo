import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "research"))
import numpy as np
import pandas as pd
import reddit_will_perry_20261115 as run

def test_two_session_confirmation():
    idx=pd.bdate_range('2020-01-01',periods=11)
    data=pd.DataFrame({'vol20':[.2]*11,'ratio':[.9]*11,'dd':[-.05]*3+[-.25]*3+[-.05]*5,'spy_rel':[0]*11},index=idx)
    sig=run.signals(data)
    assert sig.perry.iloc[0]==0
    assert sig.perry.iloc[1]==1
    assert sig.perry.iloc[3]==1
    assert sig.perry.iloc[4]==0
    assert sig.perry.iloc[6]==0
    assert sig.perry.iloc[7]==1

def test_delayed_execution_cannot_use_same_bar():
    idx=pd.bdate_range('2020-01-01',periods=4)
    close=pd.DataFrame({'TQQQ':[100,100,150,150],'BIL':[100]*4},index=idx)
    s=pd.Series([0,1,1,1],index=idx,dtype=float)
    later,weights,ret=run.equity_curve(close,s,delay=2,cost=0)
    assert weights.iloc[2]==0 and later.iloc[-1]==1
    look,w2,ret2=run.equity_curve(close,s,delay=1,cost=0)
    assert look.iloc[-1]==1.5

def test_hysteresis_200day_band():
    idx=pd.bdate_range('2020-01-01',periods=6)
    d=pd.DataFrame({'vol20':[.5]*6,'ratio':[1.5]*6,'dd':[-.4]*6,
                    'spy_rel':[.03,.05,.01,-.02,-.04,.0]},index=idx)
    sig=run.signals(d)
    assert sig.spy_band.tolist()==[0,1,1,1,0,0]

def test_no_future_data_changes_past_signal():
    idx=pd.bdate_range('2020-01-01',periods=300)
    x=pd.DataFrame({'QQQ':np.linspace(100,130,300), 'SPY':np.linspace(100,115,300)},index=idx)
    original=run.signals(run.indicators(x)).iloc[-2].perry
    x.iloc[-1,0]=10
    changed=run.signals(run.indicators(x)).iloc[-2].perry
    assert original==changed
