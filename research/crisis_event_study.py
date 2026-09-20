from __future__ import annotations
import numpy as np, pandas as pd, math
import research.nasdaq1985_final_tournament as n

EVENTS = {
 'Black_Monday_1987':('1987-08-25','1988-07-31'),
 'Dotcom_2000_02':('2000-03-27','2002-10-09'),
 'GFC_2007_09':('2007-10-09','2009-03-09'),
 'COVID_2020':('2020-02-19','2020-08-18'),
 'Rate_shock_2022':('2022-01-03','2022-12-30'),
}
STRATS = {
 'NDX_1x':('fixed',(1.0,)),
 '2x_200DMA_cash':('sma',(200,2.0,0.0,0.0)),
 'Vol25':('vol',(0.25,)),
 'TV35_0_200DMA':('tv',(200,0.35,0.0,0.30)),
 'TV35_12_200DMA':('tv',(200,0.35,0.12,0.30)),
}

def stats(r):
    g=np.maximum(1+np.asarray(r,float),1e-12)
    eq=np.cumprod(g); dd=eq/np.maximum.accumulate(eq)-1
    return {'return':eq[-1]-1,'max_dd':dd.min(),'ann_vol':np.std(r)*math.sqrt(252)}

def main():
    df=n.load_data(.007); ndx=df.ndx.to_numpy(); rf=df.rf.to_numpy()
    px=np.cumprod(1+ndx)
    qld=n.synth_letf(ndx,rf,2,0.0095,0.0050); tq=n.synth_letf(ndx,rf,3,0.0088,0.0065)
    out={}
    for name,(kind,p) in STRATS.items():
        c=n.Cand(name,kind,p); out[name]=n.evaluate(c,ndx,rf,px,qld,tq)
    rows=[]
    for ev,(a,b) in EVENTS.items():
        mask=(df.index>=pd.Timestamp(a))&(df.index<=pd.Timestamp(b))
        for name,r in out.items():
            s=stats(np.asarray(r)[mask])
            rows.append({'event':ev,'start':a,'end':b,'strategy':name,**s})
    z=pd.DataFrame(rows); z.to_csv('results/crisis_event_study.csv',index=False)
    print(z.pivot(index='strategy',columns='event',values=['return','max_dd']).round(4).to_string())

if __name__=='__main__': main()
