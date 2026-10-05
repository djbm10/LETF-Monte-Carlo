from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import research.hostile_strategy_tournament_fast as fast
base=fast.base
orig=base.build_strategies

def build(bm,br):
    out=orig(bm,br)
    r1=base.levret(bm,br,1); r3=base.levret(bm,br,3); px=(1+r1).clip(min=1e-12).cumprod(axis=1)
    a=base.desired_s9(px,r3,.35,.12,200,0)
    out['s9_200_35_12_band10']=base.alloc_returns(fast.hysteresis_fast(a,.10),r3,br)
    return out
base.build_strategies=build
base.STRATEGIES=('1x_buy_hold','repo_s8_composite','s9_200_35_12_band5','s9_200_35_12_band10')

def main():
    outdir=Path('results/hostile_band10_compare');outdir.mkdir(parents=True,exist_ok=True)
    ff,meta=base.load_ff();parts=[]
    for i,(name,sc) in enumerate(base.SCENARIOS.items()):
        parts.append(base.run_scenario(ff,name,sc,10_000,base.SEED+10_000*i,batch_size=128))
    d=pd.concat(parts,ignore_index=True);agg=base.aggregate(d);d.to_csv(outdir/'scenario_results.csv',index=False);agg.to_csv(outdir/'robust_ranking.csv',index=False)
    (outdir/'manifest.json').write_text(json.dumps({'data':meta,'paths_per_scenario':10000,'seed':base.SEED,'strategies':list(base.STRATEGIES),'scenarios':base.SCENARIOS},indent=2))
    print(agg.to_string(index=False))
if __name__=='__main__':main()
