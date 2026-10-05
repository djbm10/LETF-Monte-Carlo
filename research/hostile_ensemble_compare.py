from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import research.hostile_strategy_tournament_fast as fast
base=fast.base; orig=base.build_strategies

def build(bm,br):
 o=orig(bm,br); r1=o['1x_buy_hold']; s8=o['repo_s8_composite']; b5=o['s9_200_35_12_band5']
 # add 10-point band directly
 rr1=base.levret(bm,br,1); rr3=base.levret(bm,br,3);px=np.cumprod(np.maximum(1+rr1,1e-12),axis=1);a=base.desired_s9(px,rr3,.35,.12,200,0);b10=base.alloc_returns(fast.hysteresis_fast(a,.10),rr3,br);o['s9_band10']=b10
 o['blend_50_s8_50_s9']=.5*s8+.5*b10
 o['blend_50_1x_50_s9']=.5*r1+.5*b10
 o['blend_50_1x_50_s8']=.5*r1+.5*s8
 o['blend_50_1x_25_s8_25_s9']=.5*r1+.25*s8+.25*b10
 o['blend_33_each']=(r1+s8+b10)/3
 o['blend_25_1x_375_s8_375_s9']=.25*r1+.375*s8+.375*b10
 return o
base.build_strategies=build
base.STRATEGIES=('1x_buy_hold','repo_s8_composite','s9_band10','blend_50_s8_50_s9','blend_50_1x_50_s9','blend_50_1x_50_s8','blend_50_1x_25_s8_25_s9','blend_33_each','blend_25_1x_375_s8_375_s9')

def main():
 out=Path('results/hostile_ensemble_compare');out.mkdir(parents=True,exist_ok=True);ff,meta=base.load_ff();parts=[]
 for i,(n,s) in enumerate(base.SCENARIOS.items()):parts.append(base.run_scenario(ff,n,s,10_000,base.SEED+10_000*i,batch_size=128))
 d=pd.concat(parts,ignore_index=True);agg=base.aggregate(d);d.to_csv(out/'scenario_results.csv',index=False);agg.to_csv(out/'robust_ranking.csv',index=False);(out/'manifest.json').write_text(json.dumps({'data':meta,'paths':10000,'strategies':list(base.STRATEGIES),'scenarios':base.SCENARIOS},indent=2));print(agg.to_string(index=False))
if __name__=='__main__':main()
