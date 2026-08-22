from __future__ import annotations
import json
from pathlib import Path
import research.nasdaq1985_final_tournament as n

OUT=Path('results/nasdaq1985_phase1'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    df=n.load_data(.007)
    screen,folds,cs=n.historical_screen(df)
    by={c.name:c for c in cs}
    # Strictly pre-2010 selection. No post-2010 CAGR/block metric enters the score.
    pre=screen[(screen.train_dd>-0.85)&screen.train_cagr.notna()].copy()
    pre['pre2010_score']=pre.train_cagr + 0.10*pre.train_dd  # mild drawdown penalty; train-only
    train_selected=list(pre.sort_values('pre2010_score',ascending=False).strategy.head(12))
    frozen=['ndx_1x','fut_fixed_2x','fut_sma200_2x_b0_buf0','fut_sma200_3x_b0_buf0',
            'fut_tv_w200_u35_d12_b30','fut_tv_w200_u35_d0_b30','fut_vol25','fut_mom252_2x',
            'qld_s9_w200_u35_d12_b10','tqqq_s9_w200_u35_d12_b10']
    finalist_names=[]
    for x in train_selected+frozen:
        if x in by and x not in finalist_names: finalist_names.append(x)
    tax=n.taxable_history(df,screen,cs)
    div=n.dividend_sensitivity(); veh=n.cross_vehicle()
    screen.to_csv(OUT/'historical_screen.csv',index=False);folds.to_csv(OUT/'five_year_blocks.csv',index=False)
    tax.to_csv(OUT/'taxable_futures.csv',index=False);div.to_csv(OUT/'pre1999_dividend_sensitivity.csv',index=False);veh.to_csv(OUT/'cross_vehicle_1988plus.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'candidate_count':len(cs),'finalists':finalist_names,
        'pre2010_selected':train_selected,'data_start':str(df.index.min().date()),'data_end':str(df.index.max().date()),
        'selection_rule':'STRICT pre-2010 only: train_cagr + 0.10*train_dd, require train max DD > -85%; 2010+ untouched for selection'},indent=2))
    print('CANDIDATES',len(cs));print('STRICT PRE2010 SELECTED',train_selected);print('FINALISTS',finalist_names)
    print('\nTOP STRICT PRE-2010 SCORE\n',pre.sort_values('pre2010_score',ascending=False).head(30).to_string(index=False))
    print('\n2010+ HOLDOUT (NOT USED TO SELECT)\n',screen[screen.strategy.isin(finalist_names)].sort_values('holdout_cagr',ascending=False).to_string(index=False))
    print('\nTAXABLE\n',tax.to_string(index=False));print('\nDIVIDEND SENSITIVITY\n',div.to_string(index=False));print('\nCROSS VEHICLE\n',veh.to_string(index=False))

if __name__=='__main__': main()
