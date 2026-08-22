from __future__ import annotations
import json
from pathlib import Path
import research.nasdaq1985_final_tournament as n

OUT=Path('results/nasdaq1985_phase1'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    df=n.load_data(.007)
    screen,folds,cs=n.historical_screen(df)
    finalists,train_selected=n.choose_finalists(screen,cs)
    tax=n.taxable_history(df,screen,cs)
    div=n.dividend_sensitivity()
    veh=n.cross_vehicle()
    screen.to_csv(OUT/'historical_screen.csv',index=False)
    folds.to_csv(OUT/'five_year_blocks.csv',index=False)
    tax.to_csv(OUT/'taxable_futures.csv',index=False)
    div.to_csv(OUT/'pre1999_dividend_sensitivity.csv',index=False)
    veh.to_csv(OUT/'cross_vehicle_1988plus.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({
        'candidate_count':len(cs),'finalists':[c.name for c in finalists],
        'pre2010_selected':train_selected,'data_start':str(df.index.min().date()),
        'data_end':str(df.index.max().date()),'selection_rule':'pre-2010 train + fixed 5y block robustness; 2010+ held out'
    },indent=2))
    print('CANDIDATES',len(cs))
    print('FINALISTS',[c.name for c in finalists])
    print('\nTOP PRE-2010 TRAIN\n',screen.sort_values(['train_cagr','p10_5y_cagr'],ascending=False).head(30).to_string(index=False))
    print('\nTOP 2010+ HOLDOUT (NOT USED TO SELECT)\n',screen.sort_values('holdout_cagr',ascending=False).head(30).to_string(index=False))
    print('\nTAXABLE\n',tax.to_string(index=False))
    print('\nDIVIDEND SENSITIVITY\n',div.to_string(index=False))
    print('\nCROSS VEHICLE\n',veh.to_string(index=False))

if __name__=='__main__': main()
