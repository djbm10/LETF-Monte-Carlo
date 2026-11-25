"""Same-path canonical daily versus exact monthly lookup reconciliation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.macro.artifacts import ArtifactStore,canonical
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage
from letf.macro.daily_tails import daily_bridge
from letf.macro.precision import MonthlyLookup
from letf.reconstruction import SourceRecord,daily_reset_returns


def run(experiment,output):
    if output.exists():raise ValueError('No overwrite')
    registration=json.loads((experiment/'registration.json').read_text());store=ArtifactStore(experiment/'artifacts')
    m,a=store.load('models',registration['model_id']);model=MacroModel(m['kind'],m['columns'],a['training'],m['parameters'])
    monthly,lineage=sample_with_lineage(model,480,128,9321)
    start=time.perf_counter();lookup=MonthlyLookup.build(model,a['templates'],a['lengths']);setup=time.perf_counter()-start
    start=time.perf_counter();fast,_=lookup.evaluate(lineage,np.full(128,lookup.rate_indices[-1]),44);fast_time=time.perf_counter()-start
    logs=daily_bridge(monthly,lineage,model.columns,a['templates'],a['lengths'],44)
    dates=pd.date_range('2000-01-01',periods=logs.shape[1],freq='D')
    source=SourceRecord('fixture','https://example.invalid','2026-11-25','reconciliation','a'*64,str(dates[0].date()),'USD','Hypothetical daily values','daily',False)
    errors=[];start=time.perf_counter();j=model.columns.index('short_rate')
    for p in range(128):
        rate=pd.Series(np.repeat(np.r_[model.train_values[-1,j],monthly[p,:-1,j]],21),index=dates)
        for l in (1,2,3):
            result=daily_reset_returns(pd.Series(np.expm1(logs[p]),index=dates),l,source=source,
                annual_fee=.01 if l>1 else 0.,funding_rate_at_previous_close=rate,borrowing_spread=.015)
            growth=np.cumprod(1+result['return'].to_numpy());peak=np.maximum.accumulate(np.r_[1.,growth])[1:]
            for h in (5,10,20,30,40):
                f=fast[h,l];last=252*h
                errors.append([abs(f['log_nav'][p]-np.log(growth[last-1])),
                    abs(f['drawdown'][p]-np.max(1-growth[:last]/peak[:last]))])
    slow_time=time.perf_counter()-start;maximum=np.max(errors,axis=0)
    if maximum[0]>1e-10 or maximum[1]>1e-12:raise AssertionError('Acceleration failed canonical same-path reconciliation')
    report={'model_id':registration['model_id'],'same_paths':128,'years':40,'leverages':3,'horizons':5,
        'independent_comparisons':128*3*5,'max_abs_log_nav_error':maximum[0],'max_abs_drawdown_error':maximum[1],
        'lookup_setup_seconds':setup,'lookup_evaluation_seconds':fast_time,'canonical_daily_evaluation_seconds':slow_time,
        'amortized_kernel_speedup':slow_time/fast_time,'all_numerical_checks_passed':True,
        'support_minimum_daily_gross_by_leverage':[float(lookup.summaries[l,:,:,4].min()) for l in range(3)],
        'support_contains_liquidation_by_leverage':[bool(lookup.summaries[l,:,:,5].any()) for l in range(3)],
        'scope':'Numerical conformance and conditional finite support, not actual future market/liquidation risk'}
    output.write_bytes(canonical(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.experiment,a.output),indent=2))
