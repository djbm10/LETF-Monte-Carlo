"""Historical macro research CLI: sources, counterfactuals, walk-forward tests."""
import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path

from letf.macro.data import HistoricalSources
from letf.macro.learning import ModelRegistry,walkforward_study
from letf.macro.counterfactual import historical_counterfactual


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('catalog','counterfactual','walkforward','issue','evaluate-issued','evaluate-pending','parameter-sensitivity','explore'):
        q=sub.add_parser(name);q.add_argument('--sources',required=True,type=Path)
        if name!='catalog':q.add_argument('--output',required=True,type=Path)
        if name=='counterfactual':q.add_argument('--start',default='1930-01-01');q.add_argument('--end')
        if name in ('walkforward','issue','evaluate-issued','evaluate-pending'):
            q.add_argument('--registry',required=True,type=Path)
            q.add_argument('--protocol',default=Path('docs/macro-learning-2026-11-24/protocol.json'),type=Path)
        if name=='walkforward':q.add_argument('--paths',type=int)
        if name=='issue':q.add_argument('--require-existing',action='store_true')
        if name=='evaluate-issued':q.add_argument('--forecasts',nargs='+',required=True)
        if name=='parameter-sensitivity':
            q.add_argument('--calibration-end',required=True)
            q.add_argument('--replicates',type=int,default=12)
            q.add_argument('--paths',type=int,default=256)
        if name=='explore':
            q.add_argument('--calibration-end',required=True)
            q.add_argument('--paths',type=int,default=16)
            q.add_argument('--horizons',nargs='+',type=int,default=[5,10,20,30,40])
            q.add_argument('--seed',type=int,default=20261124)
            q.add_argument('--model',choices=['joint_block','regime_block','conditional_var_t'],default='conditional_var_t')
            q.add_argument('--initial-capital',type=float,default=10000.)
            q.add_argument('--annual-contribution',type=float,default=0.)
            q.add_argument('--monthly-withdrawal',type=float,default=0.)
            q.add_argument('--roth-birth-date')
            q.add_argument('--roth-first-tax-year',type=int)
            q.add_argument('--roth-regular-basis',type=float)
            q.add_argument('--ordinary-income',type=float,default=100000.)
            q.add_argument('--filing-status',default='single',choices=['single','married','head_of_household','married_filing_separately'])
            q.add_argument('--state',choices=['FEDERAL_ONLY','MA','TX','FL','NV'],required=True)
            q.add_argument('--freeze-tax-year',type=int,required=True)
            for arg in ['rate','inflation','production-growth','equity-drift','equity-cagr-constraint','stock-bond-correlation','crash-return']:q.add_argument('--'+arg,type=float)
            q.add_argument('--crash-month',type=int)
            q.add_argument('--volatility-scale',type=float,default=1.)
            q.add_argument('--annual-fee',type=float,default=.01)
            q.add_argument('--borrowing-spread',type=float,default=.015)
            q.add_argument('--distribution-yield',type=float,default=.01)
    a=p.parse_args();sources=HistoricalSources(a.sources)
    if a.command=='parameter-sensitivity':
        from letf.macro.uncertainty import parameter_sensitivity
        r=parameter_sensitivity(sources,calibration_end=a.calibration_end,output=a.output,replicates=a.replicates,paths=a.paths)
        print(json.dumps(r['results'],indent=2));return
    if a.command=='catalog':
        print(json.dumps(sources.catalog,indent=2));return
    if a.command=='counterfactual':
        r=historical_counterfactual(sources,start=a.start,end=a.end,output=a.output)
        print(json.dumps({'exposures':list(r['exposures']),'evidence':r['evidence'],'report':str(a.output/'report.json')}));return
    if a.command=='explore':
        from letf.macro.explorer import explore
        r=explore(sources,calibration_end=a.calibration_end,output=a.output,paths=a.paths,horizons=a.horizons,seed=a.seed,
            model_type=a.model,initial_capital=a.initial_capital,annual_contribution=a.annual_contribution,monthly_withdrawal=a.monthly_withdrawal,
            roth_birth_date=a.roth_birth_date,roth_first_tax_year=a.roth_first_tax_year,roth_regular_basis=a.roth_regular_basis,
            ordinary_income=a.ordinary_income,state=a.state,filing_status=a.filing_status,frozen_tax_year=a.freeze_tax_year,rate=a.rate,inflation=a.inflation,
            production_growth=a.production_growth,equity_drift=a.equity_drift,volatility_scale=a.volatility_scale,
            equity_cagr_constraint=a.equity_cagr_constraint,
            stock_bond_correlation=a.stock_bond_correlation,crash_month=a.crash_month,crash_return=a.crash_return,
            annual_fee=a.annual_fee,borrowing_spread=a.borrowing_spread,distribution_yield=a.distribution_yield)
        print(json.dumps({'status':r['evidence'],'results':r['results']},indent=2));return
    protocol=json.loads(a.protocol.read_text())
    if a.command=='issue':
        from letf.macro.retraining import issue_candidates
        print(json.dumps(issue_candidates(sources,ModelRegistry(a.registry),protocol,output=a.output,require_existing=a.require_existing),indent=2));return
    if a.command=='evaluate-issued':
        from letf.macro.retraining import evaluate_issued
        print(json.dumps(evaluate_issued(sources,ModelRegistry(a.registry),protocol,a.forecasts,output=a.output),indent=2));return
    if a.command=='evaluate-pending':
        from letf.macro.retraining import evaluate_pending
        print(json.dumps(evaluate_pending(sources,ModelRegistry(a.registry),protocol,output=a.output),indent=2));return
    r=walkforward_study(sources,ModelRegistry(a.registry),protocol,output=a.output,paths=a.paths)
    print(json.dumps({'forecasts':len(r['forecast_ids']),'blocked':len(r['blocked_models']),
        'decisions':r['model_comparison'],'report':str(a.output/'report.json')},indent=2))


if __name__=='__main__':main()
