"""Paired daily bootstrap routed through the same account replay as history.

Future tax law is unknown: a supplied frozen tax year is an explicit stress
assumption, not historical law. This path supports non-distributing stock
scenarios only. No inferred ETF distribution classifications are manufactured.
"""
import argparse
from dataclasses import replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from letf.account_cli import load_bundle, allocator_for, _serialize
from letf.account_backtest import run_account_backtest
from letf.accounts import Security, TAXABLE_CASH, CONSERVATIVE_ROTH
from letf.accuracy import UnsupportedModelError
from letf.simulation.paired import paired_block_paths, paired_mean_difference_ci


def compare_scenarios(bundle, *, paths, horizons, seed, freeze_tax_year,
                       initial_capital=100000., ordinary_income=100000., output=None):
    market, manifest, distributions, _ = load_bundle(bundle)
    if distributions or manifest.get('instrument_kind') != 'stock':
        raise UnsupportedModelError('Dividend/ETF scenarios require joint ex-price, distribution and tax-character paths')
    if freeze_tax_year not in (2024,2025,2026):
        raise UnsupportedModelError('Choose an explicit verified frozen tax-law scenario year')
    horizons=sorted(set(horizons))
    if not horizons or set(horizons)-{5,10,20,30,40}:
        raise ValueError('Supported horizons are 5,10,20,30,40')
    symbol=manifest['symbol']
    op=market[f'{symbol}_Open'];cl=market[f'{symbol}_Close']
    observations=pd.DataFrame({'gap':op/cl.shift(1)-1,'session':cl/op-1}).iloc[1:]
    cutoff=observations.index[-1]
    calendar=xcals.get_calendar('XNYS',start='2026-12-01',end=f'{2028+max(horizons)}-12-31')
    sessions=calendar.sessions[calendar.sessions >= pd.Timestamp('2027-01-04')].tz_localize(None)
    rows=[]
    for path in paired_block_paths(observations,paths=paths,years=max(horizons),block_sessions=21,
            seed=seed,training_end=cutoff,evaluation_start='2027-01-04',provenance_verified=True):
        prices=[];previous=100.
        for gap,ret in path[['gap','session']].itertuples(index=False,name=None):
            opening=previous*(1+gap);closing=opening*(1+ret)
            prices.append((opening,closing));previous=closing
        generated=pd.DataFrame(prices,index=sessions[:len(path)],columns=[f'{symbol}_Open',f'{symbol}_Close'])
        for horizon in horizons:
            sample=generated.iloc[:252*horizon]
            payments={y:calendar.date_to_session(pd.Timestamp(date(y+1,4,15)),direction='next').date()
                      for y in range(sample.index[0].year,sample.index[-1].year)}
            outcomes={}
            for account,base in [('taxable',TAXABLE_CASH),('roth',CONSERVATIVE_ROTH)]:
                result=run_account_backtest(sample,[Security(symbol,kind='stock')],
                    replace(base,spread_bps=2.,slippage_bps=1.),initial_capital=initial_capital,
                    allocator=allocator_for(symbol,'buy_hold',50),price_convention='as_traded',
                    distributions_complete=True,settlement_policy='us_equity',terminal_liquidation=True,
                    tax_inputs={'ordinary_income':ordinary_income,'federal_only':True,
                                'no_external_wash_replacements_after_end':True},
                    tax_payment_dates=payments,freeze_tax_year=freeze_tax_year)
                outcomes[account]={'ending_wealth':result['ending_wealth_net_payable'],
                    'metrics':result['after_tax_metrics'],'tax_paid':result['tax_paid'],
                    'terminal_tax_payable':result['terminal_tax_payable'],'costs':result['trading_costs']}
            rows.append({'path':path.attrs['path_id'],'horizon_years_252_sessions':horizon,
                'actual_calendar_years':(sample.index[-1]-sample.index[0]).days/365.2425,
                'sample_indices_sha256':sha256(np.asarray(path.attrs['source_indices'][:252*horizon],dtype='<i8').tobytes()).hexdigest(),
                'accounts':outcomes})
    ci={}
    if paths>=2:
        for h in horizons:
            selected=[r for r in rows if r['horizon_years_252_sessions']==h]
            ci[str(h)]=paired_mean_difference_ci([r['accounts']['roth']['ending_wealth'] for r in selected],
                [r['accounts']['taxable']['ending_wealth'] for r in selected],seed=seed)
    report={'evidence':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'seed':seed,'independent_paths':paths,'frozen_tax_year':freeze_tax_year,
        'source_manifest_sha256':sha256((Path(bundle)/'manifest.json').read_bytes()).hexdigest(),
        'strategy':'buy_hold','block_sessions':21,'rows':rows,'paired_terminal_difference_ci':ci,
        'limitations':['All future law is the explicit frozen-year stress assumption',
            'Generic US cash accounts, federal regular tax/NIIT only; no state or future eligibility predictions',
            'Resampled stock assumes no future dividends/splits/closures; not an investable forecast',
            'Calendar is a projected XNYS calendar; unexpected closures are not modeled',
            'Intervals quantify sampling variation conditional on one short source history, not model/source uncertainty',
            'Small demonstration path counts do not establish Monte Carlo convergence; no full strategy rankings authorized']}
    if output:
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as f:json.dump(report,f,indent=2,default=_serialize,allow_nan=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paths',type=int,default=2);p.add_argument('--horizons',nargs='+',type=int,default=[5,10,20,30,40])
    p.add_argument('--seed',type=int,default=20261124);p.add_argument('--freeze-tax-year',type=int,required=True)
    a=p.parse_args();r=compare_scenarios(a.bundle,paths=a.paths,horizons=a.horizons,seed=a.seed,
        freeze_tax_year=a.freeze_tax_year,output=a.output)
    print(json.dumps({'evidence':r['evidence'],'independent_paths':r['independent_paths'],
        'account_runs':2*len(r['rows']),'investment_performance_certification':'BLOCKED'}))


if __name__=='__main__':main()
