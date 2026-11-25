"""Run one common observed-fund/account cohort; never mix reinvestment policies.

Each bundle must independently pass the canonical loader's source, coverage,
distribution and tax-state gates. This is an execution/accounting experiment,
not a cross-strategy performance certification or a recommendation.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.account_cli import compare_accounts,_serialize


def compare_cohort(bundles,*,output,start='2024-01-02',end='2025-12-31',state='FEDERAL_ONLY',initial_capital=100000.):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows={};checks={}
    for symbol,bundle in sorted(bundles.items()):
        target=output/symbol
        summary=compare_accounts(bundle,strategy='buy_hold',start=start,end=end,
            state=state,initial_capital=initial_capital,ordinary_income=100000.,
            filing_status='single',buy_hold_initial_only=True,output=target)
        if summary['strategy']['symbol']!=symbol:raise ValueError('Bundle symbol mismatch')
        checks[symbol]={}
        if (target/'taxable_pretax_fills.csv').read_bytes()!=(target/'roth_pretax_fills.csv').read_bytes():
            raise AssertionError('Account-neutral fills differ')
        for account in ('taxable','roth'):
            path=target/f'{account}_daily_ledger.csv';frame=pd.read_csv(path)
            # Independent cash/position/receivable/NAV algebra on exported data.
            residual=frame.nav-(frame.cash+frame.unsettled+frame.position_value+frame.receivable_dividends)
            if residual.abs().max()>1e-7:raise AssertionError('Exported NAV failed independent identity')
            values=summary['accounts'][account]
            terminal=frame.nav.iloc[-1]-values['terminal_tax_payable']
            if not np.isclose(terminal,values['ending_wealth_net_payable'],atol=1e-7,rtol=0):raise AssertionError('Terminal wealth mismatch')
            if account=='roth' and (values['tax_paid']!=0 or values['terminal_tax_payable']!=0):raise AssertionError('Routine Roth tax')
            checks[symbol][account]={'daily_rows':len(frame),'max_independent_nav_residual':float(residual.abs().max()),
                'net_terminal_identity':float(terminal),'ledger_sha256':sha256(path.read_bytes()).hexdigest()}
        rows[symbol]=summary
        print('cohort verified',symbol,flush=True)
    policies={s['cash_distribution_policy'] for s in rows.values()}
    if len(policies)!=1:raise AssertionError('Incompatible cash policies')
    report={'data_status':'OBSERVED','execution_status':'HYPOTHETICAL_STRESS','investment_performance_certification':'BLOCKED',
        'common_inputs':{'start':start,'end':end,'initial_capital':initial_capital,'state':state,
            'filing_status':'single','ordinary_income':100000.,'allocation':'Initial lump-sum buy, hold, terminal sell',
            'cash_distributions':'Accumulate in cash in BOTH accounts for EVERY fund','cash_yield':0.,
            'spread_bps':2.,'slippage_bps':1.,'fees':'Observed fund costs embedded in prices, never deducted twice'},
        'funds':rows,'independent_checks':checks,
        'limitations':['Hypothetical daily OHLC execution, no contemporaneous NBBO verification',
            'Existing Roth balance; terminal internal sale is not a personal withdrawal',
            'Salary withholding/safe harbor supplied assumption, not complete household tax advice',
            'Initial-only cash-distribution policy makes the cohort consistent; no dividend reinvestment claim',
            'No investment ranking is certified; different fund risks are not matched portfolios']}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=_serialize)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',action='append',required=True,help='SYMBOL=directory; repeat per fund')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',default='2024-01-02');p.add_argument('--end',default='2025-12-31')
    p.add_argument('--state',required=True,choices=['FEDERAL_ONLY','MA','TX','FL','NV'])
    a=p.parse_args();pairs=[s.split('=',1) for s in a.bundle]
    if any(len(s)!=2 for s in pairs) or len({s[0] for s in pairs})!=len(pairs):p.error('Unique SYMBOL=directory inputs required')
    compare_cohort(dict(pairs),output=a.output,start=a.start,end=a.end,state=a.state)


if __name__=='__main__':main()
