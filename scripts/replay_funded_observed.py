"""Observed fund, identical contributions/reinvestment, canonical taxable/Roth.

An existing constant 100% allocation, not a new strategy. Real source actions
and supported historical tax years are required. No internal fund cost is
deducted again. Trades are assumed daily-open executions with declared costs.
"""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
from datetime import date
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from letf.accounts import Security,TAXABLE_CASH,CONSERVATIVE_ROTH
from letf.account_backtest import run_account_backtest,equity_calendar
from letf.account_cli import load_bundle
from letf.account_stress import independent_account_audit,write_evidence_json
from letf.macro.precision_run import file_hash


def run(bundle,output):
    output.mkdir(parents=True,exist_ok=False)
    market,manifest,distributions,cpi=load_bundle(bundle);symbol=manifest['symbol']
    selected=market.loc['2024-01-02':'2025-12-31']
    if selected.index[0]!=pd.Timestamp('2024-01-02') or selected.index[-1]!=pd.Timestamp('2025-12-31'):
        raise ValueError('Complete common 2024-2025 cohort required')
    calendar=equity_calendar(selected.index[0].date(),selected.index[-1].date())
    rows={};fills={};payments={2024:calendar.date_to_session('2025-04-15',direction='next').date()}
    for name,base,taxed in [('taxable',TAXABLE_CASH,True),('roth',CONSERVATIVE_ROTH,True),('taxable_pre_tax_control',TAXABLE_CASH,False)]:
        flows=[{'date':str(day.date()),'kind':'contribution' if base.account_type=='roth' else 'deposit',
                'amount':3000.,'magi':100000.,'compensation':100000.}
               for day in (selected.index[0],selected.loc['2025'].index[0])]
        result=run_account_backtest(market,[Security(symbol)],replace(base,spread_bps=2.,slippage_bps=1.),
            initial_capital=10000.,allocator=lambda history,weights:{symbol:1.},price_convention='as_traded',
            distributions_complete=manifest['distributions_complete'],distributions=distributions,
            rebalance_every=63,settlement_policy='us_equity',terminal_liquidation=True,
            tax_inputs={'ordinary_income':100000.,'federal_only':True,'no_external_wash_replacements_after_end':True},
            tax_payment_dates=payments,split_events=manifest.get('split_events',[]),inflation_index=cpi,
            cash_flows=flows,roth_history_config={'birth_date':'1980-01-01','first_tax_year':2010,'regular_basis':10000.},
            source_manifest=manifest,trade_start='2024-01-02',qualified_holding_policy='track_unhedged_common',
            apply_taxes=taxed,audit_detail=True)
        audit=independent_account_audit(result,10000.);directory=output/name;directory.mkdir()
        result['reconciliation'].to_csv(directory/'daily-account.csv')
        pd.DataFrame({'wealth':result['equity'],'unit_wealth':result['unit_equity'],
            'terminal_net_unit_wealth':result['terminal_net_unit_equity']}).to_csv(directory/'wealth.csv')
        write_evidence_json(directory/'events.json',result['ledger'].events);write_evidence_json(directory/'annual-tax.json',result['tax_rows'])
        trades=[(str(e['date']),e['action'],e['security'],e['shares'],e['price'],e['commission'])
                for e in result['ledger'].events if e['action']in ('BUY','SELL')]
        fills[name]=trades
        rows[name]={'net_ending_wealth':result['ending_wealth_net_payable'],'metrics':result['after_tax_metrics'],
            'tax_paid':result['tax_paid'],'terminal_tax_payable':result['terminal_tax_payable'],
            'trading_costs':result['trading_costs'],'tax_funding_sales':sum(e.get('reason')=='TAX_FUNDING' for e in result['ledger'].events),
            'split_events':sum(e['action']=='SPLIT' for e in result['ledger'].events),
            'drawdown_convention':result['drawdown_convention'],'independent_audit':audit,
            'files':{p.name:file_hash(p) for p in directory.iterdir()},'annual_tax':result['tax_rows']}
    a,b=fills['roth'],fills['taxable_pre_tax_control']
    if len(a)!=len(b) or any(x[:3]!=y[:3] or not np.allclose(x[3:],y[3:],rtol=1e-12,atol=1e-9) for x,y in zip(a,b,strict=True)):
        raise AssertionError('Paired pre-tax control trades do not reconcile')
    report={'source_evidence':'OBSERVED','execution_exact':False,'symbol':symbol,
        'source_manifest_sha256':file_hash(bundle/'manifest.json'),'source_manifest':manifest,
        'period':['2024-01-02','2025-12-31'],'initial':10000.,'contributions':6000.,
        'ordinary_income':100000.,'state':'FEDERAL_ONLY','filing_status':'single',
        'tax_years':[2024,2025],'frozen_future_tax_year':None,'reinvestment':'Existing constant 100% allocation, every 63 sessions',
        'identical_roth_and_account_neutral_pre_tax_fills':True,'pre_tax_control_trade_count':len(a),'accounts':rows,
        'investment_prediction_certification':'BLOCKED','limitations':['Observed data plus modeled execution and supplied household tax inputs',
            'Endpoint real returns use observed CPI endpoints; missing interior CPI is not filled',
            'No actual-fund internal fee/financing charged twice','No unhedged-common QDI inference for complex option hedges',
            'Roth internal liquidation, not an owner qualified withdrawal; contributions checked against actual 2024/2025 tables',
            'Drawdown uses cash tax payments and terminal payable, not daily hypothetical liquidation tax']}
    write_evidence_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.bundle,a.output);print(json.dumps({k:{n:v for n,v in x.items() if n in ('net_ending_wealth','tax_paid','terminal_tax_payable','tax_funding_sales')}for k,x in r['accounts'].items()},indent=2))
