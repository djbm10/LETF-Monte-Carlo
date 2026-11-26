"""Portable synthetic software demonstration, never actual ETF market data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.account_backtest import equity_calendar
from letf.lab.catalog import file_hash,private_bytes,private_json
from letf.lab.portfolio import run_observed
from letf.lab.html import render_report


def demo(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    calendar=equity_calendar(pd.Timestamp('2024-01-02').date(),pd.Timestamp('2025-12-31').date())
    dates=calendar.sessions_in_range('2024-01-02','2025-12-31').tz_localize(None)
    bundles={}
    for i,symbol in enumerate(('FIXTURE_EQUITY','FIXTURE_DEFENSIVE')):
        folder=output/symbol;folder.mkdir(mode=0o700)
        t=np.arange(len(dates))
        values=100*np.exp((.00025 if i==0 else .00010)*t+(.10 if i==0 else .025)*np.sin(t/37))
        price=pd.DataFrame({'Open':values,'Close':values,'Dividends':0.,'Stock Splits':0.},index=dates)
        payouts=[]
        for year in (2024,2025):
            ex=pd.Timestamp(year,6,3);pay=calendar.session_offset(ex,2)
            price.loc[ex,'Dividends']=.5
            payouts.append({'ex_date':str(ex.date()),'pay_date':str(pay.date()),'amount_per_share':.5,
                            'category':'ordinary_dividend','source':'SOFTWARE_FIXTURE: explicitly stipulated cash payment'})
        private_bytes(folder/'prices.csv',price.to_csv().encode())
        private_bytes(folder/'distributions.csv',pd.DataFrame(payouts).set_index('ex_date').to_csv().encode())
        files={name:{'path':name+'.csv','sha256':file_hash(folder/(name+'.csv')),
                     'provider':'SOFTWARE_FIXTURE','url':'https://example.invalid/letf-software-fixture'}
               for name in ('prices','distributions')}
        private_json(folder/'manifest.json',{'symbol':symbol,'data_status':'SOFTWARE_FIXTURE','classification':'SOFTWARE_FIXTURE',
            'currency':'USD','frequency':'daily','price_convention':'as_traded','distributions_complete':True,
            'account_coverage_start':'2024-01-02','account_coverage_end':'2025-12-31','supported_tax_states':['FEDERAL_ONLY'],
            'files':files,'limitations':['Invented prices/distributions solely for reproducible software mechanics; no forecast or actual investment claim']})
        bundles[symbol]=str(folder)
    request={'schema':1,'mode':'fixture','bundles':bundles,'start':'2024-01-02','end':'2025-12-31',
        'strategy':{'id':'fixed_weights','version':'cash_v1','weights':{'FIXTURE_EQUITY':.6,'FIXTURE_DEFENSIVE':.3}},
        'initial_capital':10000.,'rebalance_sessions':63,'cash_sweep_annual_rate':.02,
        'tax':{'state':'FEDERAL_ONLY','ordinary_income':100000.,'filing_status':'single','law_mode':'ACTUAL_DATED',
               'external_wash_policy':'NO_OTHER_OWNER_REPLACEMENTS'},
        'cash_flows':[{'date':'2025-01-02','kind':'contribution','amount':1000.,'magi':100000.,'compensation':100000.}],
        'roth_owner':{'birth_date':'1980-01-01','first_tax_year':2010,'regular_basis':10000.}}
    private_json(output/'request.json',request);private_json(output/'profiles.json',{'Portable software fixture':request})
    result=run_observed(request,output=output/'run');render_report(output/'run/report.json',output/'report.html')
    return {'classification':'SOFTWARE_FIXTURE','report':str(output/'report.html'),
            'configuration':str(output/'request.json'),'profiles':str(output/'profiles.json'),
            'independent_account_audits_passed':all(a['independent_audit']['passed'] for a in result['accounts'].values()),
            'observed_market_performance':False}
