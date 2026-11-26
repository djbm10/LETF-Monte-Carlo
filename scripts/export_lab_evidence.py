"""Export aggregate audit evidence; never copy raw provider quotes/ensembles.

Requires the private studies named in the reproduction guide. Their original
bytes remain authoritative; this exporter preserves producer-input hashes.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letf.lab.catalog import SourceLake,file_hash


def export(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    artifacts={};written={}
    def read(relative):
        path=root/relative;artifacts[relative]=file_hash(path)
        return json.loads(path.read_text())
    def emit(name,record):
        blob=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode();path=output/name
        if path.exists() and path.read_bytes()!=blob:raise FileExistsError('Preserve evidence: '+str(path))
        if not path.exists():
            with path.open('xb') as stream:stream.write(blob)
        written[name]=sha256(blob).hexdigest()
    lake=SourceLake(root/'audit-output/source-lake-v1').audit()
    allowed=('provider','url','retrieved_at','publication_vintage','license','license_url','convention',
        'frequency','currency','first_observation','last_observation','rows','records','point_in_time_verified','classification')
    emit('final-source-catalog.json',{'classification':lake['classification'],'source_integrity_only':True,
        'snapshots_verified':lake['snapshots_verified'],'financial_certification':'BLOCKED',
        'sources':[dict(snapshot=r['snapshot'],name=r['name'],raw_sha256=r['raw_sha256'],raw_bytes=r['raw_bytes'],
            metadata={k:r['metadata'].get(k,'UNKNOWN') for k in allowed}) for r in lake['sources']]})
    cohort=read('audit-output/observed-five-year-cohort-v4-final-engine/report.json')
    emit('final-observed-accounts.json',{'classification':'OBSERVED_MARKET','execution_classification':'CONDITIONAL_STRESS',
        'common_inputs':cohort['common_inputs'],'limitations':cohort['limitations'],
        'investment_performance_certification':'BLOCKED','independent_checks':cohort['independent_checks'],
        'funds':{s:{'source_manifest_sha256':v['source_manifest_sha256'],
            'identical_pre_tax_trades':v['identical_pre_tax_trades'],'accounts':v['accounts']} for s,v in cohort['funds'].items()}})
    s6=read('audit-output/s6-observed-v3-final-engine/report.json')
    emit('final-s6-flow-comparison.json',{'classification':s6['classification'],'accounts':s6['accounts'],
        'same_pre_tax_policy_and_fills':s6['same_pre_tax_policy_and_fills'],
        'implementation_commit':s6['implementation_commit'],'implementation_source_sha256':s6['implementation_source_sha256'],
        'scope':'2024–25 $10000 initial plus $3000/year; S6 causal cash21-session wrapper, observed input / hypothetical execution'})
    pair=read('audit-output/s6-hfea-common-v2/report.json')
    emit('final-s6-hfea-flow-cohort.json',{'classification':pair['classification'],
        'common_inputs_verified':pair['common_inputs_verified'],'all_requested_cells_supported':pair['all_requested_cells_supported'],
        'rankings_released':False,'rebalance_sessions':21,'start':'2024-01-02','end':'2025-12-31',
        'cells':[{'strategy':c['strategy'],'accounts':c['accounts'],'report_sha256':c['report_sha256']} for c in pair['cells']]})
    cash=read('audit-output/all-cash-v1/report.json')
    if any(v['ending_wealth_net_payable']!=16000. for v in cash['accounts'].values()):raise AssertionError('Zero-yield cash-flow identity failed')
    emit('final-all-cash-conformance.json',{'classification':'SOFTWARE_FIXTURE',
        'expected_wealth':16000.,'initial':10000.,'deposits':6000.,'cash_yield_assumption':0.,
        'account_wealth':{k:v['ending_wealth_net_payable'] for k,v in cash['accounts'].items()},
        'scope':'Independent cash-flow arithmetic against canonical observed-source plumbing; no market exposure or interest forecast'})
    cf=read('audit-output/1930-counterfactual-v2/report.json');exposures={}
    for key,value in cf['exposures'].items():
        records=[]
        for original in value['results']:
            row=dict(original)
            if 'historical_windows'in row:
                windows=row['historical_windows'];rolling={}
                for years,items in windows['rolling_year_start_windows'].items():
                    observations=items['windows']
                    rolling[years]={k:v for k,v in items.items() if k!='windows'}
                    rolling[years].update(windows_count=len(observations),
                        first_period_start_close=observations[0]['period_start_close'] if observations else None,
                        last_period_end=observations[-1]['end'] if observations else None,
                        worst_cagr=min((x['cagr'] for x in observations),default=None),
                        worst_drawdown=max((x['max_drawdown'] for x in observations),default=None))
                row['historical_windows']={'eras':windows['eras'],'rolling_year_start_windows':rolling,
                    'individual_window_records':'Retained in hashed private producer report; aggregate export only'}
            records.append(row)
        exposures[key]={'first_genuine_observation':value['first_genuine_observation'],
            'source_sha256':value['underlying_source_sha256'],'proxy_identity':value['proxy_identity'],
            'results':records}
    emit('final-counterfactual-summary.json',{'classification':'HISTORICAL_COUNTERFACTUAL','exposures':exposures,
        'method_version':cf['method_version'],
        'source_catalog_sha256':cf['source_catalog_sha256'],'financing_proxy':cf['financing_proxy'],
        'closure_evidence':cf['closure_evidence'],'limitations':cf['limitations'],'account_history_certification':'BLOCKED'})
    future=read('audit-output/canonical-future-v1/laboratory-report.json')
    emit('final-future-software-demonstration.json',{k:future[k] for k in (
        'classification','model_approved','model_type','model_id','forecast_id','source_catalog_sha256',
        'conditional_distributions','sampling_uncertainty','structural_uncertainty','elapsed_seconds',
        'process_peak_rss_bytes','runtime_scope','rankings_released')})
    assessments=list((root/'audit-output/full-account-precision-v1').glob('assessment-*.json'))
    complete=[p for p in assessments if json.loads(p.read_text())['study_complete']]
    if len(complete)!=1:raise ValueError('Exactly one completed registered precision assessment required')
    emit('final-full-account-precision.json',read(str(complete[0].relative_to(root))))
    emit('final-fixed-parameter-nav-diagnostic.json',read('audit-output/refreshed-nav-diagnostic-v2.json'))
    tournament=read('audit-output/cohort-2025-v4-wash-split/report.json')
    emit('final-strategy-cohort.json',{'classification':tournament['classification'],'rankings_released':False,
        'all_requested_cells_supported':tournament['all_requested_cells_supported'],
        'common_inputs_verified':tournament['common_inputs_verified'],
        'policy':'2025 common universe, $10000, no flows, FEDERAL_ONLY; original VIX/IRX observations; explicit1-session rebalance override, not original S6/HFEA execution frequency',
        'cells':[{k:r[k] for k in ('strategy','status','classification','reason','comparison_fingerprint','report_sha256') if k in r}
                 for r in tournament['cells']]})
    first=read('audit-output/master-operations-v1/operations-report.json')
    refreshed=read('audit-output/refreshed-operations-v1/operations-report.json')
    emit('final-prospective-operations.json',{'classification':'CONDITIONAL_STRESS',
        'frozen_candidate_issue':first['candidate_issue'],'first_source_catalog_sha256':first['source_catalog_sha256'],
        'refreshed_source_catalog_sha256':refreshed['source_catalog_sha256'],
        'refreshed_issuance_status':refreshed['candidate_issue']['status'],
        'evaluated_months':refreshed['prospective_evaluated_months'],'promotion_mode':refreshed['promotion_mode'],
        'promoted':False,'local_archive':refreshed['local_archive'],'local_recovery':refreshed['local_recovery'],
        'off_host_verified':False})
    emit('final-provider-demo-inspection.json',read('audit-output/free-derivative-samples-v1/sample-inspection.json'))
    emit('final-legacy-realism.json',{'classification':'SOFTWARE_FIXTURE',**read('audit-output/master-mission-baseline/legacy-realism-v1.json')})
    preserved=read('docs/financial-realism-2026-11-25/evidence/preserved-hashes.json')
    mismatches=[name for name,h in preserved.items() if not Path(name).is_file() or file_hash(name)!=h]
    emit('final-preservation.json',{'classification':'SOFTWARE_FIXTURE','old_artifacts_verified':len(preserved),'mismatches':mismatches})
    if mismatches:raise ValueError('Protected research changed')
    issuer=root/'audit-output/extended-issuer-v1'
    emit('final-issuer-extension-sources.json',{'classification':'OBSERVED_MARKET',
        'supersedes_initial_importer_limits':True,'QLD_2023':'BLOCKED: positive vendor payout conflicts with affirmative zero ICI/API',
        'SSO_2021_2025':'Reviewed explicit zero rows, split factors and all20 cash-source checks',
        'sources':[{'year':y,'url':f'https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_{y}.{ext}',
            'sha256':file_hash(issuer/f'letf-ici-primary-{y}.{ext}'),'license':'Issuer copyright; private raw preservation only'}
            for y,ext in ((2021,'xlsx'),(2022,'xls'),(2023,'xlsx'))]})
    manifest={'classification':'BLOCKED','purpose':'Mixed evidence pack; no investment-prediction certification',
        'exporter_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'exporter_sha256':file_hash(__file__),'private_producer_artifact_hashes':artifacts,'public_aggregate_hashes':dict(written)}
    emit('final-manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=export(a.root.resolve(),a.output);print(json.dumps({'aggregate_reports':len(r['public_aggregate_hashes']),
        'raw_provider_data_exported':False,'financial_certification':'BLOCKED'}))
