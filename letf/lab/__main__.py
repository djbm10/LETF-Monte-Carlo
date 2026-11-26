"""python -m letf.lab: audited integration of existing research workflows."""
import argparse
import json
from pathlib import Path
import sys

from letf.accuracy import UnsupportedModelError
from letf.account_cli import _serialize
from letf.lab.catalog import SourceLake, private_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('observed','tournament','simulate','cycle'):
        p=sub.add_parser(name);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('account-precision');p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-new-runs',type=int,default=1)
    p=sub.add_parser('sources');p.add_argument('--lake',required=True,type=Path);p.add_argument('--output',type=Path)
    p=sub.add_parser('ingest');p.add_argument('--lake',required=True,type=Path);p.add_argument('--folder',required=True,type=Path)
    p.add_argument('--catalog-file',default='catalog.json');p.add_argument('--namespace',required=True)
    p=sub.add_parser('refresh');p.add_argument('--lake',required=True,type=Path);p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True,type=Path)
    sub.add_parser('strategies')
    p=sub.add_parser('derivative-inputs');p.add_argument('--folder',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--decision-at')
    p=sub.add_parser('cboe-sample');p.add_argument('--archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('demo');p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('counterfactual');p.add_argument('--sources',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--start',default='1930-01-01');p.add_argument('--end')
    p=sub.add_parser('stress');p.add_argument('--experiment',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--protocol',type=Path,default=Path('docs/financial-realism-2026-11-25/protocol.json'))
    p.add_argument('--cases',nargs='+');p.add_argument('--anchors',type=int,choices=[1,2])
    p=sub.add_parser('render');p.add_argument('--report',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('serve');p.add_argument('--profiles',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path);p.add_argument('--port',type=int,default=8765)
    a=parser.parse_args(argv)
    try:
        if a.command=='sources':
            result=SourceLake(a.lake).audit()
            if a.output: private_json(a.output,result)
            print(json.dumps({'snapshots_verified':result['snapshots_verified'],'classification':result['classification'],'financial_certification':'BLOCKED'}));return 0
        if a.command=='ingest':
            identifier,record=SourceLake(a.lake).ingest(a.folder,catalog_file=a.catalog_file,namespace=a.namespace)
            print(json.dumps({'import_id':identifier,'sources':len(record['sources']),'blocked':record['blocked']}));return 0
        if a.command=='refresh':
            from letf.lab.public_sources import refresh
            result=refresh(a.lake,a.names);private_json(a.output,result)
            print(json.dumps(result));return 0 if result['complete'] else 2
        if a.command=='strategies':
            from letf.lab.strategies import registry
            print(json.dumps(registry(),indent=2));return 0
        if a.command=='derivative-inputs':
            from letf.lab.derivative_inputs import load_inputs
            result=load_inputs(a.folder,decision_at=a.decision_at)['summary'];private_json(a.output,result)
            print(json.dumps(result));return 0
        if a.command=='cboe-sample':
            from letf.lab.derivative_inputs import inspect_cboe_sample
            result=inspect_cboe_sample(a.archive);private_json(a.output,result);print(json.dumps(result));return 0
        if a.command=='demo':
            from letf.lab.fixtures import demo
            result=demo(a.output);print(json.dumps(result));return 0
        if a.command=='observed':
            from letf.lab.portfolio import run_observed
            result=run_observed(json.loads(a.config.read_text()),output=a.output)
            print(json.dumps({'classification':result['classification'],'accounts':{k:v['ending_wealth_net_payable'] for k,v in result['accounts'].items()},'report':str(a.output/'report.json')}));return 0
        if a.command=='counterfactual':
            from letf.macro.data import HistoricalSources
            from letf.macro.counterfactual import historical_counterfactual
            r=historical_counterfactual(HistoricalSources(a.sources),start=a.start,end=a.end,output=a.output)
            envelope={'classification':'HISTORICAL_COUNTERFACTUAL','financial_certification':'BLOCKED',
                'report':'report.json','report_source_status':r['evidence'],'literal_products_existed':False,
                'source_catalog_sha256':r['source_catalog_sha256'],'account_history_certification':'BLOCKED'}
            private_json(a.output/'laboratory-evidence.json',envelope);print(json.dumps(envelope));return 0
        if a.command=='stress':
            from letf.account_stress import run_study
            result=run_study(a.experiment,json.loads(a.protocol.read_text()),a.output,cases=a.cases,anchors=a.anchors)
            envelope={'classification':'CONDITIONAL_STRESS','model_approved':False,'event_probabilities':'UNSPECIFIED',
                      'registered_cases':len(result['results']),'all_cases_supported':result['all_account_reconciliations_passed']}
            private_json(a.output/'laboratory-evidence.json',envelope);print(json.dumps(envelope))
            return 0 if result['all_account_reconciliations_passed'] else 2
        if a.command=='simulate':
            from letf.lab.workflows import simulate
            result=simulate(json.loads(a.config.read_text()),output=a.output)
            print(json.dumps({'classification':result['classification'],'model_approved':False,'report':str(a.output/'laboratory-report.json')}));return 0
        if a.command=='account-precision':
            from letf.lab.account_precision import study
            result=study(json.loads(a.config.read_text()),a.output,max_new_runs=a.max_new_runs)
            print(json.dumps(result));return 0 if result['account_convergence_certified'] else 2
        if a.command=='tournament':
            from letf.lab.workflows import tournament
            result=tournament(json.loads(a.config.read_text()),output=a.output)
            print(json.dumps(result,default=_serialize));return 0 if result['all_requested_cells_supported'] else 2
        if a.command=='cycle':
            from letf.lab.operations import cycle
            result=cycle(json.loads(a.config.read_text()),output=a.output);print(json.dumps(result,default=_serialize));return 0
        if a.command=='render':
            from letf.lab.html import render_report
            render_report(a.report,a.output);print(a.output);return 0
        if a.command=='serve':
            from letf.lab.server import serve
            serve(json.loads(a.profiles.read_text()),a.output_root,port=a.port);return 0
    except (ValueError,KeyError,FileNotFoundError,UnsupportedModelError) as error:
        print(json.dumps({'classification':'BLOCKED','reason':str(error)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
