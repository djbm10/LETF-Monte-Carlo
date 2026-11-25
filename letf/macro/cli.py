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
    for name in ('catalog','counterfactual','walkforward'):
        q=sub.add_parser(name);q.add_argument('--sources',required=True,type=Path)
        if name!='catalog':q.add_argument('--output',required=True,type=Path)
        if name=='counterfactual':q.add_argument('--start',default='1930-01-01');q.add_argument('--end')
        if name=='walkforward':
            q.add_argument('--registry',required=True,type=Path)
            q.add_argument('--protocol',default=Path('docs/macro-learning-2026-11-24/protocol.json'),type=Path)
            q.add_argument('--paths',type=int)
    a=p.parse_args();sources=HistoricalSources(a.sources)
    if a.command=='catalog':
        print(json.dumps(sources.catalog,indent=2));return
    if a.command=='counterfactual':
        r=historical_counterfactual(sources,start=a.start,end=a.end,output=a.output)
        print(json.dumps({'exposures':list(r['exposures']),'evidence':r['evidence'],'report':str(a.output/'report.json')}));return
    protocol=json.loads(a.protocol.read_text())
    r=walkforward_study(sources,ModelRegistry(a.registry),protocol,output=a.output,paths=a.paths)
    print(json.dumps({'forecasts':len(r['forecast_ids']),'blocked':len(r['blocked_models']),
        'decisions':r['model_comparison'],'report':str(a.output/'report.json')},indent=2))


if __name__=='__main__':main()
