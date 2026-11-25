"""Registered cost/history sensitivities, not selection on future outcomes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letf.macro.data import HistoricalSources
from letf.macro.precision_run import run_precision,json_safe
from letf.macro.artifacts import canonical


def run(sources,protocol,sensitivity,output):
    output.mkdir(parents=True,exist_ok=False);results={}
    for name,changes in sensitivity['cases'].items():
        child=dict(protocol,**changes,checkpoints_per_seed=sensitivity['checkpoints'],
            family_alpha_quantiles=.025/len(sensitivity['cases']),family_alpha_probabilities=.025/len(sensitivity['cases']))
        r=run_precision(sources,child,output=output/name)
        results[name]={'passed_precision':r['passed'],'paths_per_seed':r['paths_per_seed'],
            'model_approved':False,'changes':changes,'results':r['results']}
    report={'evidence':'HYPOTHETICAL_STRESS','purpose':sensitivity['purpose'],'specification_sensitivity':results,
        'no_approved_model':True,'no_probability_weights_assigned_to_specifications':True,
        'source_uncertainty':'Historical vintage and missing dated financing uncertainty is not identified by this finite set of specifications'}
    (output/'report.json').write_bytes(canonical(json_safe(report)));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('sources','protocol','sensitivity','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(HistoricalSources(a.sources),json.loads(a.protocol.read_text()),json.loads(a.sensitivity.read_text()),a.output)
