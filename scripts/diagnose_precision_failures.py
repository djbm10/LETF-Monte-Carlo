"""Read-only diagnosis of all original 15 failed sampling gates."""
from hashlib import sha256
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import norm
from letf.macro.artifacts import ArtifactStore,canonical
from letf.macro.precision import quantile_interval
from letf.macro.precision_run import json_safe


def diagnose(registry,report,output):
    if output.exists():raise ValueError('Never overwrite existing diagnosis')
    old=json.loads(report.read_text());store=ArtifactStore(registry/'artifacts');samples=[]
    for fid in old['forecast_ids']:
        choices=[]
        for path in (store.root/'evaluations').iterdir():
            meta,a=store.load('evaluations',path.name)
            if meta['forecast_id']==fid:choices.append(a)
        if len(choices)!=1:raise ValueError('Ambiguous/missing original samples')
        samples.append(choices[0])
    rows=[];z=norm.ppf(1-.025/(45*6)/2)
    for original in old['results']:
        h=original['horizon'];l=original['leverage'];key=f'{h}y_{l}x_unit_nav'
        pooled=np.log(np.concatenate([a[key] for a in samples]));intervals=[]
        for q in (.1,.5,.9):
            r=quantile_interval(pooled,q,.05)
            density=float(np.mean(abs(pooled-r['estimate'])<=.02)/.04)
            intervals.append(dict(r,quantile=q,pilot_local_log_density=density,
                approximate_pooled_paths_for_registered_simultaneous_half_width=float((z*np.sqrt(q*(1-q))/(density*.02))**2) if density else None))
        rows.append({'horizon':h,'leverage':l,'original_failed':not original['convergence_passed'],
            'original_log_quantile_discrepancies':original['maximum_abs_log_quantile_differences'],
            'sampling_intervals95':intervals,'minimum_observed_unit_nav':float(np.exp(pooled.min())),
            'actual_liquidations_in_original_sample':sum(int(a[f'{h}y_{l}x_liquidated'].sum()) for a in samples),
            'scope':'Density-based path counts are pilot approximations, not guarantees or future risk calibration'})
    r={'original_report_sha256':sha256(report.read_bytes()).hexdigest(),'combinations':rows,
        'causes':['1024/2048 prefixes are individually noisy and remain fixed in the old gate',
            'Log wealth dispersion increases with leverage/horizon; probability precision does not imply percentile precision',
            'No actual liquidation atom in these samples; this does not establish real-world absence of liquidation',
            'Original routine used cumprod and no percentile sampling intervals; new engine retains finite log NAV below floating underflow'],
        'original_results_modified':False}
    output.write_bytes(canonical(json_safe(r)));return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry','report','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=diagnose(a.registry,a.report,a.output);print('diagnosed',len(r['combinations']))
