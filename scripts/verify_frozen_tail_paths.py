"""Regenerate frozen monthly ensembles/lineage without reading any outcomes.

Numerical reproducibility is distinct from independent predictive validation.
This command verifies identities and leaves the source registry read-only.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from letf.macro.artifacts import ArtifactStore,canonical
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage


def verify(registry,report,output):
    if output.exists():raise ValueError('Existing reproduction evidence cannot be overwritten')
    store=ArtifactStore(registry/'artifacts');r=json.loads(report.read_text());rows=[]
    for fid in r['receipt']['forecast_ids']:
        f,a=store.load('forecasts',fid);m,x=store.load('models',f['model_id'])
        model=MacroModel(f['kind'],m['columns'],x['training'],m['parameters'])
        gap=f['unobserved_state_bridge_months']
        monthly,lineage=sample_with_lineage(model,len(f['dates'])+gap,f['paths'],f['seed'])
        if not np.array_equal(monthly[:,gap:],a['monthly']) or not np.array_equal(lineage[:,gap:],a['lineage']):
            raise AssertionError('Frozen monthly prediction/lineage failed exact reproduction')
        rows.append({'forecast_id':fid,'model_id':f['model_id'],'monthly_exact':True,'lineage_exact':True,
            'monthly_sha256':sha256(a['monthly'].tobytes()).hexdigest(),
            'lineage_sha256':sha256(a['lineage'].tobytes()).hexdigest()})
    result={'forecasts_reproduced_exactly':len(rows),'all_passed':True,'rows':rows,
        'scope':'Numerical regeneration of sealed predictions, no outcome loading or new validation claim'}
    output.write_bytes(canonical(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('registry','report','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();r=verify(a.registry,a.report,a.output);print(r['forecasts_reproduced_exactly'])
