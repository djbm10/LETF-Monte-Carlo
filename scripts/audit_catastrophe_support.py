"""Enumerate old fixed-clock support without assigning unprecedented-shock risk."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from letf.macro.artifacts import ArtifactStore,canonical
from letf.macro.models import MacroModel
from letf.macro.precision import MonthlyLookup
from letf.macro.precision_run import file_hash


def audit(experiment,output):
    if output.exists():raise ValueError('Existing support audit cannot be overwritten')
    registration=json.loads((experiment/'registration.json').read_text())
    meta,a=ArtifactStore(experiment/'artifacts').load('models',registration['model_id'])
    model=MacroModel(meta['kind'],meta['columns'],a['training'],meta['parameters'])
    p=registration['protocol'];lookup=MonthlyLookup.build(model,a['templates'],a['lengths'],fee=p['annual_fee_leveraged'],spread=p['borrowing_spread'])
    rows=[]
    for leverage in p['leverages']:
        gross=lookup.summaries[leverage-1,:,:,4]
        rows.append({'leverage':leverage,'minimum_daily_gross':float(gross.min()),
            'support_contains_one_session_extinction':bool((gross<=0).any()),
            'extinction_threshold_underlying_at_rate_min':(-1+(p['annual_fee_leveraged']/252 if leverage>1 else 0)+(leverage-1)*(lookup.rates.min()+p['borrowing_spread'])/252)/leverage,
            'extinction_threshold_underlying_at_rate_max':(-1+(p['annual_fee_leveraged']/252 if leverage>1 else 0)+(leverage-1)*(lookup.rates.max()+p['borrowing_spread'])/252)/leverage})
    result={'model_id':registration['model_id'],'registration_sha256':file_hash(experiment/'registration.json'),
        'training_period':model.parameters['training_end'],'calendar_window_months':len(model.train_values),
        'finite_daily_shapes':len(lookup.shapes),'lagged_rate_states':len(lookup.rates),
        'minimum_underlying_daily_return_in_transformed_support':float(np.expm1(lookup.shapes).min()),
        'results':rows,'model_approved':False,
        'limitations':['Rescaled historical templates are finite hypothetical support, not all feasible market shocks',
            'Stationary blocks can concatenate long bears but cannot identify unprecedented shock frequencies',
            'Zero event count is structural for fund extinction under this support, not a calibrated market probability',
            'Actual fund closure, market suspension, counterparty failure, tracking discontinuities and future tax law are additional risks',
            'New stress cases are unweighted interventions; they do not alter frozen forecasts or their probabilities']}
    with output.open('xb') as f:f.write(canonical(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.experiment,a.output),indent=2))
