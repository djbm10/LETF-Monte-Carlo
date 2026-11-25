import json

import pytest

from letf.lab.account_precision import assess_samples,study
from letf.lab.catalog import file_hash,private_json


def rows(n,offset=0):
    return [dict(path=i,horizon=5,strategy='US1X_CF',account=a,net_terminal_wealth=10000.+offset,
        fund_liquidation=False,metrics={'max_drawdown':0.}) for a in ('taxable','roth') for i in range(n)]


def test_sampling_acceptance_needs_independent_roots_stability_and_probabilities():
    one=assess_samples({1:{11:rows(1),22:rows(1)}},planned_looks=2,seeds=[11,22])
    assert not one['account_sampling_gates_passed'] and not one['model_approved']
    constant=assess_samples({1000:{11:rows(1000),22:rows(1000)},2000:{11:rows(2000),22:rows(2000)}},planned_looks=2,seeds=[11,22])
    assert constant['account_sampling_gates_passed'] and not constant['model_approved']
    shifted=assess_samples({1000:{11:rows(1000),22:rows(1000)},2000:{11:rows(2000),22:rows(2000,1000)}},planned_looks=2,seeds=[11,22])
    assert not shifted['account_sampling_gates_passed']
    with pytest.raises(ValueError,match='independent'):assess_samples({},planned_looks=2,seeds=[11,11])
    with pytest.raises(ValueError,match='duplicate'):assess_samples({2:{11:rows(1),22:rows(2)}},planned_looks=2,seeds=[11,22])


def test_full_ledger_study_checkpoints_resume_without_overwriting_or_changing_protocol(tmp_path):
    sources=tmp_path/'sources';sources.mkdir();(sources/'raw.csv').write_text('date,X\n2025-01-02,1\n')
    (sources/'catalog.json').write_text(json.dumps({'sources':{'X':{'path':'raw.csv','sha256':file_hash(sources/'raw.csv')}}}))
    spec=dict(schema=1,experiment={'sources':str(sources),'schema':1,'mode':'future'},counts=[1,2],root_seeds=[11,22])
    calls=[]
    def fixture_runner(request,*,output):
        calls.append((request['paths'],request['seed']));output.mkdir()
        private_json(output/'report.json',{'path_results':rows(request['paths'])})
    root=tmp_path/'study'
    first=study(spec,root,max_new_runs=1,runner=fixture_runner)
    assert len(calls)==1 and len(first['pending'])==3 and not first['study_complete']
    receipt=(root/'paths-1-seed-11/completion.json').read_bytes()
    second=study(spec,root,max_new_runs=3,runner=fixture_runner)
    assert len(calls)==4 and second['study_complete'] and not second['account_convergence_certified']
    assert (root/'paths-1-seed-11/completion.json').read_bytes()==receipt
    assert len(list(root.glob('assessment-*.json')))==2
    with pytest.raises(ValueError,match='Registered'):
        study(dict(spec,root_seeds=[11,33]),root,runner=fixture_runner)
    (root/'paths-1-seed-11/report.json').write_text('{}')
    with pytest.raises(ValueError,match='corrupted'):study(spec,root,runner=fixture_runner)
