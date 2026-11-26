import json

from letf.lab.catalog import SourceLake
from letf.lab.public_sources import inspect
from scripts.diagnose_refreshed_nav import diagnose


def test_new_nav_diagnostic_does_not_refit_or_replace_original_gate(tmp_path):
    lake=SourceLake(tmp_path/'lake');snapshots={}
    blobs={
        'TQQQ_nav':b'Date,NAV\n12/31/2025,100\n01/02/2026,100\n01/05/2026,100\n',
        'TQQQ_cash_2026':b'[]',
        'NASDAQXNDX':b'observation_date,NASDAQXNDX\n2025-12-31,100\n2026-01-02,100\n2026-01-05,100\n',
        'DFF':b'observation_date,DFF\n2025-12-30,0\n2025-12-31,0\n2026-01-01,0\n2026-01-02,0\n2026-01-03,0\n'}
    for name,blob in blobs.items():
        snapshots[name]=lake.put(name,blob,dict(provider='SOFTWARE_FIXTURE',url='https://example.invalid/',
            retrieved_at='2026-01-06T00:00:00Z',license='Authored fixture',convention='USD nominal',
            first_observation='2025-12-30',publication_vintage='Fixture'))
    old=tmp_path/'original.json'
    prior={'products':{'TQQQ':{'annual_fee_scenario':0.,'fitted_training_spread':0.,'errors':{'cagr_error':.024270821740895565}}},
        'gate_sha256':'ORIGINAL_GATE','original_gate':{'min_observations':504}}
    old.write_text(json.dumps(prior));before=old.read_bytes()
    r=diagnose(tmp_path/'lake',snapshots,old,tmp_path/'new.json',end='2026-01-05')
    assert r['new_errors']['cagr_error']==0.
    assert r['old_INSPECTED_2020_2025_errors']==prior['products']['TQQQ']['errors']
    assert not r['new_period_meets_original_minimum'] and not r['provenance_certified']
    assert old.read_bytes()==before


def test_cash_capture_does_not_establish_final_tax_classes():
    r=inspect('TQQQ_cash_2026',b'[{"ExDate":"2026-03-25","PayableDate":"2026-04-01","CashDividendPerShare":0.1}]')
    assert r['rows']==1 and not r['final_tax_classes_verified']
