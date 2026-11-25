from datetime import datetime,timezone
import json

import pandas as pd
import pytest

from letf.accuracy import UnsupportedModelError
from letf.lab.catalog import file_hash
from letf.lab.derivative_inputs import load_inputs,quote_at


def bundle(root,kind='option',**changes):
    root.mkdir(exist_ok=True)
    if kind=='option':
        spec=dict(symbol='X_CALL',underlying='X',strike=100.,expiry='2025-01-17',right='call',
            style='american',settlement='physical',multiplier=100,adjusted=False,available_at='2025-01-01T00:00:00Z')
        rows=[dict(timestamp='2025-01-03T15:00:00Z',contract_symbol='X_CALL',bid=2.,ask=2.5,
            bid_size=2,ask_size=1,source='SOFTWARE_FIXTURE quote',available_at='2025-01-03T15:00:01Z')]
        role='quotes'
    else:
        spec=dict(symbol='MESZ2025',expiry='2025-12-19',multiplier=5.,tick_size=.25,initial_margin=1500.,
            maintenance_margin=1200.,margin_as_of='2025-01-01',source='SOFTWARE_FIXTURE margins, CME contract units',
            section1256=True,settlement_type='cash',first_notice='',currency='USD',available_at='2025-01-01T00:00:00Z')
        rows=[dict(date='2025-01-03',contract='MESZ2025',expiry='2025-12-19',settlement=5000.,available_at='2025-01-03T22:00:00Z')]
        role='settlements'
    rows[0].update(changes)
    pd.DataFrame([spec]).to_csv(root/'contracts.csv',index=False)
    pd.DataFrame(rows).to_csv(root/(role+'.csv'),index=False)
    files={k:dict(path=k+'.csv',sha256=file_hash(root/(k+'.csv')),provider='SOFTWARE_FIXTURE',
        license='Test authored records',retrieved_at='2025-01-04T00:00:00Z',convention='USD nominal dated contract fields') for k in ('contracts',role)}
    (root/'manifest.json').write_text(json.dumps(dict(schema_version=1,kind=kind,classification='SOFTWARE_FIXTURE',files=files)))
    return root


def test_option_feed_preserves_depth_publication_and_canonical_quote_objects(tmp_path):
    inputs=load_inputs(bundle(tmp_path/'quotes'))
    with pytest.raises(UnsupportedModelError,match='available'):
        quote_at(inputs,'X_CALL','2025-01-03T15:00:00Z')
    decision=datetime(2025,1,3,15,0,1,tzinfo=timezone.utc)
    quote=quote_at(inputs,'X_CALL',decision)
    assert quote.fill(decision,'BUY',1)==2.5
    with pytest.raises(UnsupportedModelError,match='size'):quote.fill(decision,'BUY',2)
    assert inputs['summary']['historical_execution_certification']=='BLOCKED'
    with pytest.raises(UnsupportedModelError):load_inputs(tmp_path/'quotes',decision_at='2025-01-03T15:00:00Z')


@pytest.mark.parametrize('changes',[{'ask_size':1.5},{'ask_size':''},{'bid':3.},{'available_at':'2025-01-03T14:59:00Z'}])
def test_missing_depth_crossed_and_unavailable_quote_records_reject(tmp_path,changes):
    with pytest.raises((ValueError,UnsupportedModelError)):load_inputs(bundle(tmp_path/'quotes',**changes))


def test_futures_are_dated_ticks_and_source_hashes_not_continuous_charts(tmp_path):
    root=bundle(tmp_path/'future','future');inputs=load_inputs(root)
    c=inputs['contracts']['MESZ2025'];assert c.tick_size*c.multiplier==1.25
    with pytest.raises(ValueError,match='checksum'):
        (root/'settlements.csv').write_text('corrupt');load_inputs(root)
    with pytest.raises(ValueError,match='grid'):load_inputs(bundle(tmp_path/'offtick','future',settlement=5000.1))
    with pytest.raises(ValueError,match='expiry'):load_inputs(bundle(tmp_path/'expiry','future',expiry='2026-03-20'))


def test_provider_demo_and_two_entitlement_variants_never_certify_actual_option_history(tmp_path):
    import zipfile
    from letf.lab.derivative_inputs import inspect_cboe_sample
    row=dict(underlying_symbol='SPY',quote_date='2023-08-25',root='SPY',expiration='2023-09-15',strike=400.,option_type='C',delivery_code='UNKNOWN',
        bid_size_1545=1,bid_1545=2.,ask_size_1545=1,ask_1545=3.,bid_size_eod=0,bid_eod=0.,ask_size_eod=0,ask_eod=0.)
    p=tmp_path/'sample.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('OptionEOD_Sample_ReadMe.txt','Sample files are for demonstration purposes only.')
        z.writestr('UnderlyingOptionsEODCalcs_2023-08-25_no_cgi_subscription.csv',pd.DataFrame([row]).to_csv(index=False))
    r=inspect_cboe_sample(p)
    assert r['classification']=='SOFTWARE_FIXTURE' and r['rows']==1
    assert not r['other_entitlement_variant_is_independent_data']
    assert r['historical_execution_certification']=='BLOCKED'
