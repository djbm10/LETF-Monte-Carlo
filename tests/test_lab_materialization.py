import json
from pathlib import Path

import pytest

from letf.lab.catalog import SourceLake,file_hash
from letf.lab.operations import materialize_sources


def test_refresh_materializes_new_vintage_without_changing_any_original_bytes(tmp_path):
    original=tmp_path/'old';original.mkdir();raw=original/'DFF.csv';raw.write_text('date,DFF\n2024-01-02,5.33\n')
    meta=dict(path='DFF.csv',sha256=file_hash(raw),provider='SOFTWARE_FIXTURE',url='https://example.invalid/DFF',
        retrieved_at='2024-01-03T00:00:00+00:00',license='Fixture only',convention='Observed test APR, percent')
    (original/'catalog.json').write_text(json.dumps({'sources':{'DFF':meta}}));before=file_hash(original/'catalog.json')
    lake=SourceLake(tmp_path/'lake')
    new_meta=dict(meta,retrieved_at='2024-01-04T00:00:00+00:00',format='csv')
    ident=lake.put('DFF',b'date,DFF\n2024-01-02,5.33\n2024-01-03,5.33\n',new_meta)
    updated=materialize_sources(original,lake,{'DFF':ident},tmp_path/'new')
    assert file_hash(original/'catalog.json')==before
    assert raw.read_text()=='date,DFF\n2024-01-02,5.33\n'
    assert updated.catalog['parent_catalog_sha256']==before
    assert updated.bytes('DFF')[0].endswith(b'2024-01-03,5.33\n')
    with pytest.raises(FileExistsError):materialize_sources(original,lake,{'DFF':ident},tmp_path/'new')
    wrong=lake.put('OTHER',b'date,OTHER\n2024-01-02,5.33\n',new_meta)
    with pytest.raises(ValueError,match='identity'):materialize_sources(original,lake,{'DFF':wrong},tmp_path/'bad')
