import pytest

from letf.lab.catalog import SourceLake
from letf.research_backup import create_bundle,restore_bundle


def test_source_lake_extensionless_objects_are_backed_up_only_with_matching_content_hash(tmp_path):
    lake=SourceLake(tmp_path/'lake')
    lake.put('X',b'Authored fixture bytes',dict(provider='SOFTWARE_FIXTURE',url='https://example.invalid/',
        retrieved_at='2026-01-01T00:00:00Z',license='Authored fixture',convention='Fixture'))
    receipt=create_bundle({'lake':lake.root},tmp_path/'private.tar.gz')
    result=restore_bundle(tmp_path/'private.tar.gz',receipt['archive_sha256'],tmp_path/'restored')
    assert not result['off_host_verified'] and SourceLake(tmp_path/'restored/lake').audit()['snapshots_verified']==1
    (lake.root/'objects'/('0'*64)).write_bytes(b'Not the content addressed by its filename')
    with pytest.raises(ValueError,match='Unreviewed'):create_bundle({'lake':lake.root},tmp_path/'bad.tar.gz')
