import json
from pathlib import Path

import pytest

from letf.lab.catalog import SourceLake, file_hash, require_available_at
from letf.lab.public_sources import refresh
from letf.lab.evidence import evidence
from letf.accuracy import UnsupportedModelError


def metadata():
    return dict(provider='IRS fixture',url='https://www.irs.gov/',retrieved_at='2026-11-25T12:00:00+00:00',
                license='Public domain fixture',convention='test-only observations')


def test_source_revisions_deduplicate_bytes_but_keep_vintage_and_detect_tampering(tmp_path):
    lake=SourceLake(tmp_path/'lake');meta=metadata()
    first=lake.put('X',b'1,2\n',meta)
    assert lake.put('X',b'1,2\n',meta)==first
    second=lake.put('X',b'1,2\n',dict(meta,publication_vintage='later'))
    assert second!=first and len(list((lake.root/'objects').iterdir()))==1
    assert lake.audit()['snapshots_verified']==2
    _,path=lake.load(first)
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):lake.audit()


def test_catalog_adapter_preserves_missing_provenance_and_checks_original_hash(tmp_path):
    source=tmp_path/'inputs';source.mkdir();raw=source/'x.csv';raw.write_text('date,X\n2024-01-02,1\n')
    catalog={'sources':{'X':dict(metadata(),path='x.csv',sha256=file_hash(raw))},'blocked':{'Y':'unavailable'}}
    (source/'catalog.json').write_text(json.dumps(catalog))
    lake=SourceLake(tmp_path/'lake');_,record=lake.ingest(source,namespace='original')
    result,_=lake.load(record['sources']['X'])
    assert not result['metadata']['point_in_time_verified'] and record['blocked']==catalog['blocked']
    assert lake.audit()['snapshots_verified']==2
    raw.write_text('changed')
    with pytest.raises(ValueError,match='checksum'):lake.ingest(source,namespace='original')


@pytest.mark.parametrize('relative',['../outside.csv','/etc/passwd','x/../x.csv','./x.csv','x\\csv'])
def test_import_never_reads_traversal(tmp_path,relative):
    root=tmp_path/'inputs';root.mkdir();(root/'catalog.json').write_text(json.dumps({'sources':{'x':dict(metadata(),path=relative,sha256='0'*64)}}))
    with pytest.raises(ValueError):SourceLake(tmp_path/'lake').ingest(root,namespace='bad')


def test_symlink_source_rejected_even_inside_bundle(tmp_path):
    raw=tmp_path/'raw.csv';raw.write_bytes(b'X')
    (tmp_path/'link.csv').symlink_to(raw)
    (tmp_path/'catalog.json').write_text(json.dumps({'sources':{'x':dict(metadata(),path='link.csv',sha256=file_hash(raw))}}))
    with pytest.raises(ValueError,match='symlink'):SourceLake(tmp_path/'lake').ingest(tmp_path,namespace='bad')


def test_release_availability_is_not_inferred_from_download_or_vintage():
    with pytest.raises(UnsupportedModelError):require_available_at(metadata(),'2026-12-01T00:00:00+00:00')
    meta=dict(metadata(),point_in_time_verified=True,available_at='2026-12-10T13:30:00+00:00')
    with pytest.raises(UnsupportedModelError):require_available_at(meta,'2026-12-10T13:29:00+00:00')
    require_available_at(meta,'2026-12-10T13:30:00+00:00')


def test_public_refresh_keeps_missing_rows_and_never_implies_tax_classification(tmp_path):
    result=refresh(tmp_path/'lake',['DFF'],downloader=lambda url:(b'observation_date,DFF\n2024-01-02,5.33\n2024-01-03,.\n',{}))
    assert result['complete'] and result['sources']['DFF']['missing_values']==1
    with pytest.raises(ValueError):refresh(tmp_path/'lake',['PRIVATE_URL'])
    broken=refresh(tmp_path/'lake',['DFF'],downloader=lambda url:(b'<html>blocked</html>',{}))
    assert not broken['complete'] and broken['sources']['DFF']['status']=='BLOCKED'
    with pytest.raises(ValueError):evidence('RECONSTRUCTED_WITH_VALIDATION',model_approved=True)
