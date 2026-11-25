import io,json,os,tarfile
from hashlib import sha256
import pytest
from letf.research_backup import create_bundle,verify_bundle,restore_bundle,publish_recovery


def test_private_recovery_preserves_all_bytes_and_fails_on_tampering_or_overwrite(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'a.csv').write_bytes(b'original source\n')
    (source/'b.json').write_text('{"frozen":true}')
    archive=tmp_path/'sealed.tar.gz';receipt=create_bundle({'data':source},archive,provenance={'commit':'immutable-test'})
    assert not receipt['off_host_verified']
    assert archive.stat().st_mode&0o777==0o600
    expected=receipt['archive_sha256'];verify_bundle(archive,expected)
    target=tmp_path/'fresh';r=restore_bundle(archive,expected,target)
    assert r['status']=='VERIFIED_LOCAL_RECOVERY' and not r['off_host_verified']
    assert (target/'data/a.csv').read_bytes()==(source/'a.csv').read_bytes()
    assert all(p.stat().st_mode&0o777==(0o700 if p.is_dir() else 0o600) for p in [target,*target.rglob('*')])
    with pytest.raises(ValueError):restore_bundle(archive,expected,target)
    with pytest.raises(ValueError):create_bundle({'data':source},archive)
    with archive.open('ab') as stream:stream.write(b'tampering')
    with pytest.raises(ValueError,match='trusted checksum'):restore_bundle(archive,expected,tmp_path/'other')
    assert not (tmp_path/'other').exists()


@pytest.mark.parametrize('name,kind',[('../outside',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),
    ('data/link',tarfile.SYMTYPE),('data/hard',tarfile.LNKTYPE),('data\\escape',tarfile.REGTYPE)])
def test_recovery_rejects_unsafe_tar_members_even_with_matching_archive_hash(tmp_path,name,kind):
    archive=tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        header=tarfile.TarInfo(name);header.type=kind;header.linkname='../outside'
        tar.addfile(header,io.BytesIO(b''))
    digest=sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError):restore_bundle(archive,digest,tmp_path/'target')
    assert not (tmp_path/'target').exists()


def test_backup_scope_rejects_credentials_and_symlinks_instead_of_collecting_home(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'credentials.json').write_text('FAKE_TEST_ONLY')
    with pytest.raises(ValueError,match='credential'):create_bundle({'input':source},tmp_path/'one.tar.gz')
    (source/'credentials.json').unlink();(source/'link.csv').symlink_to('/etc/hosts')
    with pytest.raises(ValueError,match='symlink'):create_bundle({'input':source},tmp_path/'two.tar.gz')


def test_atomic_publication_cannot_replace_even_a_new_empty_target(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();(stage/'data.csv').write_text('preserve')
    target=tmp_path/'target';target.mkdir();identity=target.stat().st_ino
    with pytest.raises(FileExistsError):publish_recovery(stage,target)
    assert target.stat().st_ino==identity and list(target.iterdir())==[]
    assert (stage/'data.csv').read_text()=='preserve'
