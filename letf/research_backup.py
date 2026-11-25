"""Private local research bundles and safe verified recovery; no network transfer.

A checksum-authenticated restore drill is not evidence of an off-host backup.
Only explicitly selected research roots are read. Credentials, links and unknown
file types fail closed; nothing is discovered from home/cloud configuration.
"""
from datetime import datetime,timezone
import argparse,json,os,re,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath
from hashlib import sha256

from letf.macro.artifacts import canonical,fsync_directory
from letf.macro.precision_run import file_hash

ALLOWED_SUFFIXES={'.json','.jsonl','.csv','.txt','.npz','.zip','.bin','.pdf','.xls','.xlsx','.md'}


def safe_name(name):
    p=PurePosixPath(name)
    if (not name or p.is_absolute() or str(p)!=name or '..' in p.parts or '\\' in name or ':' in name
            or len(name)>512):raise ValueError('Unsafe research archive path')
    return p


def private_file(path):
    return os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb')


def publish_recovery(stage,target):
    """Linux atomic no-replace directory publication; never replace a rival."""
    import ctypes
    library=ctypes.CDLL(None,use_errno=True)
    rename=getattr(library,'renameat2',None)
    if rename is None:raise NotImplementedError('Verified atomic research restore currently requires Linux renameat2')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(stage),-100,os.fsencode(target),1):
        code=ctypes.get_errno();raise OSError(code,os.strerror(code),str(target))


def create_bundle(roots,output,*,provenance=None):
    output=Path(output)
    if output.exists():raise ValueError('Existing private bundle cannot be overwritten')
    inventory={};sources={}
    for label,root in roots.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',label):raise ValueError('Simple unique research root label required')
        root=Path(root)
        if not root.exists() or root.is_symlink():raise ValueError('Explicit existing non-symlink research root required')
        for p in sorted(root.rglob('*') if root.is_dir() else [root]):
            if p.is_symlink():raise ValueError('Research archive cannot contain symlinks')
            if p.is_dir():continue
            if p.name=='.lock':continue
            if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES or any(s in p.name.lower() for s in ('.env','credential','secret','id_rsa','token')):
                raise ValueError('Unreviewed or credential-like research file')
            name=label+'/'+(str(p.relative_to(root)) if root.is_dir() else p.name)
            safe_name(name);sources[name]=p;inventory[name]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    if not inventory:raise ValueError('Empty research backup')
    manifest={'schema':'private-research-recovery-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'files':inventory,'provenance':provenance or {},'off_host_verified':False,
        'license':'Private preservation only; no redistribution rights inferred'}
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,temporary=tempfile.mkstemp(prefix='.research-pending-',dir=output.parent);pending=Path(temporary)
    try:
        with os.fdopen(fd,'wb') as stream:
            with tarfile.open(fileobj=stream,mode='w:gz') as archive:
                from io import BytesIO
                blob=canonical(manifest);header=tarfile.TarInfo('recovery-manifest.json');header.size=len(blob);header.mode=0o600
                archive.addfile(header,BytesIO(blob))
                for name,item in inventory.items():
                    header=tarfile.TarInfo(name);header.size=item['bytes'];header.mode=0o600
                    with sources[name].open('rb') as source:archive.addfile(header,source)
            stream.flush();os.fsync(stream.fileno())
        digest=file_hash(pending);verify_bundle(pending,digest)
        # A hard link publishes an immutable filename without an overwrite race.
        os.link(pending,output);pending.unlink();fsync_directory(output.parent)
        return {'archive_sha256':digest,'archive_bytes':output.stat().st_size,'files':len(inventory),
            'uncompressed_bytes':sum(v['bytes'] for v in inventory.values()),'verified_member_hashes':True,
            'off_host_verified':False,'approved_remote_destination':None,'created_at':manifest['created_at']}
    finally:
        if pending.exists():pending.unlink()


def verify_bundle(archive,expected_sha256,*,maximum_bytes=20_000_000_000):
    if not re.fullmatch('[a-f0-9]{64}',expected_sha256 or '') or file_hash(archive)!=expected_sha256:
        raise ValueError('Private archive does not match externally retained trusted checksum')
    # Compressed-tar random seeks repeatedly decompress the archive, becoming
    # quadratic when canonical manifest order differs from physical file order.
    # Read every member once in physical order, with bounded manifest/size checks.
    with tarfile.open(archive,'r|gz') as tar:
        first=tar.next()
        if first is None or first.name!='recovery-manifest.json' or first.type!=tarfile.REGTYPE:
            raise ValueError('Missing first ordinary recovery manifest')
        if not 0<=first.size<=50_000_000:raise ValueError('Oversized manifest')
        manifest=json.load(tar.extractfile(first));files=manifest.get('files',{})
        if manifest.get('schema')!='private-research-recovery-v1' or not files or len(files)>99999:
            raise ValueError('Recovery manifest inventory mismatch')
        for name,identity in files.items():
            safe_name(name)
            if name=='recovery-manifest.json' or type(identity.get('bytes'))is not int or identity['bytes']<0 or not re.fullmatch('[a-f0-9]{64}',identity.get('sha256','')):
                raise ValueError('Invalid recovery identity')
        if sum(v['bytes'] for v in files.values())+first.size>maximum_bytes:raise ValueError('Recovery exceeds reviewed size budget')
        seen=set()
        for member in tar:
            if member is first:continue  # tar's iterator can yield its cached first header
            safe_name(member.name)
            if member.name in seen or member.name not in files:raise ValueError('Duplicate or undeclared archive member')
            if member.type!=tarfile.REGTYPE or member.size!=files[member.name]['bytes']:raise ValueError('Nonordinary file or recovery size mismatch')
            seen.add(member.name);h=sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=files[member.name]['sha256']:raise ValueError('Private research member checksum mismatch')
        if seen!=set(files):raise ValueError('Missing recovery files')
    return manifest


def restore_bundle(archive,expected_sha256,target):
    """Verify completely before writing, then restore regular files privately."""
    target=Path(target)
    if target.exists() or target.is_symlink():raise ValueError('Recovery target must be new; no replacement of research')
    manifest=verify_bundle(archive,expected_sha256)
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    stage=Path(tempfile.mkdtemp(prefix='.recovery-pending-',dir=target.parent))
    try:
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if member.name=='recovery-manifest.json':continue
                name=member.name;identity=manifest['files'][name]
                destination=stage.joinpath(*safe_name(name).parts)
                parent=destination.parent
                # mkdir(parents=True) need not apply its mode to intermediate
                # directories. Set each private archive-created directory.
                relative=parent.relative_to(stage);cursor=stage
                for part in relative.parts:
                    cursor=cursor/part;cursor.mkdir(exist_ok=True,mode=0o700)
                with tar.extractfile(member) as source,private_file(destination) as out:
                    shutil.copyfileobj(source,out,1024*1024);out.flush();os.fsync(out.fileno())
                if file_hash(destination)!=identity['sha256']:raise ValueError('Restored research bytes failed verification')
        for directory in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda x:len(x.parts),reverse=True):fsync_directory(directory)
        fsync_directory(stage)
        if target.exists():raise ValueError('Recovery target appeared during restore')
        publish_recovery(stage,target);fsync_directory(target.parent)
        return {'status':'VERIFIED_LOCAL_RECOVERY','files':len(manifest['files']),
            'archive_sha256':expected_sha256,'off_host_verified':False,
            'file_mode':'0600','directory_mode':'0700','provenance':manifest['provenance']}
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('create','verify','restore'))
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--trusted-sha256')
    parser.add_argument('--roots',type=Path);parser.add_argument('--target',type=Path);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if args.receipt.exists():raise ValueError('Receipt must be a new immutable file')
    if args.command=='create':
        specification=json.loads(args.roots.read_text());result=create_bundle(specification['roots'],args.archive,provenance=specification.get('provenance'))
    elif args.command=='restore':result=restore_bundle(args.archive,args.trusted_sha256,args.target)
    else:
        m=verify_bundle(args.archive,args.trusted_sha256);result={'status':'VERIFIED_LOCAL_BUNDLE','files':len(m['files']),'off_host_verified':False}
    with private_file(args.receipt) as stream:stream.write(canonical(result));stream.flush();os.fsync(stream.fileno())
    fsync_directory(args.receipt.parent);print(json.dumps(result,indent=2))
