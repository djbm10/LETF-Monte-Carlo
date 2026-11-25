"""Content-addressed immutable research artifacts; no pickle or live authority."""
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
from hashlib import sha256
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def fingerprint(frame):
    payload={'index':[str(d) for d in frame.index],'columns':list(frame.columns),
             'values_sha256':sha256(np.ascontiguousarray(frame.to_numpy(float),dtype='<f8').tobytes()).hexdigest()}
    return sha256(canonical(payload)).hexdigest()


@contextmanager
def lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


class ArtifactStore:
    def __init__(self,root):self.root=Path(root)

    def put(self,kind,metadata,arrays=None):
        if kind not in ('models','forecasts','evaluations','batches','decisions'):raise ValueError('Unknown artifact kind')
        arrays=arrays or {}
        hashes={k:sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
        identity={'metadata':metadata,'arrays':{k:{'sha256':hashes[k],'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}}
        identifier=sha256(canonical(identity)).hexdigest()
        directory=self.root/kind/identifier
        with lock(self.root):
            if directory.exists():
                self.load(kind,identifier);return identifier
            directory.parent.mkdir(parents=True,exist_ok=True)
            pending=self.root/'.pending';pending.mkdir(exist_ok=True)
            staging=Path(tempfile.mkdtemp(dir=pending))
            try:
                data_path=staging/'arrays.npz'
                with data_path.open('xb') as f:
                    np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                envelope={'id':identifier,'identity':identity,'arrays_file_sha256':sha256(data_path.read_bytes()).hexdigest(),
                          'created_at':datetime.now(timezone.utc).isoformat()}
                with (staging/'manifest.json').open('xb') as f:
                    f.write(canonical(envelope));f.flush();os.fsync(f.fileno())
                # Publish both files atomically. A failed/terminated write can
                # never expose a half-written content-addressed artifact.
                os.rename(staging,directory)
                descriptor=os.open(directory.parent,os.O_RDONLY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
            finally:
                if staging.exists():shutil.rmtree(staging)
        return identifier

    def load(self,kind,identifier):
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid artifact identifier')
        directory=self.root/kind/identifier
        envelope=json.loads((directory/'manifest.json').read_text())
        if sha256(canonical(envelope['identity'])).hexdigest()!=identifier:raise ValueError('Artifact metadata tampered')
        path=directory/'arrays.npz'
        if sha256(path.read_bytes()).hexdigest()!=envelope['arrays_file_sha256']:raise ValueError('Artifact data tampered')
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        for key,value in arrays.items():
            spec=envelope['identity']['arrays'][key]
            if sha256(np.ascontiguousarray(value).tobytes()).hexdigest()!=spec['sha256'] or list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype']:
                raise ValueError('Artifact array identity mismatch')
        return envelope['identity']['metadata'],arrays
