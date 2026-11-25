"""Private content-addressed data lake adapting existing source catalogs.

Original bytes, revisions, missing fields and licenses are retained. Hashing
does not establish publication-time availability, licensed redistribution or
financial correctness. No source is interpolated or promoted by ingestion.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from letf.accuracy import UnsupportedModelError
from letf.macro.artifacts import canonical


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def private_json(path, value):
    """Atomic no-replacement publication, with private staging and file mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = canonical(value)
    private_bytes(path, data)


def private_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # fails even if an existing target is empty
        handle = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(handle)
        finally: os.close(handle)
    finally:
        os.unlink(temporary)


def inside_file(folder, relative):
    """Reject traversal and symlinks, even ones currently pointing inside."""
    folder = Path(folder).resolve()
    name = PurePosixPath(relative)
    if (not relative or '\\' in relative or ':' in relative or name.is_absolute()
            or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise ValueError('Source path must be a relative nontraversing filename')
    path = folder
    for component in name.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError('Source symlinks are not immutable catalog members')
    if not path.resolve().is_relative_to(folder) or not path.is_file():
        raise ValueError('Missing source or path outside bundle')
    return path


def digest_id(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a SHA-256 identifier')
    return value


class SourceLake:
    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink(): raise ValueError('Source lake cannot be a symlink')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Do not change permissions on an existing user directory silently.
        if self.root.stat().st_mode & 0o077:
            raise ValueError('Source lake root must have owner-only permissions (0700)')

    def put(self, name, blob, metadata):
        if not isinstance(blob, bytes) or not blob:
            raise ValueError('Nonempty original source bytes required')
        for key in ('provider', 'url', 'retrieved_at', 'license', 'convention'):
            if not metadata.get(key): raise ValueError(f'Missing source metadata: {key}')
        if not name or not isinstance(name, str): raise ValueError('Source name required')
        digest = sha256(blob).hexdigest()
        identity = {'schema': 1, 'name': name, 'raw_sha256': digest,
                    'raw_bytes': len(blob), 'metadata': metadata}
        identifier = sha256(canonical(identity)).hexdigest()
        raw = self.root / 'objects' / digest
        if raw.exists():
            if file_hash(raw) != digest: raise ValueError('Existing source object corrupted')
        else:
            try: private_bytes(raw, blob)
            except FileExistsError:
                if file_hash(raw) != digest: raise ValueError('Concurrent corrupt source object')
        target = self.root / 'snapshots' / (identifier + '.json')
        if target.exists():
            if target.read_bytes() != canonical(identity): raise ValueError('Source identity tampered')
        else:
            try: private_json(target, identity)
            except FileExistsError:
                if target.read_bytes() != canonical(identity): raise ValueError('Concurrent metadata tampering')
        return identifier

    def load(self, identifier):
        digest_id(identifier)
        path = inside_file(self.root, f'snapshots/{identifier}.json')
        identity = json.loads(path.read_text())
        if sha256(canonical(identity)).hexdigest() != identifier:
            raise ValueError('Snapshot metadata checksum mismatch')
        digest_id(identity['raw_sha256'])
        raw = inside_file(self.root, 'objects/' + identity['raw_sha256'])
        if raw.stat().st_size != identity['raw_bytes'] or file_hash(raw) != identity['raw_sha256']:
            raise ValueError('Snapshot original bytes corrupted')
        return identity, raw

    def ingest(self, folder, *, catalog_file='catalog.json', namespace):
        """Adapt old sources/files manifests without claiming missing provenance."""
        folder = Path(folder)
        original = inside_file(folder, catalog_file)
        catalog = json.loads(original.read_text())
        members = catalog.get('sources', catalog.get('files'))
        if not isinstance(members, dict) or not members:
            raise ValueError('Expected existing sources/files catalog')
        ids = {}
        for name, item in sorted(members.items()):
            path = inside_file(folder, item['path'])
            if file_hash(path) != digest_id(item['sha256']):
                raise ValueError(f'Original source checksum mismatch: {name}')
            meta = dict(item)
            meta.pop('path', None)
            meta.setdefault('retrieved_at', catalog.get('retrieved_at', catalog.get('original_vendor_capture', 'UNKNOWN')))
            meta.setdefault('license', 'UNKNOWN: local preservation only; redistribution not approved')
            meta.setdefault('convention', catalog.get('price_convention', 'UNKNOWN: original catalog does not specify'))
            meta['original_catalog_sha256'] = file_hash(original)
            meta['original_member'] = name
            meta['point_in_time_verified'] = item.get('point_in_time_verified', False)
            meta['source_integrity_only'] = True
            ids[name] = self.put(namespace + ':' + name, path.read_bytes(), meta)
        # Store the actual catalog too; interpretation can be reproduced later.
        catalog_id = self.put(namespace + ':catalog', original.read_bytes(), {
            'provider': 'Repository source adapter', 'url': 'local:' + catalog_file,
            'retrieved_at': catalog.get('retrieved_at', 'UNKNOWN'),
            'license': 'Contains provider metadata; original member rights retained',
            'convention': 'Original source catalog bytes; no normalized observations'})
        record = {'schema': 1, 'namespace': namespace, 'catalog_snapshot': catalog_id,
                  'catalog_sha256': file_hash(original), 'sources': ids,
                  'blocked': catalog.get('blocked', {}), 'classification': 'OBSERVED_MARKET',
                  'source_integrity_only': True, 'financial_certification': 'BLOCKED'}
        identifier = sha256(canonical(record)).hexdigest()
        target = self.root / 'imports' / (identifier + '.json')
        if not target.exists(): private_json(target, record)
        return identifier, record

    def audit(self):
        rows = []
        for path in sorted((self.root / 'snapshots').glob('*.json')):
            record, _ = self.load(path.stem)
            rows.append({'snapshot': path.stem, **record})
        for path in sorted((self.root / 'imports').glob('*.json')):
            record = json.loads(path.read_text())
            if sha256(canonical(record)).hexdigest() != path.stem:
                raise ValueError('Import manifest tampered')
            for identifier in [record['catalog_snapshot'], *record['sources'].values()]: self.load(identifier)
        return {'classification': 'OBSERVED_MARKET', 'source_integrity_only': True,
                'snapshots_verified': len(rows), 'sources': rows, 'financial_certification': 'BLOCKED'}


def require_available_at(metadata, decision_time):
    """Do not infer old releases from a current download or vintage label."""
    if not metadata.get('point_in_time_verified') or not metadata.get('available_at'):
        raise UnsupportedModelError('No independently verified release-time availability')
    available = datetime.fromisoformat(metadata['available_at'].replace('Z', '+00:00'))
    decision = datetime.fromisoformat(str(decision_time).replace('Z', '+00:00'))
    if available.tzinfo is None or decision.tzinfo is None:
        raise ValueError('Release and decision timestamps need explicit time zones')
    if available > decision:
        raise UnsupportedModelError('Observation was not released at the strategy decision')
