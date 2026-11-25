"""Regenerate registered first/last stream blocks; verify all frozen sample bytes.

This checks numerical reproducibility, not predictive accuracy. Source ensembles
are opened read-only; no outcomes are loaded and no existing evidence is replaced.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from letf.macro.artifacts import ArtifactStore, canonical
from letf.macro.models import MacroModel
from letf.macro.precision import MonthlyLookup
from letf.macro.precision_run import file_hash, generate_block


def verify(experiment, output):
    if output.exists():
        raise ValueError('Existing reproduction evidence cannot be overwritten')
    report = json.loads((experiment / 'report.json').read_text())
    registration = report['registration']
    protocol = registration['protocol']
    metadata, arrays = ArtifactStore(experiment / 'artifacts').load(
        'models', registration['model_id'])
    model = MacroModel(metadata['kind'], metadata['columns'],
                       arrays['training'], metadata['parameters'])
    lookup = MonthlyLookup.build(model, arrays['templates'], arrays['lengths'],
                                 fee=protocol['annual_fee_leveraged'],
                                 spread=protocol['borrowing_spread'])
    size = protocol['fixed_stream_block_paths']
    blocks = sorted({0, report['paths_per_seed'] // size - 1})
    rows = []
    for seed, identity in zip(protocol['seeds'], report['samples'], strict=True):
        path = experiment / identity['path']
        if file_hash(path) != identity['sha256']:
            raise AssertionError('Frozen sample checksum mismatch')
        sample = np.memmap(path, dtype=identity['dtype'], mode='r',
                           shape=tuple(identity['shape']))
        for block in blocks:
            expected = generate_block(model, lookup, protocol, seed, block)
            if not np.array_equal(expected, sample[block*size:(block+1)*size]):
                raise AssertionError('Registered stream block failed exact regeneration')
        rows.append({'seed': seed, 'sample_sha256': identity['sha256'],
                     'all_sample_bytes_verified': True,
                     'regenerated_blocks': blocks, 'regenerated_paths': size*len(blocks),
                     'all_regenerated_fields_exact': True})
    result = {'model_id': registration['model_id'], 'passed': True, 'rows': rows,
              'scope': 'Full frozen sample hashes and first/last blocks; no outcome loading or forecast validation'}
    with output.open('xb') as stream:
        stream.write(canonical(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.experiment, arguments.output), indent=2))
