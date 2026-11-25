import json

import numpy as np
import pytest

from letf.macro.artifacts import ArtifactStore
from letf.macro.uncertainty import wilson_interval,parameter_sensitivity
from test_macro_explorer import FixtureSources,training


def test_wilson_conditional_sampling_interval_does_not_certify_tiny_ensembles():
    assert wilson_interval(50,100)==pytest.approx([.40383153,.59616847],abs=1e-8)
    low,high=wilson_interval(0,2)
    assert low==0. and high>.65
    assert wilson_interval(2,2)[0]<.35


def test_parameter_refits_preserve_joint_blocks_and_exclude_join_transitions(tmp_path):
    folder=tmp_path/'parameter'
    report=parameter_sensitivity(FixtureSources(),calibration_end='2024-12-31',output=folder,
        replicates=2,paths=8,horizons=[1],seed=77)
    assert report['certification']=='BLOCKED' and len(report['results'])==1
    store=ArtifactStore(folder/'artifacts')
    meta,arrays=store.load('models',report['refits'][0]['model_id'])
    original=training().tail(240).to_numpy()
    first=meta['selected_block_starts'][0]
    np.testing.assert_array_equal(arrays['training'][:12],original[first:first+12])
    assert np.isnan(arrays['training'][12]).all()
    assert meta['parameters']['observed_adjacent_transitions']==201
    assert len(report['results'][0]['across_refit_p10_p50_p90_of_conditional_medians'])==3
