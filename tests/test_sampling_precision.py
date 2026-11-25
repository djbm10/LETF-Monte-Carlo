import numpy as np
import pytest
from letf.macro.sampling import precision_report


def policy():return {'master_paths':1000,'prefix_counts':[250,500,1000],'quantiles':[.1,.5,.9],
    'maximum_abs_log_wealth_quantile_difference':.02,'maximum_probability_wilson_half_width':.04}


def test_sampling_precision_is_not_silently_declared_for_wealth_quantile_instability():
    # Repeated two-point populations have identical empirical quantiles at all
    # registered counts, and 500/1000 losses has an independent Wilson interval.
    a=np.tile([.5,2.],500);r=precision_report([a,a.copy()],policy())
    z=1.959963984540054
    independent_half_width=z/(2*np.sqrt(1000+z*z))
    assert r['convergence_passed'] and r['maximum_wilson_half_width']==pytest.approx(independent_half_width,abs=1e-12)
    b=a.copy();b[:500]*=5
    assert not precision_report([b,a],policy())['quantile_convergence_passed']
    zero=np.zeros(1000)
    r=precision_report([zero,zero],policy())
    assert not r['quantile_convergence_passed']
    assert r['unit_nav_p10_p50_p90_by_seed_and_prefix'][0]['1000']==[0.,0.,0.]
    with pytest.raises(ValueError):precision_report([a[:-1],a],policy())
