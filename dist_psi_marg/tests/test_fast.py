"""Production dispatch accuracy, empirical gate boundaries, and API limits."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dpmarg import (lnL_marg_ref, lnL_marg_m2, lnL_marg_m2_table,
                    lnL_marg_m3_2d, lnL_marg_fast, build_table, load_table,
                    reduce_params, snr)
from dpmarg.fast import (lnL_marg_fast_array, fast_diagnostics,
                         fast_array_diagnostics, laplace_gate, rho_break,
                         LAPLACE_MARGIN, LAPLACE_INTERIOR_SIGMA)
from dpmarg.method2 import lnL_marg_m2_table_array
from dpmarg.params import expand_params

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('A','B','C','D','E','xmin','xmax')


@pytest.fixture(scope='module')
def tables():
    out = {}
    for hi in (100.,10.):
        path = ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz'
        out[hi] = load_table(path) if path.exists() else build_table(.1,hi)
    return out


@pytest.mark.parametrize('filename',['regression_points','method2_random_reference'])
def test_saved_accuracy_and_diagnostics(filename,tables,record_property):
    cases = json.loads((ROOT/f'data/{filename}.json').read_text())
    errors = []
    for hi,table in tables.items():
        group = [c for c in cases if c['xmax']==hi]
        if not group:
            continue
        ps = np.array([[c[k] for k in KEYS] for c in group])
        values,branches = fast_array_diagnostics(ps,table)
        np.testing.assert_array_equal(values,lnL_marg_fast_array(ps,table))
        np.testing.assert_array_equal(lnL_marg_fast_array(ps,table,False),
                                      lnL_marg_m2_table_array(ps,table))
        off_values,off_branches = fast_array_diagnostics(ps,table,False)
        np.testing.assert_array_equal(off_values,lnL_marg_m2_table_array(ps,table))
        assert not off_branches.any()
        for c,p,value,branch in zip(group,ps,values,branches):
            expected = laplace_gate(*p) and math.isfinite(lnL_marg_m3_2d(*p))
            assert branch == int(expected)
            if branch:
                l1,l2,R,eta = reduce_params(*p[:5])
                s,t = R*math.cos(eta)/l1,R*math.sin(eta)/l2
                radius = math.hypot(s,t)
                assert l2>0 and l2/l1>=.001
                assert p[5]+LAPLACE_INTERIOR_SIGMA/math.sqrt(l2)<radius
                assert radius<p[6]-LAPLACE_INTERIOR_SIGMA/math.sqrt(l2)
                assert math.sqrt(l1*s*s+l2*t*t)>=LAPLACE_MARGIN*rho_break(l2/l1)
            errors.append((abs(value-c['lnL_marg_ref']),c['name']))
    worst,name = max(errors)
    record_property('worst_absolute_log_error',worst)
    record_property('worst_case',name)
    print(f'{filename}: worst error {worst:.12g} ({name})')
    assert worst<1e-3,(name,worst)


def test_threshold_interpolation():
    assert math.isinf(rho_break(.000999))
    for q,rho in [(1.,18.),(.3,18.),(.1,42.),(.01,133.),(.001,422.)]:
        assert rho_break(q)==pytest.approx(rho,rel=2e-15)
    samples = np.array([rho_break(q) for q in np.geomspace(.001,1,501)])
    assert np.all(np.diff(samples)<=0)


@pytest.mark.parametrize('q',[.001,.01,.1,.3,1.])
def test_gate_snr_boundaries(q,tables):
    threshold = LAPLACE_MARGIN*rho_break(q)
    # Fix an exactly representable quadratic form (including q=.001).
    # Both circles are more than three weak-axis sigmas from these peaks.
    for factor,expected in [(.999,False),(1.001,True)]:
        rho = threshold*factor
        coeff = expand_params(10000.,q*10000.,rho*100.,0.)
        assert laplace_gate(*coeff,.1,100.) == expected
        value,branch = fast_diagnostics(*coeff,.1,100.,tables[100.])
        assert branch==int(expected)
        assert math.isfinite(value)
        if expected:
            assert abs(value-lnL_marg_ref(*coeff,.1,100.))<1e-3


@pytest.mark.parametrize('reduced',[(10.,0.,1.,0.),(10.,0.,1.,math.pi/2),
    (20.,3.,0.,0.),(1e8,1e8,5e6,.3),(1.,1.,200.,.2)])
def test_extremes_finite_and_general_branch(reduced,tables):
    coeff = expand_params(*reduced)
    value,branch = fast_diagnostics(*coeff,.1,100.,tables[100.])
    assert math.isfinite(value)
    assert branch==0
    assert abs(value-lnL_marg_ref(*coeff,.1,100.))<1e-3


def test_zero(tables):
    for hi,table in tables.items():
        for enabled in (True,False):
            assert lnL_marg_fast(0.,0.,0.,0.,0.,.1,hi,table,enabled)==0.


@pytest.mark.parametrize('coeff',[
    (-.5984956658182099,.05940349086388996,-2.291486743920588,
     -2.064735554121742,3.0900921157124963),
    expand_params(.5982561314887299,0.,.2631884412553095,1.1737617861221479)])
def test_hidden_large_radius_island(coeff,tables):
    # A broad near-xmin contribution and a narrow weak-axis island at large
    # radius: the original 64-node uniform rule erred by .59 and .012.
    ref = lnL_marg_ref(*coeff,.1,100.)
    assert abs(lnL_marg_m2(*coeff,.1,100.)-ref)<1e-4
    assert abs(lnL_marg_fast(*coeff,.1,100.,tables[100.])-ref)<1e-4


@pytest.mark.parametrize('coeff',[
    (16.948711265673715,27.607321753023136,-2.1348128164665057,
     4.483601570832346,6.633182378713948),
    (21.395814952434804,18.805908983120858,1.244753479681153,
     5.817197177739085,16.896495387635404),
    (-.4050033597832454,3.732158650815139,-.08419409158783564,
     .0031788339344747253,.32508833883254384)])
def test_weak_seed_clipping_is_not_a_narrow_mode(coeff,tables):
    ref = lnL_marg_ref(*coeff,.1,100.)
    assert abs(lnL_marg_fast(*coeff,.1,100.,tables[100.])-ref)<1e-3


def test_invalid_and_empty(tables):
    tab = tables[100.]
    for enabled in (True,False):
        # A mismatched table must be rejected even for a Laplace-eligible point.
        with pytest.raises(ValueError,match='table bounds'):
            lnL_marg_fast(10000.,0.,0.,0.,10000.,.1,10.,tab,enabled)
        with pytest.raises(ValueError):
            lnL_marg_fast(0.,0.,2.,0.,1.,.1,100.,tab,enabled)
        with pytest.raises(ValueError):
            lnL_marg_fast_array(np.zeros((2,6)),tab,enabled)
        assert lnL_marg_fast_array(np.empty((0,7)),tab,enabled).shape==(0,)
    with pytest.raises(ValueError):
        laplace_gate(0.,0.,0.,0.,0.,0.,100.)
