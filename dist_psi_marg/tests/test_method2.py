"""Reference accuracy, geometry identities and adaptive node diagnostics."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dpmarg.kernel1d import fmax_1d, lnJ_pair
from dpmarg.method2 import (_compute, _geometry, _proxy,
                            lnL_marg_m2, lnL_marg_m2_array, method2_nodes)
from dpmarg.params import expand_params, reduce_params
from dpmarg.reference import lnI_ref, lnL_marg_ref
from dpmarg.regimes import random_params

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')
REGRESSION = json.loads((ROOT/'data/regression_points.json').read_text())


@pytest.mark.parametrize('a,b', [(0.,0.), (.1,0.), (-.1,0.), (1e5,0.),
                                  (1.,10.), (-100.,3.), (1e5,1e8),
                                  (100.,1e-3), (20.,1.), (400.,1.)])
@pytest.mark.parametrize('hi', [10.,100.])
def test_pair_reference(a, b, hi):
    reference = np.logaddexp(fmax_1d(a,b,.1,hi)+lnI_ref(a,b,.1,hi),
                            fmax_1d(-a,b,.1,hi)+lnI_ref(-a,b,.1,hi))
    assert abs(lnJ_pair(a,b,.1,hi)-reference) < 2e-8
    assert lnJ_pair(a,b,.1,hi) == lnJ_pair(-a,b,.1,hi)


def test_regression():
    parameters = np.array([[c[k] for k in KEYS] for c in REGRESSION])
    values = lnL_marg_m2_array(parameters)
    errors = abs(values-np.array([c['lnL_marg_ref'] for c in REGRESSION]))
    worst = int(errors.argmax())
    print(f'Worst regression error: {errors[worst]:.9g}, {REGRESSION[worst]["name"]}')
    assert errors.max() <= 1e-3, (REGRESSION[worst]['name'], errors[worst])


@pytest.mark.parametrize('pars', [(3.,.4,5.,.7), (1e8,0.,.5,0.),
                                 (20.,1.,0.,0.), (1e6,1.,5000.,0.)])
def test_symmetries(pars):
    values = []
    for xi in (0.,.27,1.3,2.8):
        A,B,C,D,E = expand_params(*pars,xi=xi)
        value = lnL_marg_m2(A,B,C,D,E,.1,100.)
        values.append(value)
        reflected = lnL_marg_m2(A,-B,C,-D,E,.1,100.)
        assert abs(value-reflected) < 1e-9
    assert np.ptp(values) < 1e-7


def test_random_convergence():
    cases = random_params(50,5)
    parameters = np.array([[c[k] for k in KEYS] for c in cases])
    errors = abs(lnL_marg_m2_array(parameters,64)-lnL_marg_m2_array(parameters,128))
    assert errors.max() < 1e-4


@pytest.mark.parametrize('bounds', [(.1,100.),(.1,10.),(1.,2.)])
def test_zero_and_finite_limits(bounds):
    assert abs(lnL_marg_m2(0.,0.,0.,0.,0.,*bounds)) < 1e-12
    for coeff in ((0.,0.,3.,0.,3.), (1.,.1,0.,0.,0.), (0.,0.,0.,0.,1e8)):
        assert math.isfinite(lnL_marg_m2(*coeff,*bounds))


@pytest.mark.parametrize('pars', [(1.,.2,.3,.4), (20.,3.,0.,0.), (0.,0.,.1,.3)])
def test_full_circle_pair_and_rotation_identity(pars):
    # A dense smooth angular check uses the unfurled raw likelihood and
    # distance reference, independently of method2's node placement.
    coeff = expand_params(*pars,xi=.37)
    l1,l2,R,eta = reduce_params(*coeff)
    A,B,C,D,E = coeff
    theta = np.arange(512)*math.pi/256
    full = []
    for t in theta:
        a = A*math.cos(t)+B*math.sin(t)
        b = C*math.cos(2*t)+D*math.sin(2*t)+E
        full.append(math.exp(fmax_1d(a,b,.1,10.)+lnI_ref(a,b,.1,10.)))
    paired = [math.exp(lnJ_pair(R*math.cos(t-eta),
                  l2+(l1-l2)*math.cos(t)**2,.1,10.)) for t in theta[:256]]
    assert abs(math.log(sum(full)/sum(paired))) < 1e-11
    for t in (.1,.7,1.4):
        def h(u):
            return lnJ_pair(R*math.cos(u-eta),l2+(l1-l2)*math.cos(u)**2,.1,10.)
        assert abs(h(t)-h(t+math.pi)) < 1e-12


def test_boundary_bimodality_and_prior_proxy():
    # At xmin, |cos(theta)|=R/((lam1-lam2)*xmin) gives two comparable
    # off-axis modes. Neither the single exponent peak nor one sinh patch
    # suffices. The r^-4 proxy must retain both modes.
    pars = (1e6,1.,5000.,0.)
    coeff = expand_params(*pars)
    centers,widths,bounds = _geometry(*pars,.1,100.)
    assert len(centers) == 2
    assert widths.max() < .02
    assert bounds[0] < centers[0] < bounds[1] < centers[1] < bounds[2]
    actual = lnL_marg_m2(*coeff,.1,100.)
    assert abs(actual-lnL_marg_ref(*coeff,.1,100.)) < 1e-4
    t = centers[0]
    a = abs(pars[2]*math.cos(t))
    b = pars[1]+(pars[0]-pars[1])*math.cos(t)**2
    r = np.clip(a/b,.1,100.)
    assert _proxy(t,*pars,.1,100.) == pytest.approx(fmax_1d(a,b,.1,100.)-4*math.log(r))


@pytest.mark.parametrize('n', [16,24,32,48,64,66,96,128])
def test_nodes_and_counts(n):
    for coeff in (expand_params(1.,.2,.3,.4), expand_params(1e6,1.,5000.,0.)):
        value,count,mode,nodes,weights = _compute(*coeff,.1,100.,n)
        ts,ws,m = method2_nodes(*coeff,.1,100.,n)
        np.testing.assert_array_equal(ts,nodes)
        np.testing.assert_array_equal(ws,weights)
        assert m == mode and 16 <= len(nodes) <= n
        assert len(nodes) <= count <= 2*n
        l1,l2,R,eta = reduce_params(*coeff)
        logs = np.array([lnJ_pair(R*math.cos(t-eta),
                l2+(l1-l2)*math.cos(t)**2,.1,100.) for t in ts])
        norm = math.log(3/(2*math.pi*(.1**-3-100.**-3)))
        rebuilt = norm+logs.max()+math.log(np.dot(ws,np.exp(logs-logs.max())))
        assert abs(value-rebuilt) < 1e-11


def test_invalid_inputs():
    for args in ((0.,0.,2.,0.,1.,.1,100.),(0.,0.,0.,0.,0.,0.,1.)):
        with pytest.raises(ValueError):
            lnL_marg_m2(*args)
    with pytest.raises(ValueError):
        lnL_marg_m2(0.,0.,0.,0.,0.,.1,100.,8)
    with pytest.raises(ValueError):
        lnL_marg_m2_array(np.zeros((2,6)))
