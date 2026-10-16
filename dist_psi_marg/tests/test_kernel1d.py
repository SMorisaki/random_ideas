"""Distance kernel accuracy, endpoint geometry, and angular consistency."""

import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from numba import njit
import pytest
from scipy import special

from dpmarg.kernel1d import (GL_NODES, GL_WEIGHTS, _drop_width,
                             fmax_1d, lnI_direct, lnI_direct_array)
from dpmarg.params import _radial_peak
from dpmarg import reference

ROOT = Path(__file__).resolve().parents[1]
REGRESSION = json.loads((ROOT/'data/regression_points.json').read_text())


def parameter_points(hi):
    """100 points: exact degeneracies, moving boundaries, weak curvature, random."""
    points = [(a, b) for a in (-1e7, 0., 1e7) for b in (0., 1e-8, 1., 1e8)]
    points.extend((b*r*(1+offset), b) for b in (1e-8, 1., 1e8)
                  for r in (0.1, hi) for offset in (-1e-10, 0., 1e-10))
    small = 1e-6/(hi-0.1)**2
    points.extend((a, small) for a in (-1., 0., 1., small*0.1, small*hi))
    rng = np.random.default_rng(42)
    for _ in range(100-len(points)):
        a = rng.choice([-1., 1.])*10.**rng.uniform(-4, 7)
        b = 0. if rng.random() < .2 else 10.**rng.uniform(-8, 8)
        points.append((a, b))
    return [(a, b, .1, hi) for a, b in points]


POINTS = parameter_points(100.)+parameter_points(10.)


@pytest.mark.parametrize('a,b,lo,hi', POINTS)
def test_kernel_against_reference(a, b, lo, hi):
    assert abs(lnI_direct(a, b, lo, hi)-reference.lnI_ref(a, b, lo, hi)) <= 1e-8
    assert fmax_1d(a, b, lo, hi) == _radial_peak(a, b, lo, hi)[0]


def test_gauss_constants():
    # Moments check the hard-coded rule independently of the integrand.
    for degree in range(48):
        exact = 0. if degree % 2 else 2/(degree+1)
        assert abs(np.dot(GL_WEIGHTS, GL_NODES**degree)-exact) < 2e-14


@njit(cache=True)
def primitives(v):
    """The two special primitives used by the peak-centred log transform."""
    return math.expm1(v), math.log1p(v)


def test_special_primitives_against_scipy():
    # No erfcx/erfcinv approximation is used. Check the actual libm operations
    # compiled by numba, including near zero and near log1p's domain boundary.
    values = np.r_[-1+1e-14, -0.5, -np.logspace(-15, -1, 40),
                    0., np.logspace(-15, 2, 80)]
    actual = np.array([primitives(v) for v in values])
    expected = np.array([special.expm1(values), special.log1p(values)]).T
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_drop_width():
    for d in (0., 1e-100, 1., 1e10, 1e150):
        for sqrtb in (0., 1e-100, 1., 1e10, 1e150):
            width = _drop_width(d, sqrtb, 80.)
            if d == sqrtb == 0:
                assert math.isinf(width)
            else:
                assert abs(d*width+0.5*(sqrtb*width)**2-80.) < 1e-12


def test_array():
    points = np.array(POINTS[:100])
    expected = np.array([lnI_direct(a, b, lo, hi) for a, b, lo, hi in points])
    actual = lnI_direct_array(points[:, 0], points[:, 1], .1, 100.)
    np.testing.assert_array_equal(actual, expected)
    assert lnI_direct_array(np.array([]), np.array([]), .1, 100.).size == 0
    with pytest.raises(ValueError):
        lnI_direct_array(np.ones(2), np.ones(3), .1, 100.)


def test_no_failure_at_finite_float_extremes():
    largest = np.finfo(float).max
    smallest = np.nextafter(0., 1.)
    for hi in (10., 100.):
        for a in (-largest, -1e200, 0., 1e200, largest):
            for b in (0., smallest, 1e200, largest):
                assert math.isfinite(lnI_direct(a, b, .1, hi)), (a, b, hi)


@pytest.mark.parametrize('args', [(math.nan, 0., .1, 100.), (0., -1., .1, 100.),
                                  (0., 0., 0., 100.), (0., 0., 1., 1.)])
def test_invalid_domain(args):
    with pytest.raises(ValueError):
        lnI_direct(*args)
    with pytest.raises(ValueError):
        reference.lnI_ref(*args)


def test_reference_refactor_regression():
    # All 40 saved iteration-001 results, also compared against an unmodified
    # reference snapshot during development (maximum change exactly zero).
    for case in REGRESSION:
        args = [case[k] for k in ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')]
        assert abs(reference.lnL_marg_ref(*args)-case['lnL_marg_ref']) <= 1e-12


CONSISTENCY_NAMES = ['zero', 'isotropic_high', 'degenerate_eta_pi2',
                     'large_lam1_rho_0.5', 'large_lam1_rho_300',
                     'peak_below_rho_100', 'peak_above_rho_100',
                     'two_comparable_lobes', 'strong_anisotropy', 'random_05']


@pytest.mark.parametrize('name', CONSISTENCY_NAMES)
def test_angular_consistency(name, monkeypatch):
    case = next(c for c in REGRESSION if c['name'] == name)
    args = [case[k] for k in ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')]
    expected = reference.lnL_marg_ref(*args)

    def direct_scaled(a, b, lo, hi, rtol):
        return lnI_direct(a, b, lo, hi)+4*math.log(lo)

    monkeypatch.setattr(reference, '_lnI_scaled', direct_scaled)
    assert abs(reference.lnL_marg_ref(*args)-expected) <= 1e-7


def test_soft_runtime():
    rng = np.random.default_rng(7)
    a = rng.choice([-1., 1.], 4000)*10.**rng.uniform(-4, 7, 4000)
    b = 10.**rng.uniform(-8, 8, a.size)
    b[::4] = 0.
    lnI_direct_array(a, b, .1, 100.)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        lnI_direct_array(a, b, .1, 100.)
        samples.append((time.perf_counter()-start)/a.size)
    median = np.median(samples)
    if median > 20e-6:
        warnings.warn(f'lnI_direct median {median*1e6:.2f} us/call exceeds 20 us')
