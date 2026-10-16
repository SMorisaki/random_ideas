"""Normalization, symmetry and independent-integral checks for the reference."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e

from dpmarg import (check_coefficients, expand_params, lnL_max, lnL_marg_mp,
                    lnL_marg_ref, reduce_params, snr)
from dpmarg.regimes import named_cases, random_params


def test_zero():
    for bounds in [(0.1, 100), (0.1, 10), (1, 2)]:
        assert abs(lnL_marg_ref(0, 0, 0, 0, 0, *bounds)) < 1e-12
        assert abs(lnL_marg_mp(0, 0, 0, 0, 0, *bounds)) < 1e-12


@pytest.mark.parametrize("pars", [(3, 0.4, 5, 0.7), (1e8, 0, 0.5, 0),
                                  (20, 1, 0, 0)])
def test_symmetries(pars):
    values = []
    for xi in (0, 0.27, 1.3, 2.8):
        A, B, C, D, E = expand_params(*pars, xi=xi)
        values.append(lnL_marg_ref(A, B, C, D, E, 0.1, 100))
        reflected = lnL_marg_ref(A, -B, C, -D, E, 0.1, 100)
        assert abs(values[-1]-reflected) < 1e-9
    assert np.ptp(values) < 1e-9


@pytest.mark.parametrize("xi", [0, 0.2, 1.7, 3.1])
@pytest.mark.parametrize("eta", [0, 0.3, math.pi/4, math.pi/2, -0.3, 3.3])
def test_parameter_round_trip(xi, eta):
    pars = reduce_params(*expand_params(7, 0.2, 3, eta, xi))
    folded = min(eta % math.pi, math.pi-eta % math.pi)
    np.testing.assert_allclose(pars, (7, 0.2, 3, folded), rtol=0, atol=1e-12)


def test_coefficient_validation():
    assert check_coefficients(0, 0, 3, 4, 5)
    assert check_coefficients(0, 0, 3, 4, 5-1e-13)
    assert reduce_params(0, 0, 3, 4, 5-1e-13)[1] == 0
    assert not check_coefficients(0, 0, 3, 4, 4.9)
    with pytest.raises(ValueError):
        reduce_params(0, 0, 3, 4, 4.9)
    with pytest.raises(ValueError):
        lnL_marg_ref(0, 0, 0, 0, 0, 0, 1)


def test_isotropic_no_linear_term():
    # This constant angular profile must not turn floating-point noise into
    # hundreds of false narrow modes at large E.
    for E in (1, 1e8):
        actual = lnL_marg_ref(0, 0, 0, 0, E, .1, 100)
        width = 1/(E*.1)
        pts = sorted(set([.1, 100] + [.1+k*min(width,1/math.sqrt(E))
                                     for k in (1,3,10,30)] +
                         list(np.geomspace(.1, 100, 30))))
        val = quad(lambda r: (.1/r)**4 * math.exp(-.5*E*(r-.1)*(r+.1)),
                   .1, 100, points=pts[1:-1], epsabs=0, epsrel=1e-11)[0]
        expected = math.log(3/(1-(.1/100)**3)/.1)-.5*E*.1**2+math.log(val)
        assert abs(actual-expected) < 1e-8


def _bessel_integral(R, E, lo, hi):
    """Independent 1-D integral of r^-4 i0(R*r) exp(-E*r²/2), log scaled."""
    def logf(r):
        return math.log(i0e(R*r))+R*r-E*r*r/2-4*math.log(r)

    # Locate every candidate, including a possible prior-dominated lower mode.
    grid = np.geomspace(lo, hi, 2048)
    best = grid[np.argmax([logf(r) for r in grid])]
    j = np.searchsorted(grid, best)
    result = minimize_scalar(lambda r: -logf(r), bounds=(grid[max(0,j-1)],
                             grid[min(len(grid)-1,j+1)]), method="bounded")
    center = result.x
    shift = max(logf(lo), logf(hi), logf(center))
    points = [lo, hi, center]
    for k in (1, 3, 10, 30):
        points.extend((center-k/math.sqrt(E), center+k/math.sqrt(E)))
    points.extend(np.geomspace(lo, hi, 24))
    points = sorted(set(p for p in points if lo <= p <= hi))
    val = quad(lambda r: math.exp(logf(r)-shift), lo, hi,
               points=points[1:-1], epsabs=0, epsrel=1e-11, limit=300)[0]
    return math.log(3/(lo**-3-hi**-3))+shift+math.log(val)


@pytest.mark.parametrize("R,E", [(0.3, 1), (160, 4), (5450000, 1e8)])
def test_isotropic_bessel(R, E):
    actual = lnL_marg_ref(R*math.cos(0.7), R*math.sin(0.7), 0, 0, E, .1, 100)
    assert abs(actual-_bessel_integral(R, E, .1, 100)) < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("name", ["isotropic_low", "degenerate_eta_pi2",
                                  "two_comparable_lobes"])
def test_mpmath_agreement(name):
    case = next(c for c in named_cases() if c["name"] == name)
    args = [case[k] for k in ("A", "B", "C", "D", "E", "xmin", "xmax")]
    assert abs(lnL_marg_ref(*args)-lnL_marg_mp(*args)) < 1e-8


@pytest.mark.parametrize("pars", [(3, .2, 4, .7), (0, 0, 1, .3),
                                  (10, 0, 1, 0), (20, 3, 0, 0)])
def test_maximum_grid(pars):
    coeff = expand_params(*pars, xi=.4)
    lo, hi = .1, 10
    value, r, theta = lnL_max(*coeff, lo, hi)
    A, B, C, D, E = coeff
    t = np.arange(4096)*2*math.pi/4096
    rr = np.linspace(lo, hi, 2048)[:, None]
    a = A*np.cos(t)+B*np.sin(t)
    b = C*np.cos(2*t)+D*np.sin(2*t)+E
    grid_max = np.max(rr*a-.5*rr*rr*b)
    # Conservative Lipschitz error bound for a half grid step in each axis.
    dr = (hi-lo)/(2048-1)/2
    dt = math.pi/4096
    bound = (math.hypot(A,B)+hi*(E+math.hypot(C,D)))*dr
    bound += (hi*math.hypot(A,B)+hi*hi*math.hypot(C,D))*dt
    assert -1e-10 <= value-grid_max <= bound
    assert lo <= r <= hi and 0 <= theta < 2*math.pi
    assert abs(value-(r*(A*math.cos(theta)+B*math.sin(theta))
               -.5*r*r*(C*math.cos(2*theta)+D*math.sin(2*theta)+E))) < 1e-10
    assert snr(*coeff, lo, hi) == pytest.approx(math.sqrt(2*max(0,value)))


def test_regime_reproducibility():
    assert random_params(20, 1) == random_params(20, 1)
    assert len(named_cases()) >= 20
    for case in named_cases()+random_params(20, 1):
        assert check_coefficients(*(case[k] for k in ("A", "B", "C", "D", "E")))
