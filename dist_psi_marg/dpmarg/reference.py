"""Independent adaptive float64 and arbitrary-precision annular integrals.

Both integrate r^-4 exp(a*r-b*r²/2) with the normalized volume prior,
for finite positive xmin < xmax and positive-semidefinite quadratic form.
They share coefficient conventions and geometric breakpoints, not quadrature
values. No asymptotic likelihood approximation or tail truncation is used.
"""

from functools import lru_cache
import math

import mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq

from .params import TAU, _bounds, _periodic_maxima, _radial_peak, reduce_params


def _radial_points(a, b, xmin, xmax, multiples=(1, 3, 10, 30)):
    """Resolve Gaussian peaks, exponential endpoint layers, and the r^-4 prior."""
    _, r = _radial_peak(a, b, xmin, xmax)
    width = min(xmax-xmin, 1/math.sqrt(b) if b else xmax-xmin,
                1/abs(a-b*r) if a != b*r else xmax-xmin)
    points = [xmin, xmax, r]
    for k in multiples:
        points.extend((r-k*width, r+k*width))
    p = xmin * 2
    while p < xmax:
        points.append(p)
        p *= 4
    return sorted(set(p for p in points if xmin <= p <= xmax))


def _angular_points(func, peaks, scale):
    """Locate unit-log-drop widths, also when a peak is sharper than 1/rho."""
    points = [i * TAU/8 for i in range(9)]
    top = peaks[0][1]
    for center, height in peaks:
        if height < top - 65:
            continue
        points.append(center)
        for sign in (-1, 1):
            width = max(1e-10, min(0.1, scale))
            while width < math.pi and func((center+sign*width) % TAU) > height - 0.5:
                width *= 2
            width = min(width, math.pi)
            if func((center+sign*width) % TAU) < height - 0.5:
                width = brentq(lambda w: func((center+sign*w) % TAU)-height+0.5,
                               0, width, xtol=1e-14)
            for k in (1, 3, 10, 30):
                points.append((center+sign*k*width) % TAU)
    return sorted(set(points))


def lnL_marg_ref(A, B, C, D, E, xmin, xmax, rtol=1e-10):
    """Float64 nested quad of the normalized r^-4 exp(lnL) integral.

    Each radial integral subtracts its analytic exponent maximum and uses
    Gaussian/endpoint widths as breakpoints. A >=720-point scan plus local
    refinement locates maxima of the radial log integral g(theta); the outer
    quad subtracts max(g) and resolves all significant lobes. Specifically the
    returned value is log[3/(2*pi*(xmin^-3-xmax^-3)) * integral dr dtheta
    r^-4 exp(a*r-b*r²/2)]. The quadratic
    is evaluated in principal axes, including lam2=0 and R=0.
    """
    _bounds(xmin, xmax)
    lam1, lam2, R, eta = reduce_params(A, B, C, D, E)
    if not 0 < rtol < 1:
        raise ValueError("require 0 < rtol < 1")
    if lam1 == 0 and R == 0:
        return 0.0

    @lru_cache(maxsize=32768)
    def g(theta):
        a = R * math.cos(theta-eta)
        b = lam2 + (lam1-lam2) * math.cos(theta)**2
        f, r = _radial_peak(a, b, xmin, xmax)
        points = _radial_points(a, b, xmin, xmax)

        def integrand(x):
            delta = x-r
            return (xmin/x)**4 * math.exp(delta*(a-b*r)-0.5*b*delta*delta)

        integral = quad(integrand, xmin, xmax, points=points[1:-1],
                        epsabs=0, epsrel=rtol*0.2, limit=400)[0]
        return f + math.log(integral)

    peaks = _periodic_maxima(g, n=720,
                             extra=(0, math.pi/2, math.pi, 3*math.pi/2, eta))
    shift = peaks[0][1]
    scale = 1/max(1.0, math.sqrt(lam1)*xmin, R/math.sqrt(lam1) if lam1 else R*xmax)
    points = _angular_points(g, peaks, scale)
    integral = quad(lambda t: math.exp(g(t)-shift), 0, TAU,
                    points=points[1:-1], epsabs=0, epsrel=rtol, limit=500)[0]
    # Inner integrand uses (xmin/r)^4 instead of r^-4.
    norm = math.log(3/(2*math.pi)) - math.log1p(-(xmin/xmax)**3) - math.log(xmin)
    return float(norm + shift + math.log(integral))


def lnL_marg_mp(A, B, C, D, E, xmin, xmax, dps=25):
    """Independent nested mpmath tanh-sinh integral, returned as a float.

    Computes log[3/(2*pi*(xmin^-3-xmax^-3)) * integral dr dtheta
    r^-4 exp(a*r-b*r²/2)]. The full finite annulus is integrated, with a single constant exponent
    shift. Intervals split at peaks and +/- (1,3,10,30) Gaussian or endpoint
    widths. Float geometric searches only choose partitions: all integrand
    arithmetic, exponent shifts and normalization use ``dps`` decimal digits.
    As in the float reference, tiny negative eigenvalues are clipped to zero.
    """
    _bounds(xmin, xmax)
    pars = reduce_params(A, B, C, D, E)
    l1, l2, radius, angle = pars
    if l1 == 0 and radius == 0:
        return 0.0
    if dps < 15:
        raise ValueError("dps must be >= 15")

    # A geometric estimate guides partitions only; no scipy integral is used.
    def envelope(t):
        a = radius * math.cos(t-angle)
        b = l2 + (l1-l2)*math.cos(t)**2
        f, r = _radial_peak(a, b, xmin, xmax)
        # The prior removes flat ridges of the exponent in lam2=0 cases.
        # Its omission would produce hundreds of redundant peak partitions.
        return f - 4*math.log(r)

    peaks = _periodic_maxima(envelope, n=1024,
                             extra=(0, math.pi/2, math.pi, 3*math.pi/2, angle))
    scale = 1/max(1.0, math.sqrt(l1)*xmin, radius/math.sqrt(l1) if l1 else radius*xmax)
    angular_points = _angular_points(envelope, peaks, scale)
    angular_points = sorted(set([0.0, math.pi/2] +
                               [p for p in angular_points if 0 < p < math.pi/2]))
    with mp.workdps(dps):
        lam1, lam2, R, eta = map(mp.mpf, pars)
        lo, hi = mp.mpf(xmin), mp.mpf(xmax)
        # A constant shift is sufficient in arbitrary precision, including
        # large negative maxima. Its accuracy does not affect the integral.
        # Include the prior in the constant shift: without this, peaks at
        # xmax can have amplitude (xmin/xmax)^4 and fool an absolute-error
        # stopping rule even at high precision.
        shift = mp.mpf(peaks[0][1] + 4*math.log(xmin))

        @lru_cache(maxsize=32768)
        def inner(theta):
            u = R * mp.cos(theta) * mp.cos(eta)
            v = R * mp.sin(theta) * mp.sin(eta)
            a = u+v
            b = lam1 * mp.cos(theta)**2 + lam2 * mp.sin(theta)**2
            pts = _radial_points(float(a), float(b), xmin, xmax)
            # Sum the four quadrants analytically. This is an exact identity,
            # and gives an independent integrand from the full-circle quad.
            return mp.quad(lambda r: 4*(lo/r)**4 * mp.exp(-b*r*r/2-shift)
                           * mp.cosh(u*r) * mp.cosh(v*r),
                           list(map(mp.mpf, pts)), method="tanh-sinh")

        # Inner quadrature errors cannot support an outer stopping target at
        # the last working digit. Check a relative error estimate explicitly,
        # and increase the tanh-sinh degree only when it requires refinement.
        target = mp.power(10, -min(15, dps-8))
        for degree in (4, 6, 8):
            integral, error = mp.quad(inner, list(map(mp.mpf, angular_points)),
                                      method="tanh-sinh", maxdegree=degree, error=True)
            if error <= abs(integral)*target:
                break
        else:
            raise ArithmeticError("mpmath outer quadrature did not reach its relative error target")
        norm = mp.log(3/(2*mp.pi)) - mp.log1p(-(lo/hi)**3) - mp.log(lo)
        return float(norm + shift + mp.log(integral))
