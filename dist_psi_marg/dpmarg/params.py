"""Scalar coefficient transformations and likelihood maxima on an annulus."""

import math

import numpy as np
from scipy.optimize import minimize_scalar

TAU = 2 * math.pi


def check_coefficients(A, B, C, D, E, rtol=1e-12):
    """Return whether finite coefficients obey E >= 0 and E²-C²-D² >= 0.

    A negative determinant is accepted only within ``rtol * max(E²,C²+D²)``.
    The factored comparison avoids squaring large coefficients. False reports
    a material violation; :func:`reduce_params` raises for such inputs.
    """
    if not all(math.isfinite(v) for v in (A, B, C, D, E)) or E < 0:
        return False
    S = math.hypot(C, D)
    scale = max(E, S)
    return scale == 0 or (E / scale - S / scale) * (E / scale + S / scale) >= -rtol


def reduce_params(A, B, C, D, E):
    """Return (E+S, max(0,E-S), hypot(A,B), eta), S=hypot(C,D).

    Valid for finite coefficients with E >= S, allowing roundoff at the
    boundary. With xi=atan2(D,C)/2 and zeta=atan2(B,A), eta=zeta-xi.
    In the full-circle integral theta -> theta+pi identifies eta and eta+pi;
    theta -> -theta identifies eta and -eta. Folding modulo pi therefore
    gives eta in [0,pi/2]. When R=0 its angle is chosen as zero. If S=0,
    the quadratic axes are arbitrary; xi=0 is used, so eta has no invariant
    meaning and need not round-trip an arbitrary expansion rotation.
    """
    if not check_coefficients(A, B, C, D, E):
        raise ValueError("coefficients require finite values and E >= hypot(C,D)")
    S = math.hypot(C, D)
    R = math.hypot(A, B)
    eta = (math.atan2(B, A) - 0.5 * math.atan2(D, C)) % math.pi if R else 0.0
    eta = min(eta, math.pi - eta)
    return float(E + S), float(max(0.0, E - S)), R, eta


def expand_params(lam1, lam2, R, eta, xi=0.0):
    """Return A=R cos(eta+xi), B=R sin(eta+xi), C=S cos(2xi), D=S sin(2xi), E.

    Here E=(lam1+lam2)/2, S=(lam1-lam2)/2. The domain is
    lam1 >= lam2 >= 0, R >= 0 and arbitrary finite angles; xi is a rotation.
    """
    if not all(math.isfinite(v) for v in (lam1, lam2, R, eta, xi)):
        raise ValueError("parameters must be finite")
    if not lam1 >= lam2 >= 0 or R < 0:
        raise ValueError("require lam1 >= lam2 >= 0 and R >= 0")
    S = 0.5 * (lam1 - lam2)
    return (R * math.cos(eta + xi), R * math.sin(eta + xi),
            S * math.cos(2 * xi), S * math.sin(2 * xi), 0.5 * (lam1 + lam2))


def _bounds(xmin, xmax):
    if not (math.isfinite(xmin) and math.isfinite(xmax) and 0 < xmin < xmax):
        raise ValueError("require finite 0 < xmin < xmax")


def _radial_peak(a, b, xmin, xmax):
    """Maximize a*r-b*r²/2 on a positive closed interval, including b=0."""
    r = min(xmax, max(xmin, a / b)) if b > 0 else (xmax if a > 0 else xmin)
    return r * (a - 0.5 * b * r), r


def _periodic_maxima(func, n=1024, extra=()):
    """Dense periodic scan plus bounded refinement of every sampled local peak."""
    grid = np.arange(n) * (TAU / n)
    values = np.array([func(float(t)) for t in grid])
    step = TAU / n
    indices = np.flatnonzero((values >= np.roll(values, 1)) &
                             (values >= np.roll(values, -1)))
    if np.ptp(values) <= 1e-13:
        indices = [int(np.argmax(values))]
    peaks = []
    for t in [*(grid[i] for i in indices), *extra]:
        if func(t % TAU) < max(func((t-step) % TAU), func((t+step) % TAU)):
            continue
        opt = minimize_scalar(lambda u: -func(u % TAU), bounds=(t-step, t+step),
                              method="bounded", options={"xatol": 2e-14})
        peaks.append(max([(float(t % TAU), float(func(t % TAU))),
                          (float(opt.x % TAU), float(-opt.fun))], key=lambda p: p[1]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    unique = []
    for p in peaks:
        if all(abs((p[0]-q[0]+math.pi) % TAU-math.pi) > 1e-7 for q in unique):
            unique.append(p)
    return unique


def lnL_max(A, B, C, D, E, xmin, xmax):
    """Return global (max lnL, r*, theta*) for r in [xmin,xmax], theta in [0,2pi).

    The radial maximum is analytic (clipped a/b, or a linear endpoint).
    A 2048-angle scan and bounded refinement find the angular maximum.
    Tiny negative quadratic eigenvalues are clipped as in reduce_params.
    """
    _bounds(xmin, xmax)
    lam1, lam2, R, _ = reduce_params(A, B, C, D, E)
    xi = 0.5 * math.atan2(D, C)
    eta = math.atan2(B, A) - xi

    def peak(theta):
        a = R * math.cos(theta - eta)
        b = lam2 + (lam1-lam2) * math.cos(theta)**2
        return _radial_peak(a, b, xmin, xmax)

    candidates = _periodic_maxima(lambda t: peak(t)[0], n=2048,
                                 extra=(0, math.pi/2, math.pi, 3*math.pi/2, eta))
    theta, fmax = candidates[0]
    return fmax, peak(theta)[1], (theta + xi) % TAU


def snr(A, B, C, D, E, xmin, xmax):
    """Return sqrt(2 max(0,max lnL)) on the finite annulus."""
    return math.sqrt(2 * max(0.0, lnL_max(A, B, C, D, E, xmin, xmax)[0]))
