"""Numba distance integral in peak-centred logarithmic coordinates.

For finite a, b >= 0 and 0 < xmin < xmax, compute
lnI = log integral x^-4 exp(a*x-b*x*x/2-fmax) dx.

Window bound (no special-function approximation): on either side of the
clipped exponent peak r, the exponent drop is h(t)=d*t+b*t*t/2, d>=0.
Truncate at h(L)=H. Convexity gives tail <= exp(-H)*L/H, whereas the
first L/H of the retained side contributes >= exp(-1)*L/H before
prior weighting. Thus the relative discarded mass is at most
exp(1-H)*(xmax/xmin)^4. We use H=1-log(1e-12)+4*log(xmax/xmin).
When h never reaches H use the full side. For b=0 we always integrate the
full interval, stopping only if the entire remaining integrand underflows.

Integrate v=log(x/r) with 24-point Gauss-Legendre panels. A boundary at
v=0 separates the two sides. Initial widths resolve min(r,1/sqrt(b),1/d)
in x, subsequent widths double up to 0.5 in v. expm1(v) preserves thin
endpoint layers; f-fmax is evaluated as delta*(a-b*r)-b*delta**2/2,
never as a difference of large exponents. No erfcx/erfcinv is needed.
"""

import math

import numpy as np
from numba import njit

GL_NODES = np.array([-0.9951872199970213, -0.9747285559713095, -0.9382745520027328, -0.8864155270044011, -0.820001985973903, -0.7401241915785544, -0.6480936519369755, -0.5454214713888396, -0.4337935076260451, -0.3150426796961634, -0.1911188674736163, -0.06405689286260563, 0.06405689286260563, 0.1911188674736163, 0.3150426796961634, 0.4337935076260451, 0.5454214713888396, 0.6480936519369755, 0.7401241915785544, 0.820001985973903, 0.8864155270044011, 0.9382745520027328, 0.9747285559713095, 0.9951872199970213])
GL_WEIGHTS = np.array([0.01234122979998869, 0.02853138862893356, 0.04427743881741941, 0.05929858491543636, 0.07334648141108016, 0.0861901615319532, 0.09761865210411393, 0.10744427011596556, 0.11550566805372552, 0.1216704729278033, 0.12583745634682825, 0.12793819534675202, 0.12793819534675202, 0.12583745634682825, 0.1216704729278033, 0.11550566805372552, 0.10744427011596556, 0.09761865210411393, 0.0861901615319532, 0.07334648141108016, 0.05929858491543636, 0.04427743881741941, 0.02853138862893356, 0.01234122979998869])
WINDOW_EPS = 1e-12


@njit(cache=True)
def _extreme_local_integral(a, b, r, interior):
    """Local constant-prior limit when the width/r is below 1e-150.

    Scale t=z/M, M=max(|a-b*r|,sqrt(b)), using logarithms to avoid
    overflow. The prior's relative variation is O(1e-150). At an interior
    maximum the integral is sqrt(2*pi/b)*r^-4; at an endpoint integrate
    exp(-p*z-q²*z²/2)/M with p,q<=1, max(p,q)=1. This fallback ensures
    finite results even when a-b*r overflows; its far-outside-target
    inputs have no float64 absolute-accuracy guarantee for fmax itself.
    """
    logr = math.log(r)
    logrootb = 0.5*math.log(b) if b else -math.inf
    if interior:
        return 0.5*math.log(2*math.pi)-logrootb-4*logr, 0
    logbr = math.log(b)+logr if b else -math.inf
    loga = math.log(abs(a)) if a else -math.inf
    high = max(loga, logbr)
    low = min(loga, logbr)
    if a <= 0:
        logd = high+math.log1p(math.exp(low-high))
    elif high == low:
        logd = -math.inf
    else:
        logd = high+math.log(-math.expm1(low-high))
    logscale = max(logd, logrootb)
    p = math.exp(logd-logscale)
    q = math.exp(logrootb-logscale)
    extent = _drop_width(p, q, 80.0)
    total = 0.0
    lo = 0.0
    step = 1.0
    panels = 0
    while lo < extent:
        hi = min(extent, lo+step)
        half = 0.5*(hi-lo)
        mid = 0.5*(hi+lo)
        for j in range(24):
            z = mid+half*GL_NODES[j]
            total += half*GL_WEIGHTS[j]*math.exp(-p*z-0.5*(q*z)**2)
        panels += 1
        lo = hi
        step *= 2
    return math.log(total)-logscale-4*logr, panels


@njit(cache=True)
def fmax_1d(a, b, xmin, xmax):
    """Return max(a*x-b*x*x/2), matching params._radial_peak exactly.

    Domain: finite a, b>=0, finite 0<xmin<xmax. The maximum can overflow
    float64 for extreme inputs even though the shifted lnI remains finite.
    """
    r = min(xmax, max(xmin, a / b)) if b > 0 else (xmax if a > 0 else xmin)
    return r * (a - 0.5 * b * r)


@njit(cache=True)
def _drop_width(d, sqrtb, height):
    """Positive root of d*t+b*t²/2=height; infinity for d=b=0.

    The hypot and divided expression avoid squaring d or forming b*height.
    Domain: nonnegative d, sqrtb and height.
    """
    q = math.hypot(d, math.sqrt(2*height)*sqrtb)
    denom = 0.5*d + 0.5*q
    return height/denom if denom > 0 else math.inf


@njit(cache=True)
def _side_integral(r, slope, sqrtb, extent, sign, initial):
    """Integrate exp(-3v+f(r*exp(v))-f(r)) d|v| on one side.

    extent>=0 is a logarithmic width, sign is +/-1; r is the clipped peak.
    Returns (integral, panel_count), without the common r^-3 factor.
    """
    total = 0.0
    panels = 0
    lo = 0.0
    step = initial
    # For b=0, -3v+a*r*expm1(v) is convex if a>0 and monotone if
    # a<=0. Its maximum on any remaining side is therefore at an endpoint.
    end_log = (-3*sign*extent+r*math.expm1(sign*extent)*slope
               if sqrtb == 0 else 0.0)
    while lo < extent:
        hi = min(extent, lo+step)
        mid = 0.5*(lo+hi)
        half = 0.5*(hi-lo)
        subtotal = 0.0
        for j in range(24):
            v = sign*(mid+half*GL_NODES[j])
            delta = r*math.expm1(v)
            z = sqrtb*delta
            drop = delta*slope - 0.5*z*z
            subtotal += GL_WEIGHTS[j]*math.exp(-3*v+drop)
        total += half*subtotal
        panels += 1
        lo = hi
        step = min(0.5, 2*step)
        if sqrtb == 0 and end_log < -745:
            v = sign*hi
            if -3*v+r*math.expm1(v)*slope < -745:
                break
    return total, panels


@njit(cache=True)
def _lnI_with_panels(a, b, xmin, xmax):
    """Return (lnI, count) for diagnostics; same domain as lnI_direct."""
    if not (math.isfinite(a) and math.isfinite(b) and b >= 0 and
            math.isfinite(xmin) and math.isfinite(xmax) and 0 < xmin < xmax):
        raise ValueError("require finite a, b>=0 and finite 0<xmin<xmax")
    r = min(xmax, max(xmin, a/b)) if b > 0 else (xmax if a > 0 else xmin)
    sqrtb = math.sqrt(b)
    slope = a-b*r
    height = 1-math.log(WINDOW_EPS)+4*(math.log(xmax)-math.log(xmin))
    width = _drop_width(abs(slope), sqrtb, height) if b else math.inf
    characteristic = min(0.25*r, 1/sqrtb if b else math.inf,
                         1/abs(slope) if slope else math.inf)
    initial = math.log1p(characteristic/r)
    if initial < 1e-150 or not math.isfinite(slope):
        return _extreme_local_integral(a, b, r, xmin < r < xmax)
    total = 0.0
    count = 0
    if r > xmin:
        extent = -math.log(xmin/r)
        if width < r-xmin:
            extent = -math.log1p(-width/r)
        value, panels = _side_integral(r, slope, sqrtb, extent, -1, initial)
        total += value
        count += panels
    if r < xmax:
        extent = math.log(xmax/r)
        if width < xmax-r:
            extent = math.log1p(width/r)
        value, panels = _side_integral(r, slope, sqrtb, extent, 1, initial)
        total += value
        count += panels
    return math.log(total)-3*math.log(r), count


@njit(cache=True)
def lnI_direct(a, b, xmin, xmax):
    """Return log integral x^-4 exp(a*x-b*x²/2-fmax) dx.

    Domain: finite a, b>=0 and finite 0<xmin<xmax. Validated absolute
    log error <=1e-8 for |a|<=1e7, b=0 or 1e-8<=b<=1e8, and
    (xmin,xmax)=(0.1,100) or (0.1,10). Tail bound is <=1e-12;
    this bound does not include floating-point and quadrature error.
    """
    return _lnI_with_panels(a, b, xmin, xmax)[0]


@njit(cache=True)
def lnI_direct_array(a, b, xmin, xmax):
    """Elementwise lnI_direct for equal-length one-dimensional arrays.

    Bounds are scalar; inputs obey lnI_direct's domain. Returns float64.
    """
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("a and b must be equal-length one-dimensional arrays")
    out = np.empty(a.size, dtype=np.float64)
    for i in range(a.size):
        out[i] = lnI_direct(a[i], b[i], xmin, xmax)
    return out
