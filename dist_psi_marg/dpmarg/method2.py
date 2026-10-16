"""Adaptive angular quadrature of the direct distance kernel.

The full-circle integral is folded into the pi-periodic pair h=J(a)+J(-a).
Cheap exponent/prior geometry locates up to two peaks without evaluating J.
All expensive nodes and their weights are exposed by method2_nodes.
"""

import math

import numpy as np
from numba import njit

from .kernel1d import _lnJ_pair_count
from .params import reduce_params

PI = math.pi
SCAN = 128
WIDTH_SWITCH = 0.22
CONVERGENCE_ATOL = 2e-6
SPLIT_WIDTH_SCALE = 1.5


@njit(cache=True)
def _proxy(t, l1, l2, R, eta, lo, hi):
    """max(fmax(+a),fmax(-a))-4 log(r*) for the positive-radius pair."""
    a = abs(R*math.cos(t-eta))
    b = l2+(l1-l2)*math.cos(t)**2
    r = min(hi, max(lo, a/b)) if b > 0 else (hi if a > 0 else lo)
    return r*(a-0.5*b*r)-4*math.log(r)


@njit(cache=True)
def _refine(left, right, l1, l2, R, eta, lo, hi, sign):
    """Golden-section extremum of sign*proxy on a bracketing interval."""
    g = 0.6180339887498949
    x = right-g*(right-left)
    y = left+g*(right-left)
    fx = sign*_proxy(x, l1, l2, R, eta, lo, hi)
    fy = sign*_proxy(y, l1, l2, R, eta, lo, hi)
    for _ in range(40):
        if fx > fy:
            right, y, fy = y, x, fx
            x = right-g*(right-left)
            fx = sign*_proxy(x, l1, l2, R, eta, lo, hi)
        else:
            left, x, fx = x, y, fy
            y = left+g*(right-left)
            fy = sign*_proxy(y, l1, l2, R, eta, lo, hi)
    return x if fx > fy else y


@njit(cache=True)
def _peak_width(t, l1, l2, R, eta, lo, hi):
    """Curvature width of the proxy, including r*'s interior motion."""
    co = math.cos(t-eta)
    a = abs(R*co)
    ap = -R*math.sin(t-eta)*(1 if co >= 0 else -1)
    b = l2+(l1-l2)*math.cos(t)**2
    bp = -(l1-l2)*math.sin(2*t)
    bpp = -2*(l1-l2)*math.cos(2*t)
    r = min(hi, max(lo, a/b)) if b else (hi if a > 0 else lo)
    curvature = -a*r-0.5*bpp*r*r
    if b > 0 and lo < r < hi:
        rp = (ap-bp*r)/b
        rpp = (-a-bpp*r-2*bp*rp)/b
        curvature += (ap-bp*r)*rp-4*(rpp/r-(rp/r)**2)
    return min(0.3, 1/math.sqrt(max(abs(curvature), 1e-30)))


@njit(cache=True)
def _geometry(l1, l2, R, eta, lo, hi):
    """Return peak centers/widths and valley boundaries on one period."""
    step = PI/SCAN
    values = np.empty(SCAN)
    for i in range(SCAN):
        values[i] = _proxy(i*step, l1, l2, R, eta, lo, hi)
    centers = np.empty(SCAN+2)
    heights = np.empty(SCAN+2)
    count = 0
    if np.max(values)-np.min(values) < 1e-10:
        return np.array([0.0]), np.array([0.3]), np.array([-PI/2, PI/2])
    for i in range(SCAN):
        if values[i] >= values[(i-1) % SCAN] and values[i] >= values[(i+1) % SCAN]:
            t = _refine((i-1)*step, (i+1)*step, l1, l2, R, eta, lo, hi, 1)
            centers[count] = t % PI
            heights[count] = _proxy(t, l1, l2, R, eta, lo, hi)
            count += 1
    # Extra seeds capture a narrow weak-axis mode between scan samples.
    for seed in (PI/2, eta):
        if (_proxy(seed, l1, l2, R, eta, lo, hi) >=
                max(_proxy(seed-step, l1, l2, R, eta, lo, hi),
                    _proxy(seed+step, l1, l2, R, eta, lo, hi))):
            t = _refine(seed-step, seed+step, l1, l2, R, eta, lo, hi, 1) % PI
            duplicate = False
            for j in range(count):
                if abs((t-centers[j]+PI/2) % PI-PI/2) < step/2:
                    duplicate = True
            if not duplicate:
                centers[count] = t
                heights[count] = _proxy(t, l1, l2, R, eta, lo, hi)
                count += 1
    order = np.argsort(heights[:count])[::-1]
    t1 = centers[order[0]]
    w1 = _peak_width(t1, l1, l2, R, eta, lo, hi)
    if count < 2 or heights[order[1]] < heights[order[0]]-40:
        return np.array([t1]), np.array([w1]), np.array([t1-PI/2, t1+PI/2])
    t2 = centers[order[1]]
    w2 = _peak_width(t2, l1, l2, R, eta, lo, hi)
    if t2 < t1:
        t1, t2, w1, w2 = t2, t1, w2, w1
    # Locate valleys by scanning first: the proxy need not be unimodal
    # over an entire arc (its radial clipping transitions are continuous).
    valleys = np.empty(2)
    for j in range(2):
        left, right = (t1, t2) if j == 0 else (t2, t1+PI)
        gridstep = (right-left)/64
        best, value = left, math.inf
        for k in range(1, 64):
            t = left+k*gridstep
            p = _proxy(t, l1, l2, R, eta, lo, hi)
            if p < value:
                best, value = t, p
        valleys[j] = _refine(best-gridstep, best+gridstep,
                              l1, l2, R, eta, lo, hi, -1)
    return (np.array([t1, t2]), np.array([w1, w2]),
            np.array([valleys[1]-PI, valleys[0], valleys[1]]))


@njit(cache=True)
def _nodes_reduced(l1, l2, R, eta, lo, hi, n_nodes):
    centers, widths, bounds = _geometry(l1, l2, R, eta, lo, hi)
    if np.min(widths) >= WIDTH_SWITCH:
        return np.arange(n_nodes)*PI/n_nodes, np.full(n_nodes, PI/n_nodes), 0
    if centers.size == 1:
        w, center = widths[0], centers[0]
        T = math.asinh(PI/(2*w))
        dt = 2*T/n_nodes
        tau = -T+np.arange(n_nodes)*dt
        nodes = center+w*np.sinh(tau)
        weights = dt*w*np.cosh(tau)
        # F'(-T)-F'(T)=-pi*h(endpoint), exactly by periodicity.
        weights[0] -= dt*dt*PI/12
        return nodes, weights, 1
    # A split sinh trapezoid has nonperiodic endpoints at the valleys.
    # Its endpoint derivative stencil can be unresolved near a weak-axis
    # trough, even when both peaks are resolved. Use Gauss--Legendre in
    # tau on these asymmetric intervals: it clusters nodes at both ends
    # and needs no Euler--Maclaurin correction or derivative evaluations.
    nodes = np.empty(n_nodes)
    weights = np.empty(n_nodes)
    offset = 0
    for j in range(2):
        n = n_nodes//2+(n_nodes % 2 if j == 0 else 0)
        x, q = _legendre_rule(n)
        w, center = min(0.3, SPLIT_WIDTH_SCALE*widths[j]), centers[j]
        tm = -math.asinh((center-bounds[j])/w)
        tp = math.asinh((bounds[j+1]-center)/w)
        half, mid = (tp-tm)/2, (tp+tm)/2
        for k in range(n):
            tau = mid+half*x[k]
            nodes[offset+k] = center+w*math.sinh(tau)
            weights[offset+k] = q[k]*half*w*math.cosh(tau)
        offset += n
    return nodes, weights, 2


@njit(cache=True)
def _legendre_rule(n):
    """Gauss--Legendre nodes/weights on [-1,1], via Legendre recurrence."""
    nodes, weights = np.empty(n), np.empty(n)
    for i in range((n+1)//2):
        z = math.cos(PI*(i+.75)/(n+.5))
        derivative = 0.0
        for _ in range(20):
            p, previous = 1.0, 0.0
            for k in range(1,n+1):
                p, previous = ((2*k-1)*z*p-(k-1)*previous)/k, p
            derivative = n*(z*p-previous)/(z*z-1)
            change = p/derivative
            z -= change
            if abs(change) < 2e-16:
                break
        nodes[i], nodes[n-1-i] = -z, z
        weights[i] = 2/((1-z*z)*derivative*derivative)
        weights[n-1-i] = weights[i]
    return nodes, weights


@njit(cache=True)
def _validate(A, B, C, D, E, xmin, xmax, n_nodes):
    """Validate public arguments and return canonical reduced parameters."""
    if n_nodes < 16:
        raise ValueError('n_nodes must be >=16')
    if not (math.isfinite(xmin) and math.isfinite(xmax) and 0 < xmin < xmax):
        raise ValueError('require finite 0<xmin<xmax')
    return reduce_params(A, B, C, D, E)


@njit(cache=True)
def _compute(A, B, C, D, E, xmin, xmax, n_nodes):
    l1, l2, R, eta = _validate(A, B, C, D, E, xmin, xmax, n_nodes)
    fullnodes, fullweights, mode = _nodes_reduced(l1, l2, R, eta, xmin, xmax, n_nodes)
    # Nested uniform-tau grids reuse every distance evaluation. Start at
    # >=16 nodes, then double up to the requested budget. The convergence
    # tolerance is empirical, not a rigorous error certificate; the full
    # reference sweep validates both early stopping and the maximum budget.
    stride = 1
    while mode != 2 and n_nodes % (4*stride) == 0 and n_nodes//(2*stride) >= 16:
        stride *= 2
    logs = np.full(n_nodes, np.nan)
    previous = math.inf
    count = 0
    norm = math.log(3/(2*PI))+3*math.log(xmin)-math.log1p(-(xmin/xmax)**3)
    while True:
        n = n_nodes//stride
        if stride == 1:
            nodes, weights = fullnodes, fullweights
        else:
            nodes, weights, mode = _nodes_reduced(l1, l2, R, eta, xmin, xmax, n)
        shift = -math.inf
        for i in range(n):
            if math.isnan(logs[i*stride]):
                t = nodes[i]
                a = R*math.cos(t-eta)
                b = l2+(l1-l2)*math.cos(t)**2
                logs[i*stride], calls = _lnJ_pair_count(a, b, xmin, xmax)
                count += calls
            shift = max(shift, logs[i*stride])
        total = 0.0
        for i in range(n):
            total += weights[i]*math.exp(logs[i*stride]-shift)
        value = norm+shift+math.log(total)
        if stride == 1 or abs(value-previous) < CONVERGENCE_ATOL:
            return value, count, mode, nodes, weights
        previous = value
        stride //= 2


@njit(cache=True)
def method2_nodes(A, B, C, D, E, xmin, xmax, n_nodes=64):
    """Return actual (theta, weights, mode) after nested convergence checks.

    Domain: finite physical coefficients E>=hypot(C,D), 0<xmin<xmax,
    n_nodes>=16 is a maximum budget. Nodes may lie outside [0,pi);
    h is pi-periodic. This diagnostic evaluates the distance kernels too.
    Mode 0 is uniform periodic; 1 is a corrected sinh trapezoid; 2 is
    Gauss--Legendre on two sinh intervals. Mode-1 weights include the
    first Euler--Maclaurin correction and may be signed.
    """
    _, _, mode, nodes, weights = _compute(A, B, C, D, E, xmin, xmax, n_nodes)
    return nodes, weights, mode


@njit(cache=True)
def _evaluate(A, B, C, D, E, xmin, xmax, n_nodes):
    """Return (value, actual lnI_direct evaluation count, geometry mode)."""
    value, count, mode, _, _ = _compute(A, B, C, D, E, xmin, xmax, n_nodes)
    return value, count, mode


@njit(cache=True)
def lnL_marg_m2(A, B, C, D, E, xmin, xmax, n_nodes=64):
    """Log normalized distance/polarization likelihood by theta quadrature.

    Computes log[3/(2*pi*(xmin^-3-xmax^-3))*integral_0^pi (J(a)+J(-a))
    dtheta], a=R*cos(theta-eta), b=lam2+(lam1-lam2)*cos(theta)**2.
    Domain and node semantics are those of method2_nodes. n_nodes is the
    maximum total number of paired theta nodes, including any secondary
    peak. Nested rules stop when successive log integrals differ by <2e-6.
    """
    return _evaluate(A, B, C, D, E, xmin, xmax, n_nodes)[0]


@njit(cache=True)
def lnL_marg_m2_array(parameters, n_nodes=64):
    """Numba loop over an (n,7) array of A,B,C,D,E,xmin,xmax rows."""
    if parameters.ndim != 2 or parameters.shape[1] != 7:
        raise ValueError('parameters must have shape (n,7)')
    out = np.empty(parameters.shape[0])
    for i in range(parameters.shape[0]):
        p = parameters[i]
        out[i] = lnL_marg_m2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], n_nodes)
    return out
