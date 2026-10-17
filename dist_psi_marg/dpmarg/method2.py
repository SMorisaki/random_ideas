"""Adaptive angular quadrature of direct or tabulated distance kernels.

The full-circle integral is folded into the pi-periodic pair h=J(a)+J(-a).
Cheap exponent/prior geometry locates up to two peaks without evaluating J.
All expensive nodes and their weights are exposed by method2_nodes.
"""

import math

import numpy as np
from numba import njit

from .kernel1d import _lnJ_pair_count
from .table1d import _lnJ_pair_table_count
from .params import reduce_params

PI = math.pi
GL32_X, GL32_W = np.polynomial.legendre.leggauss(32)
GL64_X, GL64_W = np.polynomial.legendre.leggauss(64)
# Eighth-order Euler--Maclaurin endpoint weights on a uniform tau grid.
# Derivative stencils are exact for degree <=7; absorb them into weights
# so partitioned sinh trapezoids need no extra radial evaluations.
_vand = np.arange(8,dtype=float)[None,:]**np.arange(8)[:,None]
SINH_ENDPOINT = np.zeros(8)
for _deriv,_coef in ((1,1/12),(3,-1/720),(5,1/30240),(7,-1/1209600)):
    _rhs=np.zeros(8);_rhs[_deriv]=math.factorial(_deriv)
    SINH_ENDPOINT+=_coef*np.linalg.solve(_vand,_rhs)
SCAN = 64
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
    for _ in range(28):
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
    if l1>0 and l2/l1<.01 and eta<.1:
        wc,kink=clipping_width(t,l1,l2,R,eta,lo,hi)
        if kink:return wc
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
def _refine_peak(seed, step, l1, l2, R, eta, lo, hi):
    """Keep a narrow seeded maximum when a broad golden bracket is multimodal.

    In the null-eigenvalue boundary layer the exponent peak at pi/2 can be
    much narrower than the scan spacing. The prior proxy also has a broad
    local peak in that same bracket. Compare broad and curvature-scaled
    refinement, retaining the seed itself as a candidate.
    """
    best = seed
    value = _proxy(seed, l1, l2, R, eta, lo, hi)
    width = _peak_width(seed, l1, l2, R, eta, lo, hi)
    for half in (step, min(step, 4*width)):
        t = _refine(seed-half, seed+half, l1, l2, R, eta, lo, hi, 1)
        candidate = _proxy(t, l1, l2, R, eta, lo, hi)
        if candidate > value:
            best, value = t, candidate
    return best


@njit(cache=True)
def _geometry(l1, l2, R, eta, lo, hi):
    """Legacy scan, augmented by analytic stationary/clipping candidates."""
    step = PI/SCAN
    values = np.empty(SCAN)
    for i in range(SCAN):
        values[i] = _proxy(i*step, l1, l2, R, eta, lo, hi)
    centers = np.empty(SCAN+30)
    heights = np.empty(SCAN+30)
    count = 0
    if np.max(values)-np.min(values) < 1e-10:
        return np.array([0.0]), np.array([0.3]), np.array([-PI/2, PI/2])
    for i in range(SCAN):
        if values[i] >= values[(i-1) % SCAN] and values[i] >= values[(i+1) % SCAN]:
            t = _refine_peak(i*step, step, l1, l2, R, eta, lo, hi)
            centers[count] = t % PI
            heights[count] = _proxy(t, l1, l2, R, eta, lo, hi)
            count += 1
    # Extra seeds capture a narrow weak-axis mode between scan samples.
    for seed in (PI/2, eta):
        if (_proxy(seed, l1, l2, R, eta, lo, hi) >=
                max(_proxy(seed-step, l1, l2, R, eta, lo, hi),
                    _proxy(seed+step, l1, l2, R, eta, lo, hi))):
            t = _refine_peak(seed, step, l1, l2, R, eta, lo, hi) % PI
            duplicate = False
            for j in range(count):
                if abs((t-centers[j]+PI/2) % PI-PI/2) < step/2:
                    duplicate = True
            if not duplicate:
                centers[count] = t
                heights[count] = _proxy(t, l1, l2, R, eta, lo, hi)
                count += 1
    # The coarse scan cannot see delta~eta weak-axis islands. Keep the
    # scan as an independent search, but seed it with the complete finite
    # analytic candidate set (stationary points and clipping crossings).
    if l1>0 and l2/l1<.01 and eta<.1:
        ac,aw,ab=analytic_geometry(l1,l2,R,eta,lo,hi)
    else:
        ac,aw=np.empty(0),np.empty(0)
    for seed,width in zip(ac,aw):
        t=_refine_peak(seed,min(step,2*width),l1,l2,R,eta,lo,hi)%PI
        h=_proxy(t,l1,l2,R,eta,lo,hi)
        duplicate=-1
        for j in range(count):
            if abs((t-centers[j]+PI/2)%PI-PI/2)<min(width,_peak_width(centers[j],l1,l2,R,eta,lo,hi)):
                duplicate=j;break
        if duplicate<0:
            centers[count],heights[count]=t,h;count+=1
        elif h>heights[duplicate]:
            centers[duplicate],heights[duplicate]=t,h
    # Reject the clipped-proxy artifact at a=b=0: the true pair is 2 V
    # and has positive angular curvature when R²>lam1.
    for j in range(count):
        t=centers[j];a=abs(R*math.cos(t-eta));b=l2+(l1-l2)*math.cos(t)**2
        if a*hi<1e-8 and b*hi*hi<1e-12 and R*R>l1:
            heights[j]=-math.inf
    order = np.argsort(heights[:count])[::-1]
    # A narrow distant-radius island can lie below the broad xmin proxy
    # peaks, and even below both adjacent scan samples. Its radial volume
    # can nevertheless make it significant (seed-808 cases 44 and 915).
    # When the scanned geometry is broad, retain this analytic weak-axis
    # seed and stretch the entire period about it. This also resolves the
    # broad contribution, without adding a quadrature-budget exception.
    broad = True
    for j in range(min(count, 2)):
        if _peak_width(centers[order[j]], l1, l2, R, eta, lo, hi) < WIDTH_SWITCH:
            broad = False
    if broad:
        seed = PI/2
        width = _peak_width(seed, l1, l2, R, eta, lo, hi)
        height = _proxy(seed, l1, l2, R, eta, lo, hi)
        if (width < WIDTH_SWITCH and height >= heights[order[0]]-40
                and height >= _proxy(seed-width, l1, l2, R, eta, lo, hi)
                and height >= _proxy(seed+width, l1, l2, R, eta, lo, hi)):
            local = _refine(seed-2*width, seed+2*width,
                            l1, l2, R, eta, lo, hi, 1)
            if _proxy(local, l1, l2, R, eta, lo, hi) > height:
                seed = local
            # Re-evaluate curvature at the refined maximum. At a minimum
            # or a radial-clipping cusp the seed's absolute curvature can
            # suggest a spurious narrow mode (mock-posterior 124 and 876).
            width = _peak_width(seed, l1, l2, R, eta, lo, hi)
            if width < WIDTH_SWITCH:
                return (np.array([seed]), np.array([width]),
                        np.array([seed-PI/2, seed+PI/2]))
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
    if n == 32:
        return GL32_X.copy(), GL32_W.copy()
    if n == 64:
        return GL64_X.copy(), GL64_W.copy()
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
def _compute_core(A, B, C, D, E, xmin, xmax, n_nodes, table=None):
    """Shared normalized angular quadrature; domain as method2_nodes.

    Return value, radial calls, mode, actual nodes/weights, fallback count.
    A None table specializes to the original direct distance kernel.
    """
    l1, l2, R, eta = _validate(A, B, C, D, E, xmin, xmax, n_nodes)
    # In the degenerate corner the historical nested 16/32 comparison
    # can agree while missing an island entirely. Retain the seeded scan
    # geometry, but share the fixed robust quadrature with the analytic
    # rule here; the ordinary legacy path and budget remain unchanged.
    if table is not None and n_nodes==64 and l1>0 and l2/l1<.01 and eta<.1:
        return _analytic_compute_details(A,B,C,D,E,xmin,xmax,table,
            ANALYTIC_GH_ORDER,0,False,ANALYTIC_SKIP_MARGIN,True)[:6]
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
    fallbacks = 0
    norm = math.log(3/(2*PI))+3*math.log(xmin)-math.log1p(-(xmin/xmax)**3)
    while True:
        n = n_nodes//stride
        if stride == 1:
            nodes, weights = fullnodes, fullweights
        else:
            nodes = fullnodes[::stride].copy()
            weights = fullweights[::stride].copy()*stride
            if mode == 1:
                # Coarse tau spacing scales the endpoint correction by stride².
                # The full rule has already supplied stride times its correction.
                # Nested budgets are even; their middle full-grid node is
                # the proxy center, which determines the original tau step.
                center=fullnodes[n_nodes//2]
                width_proxy=_peak_width(center,l1,l2,R,eta,xmin,xmax)
                dt=2*math.asinh(PI/(2*width_proxy))/n_nodes
                weights[0]-=(stride*stride-stride)*dt*dt*PI/12
        shift = -math.inf
        for i in range(n):
            if math.isnan(logs[i*stride]):
                t = nodes[i]
                a = R*math.cos(t-eta)
                b = l2+(l1-l2)*math.cos(t)**2
                if table is None:
                    logs[i*stride], calls = _lnJ_pair_count(a, b, xmin, xmax)
                else:
                    logs[i*stride], calls, fb = _lnJ_pair_table_count(a,b,table)
                    fallbacks += fb
                count += calls
            shift = max(shift, logs[i*stride])
        total = 0.0
        for i in range(n):
            total += weights[i]*math.exp(logs[i*stride]-shift)
        value = norm+shift+math.log(total)
        # Preserve the exact normalized unit likelihood despite interpolation
        # roundoff at the single a=b=0 table coordinate.
        if A == 0 and B == 0 and C == 0 and D == 0 and E == 0:
            value = 0.0
        if stride == 1 or abs(value-previous) < CONVERGENCE_ATOL:
            return value, count, mode, nodes, weights, fallbacks
        previous = value
        stride //= 2


@njit(cache=True)
def _compute(A,B,C,D,E,xmin,xmax,n_nodes):
    result=_compute_core(A,B,C,D,E,xmin,xmax,n_nodes)
    return result[0],result[1],result[2],result[3],result[4]


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


@njit(cache=True)
def method2_table_diagnostics(A,B,C,D,E,xmin,xmax,table,n_nodes=64,rule="analytic"):
    """Return (value,kernel_calls,mode,nodes,weights,fallback_count).

    rule='analytic' uses fixed, geometry-selected GH/grid/sinh nodes;
    rule='scan' retains the historical nested rule and n_nodes budget.
    """
    if xmin!=table[7] or xmax!=table[8]:
        raise ValueError('table bounds must match integration bounds')
    if n_nodes<16:
        raise ValueError('n_nodes must be >=16')
    if rule == "analytic":
        return _analytic_compute(A,B,C,D,E,xmin,xmax,table)
    if rule != "scan":
        raise ValueError("rule must be analytic or scan")
    return _compute_core(A,B,C,D,E,xmin,xmax,n_nodes,table)


@njit(cache=True)
def lnL_marg_m2_table(A,B,C,D,E,xmin,xmax,table,n_nodes=64,rule="analytic"):
    """Normalized distance/angle likelihood through the paired radial table.

    Domain: finite E>=hypot(C,D), 0<xmin<xmax matching table bounds.
    rule='analytic' selects GH12, a precomputed grid, or a fixed sinh
    fallback without scanning/nested refinement. rule='scan' preserves
    the historical n_nodes maximum budget. n_nodes is unused by analytic;
    order overrides and doubled nodes are available in diagnostic helpers.
    """
    return method2_table_diagnostics(A,B,C,D,E,xmin,xmax,table,n_nodes,rule)[0]


@njit(cache=True)
def lnL_marg_m2_table_array(parameters,table,n_nodes=64,rule="analytic"):
    """Numba array loop over (n,7) coefficient/bound rows for a fixed table."""
    if parameters.ndim!=2 or parameters.shape[1]!=7:
        raise ValueError('parameters must have shape (n,7)')
    out=np.empty(len(parameters))
    for i in range(len(parameters)):
        p=parameters[i]
        out[i]=lnL_marg_m2_table(p[0],p[1],p[2],p[3],p[4],p[5],p[6],table,n_nodes,rule)
    return out

# Analytic, non-nested rules. Import-time constants contain every trig factor
# needed by a broad fixed grid and every allowed Hermite rule.
from .geometry import (analytic_geometry, laplace_peak, proxy_derivatives,
                       interior_theta, _newton, clipping_width, candidate_seeds)
GH_ORDERS = np.array([8,12,16,20,24,32,64])
GH_X = np.zeros((7,64))
GH_LOGW = np.zeros((7,64))
for _i, _n in enumerate(GH_ORDERS):
    _x, _w = np.polynomial.hermite.hermgauss(int(_n))
    GH_X[_i,:_n] = _x
    GH_LOGW[_i,:_n] = np.log(_w)+_x*_x
GRID_SIZES = np.array([16,24,32,48,64,96,128])
GRID_COS = np.zeros((7,128))
GRID_SIN = np.zeros((7,128))
GRID_COS2 = np.zeros((7,128))
for _i, _n in enumerate(GRID_SIZES):
    _theta = np.arange(_n)*PI/_n
    GRID_COS[_i,:_n] = np.cos(_theta)
    GRID_SIN[_i,:_n] = np.sin(_theta)
    GRID_COS2[_i,:_n] = np.cos(2*_theta)
ANALYTIC_WIDTH_SWITCH = .2
ANALYTIC_GH_ORDER = 12
ANALYTIC_SKIP_MARGIN = 40.
ANALYTIC_GRID_SPACING = 1.3
ANALYTIC_SINH_WIDTH = .017
FALLBACK_N = 64


@njit(cache=True, inline='always')
def _node_proxy(a,b,lo,hi):
    """Laplace log h from a,b, or continuous q outside its valid domain.

    Uses the same radial-asymptote guards as laplace_peak, without trig or
    derivatives. The pair's omitted factor is at most two. This proxy is
    used for selection only; the node skip has a separate integral bound.
    """
    a=abs(a)
    r=min(hi,max(lo,a/b)) if b>0 else (hi if a>0 else lo)
    p=r*(a-.5*b*r)-4*math.log(r)
    if lo<r<hi:
        if min(r-lo,hi-r)*math.sqrt(b)>4 and r*math.sqrt(b)>6:
            p+=.5*math.log(2*PI/b)
    else:
        slope=abs(a-b*r)
        if slope>6*math.sqrt(b) and slope*r>20:
            p-=math.log(slope)
    return p


@njit(cache=True)
def _candidate_mass(t,w,l1,l2,R,eta,lo,hi):
    """Log angular mass estimate with finite radial width and prior pedestal.

    The radial layer is bounded by the prior scale r/3, Gaussian width
    sqrt(2*pi/b), and endpoint slope. Unlike an endpoint Laplace factor,
    this estimate stays finite at a=b=0, where h=2*(lo^-3-hi^-3)/3.
    It is only a ranking estimate; no likelihood approximation is returned.
    """
    a=abs(R*math.cos(t-eta));b=l2+(l1-l2)*math.cos(t)**2
    if a*hi<1e-8 and b*hi*hi<1e-12:
        return math.log(2*(lo**-3-hi**-3)/3)+math.log(w)
    r=min(hi,max(lo,a/b)) if b>0 else hi
    layer=min(hi-lo,r/3,math.sqrt(2*PI/max(b,1e-300)))
    if r==lo or r==hi:
        layer=min(layer,1/max(abs(a-b*r),1e-300))
    return r*(a-.5*b*r)-4*math.log(r)+math.log(layer)+math.log(w)


@njit(cache=True, inline='always')
def _node_lower_bound(a,b,lo,hi):
    """Lower bound on log h from a finite interval adjacent to argmax f.

    Choose delta<=1/sqrt(b), 1/|f'(r*)| and half the prior interval,
    integrating toward the longer side. f drops by at most 3/2 there;
    x^-4 >= (r*+delta)^-4. Valid for b>=0, 0<lo<hi.
    """
    a=abs(a)
    r=min(hi,max(lo,a/b)) if b>0 else (hi if a>0 else lo)
    delta=min((hi-lo)/2,1/math.sqrt(max(b,1e-300)),1/max(abs(a-b*r),1e-300))
    return r*(a-.5*b*r)-1.5-4*math.log(min(hi,r+delta))+math.log(delta)


@njit(cache=True, inline='always')
def _node_upper_bound(a,b,lo,hi):
    """log h <= fmax + log(2 Gmax) - 4 log lo, for b>=0.

    Gmax bounds the exponent-only integral by the interval length, whole
    Gaussian area, and (for clipped peaks) the endpoint exponential layer.
    """
    a=abs(a)
    r=min(hi,max(lo,a/b)) if b>0 else (hi if a>0 else lo)
    length=min(hi-lo,math.sqrt(2*PI/max(b,1e-300)))
    if r==lo or r==hi:
        length=min(length,1/max(abs(a-b*r),1e-300))
    return r*(a-.5*b*r)+math.log(2*length)-4*math.log(lo)


@njit(cache=True)
def _width_grid(w):
    """Smallest precomputed grid with h<=spacing*w (capped at 128)."""
    for n in GRID_SIZES[3:]:
        if PI/n<=ANALYTIC_GRID_SPACING*w:
            return n
    return 128


@njit(cache=True)
def _analytic_plan(l1,l2,R,eta,lo,hi,gh_order=ANALYTIC_GH_ORDER,grid_n=0,use_scan=False):
    """Geometry-only selection: mode 3 GH, 4 fixed grid, 5 sinh fallback.

    GH requires isolated radial asymptotes and a 16-probe pedestal mass
    below 1e-6 of the Gaussian mass outside the union of six-width windows.
    Failed GH patches use width-selected grids, or sinh64 below w=.017.
    Mode 6 partitions overlapping peaks into sinh/GL32 intervals.
    grid_n and gh_order are non-adaptive diagnostic overrides.
    """
    # A dominant, enclosed Cartesian saddle can bypass even the small
    # candidate search when low-SNR prior lobes have negligible mass.
    corner=l1>0 and l2/l1<.01 and eta<.1
    single=False
    if not use_scan and l2>0 and l2/l1>.01:
        t0=interior_theta(l1,l2,eta)
        a0=abs(R*math.cos(t0-eta));b0=l2+(l1-l2)*math.cos(t0)**2
        r0=a0/b0
        if lo<r0<hi and min(r0-lo,hi-r0)*math.sqrt(b0)>4:
            t,p,dd=_newton(t0,l1,l2,R,eta,lo,hi)
            if dd<0:
                w=min(.3,1/math.sqrt(-dd))
                single=p+math.log(w)>-4*math.log(lo)+math.log(PI)+25
                if single:
                    centers,widths,bounds=np.array([t]),np.array([w]),np.array([t-PI/2,t+PI/2])
    if not single:
        if use_scan:
            centers,widths,bounds = _geometry(l1,l2,R,eta,lo,hi)
        else:
            centers,widths,bounds = analytic_geometry(l1,l2,R,eta,lo,hi)
    prior_layer=False
    if not corner:
        for j in range(len(centers)):
            t=centers[j]
            if not clipping_width(t,l1,l2,R,eta,lo,hi)[1]:continue
            a=abs(R*math.cos(t-eta));b=l2+(l1-l2)*math.cos(t)**2
            r=lo if abs(a-b*lo)<abs(a-b*hi) else hi
            if r*math.sqrt(b)<6:
                prior_layer=True
                # There is no isolated radial saddle in this prior-
                # dominated layer: mapping its formal Gaussian width
                # through x0' creates an artificial tiny angular scale.
                # Preserve the geometric kink flag (never GH), but use
                # the broad one-sided / prior resolution for quadrature.
                left=proxy_derivatives(t-1e-7,l1,l2,R,eta,lo,hi)[2]
                right=proxy_derivatives(t+1e-7,l1,l2,R,eta,lo,hi)[2]
                widths[j]=max(min(.3,1/math.sqrt(max(abs(left),1e-30))),
                              min(.3,1/math.sqrt(max(abs(right),1e-30))))
    # Marginal radial volume can create an island where q itself has no
    # stationary maximum. Refine the closed-form exponent seed with the
    # full radial Laplace proxy before selecting widths / a broad grid.
    if corner and R>0:
        u,w,valid=laplace_peak(interior_theta(l1,l2,eta),l1,l2,R,eta,lo,hi)
        if valid:
            duplicate=False
            for j in range(len(centers)):
                if abs((u-centers[j]+PI/2)%PI-PI/2)<min(w,widths[j]):duplicate=True
            if not duplicate:
                centers=np.append(centers,u);widths=np.append(widths,w)
                order=np.argsort(centers);centers,widths=centers[order],widths[order]
                bounds=np.empty(len(centers)+1);bounds[0]=(centers[-1]-PI+centers[0])/2
                for j in range(1,len(centers)):bounds[j]=(centers[j-1]+centers[j])/2
                bounds[-1]=bounds[0]+PI
    # A high-SNR flat ridge can have a sharp xmax shoulder away from its
    # angular maximum. Its non-negligible cutoff also needs clustered
    # nodes (R/lam1=30, eta=0), even though it is not a q maximum.
    shoulders=False
    if l1>0 and l2/l1<.01 and R/l1>hi*.05:
        best=-math.inf
        for j in range(len(centers)):
            best=max(best,_candidate_mass(centers[j],widths[j],l1,l2,R,eta,lo,hi))
        cc=np.empty(len(centers)+28);ww=np.empty(len(cc));count=len(centers)
        cc[:count],ww[:count]=centers,widths
        for t in candidate_seeds(l1,l2,R,eta,lo,hi):
            co=math.cos(t-eta);a=abs(R*co);b=l2+(l1-l2)*math.cos(t)**2
            if b<=0 or abs(a-b*hi)>1e-8*max(a,b*hi,1e-300):continue
            ap=-R*math.sin(t-eta)*(1. if co>=0 else -1.)
            bp=-(l1-l2)*math.sin(2*t)
            w=min(.3,math.sqrt(b)/max(abs(ap-bp*hi),1e-300))
            mass=_candidate_mass(t,.1,l1,l2,R,eta,lo,hi)
            if hi*math.sqrt(b)<6 or mass<best-14:continue
            duplicate=False
            for j in range(count):
                if abs((t-cc[j]+PI/2)%PI-PI/2)<min(w,ww[j]):duplicate=True
            if not duplicate:
                cc[count],ww[count]=t%PI,w;count+=1;shoulders=True
        order=np.argsort(cc[:count]);centers,widths=cc[:count][order],ww[:count][order]
        bounds=np.empty(count+1);bounds[0]=(centers[-1]-PI+centers[0])/2
        for j in range(1,count):bounds[j]=(centers[j-1]+centers[j])/2
        bounds[-1]=bounds[0]+PI
    wmin = np.min(widths)
    if grid_n or wmin >= ANALYTIC_WIDTH_SWITCH:
        n = _width_grid(wmin)
        if prior_layer:n=max(64,n)
        if grid_n:
            n = grid_n
        return centers,widths,bounds,4,n
    # Disjoint GH patches are summed only when their tails are negligible;
    # close/overlapping peaks use a partition of the full period instead.
    original_centers,original_widths,original_bounds=centers,widths,bounds
    # Keep the full 40-log-unit candidate set in geometry. A candidate's
    # approximate angular mass can be negligible even while its height is
    # retained. Discard only from the GH mixture at an 18-log mass gap.
    masses = np.empty(len(centers))
    for j in range(len(centers)):
        masses[j]=_candidate_mass(centers[j],widths[j],l1,l2,R,eta,lo,hi)
    keep=masses>=np.max(masses)-18
    centers,widths=centers[keep],widths[keep]
    sharp = True
    for j in range(len(centers)):
        if clipping_width(centers[j],l1,l2,R,eta,lo,hi)[1]:
            sharp=False
            break
        u,w,valid = laplace_peak(centers[j],l1,l2,R,eta,lo,hi)
        if not valid or w >= ANALYTIC_WIDTH_SWITCH:
            sharp = False
            break
        centers[j],widths[j] = u,w
        if corner and abs((u-PI/2+PI/2)%PI-PI/2)<20*w:
            sharp=False
            break
    if sharp:
        for j in range(len(centers)):
            for k in range(j):
                distance = abs((centers[j]-centers[k]+PI/2)%PI-PI/2)
                if distance < 6*(widths[j]+widths[k]):
                    sharp = False
        idx = 0
        for k in range(len(GH_ORDERS)):
            if GH_ORDERS[k] == gh_order:
                idx = k
        if math.sqrt(2)*np.max(widths)*GH_X[idx,gh_order-1] >= PI/2:
            sharp=False
    if sharp:
        # Uniform probe factors are precomputed; no radial calls or trig.
        coseta,sineta=math.cos(eta),math.sin(eta)
        peak=-math.inf
        for j in range(len(centers)):
            p=proxy_derivatives(centers[j],l1,l2,R,eta,lo,hi,True)[0]
            peak=max(peak,p+math.log(math.sqrt(2*PI)*widths[j]))
        pedestal=0.
        for k in range(16):
            t=k*PI/16
            outside=True
            for j in range(len(centers)):
                if abs((t-centers[j]+PI/2)%PI-PI/2)<=6*widths[j]:
                    outside=False
            if outside:
                a=R*(GRID_COS[0,k]*coseta+GRID_SIN[0,k]*sineta)
                b=l2+(l1-l2)*GRID_COS[0,k]**2
                p=_node_proxy(a,b,lo,hi)
                pedestal+=math.exp(min(700.,p-peak))*PI/16
        sharp=pedestal<1e-6
    if sharp:
        # Near degeneracy, radial marginalization produces asymmetric
        # rational peaks: GH12 can leave 1e-3 errors despite tiny pedestal.
        # GH32 resolves their higher cumulants on the corner audit.
        order=32 if corner and gh_order==12 else gh_order
        return centers,widths,bounds,3,order
    # Restore the continuous-proxy geometry if a Laplace refinement failed.
    centers,widths,bounds = original_centers,original_widths,original_bounds
    # A sharp island plus a broad pedestal is best covered by a single
    # full-period sinh patch; splitting at a proxy cusp wastes nodes.
    sharp_ids = np.where(widths < ANALYTIC_WIDTH_SWITCH)[0]
    kink=shoulders
    for t in centers:
        kink=kink or clipping_width(t,l1,l2,R,eta,lo,hi)[1]
    if not corner and not kink and len(sharp_ids)>1 and np.max(widths[sharp_ids])<10*np.min(widths[sharp_ids]):
        return centers,widths,bounds,6,32
    # Grid spacing needs a conservative width: proxy curvature omits
    # non-Gaussian radial-tail structure. The calibrated 0.8 allowance
    # also ensures the 128-node grid actually resolves its threshold.
    widths=widths*(.25 if corner else .8)
    # At a nearly null weak axis a prior-dominated maximum can have an
    # unresolved core from the upper radial endpoint. Predict that scale
    # explicitly instead of applying the xmin curvature to every radius.
    for j in range(len(centers)):
        a=abs(R*math.cos(centers[j]-eta))
        b=l2+(l1-l2)*math.cos(centers[j])**2
        if a*hi<4 and b*hi*hi<4 and l1>l2:
            widths[j]=min(widths[j],1/(hi*math.sqrt(l1-l2)))
    # A weak-axis ray may contain a distant-radius island below the
    # continuous proxy's clipping maximum. Resolve its fixed-radius
    # angular scale even if it is not a stationary maximum of q.
    aw=abs(R*math.sin(eta))
    rw=min(hi,max(lo,aw/l2)) if l2>0 else (hi if aw>0 else lo)
    pw=rw*(aw-.5*l2*rw)-4*math.log(rw)
    best=-math.inf
    for t in centers:
        best=max(best,proxy_derivatives(t,l1,l2,R,eta,lo,hi)[0])
    if pw>best-40:
        # The unperturbed r^-4 prior has rms radius sqrt(3)*lo;
        # its finite mass above the exponent's clipped mode must also be
        # resolved near a null axis (where angular curvature grows as r²).
        resolution_r=rw
        if l1>10*l2 and aw*lo<4 and l2*lo*lo<4:
            resolution_r=max(rw,min(hi,math.sqrt(3)*lo))
        ww=.8/math.sqrt(max(aw*resolution_r+(l1-l2)*resolution_r**2,1e-30))
        j=np.argmin(np.abs(centers%PI-PI/2))
        widths[j]=min(widths[j],ww)
    wmin=np.min(widths)
    if grid_n or wmin>=ANALYTIC_SINH_WIDTH:
        n=_width_grid(wmin)
        if prior_layer:n=max(64,n)
        return centers,widths,bounds,4,grid_n if grid_n else n
    if kink or len(sharp_ids)>1 or (len(sharp_ids) and len(centers)>1):
        # Preserve every significant kink: one full-period patch can
        # under-resolve the opposite clipping transition on a flat ridge.
        masses=np.array([_candidate_mass(centers[j],widths[j],l1,l2,R,eta,lo,hi)
                         for j in range(len(centers))])
        keep=masses>=np.max(masses)-18
        centers,widths=centers[keep],widths[keep]
        bounds=np.empty(len(centers)+1)
        for j in range(len(centers)):
            k=(j+1)%len(centers)
            right=centers[k]+(PI if k==0 else 0.)
            # A broad pedestal and tiny island need cuts closer to the
            # island; midpoint cuts force its patch to span the pedestal.
            ratio=max(widths[j],widths[k])/min(widths[j],widths[k])
            fraction=(widths[j]**.25/(widths[j]**.25+widths[k]**.25)
                      if ratio>10 else .5)
            bounds[j+1]=centers[j]+(right-centers[j])*fraction
        bounds[0]=bounds[-1]-PI
        return centers,widths,bounds,5,FALLBACK_N
    if len(sharp_ids):
        # A tiny but negligible endpoint layer must not choose the centre
        # of the full-period patch. Rank by estimated integrated mass.
        masses=np.array([_candidate_mass(centers[j],widths[j],l1,l2,R,eta,lo,hi)
                         for j in range(len(centers))])
        j = np.argmax(masses)
        center,width = centers[j],widths[j]
        return np.array([center]),np.array([width]),np.array([center-PI/2,center+PI/2]),5,FALLBACK_N
    return centers,widths,bounds,5,FALLBACK_N


@njit(cache=True)
def _analytic_compute_details(A,B,C,D,E,xmin,xmax,table,gh_order=ANALYTIC_GH_ORDER,grid_n=0,double_nodes=False,skip_margin=ANALYTIC_SKIP_MARGIN,use_scan=False):
    """One scan-free angular rule; six legacy fields plus skipped-node count.

    The positive-radius pair lookup retains its certified J(-a) skip.
    No nested refinement is performed. GH log weights include t², and all
    branches use a log-sum-exp reduction with the normalized volume prior.
    """
    l1,l2,R,eta = _validate(A,B,C,D,E,xmin,xmax,64)
    centers,widths,bounds,mode,n = _analytic_plan(l1,l2,R,eta,xmin,xmax,gh_order,grid_n,use_scan)
    patches = len(centers) if mode != 4 else 1
    if double_nodes:
        n *= 2
    count = n*patches
    nodes,logweights,logs = np.empty(count),np.empty(count),np.empty(count)
    aa,bb,scores = np.empty(count),np.empty(count),np.empty(count)
    calls,fb = 0,0
    coseta,sineta = math.cos(eta),math.sin(eta)
    endpoint_correction=np.ones(patches+1,dtype=np.bool_)
    if mode==5 and patches>1:
        lower=-math.inf
        for t in centers:
            lower=max(lower,_node_lower_bound(R*math.cos(t-eta),l2+(l1-l2)*math.cos(t)**2,xmin,xmax))
        for j in range(patches+1):
            t=bounds[j]
            upper=_node_upper_bound(R*math.cos(t-eta),l2+(l1-l2)*math.cos(t)**2,xmin,xmax)
            endpoint_correction[j]=upper>lower-12
    idx = 0
    if mode == 4:
        for k in range(len(GRID_SIZES)):
            if GRID_SIZES[k] == n:
                idx = k
    elif mode == 3:
        for k in range(len(GH_ORDERS)):
            if GH_ORDERS[k] == n:
                idx = k
    for j in range(patches):
        if mode == 5 or mode == 6:
            stretch=1. if patches==1 else 1.5
            if mode==5 and l1>0 and l2/l1<.01 and eta<.1:
                # Grid safety widths are deliberately tighter than the
                # sinh coordinate scale; widening the latter distributes
                # enough of the 64 nodes over the interior ridge.
                stretch*=2
            w,center = min(.3,stretch*widths[j]),centers[j]
            tm = -math.asinh((center-bounds[j])/w)
            tp = math.asinh((bounds[j+1]-center)/w)
            half,mid = (tp-tm)/2,(tp+tm)/2
        for k in range(n):
            i = j*n+k
            if mode == 4:
                nodes[i] = k*PI/n
                if n<=128:
                    ct,st=GRID_COS[idx,k],GRID_SIN[idx,k]
                else:
                    ct,st=math.cos(nodes[i]),math.sin(nodes[i])
                a = R*(ct*coseta+st*sineta)
                # Squared cos avoids catastrophic cancellation at weak axis.
                b = l2+(l1-l2)*ct**2
                logweights[i] = math.log(PI/n)
            else:
                if mode == 3:
                    scale = math.sqrt(2)*widths[j]
                    nodes[i] = centers[j]+scale*GH_X[idx,k]
                    logweights[i] = math.log(scale)+GH_LOGW[idx,k]
                else:
                    if patches == 1:
                        # Corrected periodic sinh trapezoid, fixed at 64;
                        # no convergence loop and no reused/coarse nodes.
                        T=math.asinh(PI/(2*w))
                        dt=2*T/n
                        tau=-T+k*dt
                        nodes[i]=center+w*math.sinh(tau)
                        weight=dt*w*math.cosh(tau)
                        if k==0:
                            weight-=dt*dt*PI/12
                        logweights[i]=math.log(weight)
                    elif mode==5:
                        # Composite trapezoid on a nonperiodic sinh
                        # interval. Euler--Maclaurin endpoint correction
                        # uses seventh-order one-sided derivative stencils
                        # of F(tau)=h(theta)*w*cosh(tau), absorbed into
                        # positive weights; 64 nodes include both ends.
                        dt=(tp-tm)/(n-1)
                        tau=tm+k*dt
                        nodes[i]=center+w*math.sinh(tau)
                        factor=.5 if k==0 or k==n-1 else 1.
                        edge=min(k,n-1-k)
                        # Do not extrapolate a derivative stencil from
                        # the peak into a negligible endpoint. The null
                        # axis has a very thin regular core; its exact
                        # endpoint derivatives do not describe the ridge
                        # over the first several tau samples.
                        active=endpoint_correction[j if k<n//2 else j+1]
                        if edge<8 and active:
                            factor+=SINH_ENDPOINT[edge]
                        logweights[i]=math.log(dt*w*math.cosh(tau)*factor)
                    else:
                        xq,wq = (GL64_X,GL64_W) if double_nodes else (GL32_X,GL32_W)
                        tau = mid+half*xq[k]
                        nodes[i] = center+w*math.sinh(tau)
                        logweights[i] = math.log(wq[k]*half*w*math.cosh(tau))
                a = R*math.cos(nodes[i]-eta)
                b = l2+(l1-l2)*math.cos(nodes[i])**2
            aa[i],bb[i]=a,b
            scores[i]=_node_proxy(a,b,xmin,xmax)+logweights[i]
    imax=np.argmax(scores)
    cutoff=scores[imax]-skip_margin
    # Each omitted weighted term <= exp(-25)/count times this lower
    # bound on one retained term: summed relative loss <= exp(-25).
    certified=_node_lower_bound(aa[imax],bb[imax],xmin,xmax)+logweights[imax]-25-math.log(count)
    skipped=0
    for i in range(count):
        if (scores[i]<cutoff and
                _node_upper_bound(aa[i],bb[i],xmin,xmax)+logweights[i]<certified):
            logs[i]=-math.inf
            skipped+=1
        else:
            if table is None:
                logs[i],nc=_lnJ_pair_count(aa[i],bb[i],xmin,xmax)
                nf=0
            else:
                logs[i],nc,nf=_lnJ_pair_table_count(aa[i],bb[i],table,True)
            calls+=nc
            fb+=nf
            logs[i]+=logweights[i]
    shift = np.max(logs)
    total = 0.
    for i in range(count):
        total += math.exp(logs[i]-shift)
    norm = math.log(3/(2*PI))+3*math.log(xmin)-math.log1p(-(xmin/xmax)**3)
    value = norm+shift+math.log(total)
    if A==0 and B==0 and C==0 and D==0 and E==0:
        value = 0.
    return value,calls,mode,nodes,np.exp(logweights),fb,skipped


@njit(cache=True)
def _analytic_compute(A,B,C,D,E,xmin,xmax,table,gh_order=ANALYTIC_GH_ORDER,grid_n=0,double_nodes=False,skip_margin=ANALYTIC_SKIP_MARGIN):
    """Legacy six-field result; skip_margin=inf disables node skipping."""
    return _analytic_compute_details(A,B,C,D,E,xmin,xmax,table,gh_order,grid_n,double_nodes,skip_margin)[:6]


@njit(cache=True)
def method2_analytic_diagnostics(A,B,C,D,E,xmin,xmax,table,gh_order=ANALYTIC_GH_ORDER,grid_n=0):
    """Return (value,branch,N,w,kernel_calls,radial_fallbacks,skipped).

    Branch 3=GH, 4=fixed periodic grid, 5=corrected sinh, 6=split GL32.
    N is the total paired angular-node count, w the narrowest selected width
    (Laplace curvature for GH, conservative resolution width otherwise).
    Diagnostic overrides use the same selection and never refine nodes.
    """
    if xmin != table[7] or xmax != table[8]:
        raise ValueError('table bounds must match integration bounds')
    result = _analytic_compute_details(A,B,C,D,E,xmin,xmax,table,gh_order,grid_n)
    l1,l2,R,eta = reduce_params(A,B,C,D,E)
    _,w,_,_,_ = _analytic_plan(l1,l2,R,eta,xmin,xmax,gh_order,grid_n)
    return result[0],result[2],len(result[3]),np.min(w),result[1],result[5],result[6]


@njit(cache=True)
def method2_analytic_error(A,B,C,D,E,xmin,xmax,table):
    """Optional doubled-node diagnostic: (base value, doubled minus base).

    GH12->24 / corner GH32->64 (or the width-selected grid), grid N->2N,
    sinh64->128, split GL32->64 per interval. Never used in production.
    """
    if xmin!=table[7] or xmax!=table[8]:
        raise ValueError('table bounds must match integration bounds')
    base=_analytic_compute(A,B,C,D,E,xmin,xmax,table)
    if base[2]==3:
        l1,l2,R,eta=reduce_params(A,B,C,D,E)
        _,_,_,_,order=_analytic_plan(l1,l2,R,eta,xmin,xmax)
        b=_analytic_compute(A,B,C,D,E,xmin,xmax,table,2*order)[0]
    else:
        b=_analytic_compute(A,B,C,D,E,xmin,xmax,table,ANALYTIC_GH_ORDER,0,True)[0]
    return base[0],b-base[0]
