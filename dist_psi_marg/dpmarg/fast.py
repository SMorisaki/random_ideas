"""Normalized volume-prior likelihood: guarded 2-D Laplace, else method 2.

The accuracy gate is calibrated empirically for the regimes in TASK.md;
it is not a uniform error certificate for all finite coefficients or priors.
Branch codes are 0 (table-backed method 2) and 1 (order-4 2-D Laplace).
"""
import math

import numpy as np
from numba import njit

from .method2 import lnL_marg_m2_table
from .method3 import lnL_marg_m3_2d
from .params import reduce_params

LAPLACE_MARGIN = 1.5
"""Safety multiplier on the measured 1e-3 breakdown SNR."""
LAPLACE_Q_MIN = 1e-3
"""Never use Laplace below this weak/strong eigenvalue ratio."""
LAPLACE_Q_KNOTS = np.array([.001, .01, .1, .3, 1.])
"""Anisotropy knots for monotone log-log interpolation of breakdown SNR."""
LAPLACE_RHO_BREAK = np.array([422., 133., 42., 18., 18.])
"""Iteration-006 synthesis thresholds; 18 is used at q>=0.3 as requested."""
LAPLACE_INTERIOR_SIGMA = 3.
"""Require the method3 three-sigma weak-axis ellipse inside the annulus."""
BRANCH_METHOD2 = 0
BRANCH_LAPLACE = 1


@njit(cache=True)
def rho_break(q):
    """Monotone log-log SNR threshold; infinity for q<1e-3, 18 for q>=.3."""
    if q < LAPLACE_Q_MIN:
        return math.inf
    if q >= .3:
        return LAPLACE_RHO_BREAK[3]
    for j in range(3):
        if q <= LAPLACE_Q_KNOTS[j+1]:
            t = math.log(q/LAPLACE_Q_KNOTS[j])/math.log(
                LAPLACE_Q_KNOTS[j+1]/LAPLACE_Q_KNOTS[j])
            return math.exp((1-t)*math.log(LAPLACE_RHO_BREAK[j])
                            + t*math.log(LAPLACE_RHO_BREAK[j+1]))
    return LAPLACE_RHO_BREAK[3]


@njit(cache=True)
def laplace_gate(A, B, C, D, E, xmin, xmax):
    """Check positive curvature, interior ellipse, and rho>=margin*rho_break.

    For the interior Cartesian peak s=R cos(eta)/lam1, t=R sin(eta)/lam2,
    rho²=lam1*s²+lam2*t²=2 lnL_peak, avoiding a numerical maximization.
    Domain: finite physical coefficients E>=hypot(C,D), 0<xmin<xmax.
    """
    if not (math.isfinite(xmin) and math.isfinite(xmax) and 0 < xmin < xmax):
        raise ValueError('require finite 0<xmin<xmax')
    l1, l2, R, eta = reduce_params(A, B, C, D, E)
    if l2 <= 0 or l2/l1 < LAPLACE_Q_MIN:
        return False
    s = R*math.cos(eta)/l1
    t = R*math.sin(eta)/l2
    r = math.hypot(s, t)
    guard = LAPLACE_INTERIOR_SIGMA/math.sqrt(l2)
    if r-xmin <= guard or xmax-r <= guard:
        return False
    rho = math.hypot(math.sqrt(l1)*s, math.sqrt(l2)*t)
    return rho >= LAPLACE_MARGIN*rho_break(l2/l1)


@njit(cache=True)
def fast_diagnostics(A, B, C, D, E, xmin, xmax, table, use_laplace=True):
    """Return (lnL_marg, branch), 0 for method 2 or 1 for accepted Laplace.

    Computes the normalized integral defined in TASK.md §2. Table bounds
    must match even when the Laplace branch serves the call. Nonfinite
    Laplace results fall back to the general table-backed angular rule.
    """
    if xmin != table[7] or xmax != table[8]:
        raise ValueError('table bounds must match integration bounds')
    if use_laplace and laplace_gate(A, B, C, D, E, xmin, xmax):
        value = lnL_marg_m3_2d(A, B, C, D, E, xmin, xmax, 4)
        if math.isfinite(value):
            return value, BRANCH_LAPLACE
    return lnL_marg_m2_table(A, B, C, D, E, xmin, xmax, table), BRANCH_METHOD2


@njit(cache=True)
def lnL_marg_fast(A, B, C, D, E, xmin, xmax, table, use_laplace=True):
    """Normalized log likelihood via the gated order-4 Laplace/method-2 mix.

    Domain and empirical accuracy limitations are those of fast_diagnostics.
    Setting use_laplace=False gives exactly lnL_marg_m2_table at 64 nodes.
    """
    return fast_diagnostics(A, B, C, D, E, xmin, xmax, table, use_laplace)[0]


@njit(cache=True)
def fast_array_diagnostics(parameters, table, use_laplace=True):
    """Return (values, branch codes) for (n,7) coefficient/bound rows."""
    if parameters.ndim != 2 or parameters.shape[1] != 7:
        raise ValueError('parameters must have shape (n,7)')
    values = np.empty(len(parameters))
    branches = np.empty(len(parameters), dtype=np.int64)
    for i in range(len(parameters)):
        p = parameters[i]
        values[i], branches[i] = fast_diagnostics(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], table, use_laplace)
    return values, branches


@njit(cache=True)
def lnL_marg_fast_array(parameters, table, use_laplace=True):
    """Numba loop over (n,7) rows, for one fixed-prior distance table."""
    if parameters.ndim != 2 or parameters.shape[1] != 7:
        raise ValueError('parameters must have shape (n,7)')
    out = np.empty(len(parameters))
    for i in range(len(parameters)):
        p = parameters[i]
        out[i] = lnL_marg_fast(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], table, use_laplace)
    return out
