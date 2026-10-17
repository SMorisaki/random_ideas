"""Opt-in fused quadratures for method_idea §3 and §7 (Eqs. 26,45).

All flattened weights include the window and 4/T. class_map[i,j] is a
nonnegative class for m_i >= m_j, or -1 to conjugate the reverse entry.
Class zero is real and identifies equal-m groups. No fast-math reassociation
is allowed. The support/aliasing assumptions are those of the NumPy methods.
"""

import numpy as np
from numba import njit


def flatten_samples(samples, out, axis=0):
    """Use an existing contiguous backing array, otherwise copy into out.

    The prepared LAL adapter and CoarseModes.from_model already return band
    views of exactly this layout. Verify offsets and strides on every call:
    no samples or pointer plans are cached, so replacement/mutation is safe.
    Arbitrary independent bands and reordered modes take the copy path.
    """
    base = samples[0].base
    if (isinstance(base, np.ndarray) and base.shape == out.shape
            and base.dtype == out.dtype and base.flags.c_contiguous):
        pointer = base.__array_interface__['data'][0]
        offset = 0
        for part in samples:
            if (part.base is not base or part.strides != base.strides
                    or part.__array_interface__['data'][0] != pointer + offset*base.itemsize):
                break
            offset += part.shape[axis]
        else:
            if offset == base.shape[axis]:
                return base
    np.concatenate(samples, axis=axis, out=out)
    return out


@njit(cache=True, fastmath=False, inline='always')
def _cross(w, a, b, sign):
    """Real part of W conj(a)b, optionally conjugating W."""
    return (w.real*(a.real*b.real + a.imag*b.imag)
            - sign*w.imag*(a.real*b.imag - a.imag*b.real))


@njit(cache=True, fastmath=False)
def _four_groups_hh(H, Wc, class_map, representatives, groups):
    """Register-sized grouping for four m values and at most five modes.

    This common XHM case avoids a K_MB-sized grouped scratch array. The
    arbitrary-mode fallback below uses the identical formula. Ordering is
    unrestricted: both class numbers and conjugation come from class_map.
    """
    r0, r1, r2, r3 = representatives[:4]
    pairs = ((r0, r1), (r0, r2), (r0, r3), (r1, r2), (r1, r3), (r2, r3))
    classes = np.empty(6, np.int64)
    signs = np.empty(6)
    for p in range(6):
        i, j = pairs[p]
        c = class_map[i, j]
        signs[p] = 1.0
        if c < 0:
            c = class_map[j, i]
            signs[p] = -1.0
        classes[p] = c
    w01, w02, w03 = Wc[classes[0]], Wc[classes[1]], Wc[classes[2]]
    w12, w13, w23 = Wc[classes[3]], Wc[classes[4]], Wc[classes[5]]
    s01, s02, s03, s12, s13, s23 = signs
    extra, extra_group = 0, -1
    for i in range(H.shape[0]):
        if representatives[groups[i]] != i:
            extra, extra_group = i, groups[i]
    total = 0.0
    for k in range(H.shape[1]):
        a, b, c, d = H[r0, k], H[r1, k], H[r2, k], H[r3, k]
        if extra_group == 0:
            a += H[extra, k]
        elif extra_group == 1:
            b += H[extra, k]
        elif extra_group == 2:
            c += H[extra, k]
        elif extra_group == 3:
            d += H[extra, k]
        diagonal = (a.real*a.real + a.imag*a.imag + b.real*b.real + b.imag*b.imag
                    + c.real*c.real + c.imag*c.imag + d.real*d.real + d.imag*d.imag)
        cross = (_cross(w01[k], a, b, s01) + _cross(w02[k], a, c, s02)
                 + _cross(w03[k], a, d, s03) + _cross(w12[k], b, c, s12)
                 + _cross(w13[k], b, d, s13) + _cross(w23[k], c, d, s23))
        total += Wc[0, k].real*diagonal + 2.0*cross
    return total


@njit(cache=True, fastmath=False)
def pairwise_hh(H, Wc, class_map):
    """Sum W0 |G_m|² + 2 Re W_mm' conj(G_m) G_m', G_m=sum_{A:m} H_A.

    H has shape (n_modes,K_MB); bands are contiguous along the second axis.
    Discover equal-m groups from class zero once, then fuse all point work.
    """
    n = H.shape[0]
    representatives = np.empty(n, np.int64)
    groups = np.empty(n, np.int64)
    ng = 0
    for i in range(n):
        group = -1
        for g in range(ng):
            if class_map[i, representatives[g]] == 0:
                group = g
                break
        if group < 0:
            group = ng
            representatives[ng] = i
            ng += 1
        groups[i] = group
    if ng == 4 and n <= 5:
        return _four_groups_hh(H, Wc, class_map, representatives, groups)
    # Keep the point loop innermost: contiguous streams avoid per-point
    # class lookups and strided loads. Scratch holds current samples only.
    G = np.zeros((ng, H.shape[1]), np.complex128)
    for i in range(n):
        g = groups[i]
        for k in range(H.shape[1]):
            G[g, k] += H[i, k]
    total = 0.0
    for g in range(ng):
        value = 0.0
        for k in range(H.shape[1]):
            z = G[g, k]
            value += Wc[0, k].real * (z.real*z.real + z.imag*z.imag)
        total += value
        for q in range(g):
            i, j = representatives[g], representatives[q]
            c = class_map[i, j]
            sign = 1.0
            if c < 0:
                c = class_map[j, i]
                sign = -1.0
            value = 0.0
            for k in range(H.shape[1]):
                a, b, w = G[g, k], G[q, k], Wc[c, k]
                value += (w.real * (a.real*b.real + a.imag*b.imag)
                          - sign*w.imag * (a.real*b.imag - a.imag*b.real))
            total += 2.0*value
    return total


@njit(cache=True, fastmath=False)
def mode_matrix(H, Wc, class_map):
    """Return Hermitian sum_k conj(H_i) W_ij H_j INCLUDING 4/T (§8.13).

    Do not combine same-m modes: the C5 coefficients act on individual modes.
    PairwiseHH.mode_matrix removes 4/T to preserve its established API.
    """
    n = H.shape[0]
    M = np.empty((n, n), np.complex128)
    for i in range(n):
        for j in range(i, n):
            c = class_map[i, j]
            value = 0.0j
            for k in range(H.shape[1]):
                if c >= 0:
                    w = Wc[c, k]
                else:
                    w = Wc[class_map[j, i], k].conjugate()
                value += H[i, k].conjugate() * w * H[j, k]
            if i == j:
                value = complex(value.real, 0.0)
            M[i, j] = value
            M[j, i] = value.conjugate()
    return M


@njit(cache=True, fastmath=False)
def b0_hh(h, w0):
    """Return sum_k w0_k |h_k|² on the flattened Eq. (24) grid (§8.8)."""
    value = 0.0
    for k in range(h.size):
        z = h[k]
        value += w0[k] * (z.real*z.real + z.imag*z.imag)
    return value


@njit(cache=True, fastmath=False)
def b1_prepare(h, sqrt_w, x, W_even):
    """Window x[:K]=sqrt(w)h and return the integer-bin Eq. (45) sum.

    The preallocated x[K:] tail remains zero; scipy's IFFT does not overwrite x.
    B1 weights here exclude 4/T, applied by the caller once per band.
    """
    value = 0.0
    for k in range(h.size):
        z = h[k]
        x[k] = sqrt_w[k] * z
        value += W_even[k] * (z.real*z.real + z.imag*z.imag)
    return value


@njit(cache=True, fastmath=False)
def phase_multiply(y, phase):
    """Apply the §7 half-sample phase in place before scipy's forward FFT."""
    for k in range(y.size):
        y[k] *= phase[k]
