"""Model-phase lag support audits for Eq. (26) method-A blocks.

Measure tau=t_c+(2pi)^-1 d arg(h)/df and ordered lag differences
(METHOD_NOTE.md §§2,4). These are stationary tracks, not infinite window
tails. Phase at an amplitude zero is undefined and explicitly excluded.
"""

import numpy as np
from scipy.signal import savgol_filter


def phase_time_tracks(f, mode_samples, t_c=0., amplitude_floor=1e-6, smooth=11):
    """Return tau_lm=t_c+(2pi)^-1 d arg(h_lm)/df on resolved fine Hz grids.

    For this package's exp(-i Psi) convention, the requested signed
    -(2pi)^-1 d arg(h)/df is *time* t_lm=t_c-tau_lm. Thus positive pair
    lags are tau_slow-tau_fast=t_fast-t_slow. Caller must sample sufficiently
    finely to unwrap (<pi phase change/sample); this function rejects
    nonincreasing grids but cannot infer missed full phase turns.
    We remove known exp(-2pi i f t_c) before unwrapping, differentiate each
    contiguous nonzero interval, then smooth d arg/df with a cubic local
    polynomial in sample index. Return NaN where amplitude is below
    amplitude_floor times the mode's maximum, or at unresolved endpoints.
    With smooth=1 there is no smoothing. Arrays must match the 1D grid.
    """
    f = np.asarray(f, float)
    if f.ndim != 1 or len(f) < 5 or np.any(~np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError('Require a finite increasing 1D fine frequency grid')
    if not 0 < amplitude_floor < 1 or smooth < 1 or smooth % 2 != 1:
        raise ValueError('Require 0 < amplitude_floor < 1 and positive odd smooth')
    result = {}
    for lm, samples in mode_samples.items():
        h = np.asarray(samples)
        if h.shape != f.shape or np.any(~np.isfinite(h)):
            raise ValueError('Each mode must be finite and match f')
        amp = abs(h)
        valid = (amp > amplitude_floor * amp.max()) & (amp > 0)
        track = np.full(len(f), np.nan)
        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for start, stop in boundaries.reshape(-1, 2):
            if stop - start < max(5, smooth):
                continue
            phase = np.unwrap(np.angle(h[start:stop] * np.exp(2j*np.pi*f[start:stop]*t_c)))
            derivative = np.gradient(phase, f[start:stop], edge_order=2) / (2*np.pi)
            if smooth >= 5:
                derivative = savgol_filter(derivative, smooth, 3)
            track[start:stop] = derivative
        result[lm] = track
    return result


def measured_slack(f, tracks, placements):
    """Measure per-band/class min/max lag and both block clearances in seconds.

    placements is PairwiseHH.summary(); f samples must span each band's
    support, provided as extra f_lo/f_hi keys. All pairs in a class are
    audited, including distinct-l same-m pairs in class 0. NaN tracks are
    omitted and sample counts are returned. Full-grid clearances are
    informational: a full circular block has no support restriction.
    """
    f = np.asarray(f)
    modes = list(tracks)
    output = []
    for row in placements:
        cl = row['lag_class']
        pairs = ([(a, c) for a in modes for c in modes if a[1] == c[1]] if cl == 0
                 else [(a, c) for a in modes for c in modes if (a[1], c[1]) == tuple(cl)])
        lo, hi, n = np.inf, -np.inf, 0
        inside = (f >= row['f_lo']) & (f <= row['f_hi'])
        for a, c in pairs:
            lag = np.asarray(tracks[a]) - np.asarray(tracks[c])
            values = lag[inside & np.isfinite(lag)]
            if len(values):
                lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
                n += len(values)
        if n:
            sl = lo - row['block_start']
            sh = row['block_end'] - hi
            output.append(dict(b=row['b'], lag_class=cl, measured_lo=lo, measured_hi=hi,
                               slack_lo=sl, slack_hi=sh, slack=min(sl, sh), samples=n,
                               full_grid=row['full_grid'], negative=min(sl, sh) < 0))
    return output
