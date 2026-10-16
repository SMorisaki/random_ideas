#!/usr/bin/env python
"""Validate the 3,196-point scalar grid, window mass, and warmed array timing.

Run from any directory. Products contain summary numbers only; random timing
inputs (seed 2, 100,000 points) are generated afresh and never saved.
"""

import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from numba import njit
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpmarg.kernel1d import (GL_NODES, WINDOW_EPS, _drop_width,
                             _lnI_with_panels, lnI_direct_array)
from dpmarg.params import _radial_peak
from dpmarg.reference import lnI_ref, lnL_marg_ref

REGIMES = ('x0 < xmin', 'inside', 'x0 > xmax', 'b = 0')


def regime(a, b, lo, hi):
    if b == 0:
        return 'b = 0'
    return 'x0 < xmin' if a/b < lo else 'x0 > xmax' if a/b > hi else 'inside'


def discarded_fraction(a, b, lo, hi, log_integral):
    """Full-domain SciPy tail check, independent of GL quadrature.

    Resolve each cutoff's outward Gaussian/linear layer with extra points.
    The analytic bound in kernel1d additionally certifies the entire domain.
    """
    if b == 0:
        return 0.0  # The linear kernel uses the full finite interval.
    _, r = _radial_peak(a, b, lo, hi)
    slope = a-b*r
    height = 1-math.log(WINDOW_EPS)+4*math.log(hi/lo)
    width = _drop_width(abs(slope), math.sqrt(b), height)
    tail = 0.0
    for left, right, cutoff in ((lo, r-width, r-width),
                                (r+width, hi, r+width)):
        if left >= right:
            continue
        local = min(1/math.sqrt(b) if b else math.inf,
                    1/abs(a-b*cutoff) if a != b*cutoff else math.inf)
        points = [cutoff+k*sign*local for k in (1, 3, 10, 30)
                  for sign in (-1, 1)]
        points.extend(np.geomspace(left, right, 12))
        points = sorted(set(p for p in points if left < p < right))

        def integrand(x):
            delta = x-r
            return (lo/x)**4*math.exp(delta*slope-0.5*b*delta*delta)

        tail += quad(integrand, left, right, points=points,
                     epsabs=0, epsrel=1e-7, limit=200)[0]
    return tail*math.exp(-log_integral-4*math.log(lo))


@njit(cache=True)
def panel_counts(a, b, lo, hi):
    counts = np.empty(a.size, dtype=np.int64)
    for i in range(a.size):
        counts[i] = _lnI_with_panels(a[i], b[i], lo, hi)[1]
    return counts


def benchmark(lo, hi):
    """Seven batch wall-clock samples per regime, median ns/evaluation."""
    rng = np.random.default_rng(2)
    a = 10.0**rng.uniform(-4, 7, 100_000)*rng.choice([-1., 1.], 100_000)
    b = 10.0**rng.uniform(-8, 8, a.size)
    b[rng.random(a.size) < 0.25] = 0
    labels = np.array([regime(aa, bb, lo, hi) for aa, bb in zip(a, b)])
    summary = {}
    for label in REGIMES:
        aa = np.ascontiguousarray(a[labels == label])
        bb = np.ascontiguousarray(b[labels == label])
        lnI_direct_array(aa, bb, lo, hi)
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            lnI_direct_array(aa, bb, lo, hi)
            samples.append((time.perf_counter_ns()-start)/aa.size)
        counts = panel_counts(aa, bb, lo, hi)
        summary[label] = dict(count=int(aa.size), median_ns_per_call=float(np.median(samples)),
                              panel_min=int(counts.min()), panel_max=int(counts.max()),
                              panel_median=float(np.median(counts)),
                              panel_p10_p90=np.percentile(counts, [10, 90]).tolist())
        print(f'  {label:12s}: {np.median(samples):9.1f} ns/call; '
              f'panels median={np.median(counts):g}, p10/p90='
              f'{np.percentile(counts, [10,90])}, n={aa.size}', flush=True)
    return summary


def main():
    aa = np.r_[-np.logspace(-4, 7, 23)[::-1], 0., np.logspace(-4, 7, 23)]
    bb = np.r_[0., np.logspace(-8, 8, 33)]
    regression = json.loads((ROOT/'data/regression_points.json').read_text())
    refactor_change = max(abs(lnL_marg_ref(*(c[k] for k in
                         ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')))
                         -c['lnL_marg_ref']) for c in regression)
    print(f'Reference change on {len(regression)} saved points: {refactor_change:g}', flush=True)
    summary = dict(gl_nodes_per_panel=len(GL_NODES), window_eps=WINDOW_EPS,
                   grid_points_per_bounds=int(aa.size*bb.size), timing_seed=2,
                   timing_total_points_per_bounds=100_000, timing_repeats=7,
                   reference_refactor_max_change=refactor_change, bounds=[])
    maps = []
    for hi in (100., 10.):
        lo = 0.1
        errors = np.empty((bb.size, aa.size))
        worst = {key: dict(abs_error=-1.) for key in REGIMES}
        max_tail = 0.
        for j, b in enumerate(bb):
            for i, a in enumerate(aa):
                ref = lnI_ref(a, b, lo, hi)
                value, _ = _lnI_with_panels(a, b, lo, hi)
                error = value-ref
                errors[j, i] = error
                key = regime(a, b, lo, hi)
                if abs(error) > worst[key]['abs_error']:
                    worst[key] = dict(abs_error=float(abs(error)), error=float(error),
                                      a=float(a), b=float(b))
                max_tail = max(max_tail, discarded_fraction(a, b, lo, hi, ref))
        print(f'Bounds ({lo}, {hi}), worst |delta lnI|:', flush=True)
        for key, item in worst.items():
            print(f'  {key:12s}: {item["abs_error"]:.6g}; '
                  f'a={item["a"]:.6g}, b={item["b"]:.6g}', flush=True)
        print(f'  maximum discarded fraction: {max_tail:.6g}; '
              f'certified bound: {WINDOW_EPS:g}', flush=True)
        timing = benchmark(lo, hi)
        summary['bounds'].append(dict(xmin=lo, xmax=hi, worst_by_regime=worst,
                                      max_discarded_fraction=max_tail, timing=timing))
        maps.append(errors)

    limit = max(abs(m).max() for m in maps)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, hi, errors in zip(axes, (100, 10), maps):
        artist = ax.imshow(errors, origin='lower', aspect='auto', cmap='coolwarm',
                           norm=Normalize(-limit, limit))
        ax.set(title=f'xmin=0.1, xmax={hi}', xlabel='a (signed logarithmic grid)',
               ylabel='b (zero + logarithmic grid)')
        xticks = [0, 6, 14, 23, 32, 40, 46]
        ax.set_xticks(xticks, [f'{aa[i]:.0e}' if aa[i] else '0' for i in xticks], rotation=45)
        yticks = [0, 1, 9, 17, 25, 33]
        ax.set_yticks(yticks, [f'{bb[i]:.0e}' if bb[i] else '0' for i in yticks])
    fig.colorbar(artist, ax=axes, label='lnI_direct − lnI_ref')
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/kernel1d_error_map.png', dpi=180)
    plt.close(fig)
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data/kernel1d_check.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    assert all(item['abs_error'] <= 1e-8 for row in summary['bounds']
               for item in row['worst_by_regime'].values())
    assert all(row['max_discarded_fraction'] <= WINDOW_EPS for row in summary['bounds'])
    assert refactor_change <= 1e-12


if __name__ == '__main__':
    main()
