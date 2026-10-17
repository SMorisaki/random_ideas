"""Paired GW190814 call accounting: LAL / Python+NumPy bookkeeping / HH kernel.

Instrument complete warm adapter+HH calls, including flattening and validation.
The kernel bucket includes SciPy FFTs for B1. Instrumentation overhead is left
in bookkeeping; wrapper-region LAL times include their small Python dispatch.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pathlib import Path
import sys
from statistics import median
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mbquad import kernels
import mbquad.baseband as baseband
from mbquad.testcases import xhm_case
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.pairwise import PairwiseHH
from mbquad.totalweights import CentredBlockHH
from perf_iter009 import bands_for
from iter006_common import quiet_lal, write_json


def profile(ev, model, hh, modes=False, number=200):
    """Median of paired wall-time partitions in 200 complete warm calls."""
    wave = ev.coarse_modes if modes else ev.coarse_total
    hh(wave(model))  # compile/cache-load outside instrumentation
    state = dict(lal=0., kernel=0.)
    originals = []
    def wrap(obj, name, bucket):
        fn = getattr(obj, name)
        originals.append((obj, name, fn))
        def timed(*args, **kw):
            start = perf_counter()
            result = fn(*args, **kw)
            state[bucket] += perf_counter()-start
            return result
        setattr(obj, name, timed)
    for name in ('lal_one_modes', 'lal_polarizations', 'lal_tail_polarizations'):
        wrap(ev, name, 'lal')
    for name in ('pairwise_hh', 'b0_hh', 'b1_prepare', 'phase_multiply'):
        wrap(kernels, name, 'kernel')
    for name in ('ifft', 'fft'):
        wrap(baseband, name, 'kernel')
    records = []
    try:
        for _ in range(number):
            state['lal'] = state['kernel'] = 0.
            start = perf_counter()
            hh(wave(model))
            total = perf_counter()-start
            records.append(dict(total=total, lal=state['lal'], kernel=state['kernel'],
                                bookkeeping=total-state['lal']-state['kernel']))
    finally:
        for obj, name, fn in reversed(originals):
            setattr(obj, name, fn)
    return {key:median(r[key] for r in records) for key in records[0]}


def main():
    c = xhm_case('GW190814')
    b, m, p = c.bands, c.model, c.psdgrid
    ev = XHMCoarseEvaluator(m, b)
    bh = bands_for('GW190814', rule='hh_half')
    evh = XHMCoarseEvaluator(m, bh)
    with warnings.catch_warnings(), quiet_lal():
        warnings.simplefilter('ignore', RuntimeWarning)
        calls = {
            'pairwise paper bands': (ev, PairwiseHH(b, p, m.modes, impl='numba'), True),
            'B0 paper bands': (ev, CentredBlockHH(b, p, impl='numba'), False),
            'B1 paper bands': (ev, baseband.BasebandIFFTFFTHH(b, p, impl='numba'), False),
            'B0 hh_half bands': (evh, CentredBlockHH(bh, p, impl='numba'), False)}
        rows = {label:profile(adapter, m, hh, modes) for label,(adapter,hh,modes) in calls.items()}
    write_json('profile_overhead_iter009.json', dict(system='GW190814', number=200,
               K_MB=b.K_MB, K_MB_hh_half=bh.K_MB, seconds=rows))
    lines = ['# Iteration 009: GW190814-like overhead profile', '',
             'OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, '
             'OMP_WAIT_POLICY=PASSIVE; 200 warm complete adapter + numba-HH calls. '
             'Each call is partitioned before taking medians; individual medians may not sum exactly. '
             'LAL means the prepared LAL wrapper regions. Kernel means fused numba calls plus SciPy '
             'FFTs in B1. Bookkeeping is the remainder: phase, projection, inverse-index assembly, '
             'container validation, copying into flattened buffers, Python dispatch and instrumentation. '
             'This is wall-time accounting, not a claim that all bookkeeping executes in Python bytecode.', '',
             '| Route | total ms | LAL ms | NumPy/Python bookkeeping ms | kernel ms | bookkeeping % |',
             '|---|---:|---:|---:|---:|---:|']
    for label, r in rows.items():
        lines.append(f'| {label} | {r["total"]*1e3:.4f} | {r["lal"]*1e3:.4f} | '
                     f'{r["bookkeeping"]*1e3:.4f} | {r["kernel"]*1e3:.4f} | '
                     f'{100*r["bookkeeping"]/r["total"]:.1f}% |')
    lines += ['', 'No waveform results are cached. Timed wrapper regions include their dispatch; '
              'the small profiling overhead remains in bookkeeping. Use perf_iter009.md for '
              'uninstrumented steady-state medians and the C2 verdict.']
    content = '\n'.join(lines)+'\n'
    (ROOT/'data/profile_overhead_iter009.md').write_text(content)
    print(content)


if __name__ == '__main__':
    main()
