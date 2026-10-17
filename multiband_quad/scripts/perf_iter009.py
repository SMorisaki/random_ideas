"""Iteration 009: single-thread warm steps, routes, setup memory and C2.

Run from the repository: python scripts/perf_iter009.py. No network or new
packages; all products are local. JIT warm-up is excluded from steady-state
timings and measured separately from evaluator construction.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import gc
from pathlib import Path
import sys
from statistics import median
from time import perf_counter
import tracemalloc
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import numba
import scipy
from mbquad import BandSetup
from mbquad.testcases import xhm_case, xhm_systems
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.pairwise import PairwiseHH
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.totalweights import CentredBlockHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from mbquad.kernels import pairwise_hh, mode_matrix, b0_hh
from mbquad.reference import hh_exact
from iter006_common import quiet_lal, write_json, ReferenceCache, stats
from xhm_timing_iter006 import measure_components


def bands_for(name, L=5, rule='paper'):
    """Return the documented systems, changing only L and the band rule."""
    p = xhm_systems()[name]
    kw = {k:p[k] for k in ('T', 'dt', 'f_low', 'f_high', 'durations',
                          'mchirp_ref', 'm_max')}
    return BandSetup(**kw, L=L, rule=rule, L_hh=5)


def owned_nbytes(obj):
    """Count unique retained NumPy allocations, excluding caller-owned grids."""
    seen, arrays = set(), {}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            arrays[id(base)] = base.nbytes
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                visit(v)
        elif hasattr(value, '__dict__'):
            for key, v in vars(value).items():
                if key not in ('bands', 'psdgrid', '_grids'):
                    visit(v)
    visit(obj)
    return sum(arrays.values())


def precompute(factory):
    """Median of three fresh constructors; separate traced construction peak.

    Excludes caller-owned bands/PSD, waveform samples and first-call JIT.
    Includes all cached BlockWeights, flattened tables, tensors and buffers.
    """
    seconds = []
    for _ in range(3):
        gc.collect()
        start = perf_counter()
        obj = factory()
        seconds.append(perf_counter()-start)
        del obj
    gc.collect()
    tracemalloc.start()
    obj = factory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(seconds=median(seconds), blockweights_bytes=(obj.blockweights.nbytes()
                if hasattr(obj, 'blockweights') else 0), retained_array_bytes=owned_nbytes(obj),
                tracemalloc_peak_bytes=peak)


def benchmark(name, L, target, repeat, cache):
    c = xhm_case(name)
    m, p = c.model, c.psdgrid
    b, bh, b50 = bands_for(name, L), bands_for(name, L, 'hh_half'), bands_for(name, 50)
    ev = XHMCoarseEvaluator(m, b)
    cm, co = ev.coarse_modes(m), ev.coarse_total(m)
    evh, ev50 = XHMCoarseEvaluator(m, bh), XHMCoarseEvaluator(m, b50)
    coh, co50 = evh.coarse_total(m), ev50.coarse_total(m)
    prior = xhm_systems()[name]['mchirp_range']
    factories = {
        'linear': lambda: LinearInterpolationHH(b, p),
        'IFFT-FFT paper': lambda: IFFTFFTHH(b, p, impl='paper'),
        'IFFT-FFT fast': lambda: IFFTFFTHH(b, p, impl='fast'),
        'pairwise pairs': lambda: PairwiseHH(b, p, m.modes, mchirp_range=prior),
        'pairwise numba': lambda: PairwiseHH(b, p, m.modes, mchirp_range=prior, impl='numba'),
        'B0 numpy': lambda: CentredBlockHH(b, p),
        'B0 numba': lambda: CentredBlockHH(b, p, impl='numba'),
        'B1 numpy': lambda: BasebandIFFTFFTHH(b, p),
        'B1 numba': lambda: BasebandIFFTFFTHH(b, p, impl='numba'),
        'B1 L50 numpy': lambda: BasebandIFFTFFTHH(b50, p),
        'B1 L50 numba': lambda: BasebandIFFTFFTHH(b50, p, impl='numba'),
        'B0 hh_half numpy': lambda: CentredBlockHH(bh, p),
        'B0 hh_half numba': lambda: CentredBlockHH(bh, p, impl='numba')}
    evaluators = {label:f() for label,f in factories.items()}
    inputs = {label:(cm if label.startswith('pairwise') else
                     co50 if 'L50' in label else coh if 'hh_half' in label else co)
              for label in evaluators}
    calls = {label:(lambda k=k, h=inputs[label]: k(h)) for label,k in evaluators.items()}
    pair = evaluators['pairwise numba']
    H = pair.flatten(cm).copy()
    b0 = evaluators['B0 numba']
    flat_h = b0.flatten(co).copy()
    calls.update({
        'pairwise einsum': lambda: evaluators['pairwise pairs'](cm, impl='einsum'),
        'pairwise bare kernel': lambda: pairwise_hh(H, pair.Wc, pair.class_map),
        'mode matrix numpy': lambda: pair.mode_matrix(cm, impl='einsum'),
        'mode matrix numba': lambda: pair.mode_matrix(cm),
        'mode matrix bare kernel': lambda: mode_matrix(H, pair.Wc, pair.class_map),
        'B0 bare kernel': lambda: b0_hh(flat_h, b0.w0),
        'raw polarizations': lambda: ev.lal_polarizations(m),
        'raw five OneMode': lambda: ev.lal_one_modes(m),
        'LAL tail recovery': lambda: ev.lal_tail_polarizations(m),
        'coarse_total': lambda: ev.coarse_total(m),
        'coarse_modes': lambda: ev.coarse_modes(m)})
    route_keys = {'IFFT-FFT paper bands':'IFFT-FFT fast',
                  'pairwise paper bands':'pairwise numba',
                  'B1 paper bands':'B1 numba', 'B0 hh_half bands':'B0 hh_half numba'}
    for route, key in route_keys.items():
        adapter = evh if 'hh_half' in key else ev
        wave = adapter.coarse_modes if key.startswith('pairwise') else adapter.coarse_total
        calls['route '+route] = lambda k=evaluators[key], wave=wave: k(wave(m))
    # Includes cache loading/compilation, recorded separately and never in C2.
    warm = {}
    for key in ('pairwise numba', 'mode matrix numba', 'B0 numba', 'B1 numba'):
        start = perf_counter()
        calls[key]()
        warm[key] = perf_counter()-start
    times, batches = measure_components(calls, target=target, repeat=repeat)
    routes = {}
    for route, key in route_keys.items():
        wave = times['coarse_modes'] if key.startswith('pairwise') else times['coarse_total']
        if 'hh_half' in key:
            wave *= bh.K_MB/b.K_MB
        routes[route] = dict(measured_1_seconds=times['route '+route],
                            wave_seconds=wave, step_seconds=times[key],
                            sum_1_seconds=wave+times[key], sum_3_seconds=wave+3*times[key])
    baseline = routes['IFFT-FFT paper bands']
    for r in routes.values():
        for n in (1, 3):
            r[f'ratio_{n}'] = r[f'sum_{n}_seconds']/baseline[f'sum_{n}_seconds']
    A = times['pairwise numba']
    verdict = dict(A_vs_fast=A/times['IFFT-FFT fast'], A_vs_modes=A/times['coarse_modes'],
                   A_vs_raw_modes=A/times['raw five OneMode'],
                   B1_vs_fast=times['B1 numba']/times['IFFT-FFT fast'],
                   A_pass=bool(A <= times['IFFT-FFT fast']/5 and
                               A <= .1*times['raw five OneMode']),
                   B1_pass=bool(times['B1 numba'] <= times['IFFT-FFT fast']/3))
    exact = hh_exact(cache.get(name, 'analytic_lines', m, p), p)
    errors = {label:(k(inputs[label])-exact)/exact for label,k in evaluators.items()}
    setup = {key:precompute(f) for key,f in factories.items()} if name == 'BNS' else {}
    if name == 'BNS':
        setup['waveform adapter paper'] = precompute(lambda: XHMCoarseEvaluator(m, b))
        setup['waveform adapter hh_half'] = precompute(lambda: XHMCoarseEvaluator(m, bh))
    row = dict(system=name, L=L, K_MB=b.K_MB, K_MB_hh_half=bh.K_MB, K_MB_L50=b50.K_MB,
               times_seconds=times, batch_seconds=batches, routes=routes, C2=verdict,
               first_dispatch_seconds=warm, precomputation=setup,
               relative_errors=errors, error_summary=stats(list(errors.values())),
               B1_fft_lengths=evaluators['B1 numba'].fft_lengths(),
               B1_L50_fft_lengths=evaluators['B1 L50 numba'].fft_lengths())
    print(f'{name} L={L}: A={A*1e3:.4f} ms, A/fast={verdict["A_vs_fast"]:.4f}, '
          f'B1/fast={verdict["B1_vs_fast"]:.4f}', flush=True)
    return row


def report(results, metadata):
    cols = [f'{r["system"]} L={r["L"]}' for r in results]
    lines = ['# Iteration 009: final performance tables', '',
             'All step/route times are warm median wall-clock milliseconds on this machine. '
             f'{metadata["repeat"]} seeded, interleaved batches targeting {metadata["target"]} s each; '
             'OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, '
             'OMP_WAIT_POLICY=PASSIVE; SciPy FFT workers=1. '
             'No waveform samples are cached between evaluations. Numba cache=True, fastmath=False. '
             'Construction and JIT/cache loading are excluded from steps. Wrappers include '
             'validation and flattening: shared contiguous band views are used directly; '
             'independent/reordered bands copy into reusable input buffers.', '',
             '| Component (ms) | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
    for key in results[0]['times_seconds']:
        if not key.startswith('route '):
            lines.append('| '+key+' | '+' | '.join(f'{1e3*r["times_seconds"][key]:.4f}' for r in results)+' |')
    lines += ['', '| Grid | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
    for key in ('K_MB', 'K_MB_hh_half', 'K_MB_L50'):
        lines.append('| '+key+' | '+' | '.join(str(r[key]) for r in results)+' |')
    for n in (1, 3):
        lines += ['', f'## {n} detector routes', '',
                  'Component sums: one prepared waveform evaluation + '+str(n)+' HH steps. '
                  'Identical PSD costs and negligible extra detector projection are assumed. '
                  'B0 hh_half waveform cost is scaled by K_MB(hh_half)/K_MB(paper), as requested. '
                  'Other likelihood work is excluded; ratios use fast IFFT-FFT on paper bands.', '',
                  '| Route: ms / ratio | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
        for route in results[0]['routes']:
            lines.append('| '+route+' | '+' | '.join(
                f'{1e3*r["routes"][route][f"sum_{n}_seconds"]:.4f} / '
                f'{r["routes"][route][f"ratio_{n}"]:.3f}' for r in results)+' |')
    lines += ['', 'Measured complete one-detector routes are also recorded (including the actual hh_half '
              'adapter rather than its scaled estimate):', '',
              '| Route (measured ms) | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
    for route in results[0]['routes']:
        lines.append('| '+route+' | '+' | '.join(f'{1e3*r["routes"][route]["measured_1_seconds"]:.4f}' for r in results)+' |')
    lines += ['', 'BBH coarse_modes includes all four extra polarization calls recovering OneMode tails '
              'above Mf=0.3; raw five OneMode excludes recovery. The matrix timings use the same '
              'sample magnitudes; raw intrinsic modes have identical computational cost. '
              'The public matrix excludes 4/T; the bare kernel includes it.', '', '## C2 verdict', '',
              '| System | A / fast (<=0.2) | A / raw five-mode time (<=0.1) | A met | B1 / fast (<=1/3) | B1 met |',
              '|---|---:|---:|---|---:|---|']
    for r in results:
        v = r['C2']
        lines.append(f'| {r["system"]} L={r["L"]} | {v["A_vs_fast"]:.4f} | '
                     f'{v["A_vs_raw_modes"]:.4f} | {v["A_pass"]} | {v["B1_vs_fast"]:.4f} | {v["B1_pass"]} |')
    lines += ['', '## BNS precomputation', '',
              'Median of three fresh constructors without tracing; tracemalloc peak measured in a '
              'separate fresh constructor. Caller-owned bands and PSD are excluded. '
              'All internal BlockWeights caches, tensors, flattened tables and buffers are retained. '
              'MB means 10^6 bytes. Zero BlockWeights means that evaluator uses its own weights.', '',
              '| L | Evaluator | seconds | BlockWeights MB | retained arrays MB | traced peak MB | <=60s and <=200MB |',
              '|---:|---|---:|---:|---:|---:|---|']
    for r in results:
        for key,s in r['precomputation'].items():
            met = s['seconds'] <= 60 and s['tracemalloc_peak_bytes'] <= 200e6
            lines.append(f'| {r["L"]} | {key} | {s["seconds"]:.4f} | '
                         f'{s["blockweights_bytes"]/1e6:.3f} | {s["retained_array_bytes"]/1e6:.3f} | '
                         f'{s["tracemalloc_peak_bytes"]/1e6:.3f} | {met} |')
    lines += ['', 'First dispatch (seconds; process-first use may load/compile numba and includes imports):', '',
              '| Entry | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
    for key in results[0]['first_dispatch_seconds']:
        lines.append('| '+key+' | '+' | '.join(f'{r["first_dispatch_seconds"][key]:.4f}' for r in results)+' |')
    lines += ['', '## Accuracy checks', '',
              'Fixed nominal testcases.py systems, analytic aLIGO PSD + TOY_LINES; brute-force positive '
              'fine-grid Eq. (4) from the existing local XHM reference cache. Absolute likelihood '
              'error is 312.5 times relative HH error '
              '(SNR 25). Population qualifications from iterations 007–008 still apply.', '',
              '| Method | median relative HH | worst relative HH | median abs Delta ln Lambda | worst abs Delta ln Lambda |',
              '|---|---:|---:|---:|---:|']
    for label in results[0]['relative_errors']:
        s = stats([r['relative_errors'][label] for r in results])
        lines.append(f'| {label} | {s["median_relative"]:.3e} | {s["worst_relative"]:.3e} | '
                     f'{s["median_lnLambda"]:.3e} | {s["worst_lnLambda"]:.3e} |')
    lines += ['', 'Statistics above span the four documented configurations (BNS appears at L=5 and 50); '
              'per-configuration signed errors are in JSON. Numba and NumPy share the same approximation; '
              'tests require relative agreement 1e-12. B1 at L=50 selects B0 in every truncated band. '
              'B0 on paper L=5 bands remains diagnostic; the recommended total-waveform route uses hh_half.', '',
              'Defaults remain unchanged. Recommend opt-in numba for repeated evaluations, especially '
              'pairwise; promotion to the default can be considered separately because import/compilation '
              'latency and a required numba dependency are visible for short jobs. '
              'B0 with hh_half remains the preferred total-waveform configuration.']
    (ROOT/'data/perf_iter009.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=float, default=.12)
    ap.add_argument('--repeat', type=int, default=7)
    args = ap.parse_args()
    metadata = dict(target=args.target, repeat=args.repeat, numpy=np.__version__,
                    scipy=scipy.__version__, numba=numba.__version__,
                    threads={k:os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'MKL_NUM_THREADS', 'OMP_WAIT_POLICY')})
    cache = ReferenceCache()
    with warnings.catch_warnings(), quiet_lal():
        warnings.simplefilter('ignore', RuntimeWarning)
        results = [benchmark(name, L, args.target, args.repeat, cache)
                   for name,L in [('BNS',5), ('GW190814',5), ('BBH',5), ('BNS',50)]]
    write_json('perf_iter009.json', dict(metadata=metadata, results=results))
    report(results, metadata)


if __name__ == '__main__':
    main()
