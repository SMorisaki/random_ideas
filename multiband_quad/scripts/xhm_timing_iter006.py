"""Iteration 006: prepared XHM adapter and honest waveform + HH route sums."""
import os
# Control waveform OpenMP workers before importing LAL; callers may override.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
from pathlib import Path
import sys
from time import perf_counter
from statistics import median
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mbquad.testcases import xhm_case, xhm_systems
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.linear import LinearInterpolationHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.totalweights import CentredBlockHH
from iter006_common import pairwise, quiet_lal, write_json


def measure_components(calls, target=.2, repeat=7):
    """Interleave warm batches in seeded shuffled order to reduce clock drift."""
    numbers, samples = {}, {name:[] for name in calls}
    for label, fn in calls.items():
        fn()
        start = perf_counter()
        for _ in range(3):
            fn()
        estimate = (perf_counter()-start)/3
        numbers[label] = max(1, min(10000, int(target/max(estimate, 1e-7))))
    rng = Random(6006)
    for _ in range(repeat):
        labels = list(calls)
        rng.shuffle(labels)
        for label in labels:
            start = perf_counter()
            for _ in range(numbers[label]):
                calls[label]()
            samples[label].append((perf_counter()-start)/numbers[label])
    return {label:median(v) for label, v in samples.items()}, samples


def profile_python(ev, model, kind):
    """Paired wall-time accounting inside each adapter call (100 warm calls).

    Time LAL wrapper regions and the enclosing adapter on the same call;
    subtract within calls to avoid impossible negative overhead fractions
    from independent medians. Wrapper dispatch is included in LAL regions.
    """
    originals = {name:getattr(ev, name) for name in
                 ('lal_polarizations', 'lal_one_modes', 'lal_tail_polarizations')}
    state = [0.]
    def instrument(fn):
        def wrapped(m):
            start = perf_counter()
            value = fn(m)
            state[0] += perf_counter()-start
            return value
        return wrapped
    for name, fn in originals.items():
        setattr(ev, name, instrument(fn))
    records = []
    try:
        fn = getattr(ev, kind)
        fn(model)
        for _ in range(100):
            state[0] = 0.
            start = perf_counter()
            fn(model)
            total = perf_counter()-start
            records.append((total, state[0], total-state[0]))
    finally:
        for name, fn in originals.items():
            setattr(ev, name, fn)
    return dict(total_seconds=median(r[0] for r in records),
                lal_seconds=median(r[1] for r in records),
                python_seconds=median(r[2] for r in records))


def main():
    results = []
    for name in xhm_systems():
        c = xhm_case(name)
        m, b, p = c.model, c.bands, c.psdgrid
        ev = XHMCoarseEvaluator(m, b)
        cm, co = ev.coarse_modes(m), ev.coarse_total(m)
        kernels = {'IFFT-FFT': IFFTFFTHH(b, p, impl='fast'),
                   'pairwise': pairwise(b, p, m.modes, mchirp_range=xhm_systems()[name]['mchirp_range']),
                   'B1': BasebandIFFTFFTHH(b, p), 'B0': CentredBlockHH(b, p),
                   'linear': LinearInterpolationHH(b, p)}
        calls = {'raw polarizations': lambda: ev.lal_polarizations(m),
                 'raw five OneMode': lambda: ev.lal_one_modes(m),
                 'LAL tail recovery': lambda: ev.lal_tail_polarizations(m),
                 'coarse_total': lambda: ev.coarse_total(m),
                 'coarse_modes': lambda: ev.coarse_modes(m),
                 'phase vector': lambda: ev.phase(m)}
        calls.update({f'{label} HH': (lambda k=k, h=cm if label=='pairwise' else co: k(h))
                      for label, k in kernels.items()})
        calls.update({f'route {label}':
                      (lambda label=label, k=kernels[label]:
                       k(ev.coarse_modes(m) if label=='pairwise' else ev.coarse_total(m)))
                      for label in ('IFFT-FFT', 'pairwise', 'B1', 'B0')})
        times, samples = measure_components(calls)
        for label in calls:
            print(f'{name} {label}: {times[label]*1000:.4f} ms', flush=True)
        # An empty recovery helper has no LAL cost.
        has_tails = bool(ev.lal_tail_polarizations(m))
        profiles = {wave:profile_python(ev, m, wave) for wave in ('coarse_total', 'coarse_modes')}
        baseline = times['route IFFT-FFT']
        baseline3 = times['coarse_total']+3*times['IFFT-FFT HH']
        routes = {}
        for label, wave in [('IFFT-FFT', 'coarse_total'), ('pairwise', 'coarse_modes'),
                            ('B1', 'coarse_total'), ('B0', 'coarse_total')]:
            total = times['route '+label]
            total3 = times[wave]+3*times[label+' HH']
            python = profiles[wave]['python_seconds']
            routes[label] = dict(seconds=total, ratio=total/baseline,
                                 summed_seconds=times[wave]+times[label+' HH'],
                                 python_seconds=python, python_fraction=python/total,
                                 three_detector_seconds=total3, three_detector_ratio=total3/baseline3,
                                 three_detector_python_fraction=python/total3)
        results.append(dict(system=name, K_MB=b.K_MB, K_unique=len(ev.sequence.data),
                            times_seconds=times, routes=routes, additional_tail_calls=4 if has_tails else 0,
                            adapter_profiles=profiles,
                            batch_samples_seconds=samples,
                            fft_lengths=kernels['B1'].fft_lengths()))
    threading = {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_WAIT_POLICY')}
    write_json('xhm_timing_iter006.json', dict(results=results, threading=threading, refusals=[]))
    lines = ['# Iteration 006: lean XHM timings',
             'Median warm wall time, seven batches targeting 0.2 s each, interleaved in '
             'seeded shuffled component order to reduce clock drift; Python on PATH, '
             'LALSimulation 6.2.0, this machine. All times in ms. '
             f"OMP_NUM_THREADS={threading['OMP_NUM_THREADS']}, OMP_WAIT_POLICY={threading['OMP_WAIT_POLICY']}; "
             'default L=5, nominal testcases.py parameters and analytic+lines PSD. '
             'K_MB=20167 / 4865 / 4408 (BNS / GW190814 / BBH). Preparation and waveform samples '
             'are outside HH timings. No waveform results are cached between calls.',
             '| Component | BNS | GW190814-like | BBH 60+20 |', '|---|---:|---:|---:|']
    for label in results[0]['times_seconds']:
        if label.startswith('route '):
            continue
        lines.append('| '+label+' | '+' | '.join(f"{1000*r['times_seconds'][label]:.4f}" for r in results)+' |')
    for detectors in (1, 3):
        lines += ['', f'## {detectors} detector routes',
                  ('Measured complete adapter + HH calls; other likelihood work is excluded.'
                   if detectors==1 else
                   'Waveform evaluated once + HH step three times: sums of component medians. '
                   'Uses identical PSD/kernel costs and assumes negligible additional detector '
                   'projection/assembly cost; this is not a measured network evaluation.'),
                  '| Route | BNS ms / ratio / Python % | GW190814 ms / ratio / Python % | BBH ms / ratio / Python % |',
                  '|---|---:|---:|---:|']
        prefix = '' if detectors==1 else 'three_detector_'
        for label in ('IFFT-FFT', 'pairwise', 'B1', 'B0'):
            lines.append('| '+label+' | '+' | '.join(
                f"{1000*r['routes'][label][prefix+'seconds']:.4f} / "
                f"{r['routes'][label][prefix+'ratio']:.3f} / "
                f"{100*r['routes'][label][prefix+'python_fraction']:.1f}%" for r in results)+' |')
    lines += ['', 'Python fractions use paired wall-clock accounting in 100 warm adapter calls: '
              'time each LAL wrapper region and the enclosing adapter, then subtract within '
              'each call and take the median. BBH tail recovery is included in LAL regions. '
              'The wrapper regions include negligible dispatch/bookkeeping; fractions are estimates. '
              'The phase-vector cost is also measured directly; it is included in adapter time, not added twice. '
              'Component medians and complete one-detector routes are measured separately from '
              'this instrumentation. JSON also records each warm batch and component-sum route estimates; '
              'one-detector ratios use the measured complete routes, three-detector ratios use the sums.',
              'Five OneMode calls suffice for BNS and GW190814. BBH needs four additional polarization '
              'calls to preserve iteration-005 high-frequency tails; the raw five-call row excludes them, '
              'the coarse_modes and route rows include them. No tail correction was dropped for speed.',
              'Ratios below 1 are faster. Accuracy qualifications remain: B0 loses accuracy for HOM; '
              'B1 discrete interpolants have system-dependent errors (see b1_variants_iter006.md). '
              'Pairwise retains its extrinsic-matrix reuse capability. These fixed cases do not establish '
              'population criteria C1/C1\'.',
              'Threading control: with the initial OpenMP default of 4 threads and unset wait policy, '
              'direct IFFT-FFT routes measured 16.32 / 10.12 / 10.11 ms (BNS / GW190814 / BBH), '
              'well above component sums 11.19 / 3.49 / 3.62 ms. Setting one waveform thread '
              'removed most of this interaction in a paired control. The primary table uses the '
              'explicit thread settings above; it should not be compared to iteration-005 raw timings '
              'as an adapter-only speed-up. The initial control batches are retained in '
              'xhm_timing_iter006_threads_full.json (git-ignored). '
              'All changes to threading are confined to the new benchmark scripts.',
              'LAL refused parameter sets: none.']
    (ROOT/'data/xhm_timing_iter006.md').write_text('\n\n'.join(lines[:3])+'\n'+'\n'.join(lines[3:])+'\n')


if __name__ == '__main__':
    with quiet_lal():
        main()
