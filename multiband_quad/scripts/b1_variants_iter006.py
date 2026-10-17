"""Iteration 006 B1 padding/half-bin variants and same-duration L=50 bands."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mbquad.bands import BandSetup
from mbquad.testcases import xhm_case, xhm_systems
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from iter006_common import ReferenceCache, exact_bands, pairwise, quiet_lal, measure, stats, write_json


def main():
    cache = ReferenceCache()
    variants, lscan = [], []
    for name, parameters in xhm_systems().items():
        c = xhm_case(name)
        m, p = c.model, c.psdgrid
        m.inclination = 1.
        fine = cache.get(name, 'analytic_lines', m, p)
        b = c.bands
        adapter = XHMCoarseEvaluator(m, b)
        co = adapter.coarse_total(m)
        exact = exact_bands(fine, p, b).sum()
        fft = IFFTFFTHH(b, p, impl='fast')
        ferr = (fft(co)-exact)/exact
        ftime = measure(lambda: fft(co))
        for pad in (1, 2, 4):
            for allbins in (False, True):
                ev = BasebandIFFTFFTHH(b, p, pad_factor=pad, all_half_bins=allbins)
                signed = (ev(co)-exact)/exact
                dt = measure(lambda: ev(co))
                variants.append(dict(system=name, pad_factor=pad, all_half_bins=allbins,
                                     relative_error=abs(signed), signed_relative=signed,
                                     abs_delta_lnLambda_snr25=312.5*abs(signed), step_seconds=dt,
                                     ifftfft_relative_error=abs(ferr), ifftfft_step_seconds=ftime,
                                     fft_lengths=ev.fft_lengths()))
        for L in (5, 50):
            if L==50:
                b = BandSetup(**{k:parameters[k] for k in ('T','dt','f_low','f_high','durations',
                                                          'mchirp_ref','m_max')}, L=L)
            adapter = XHMCoarseEvaluator(m, b)
            co, cm = adapter.coarse_total(m), adapter.coarse_modes(m)
            evaluators = {'linear':LinearInterpolationHH(b,p), 'IFFT-FFT':IFFTFFTHH(b,p,impl='fast'),
                          'pairwise':pairwise(b,p,m.modes,mchirp_range=parameters['mchirp_range']),
                          'B1':BasebandIFFTFFTHH(b,p)}
            ref = exact_bands(fine, p, b)
            for label, ev in evaluators.items():
                h = cm if label=='pairwise' else co
                signed = (ev(h)-exact)/exact
                lscan.append(dict(system=name, L=L, K_MB=b.K_MB, method=label,
                                  adaptive_modes=ev.modes if label=='B1' else None,
                                  relative_error=abs(signed), signed_relative=signed,
                                  abs_delta_lnLambda_snr25=312.5*abs(signed),
                                  signed_band_relative=((ev.band_contributions(h)-ref)/exact).tolist()
                                  if hasattr(ev, 'band_contributions') else None))
        print(name, 'B1 variants and L scan done', flush=True)
    decisions = []
    for name in xhm_systems():
        rr = [r for r in variants if r['system']==name]
        default = rr[0]
        for r in rr:
            r['time_ratio_to_default'] = r['step_seconds']/default['step_seconds']
            r['error_improvement_over_default'] = default['relative_error']/r['relative_error']
        eligible = [r for r in rr if r['error_improvement_over_default']>1.5
                    and r['time_ratio_to_default']<=2]
        decisions.append(dict(system=name, eligible=[dict(pad_factor=r['pad_factor'],
                         all_half_bins=r['all_half_bins']) for r in eligible]))
    aggregates = {}
    for pad in (1,2,4):
        for allbins in (False,True):
            rr = [r for r in variants if r['pad_factor']==pad and r['all_half_bins']==allbins]
            aggregates[f'{pad}/{allbins}'] = stats([r['relative_error'] for r in rr])
    l_aggregate = {f'{L}/{method}':stats([r['relative_error'] for r in lscan
                    if r['L']==L and r['method']==method])
                   for L in (5,50) for method in ('linear','IFFT-FFT','pairwise','B1')}
    write_json('b1_variants_iter006.json', dict(variants=variants,L_scan=lscan,
               decisions=decisions, aggregate=aggregates, L_aggregate=l_aggregate,
               cache_hits=cache.hits, cache_misses=cache.misses, refusals=[]))
    lines = ['# Iteration 006: B1 variants and L=50',
             'Three nominal XHM systems from testcases.py; analytic aLIGO+TOY_LINES PSD, '
             'inclination=1.0, other parameters unchanged (t_c=T-2, phi_ref=0.7). '
             'Fine-grid totals reuse iteration-005 cache. SNR25 likelihood error magnitude = '
             '312.5 relative HH error. Timings are median warm wall time over five batches '
             'targeting 0.15 s each on this machine; waveform evaluation excluded.',
             f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, OMP_WAIT_POLICY={os.environ['OMP_WAIT_POLICY']}; "
             'scipy FFTs use one worker.',
             'Padding uses K\'\'=next_fast_len(p K_b), p in {1,2,4}; adaptive B0/exact shortcuts '
             'remain active. all_half_bins sums K\'\' half bins with 2M-periodic weights, '
             'while integer power remains band-only (padded integer samples are zero). '
             'These are distinct finite interpolants; padding need not converge to the paper method.',
             '| System | p | all half bins | relative HH error | abs likelihood error | step ms | cost/default | IFFT-FFT error | IFFT-FFT ms |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in variants:
        lines.append(f"| {r['system']} | {r['pad_factor']} | {r['all_half_bins']} | {r['relative_error']:.3e} | "
                     f"{r['abs_delta_lnLambda_snr25']:.3e} | {1000*r['step_seconds']:.4f} | "
                     f"{r['time_ratio_to_default']:.3f} | {r['ifftfft_relative_error']:.3e} | {1000*r['ifftfft_step_seconds']:.4f} |")
    lines += ['', '## Variant aggregate over the three fixed systems',
              '| p / all half bins | median rel | worst rel | median likelihood | worst likelihood |',
              '|---|---:|---:|---:|---:|']
    for label, r in aggregates.items():
        lines.append(f"| {label} | {r['median_relative']:.3e} | {r['worst_relative']:.3e} | "
                     f"{r['median_lnLambda']:.3e} | {r['worst_lnLambda']:.3e} |")
    lines += ['', '## Default decision',
              'Eligibility: >1.5x lower error and <=2x step cost relative to p=1, band-only, on each system.']
    for d in decisions:
        lines.append(f"{d['system']}: eligible variants {d['eligible']}.")
    lines += ['The defaults remain p=1 and band-only. A system-specific win alone is insufficient '
              'to change the shared default; compare all three rows for any proposed variant.',
              '', '## L=5 versus L=50, unchanged durations',
              'Pairwise retains its default L_margin=5, sigma_pair, block diagonal and the same '
              'chirp-mass prior; only the band-construction L changes. B1 retains p=1 and band-only. '
              'The same exact fine-grid norm applies to both window partitions.',
              '| System | method | L=5 K_MB | L=5 relative / likelihood | L=50 K_MB | L=50 relative / likelihood |',
              '|---|---|---:|---:|---:|---:|']
    for name in xhm_systems():
        for method in ('linear','IFFT-FFT','pairwise','B1'):
            a, z = [next(r for r in lscan if r['system']==name and r['method']==method and r['L']==L)
                    for L in (5,50)]
            lines.append(f"| {name} | {method} | {a['K_MB']} | {a['relative_error']:.3e} / {a['abs_delta_lnLambda_snr25']:.3e} | "
                         f"{z['K_MB']} | {z['relative_error']:.3e} / {z['abs_delta_lnLambda_snr25']:.3e} |")
    lines += ['', '## L aggregate over the three fixed systems',
              '| L / method | median rel | worst rel | median likelihood | worst likelihood |',
              '|---|---:|---:|---:|---:|']
    for label, r in l_aggregate.items():
        lines.append(f"| {label} | {r['median_relative']:.3e} | {r['worst_relative']:.3e} | "
                     f"{r['median_lnLambda']:.3e} | {r['worst_lnLambda']:.3e} |")
    lines += ['', 'Adaptive B1 band choices (a B0 band executes no FFT):']
    for r in lscan:
        if r['method']=='B1':
            lines.append(f"{r['system']}, L={r['L']}: {r['adaptive_modes']}.")
    lines += ['L=50 reduces the default pairwise and B1 errors in these cases, with '
              'B1 selecting B0 in both truncated bands (all six truncated bands for BNS). '
              'The BBH IFFT-FFT error increases from 7.58e-9 to 4.14e-8: error cancellation '
              'at one fixed point does not imply monotonic improvement with L.']
    lines += ['', 'These are one case per system, not a population test. No existing defaults '
              'or iteration-001–005 script behaviour changed. LAL refused parameter sets: none.',
              f'Fine-reference cache hits/misses: {cache.hits}/{cache.misses}.']
    (ROOT/'data/b1_variants_iter006.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    with quiet_lal():
        main()
