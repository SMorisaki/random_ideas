"""Warm waveform/HH cost breakdown and actual likelihood-side route comparison.

Each median_time batch targets one second; setup and coarse samples are
outside HH timings. Raw LAL timings share sorted unique frequencies, while
adapter timings include sorting, assembly and the t_c phase on every call.
"""
from pathlib import Path
import sys
import json
from time import perf_counter
import warnings

import lal
import lalsimulation as ls
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mbquad.testcases import xhm_case,xhm_systems
from mbquad.coarse import CoarseModes,CoarseWaveform
from mbquad.pairwise import PairwiseHH
from mbquad.linear import LinearInterpolationHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.totalweights import CentredBlockHH
from mbquad.timing import median_time


def measure(fn):
    fn()
    start=perf_counter()
    for _ in range(3): fn()
    dt=(perf_counter()-start)/3
    number=max(1,int(1/max(dt,1e-7)))
    return median_time(fn,number=number,repeat=3),number


def assessment(results):
    """Interpret measured route costs, with the accuracy qualification."""
    return ('Go/no-go: method A is a no-go as a default speed-up with this adapter: '
            + '; '.join(f"{r['system']} costs {r['routes']['A']['ratio_to_ifftfft']:.3f}x "
                        'the IFFT-FFT route' for r in results)
            + '. The raw-LAL BNS route has headroom for shared sequence preparation, '
            'but the raw-LAL GW190814 route already loses on per-mode waveform cost. '
            'B1 reduces the HH step by '
            + ' / '.join(f"{r['times_seconds']['IFFT-FFT fast HH']/r['times_seconds']['B1 adaptive HH']:.2f}x"
                         for r in results)
            + ' (BNS / GW190814), though its real-waveform errors do not yet satisfy '
            'the requested IFFT-FFT accuracy equivalence. Neither A nor B1 is an '
            'unqualified replacement under the current accuracy and total-cost criteria. '
            'Extrinsic matrix reuse remains a separate motivation for A.')


def main():
    results=[]
    for name in ('BNS','GW190814'):
        c=xhm_case(name);m,p,b=c.model,c.psdgrid,c.bands
        f=b.all_coarse_frequencies().f
        _,seq,_=m._sequence(f)
        params=m._params()
        mass=(m.m1*lal.MSUN_SI,m.m2*lal.MSUN_SI)
        def pol():
            return ls.SimInspiralChooseFDWaveformSequence(m.phi_ref,*mass,
                       0,0,m.chi1z,0,0,m.chi2z,m.f_ref,
                       m.distance_mpc*1e6*lal.PC_SI,m.inclination,params,ls.IMRPhenomXHM,seq)
        def one(lm,sign=-1):
            return ls.SimIMRPhenomXHMFrequencySequenceOneMode(seq,*mass,
                       m.chi1z,m.chi2z,lm[0],sign*lm[1],m.distance_mpc*1e6*lal.PC_SI,
                       m.phi_ref,m.f_ref,params)
        def five():
            return [one(lm) for lm in m.modes]
        def ten():
            return [one(lm,sign) for lm in m.modes for sign in (-1,1)]
        def hm():
            # GethlmModes includes HM's additional (4,3). This is a timing
            # comparison, never a replacement for XHM numerical references.
            return ls.SimIMRPhenomHMGethlmModes(seq,*mass,0,0,m.chi1z,
                        0,0,m.chi2z,m.phi_ref,0.,m.f_ref,lal.CreateDict())
        def assembled():
            return [m.mode(f,lm) for lm in m.modes]
        coarse=CoarseModes.from_model(b,m,m.modes)
        total=CoarseWaveform.from_model(b,m.total_direct)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            pair=PairwiseHH(b,p,m.modes,mchirp_range=xhm_systems()[name]['mchirp_range'])
        calls={'XHM polarizations (LAL)':pol,'XHM five OneMode -m (LAL)':five,
               'XHM ten OneMode +/-m (LAL)':ten,'HM all modes (LAL)':hm,
               'XHM assembled detector modes':assembled,'XHM total_direct adapter':lambda:m.total_direct(f),
               'linear HH':lambda:linear(total),'IFFT-FFT paper HH':lambda:paper(total),
               'IFFT-FFT fast HH':lambda:fast(total),'pairwise pairs HH':lambda:pair(coarse,impl='pairs'),
               'pairwise einsum HH':lambda:pair(coarse,impl='einsum'),
               'B0 HH':lambda:b0(total),'B1 adaptive HH':lambda:b1(total)}
        linear=LinearInterpolationHH(b,p);paper=IFFTFFTHH(b,p,impl='paper')
        fast=IFFTFFTHH(b,p,impl='fast');b0=CentredBlockHH(b,p);b1=BasebandIFFTFFTHH(b,p)
        times={};numbers={};failed={}
        for label,fn in calls.items():
            try:
                times[label],numbers[label]=measure(fn)
                print(name,label,1000*times[label], 'ms',flush=True)
            except (RuntimeError,TypeError) as e:
                if label!='HM all modes (LAL)': raise
                failed[label]=str(e)
        # Subtraction is an estimate and includes array allocation and copying.
        times['Python assembly overhead (difference)']=(times['XHM assembled detector modes']-
                                                        times['XHM five OneMode -m (LAL)'])
        baseline=times['XHM total_direct adapter']+times['IFFT-FFT fast HH']
        routes={}
        for label,wave,hh in [
            ('IFFT-FFT','XHM total_direct adapter','IFFT-FFT fast HH'),
            ('A','XHM assembled detector modes','pairwise pairs HH'),
            ('B1','XHM total_direct adapter','B1 adaptive HH'),
            ('B0','XHM total_direct adapter','B0 HH')]:
            value=times[wave]+times[hh]
            routes[label]=dict(seconds=value,ratio_to_ifftfft=value/baseline)
        # Also give the lower-level route sums requested, excluding adapters.
        raw_baseline=times['XHM polarizations (LAL)']+times['IFFT-FFT fast HH']
        raw_routes={label:dict(seconds=times[wave]+times[hh],
                              ratio_to_ifftfft=(times[wave]+times[hh])/raw_baseline)
                    for label,wave,hh in [
                        ('IFFT-FFT','XHM polarizations (LAL)','IFFT-FFT fast HH'),
                        ('A','XHM five OneMode -m (LAL)','pairwise pairs HH'),
                        ('B1','XHM polarizations (LAL)','B1 adaptive HH'),
                        ('B0','XHM polarizations (LAL)','B0 HH')]}
        results.append(dict(system=name,K_MB=b.K_MB,K_unique=len(seq.data),
                            times_seconds=times,batch_numbers=numbers,routes=routes,
                            raw_routes=raw_routes,failed=failed,fft_lengths=b1.fft_lengths()))
    (ROOT/'data/xhm_timing_iter005.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Iteration 005: waveform and HH timing',
           'Median wall clock after warm-up, three batches per entry, each targeted at ~1 second. '
           'Python on PATH; LALSimulation 6.2.0; this machine. All times below in ms. '
           'BNS K_MB=20167; GW190814 K_MB=4865 (the latter is smaller than the reviewer\'s 16064-point setup). '
           'Same sorted unique sequence for the raw LAL comparisons; adapter calls sort/unsort every time. '
           'HH inputs and PSD weights are precomputed. Five negative-m calls suffice; ten are redundant.',
           '| Component | BNS ms | GW190814 ms |','|---|---:|---:|']
    for label in results[0]['times_seconds']:
        lines.append('| '+label+' | '+' | '.join(f"{1000*r['times_seconds'].get(label,float('nan')):.4f}" for r in results)+' |')
    for title,key in [('Actual adapter routes (waveform + HH)','routes'),
                      ('Raw LAL route sums (waveform + HH; assembly omitted)','raw_routes')]:
        lines+=['',f'## {title}','| Route | BNS ms | ratio | GW190814 ms | ratio |','|---|---:|---:|---:|---:|']
        for label in ('IFFT-FFT','A','B1','B0'):
            lines.append('| '+label+' | '+' | '.join(
                f"{1000*r[key][label]['seconds']:.4f} | {r[key][label]['ratio_to_ifftfft']:.4f}" for r in results)+' |')
    lines+=['','## Interpretation',
            'Ratios are costs: below 1 is faster. These route sums include waveform generation and the quadratic term only, '
            'not (d,h), priors or sampler work. Python overhead is estimated by subtracting independently timed medians; '
            'it includes sorting, unique/unsort, copies, phase and projection, and is not an exact isolated measurement. '
            'A reuses raw modes for all extrinsics, but strict orbital-phase independence is limited by the LAL mixed-32 residual '
            'documented in the accuracy report. HM all-modes includes (4,3), is a different approximant, and its raw amplitudes '
            'need physical scaling; it is not interchangeable with XHM. No ten-call route is needed for XHM.',
            'B0 accuracy is weaker with higher modes and line-rich PSDs. B1 remains the total-waveform candidate; '
            'its accuracy relative to IFFT-FFT must be judged alongside the speed table.',
            f"HM binding failures: {[r['failed'] for r in results]}",
            '',assessment(results)]
    (ROOT/'data/xhm_timing_iter005.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
