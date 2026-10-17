"""Iteration 005: fixed XHM accuracy cases, fine-sequence references and lag audit.

Run from multiband_quad: python scripts/xhm_accuracy_iter005.py
No regular-grid waveform driver is used. q=0.1 variants keep m1 and the
nominal geometry fixed; their chirp masses can lie outside the block prior.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
from pathlib import Path
import sys
import json
import hashlib
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mbquad.testcases import xhm_case, xhm_systems
from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.pairwise import PairwiseHH
from mbquad.linear import LinearInterpolationHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.totalweights import CentredBlockHH
from mbquad.reference import hh_exact, hh_exact_banded

METHODS=('linear','IFFT-FFT','pairwise','B0','B1')


def main():
    cache_path=ROOT/'data/xhm_reference_cache.npz'
    cache=dict(np.load(cache_path)) if cache_path.exists() else {}
    rows=[]; audits=[]; slacks=[]; refusals=[]; matrix_report={}
    for name, params in xhm_systems().items():
        for psd_name in ('analytic_lines','L1_real'):
            c=xhm_case(name,psd_name);m,p,b=c.model,c.psdgrid,c.bands
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                pair=PairwiseHH(b,p,m.modes,mchirp_range=params['mchirp_range'])
            summary=pair.summary()
            slacks.append(dict(system=name,psd=psd_name,
                               minimum_slack=min(r['slack'] for r in summary),
                               minimum_truncated_slack=min(r['slack'] for r in summary if not r['full_grid']),
                               K_MB=b.K_MB))
            evaluators={'linear':LinearInterpolationHH(b,p),'IFFT-FFT':IFFTFFTHH(b,p,impl='fast'),
                        'pairwise':pair,'B0':CentredBlockHH(b,p),'B1':BasebandIFFTFFTHH(b,p)}
            variants=[(i,False) for i in (.3,1.,1.4)]
            if name!='BNS': variants.append((1.5,True))
            for iota,edge in variants:
                m.inclination=iota;m.m2=.1*m.m1 if edge else params['m2']
                spec=dict(system=name,psd=psd_name,model=vars(m),N=p.N,T=p.T,
                          lal_version='6.2.0',reference='ChooseFDWaveformSequence-v1')
                key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
                try:
                    if key not in cache:
                        cache[key]=m.total_direct(p.f_k)
                        np.savez(cache_path,**cache)
                    fine=cache[key]
                    exact=hh_exact(fine,p)
                    banded=hh_exact_banded(lambda f:fine,p,b)
                    unity=abs(banded/exact-1)
                    assert unity < 2e-14, unity
                    # Total-only methods use the same direct sequence path.
                    co=CoarseWaveform.from_model(b,m.total_direct)
                    modes=CoarseModes.from_model(b,m,m.modes)
                    for method,ev in evaluators.items():
                        value=ev(modes if method=='pairwise' else co)
                        error=abs(value/exact-1)
                        rows.append(dict(system=name,psd=psd_name,inclination=iota,q=m.m2/m.m1,
                                         edge_variant=edge,method=method,hh_exact=exact,
                                         relative_error=error,abs_delta_lnLambda_snr25=312.5*error,
                                         partition_error=unity))
                    print(name,psd_name,iota,'q=',round(m.m2/m.m1,4),'done',flush=True)
                except RuntimeError as e:
                    refusals.append(dict(system=name,psd=psd_name,iota=iota,error=str(e)))
                    continue
                if name=='GW190814' and iota==1.:
                    # Detector-frame products: exactly the real-model pair
                    # used by the quadrature; common t_c cancels.
                    ha=m.mode(p.f_k,(4,4));hb=m.mode(p.f_k,(2,1))
                    for ib in sorted(set((1,b.B-1))):
                        r=next(r for r in summary if r['b']==ib and r['lag_class']==(4,1))
                        audit=pair.blockweights.lag_audit(
                            ib,b.window(p.f_k,ib)*ha.conj()*hb,int(b.M_b[ib]),r['m0'])
                        audits.append(dict(system=name,psd=psd_name,band=ib,
                                           outside_fraction=audit['outside_fraction'],
                                           lag_centroid=audit['lag_centroid'],lag_quantiles=audit['lag_quantiles'],
                                           block=[r['block_start'],r['block_end']],
                                           predicted=[r['Lambda_min'],r['Lambda_max']],
                                           hh_folding_error_real=4/b.T*audit['folding_error'].real))
    # Explicit C5 audit, including the residual orbital-phase dependence.
    from copy import copy
    c=xhm_case('BBH');m,p,b=c.model,c.psdgrid,c.bands
    pair=PairwiseHH(b,p,m.modes,mchirp_range=(25,35))
    f=b.all_coarse_frequencies().f
    def matrix(model):
        raw=model.raw_modes(f)
        co=CoarseModes(b,m.modes,np.split(np.array(list(raw.values())),np.cumsum(b.K_b)[:-1],axis=1))
        return pair.mode_matrix(co)
    M=matrix(m);other=copy(m)
    other.t_c=14.1;other.distance_mpc=400;other.inclination=.4;other.F_plus=-.2;other.F_cross=.9
    matrix_report['other_extrinsics_max_scaled']=float(np.max(abs(matrix(other)-M))/np.max(abs(M)))
    other.phi_ref=1.2
    matrix_report['including_phi_ref_max_scaled']=float(np.max(abs(matrix(other)-M))/np.max(abs(M)))
    C=np.array(list(m.coefficients().values()))
    matrix_report['factorization_relative']=abs(pair.hh_from_matrix(M,C)/pair(CoarseModes.from_model(b,m,m.modes))-1)
    vals=[];ffts=[];fft=IFFTFFTHH(b,p,impl='fast')
    for t in (b.T-2.1,b.T-1.9):
        m.t_c=t;co=CoarseModes.from_model(b,m,m.modes)
        vals.append(pair(co));ffts.append(fft(co.total()))
    matrix_report['pairwise_tc_relative_change']=abs(vals[1]/vals[0]-1)
    matrix_report['ifftfft_tc_relative_change']=abs(ffts[1]/ffts[0]-1)
    # Test the actual raw-series orbital-phase residual independently of
    # matrix weighting, which can hide small waveform discrepancies.
    gw=xhm_case('GW190814').model
    random_f=np.random.default_rng(15).uniform(20,1024,2000)
    gw.phi_ref=0.;raw0=gw.raw_modes(random_f)
    gw.phi_ref=.7;raw1=gw.raw_modes(random_f)
    matrix_report['GW190814_raw_phase_residuals']={
        str(lm):float(np.max(abs(raw0[lm]-raw1[lm]))/np.max(abs(raw0[lm]))) for lm in gw.modes}
    aggregate=[]
    for name in xhm_systems():
        for method in METHODS:
            rr=[r for r in rows if r['system']==name and r['method']==method]
            errors=np.array([r['relative_error'] for r in rr])
            aggregate.append(dict(system=name,method=method,n=len(rr),median_relative=float(np.median(errors)),
                                  worst_relative=float(max(errors)),median_lnLambda=float(312.5*np.median(errors)),
                                  worst_lnLambda=float(312.5*max(errors))))
    result=dict(aggregate=aggregate,slack=slacks,lag_audit=audits,C5=matrix_report,refusals=refusals)
    (ROOT/'data/xhm_accuracy_iter005.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'data/xhm_accuracy_iter005_full.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Iteration 005: XHM fixed-case accuracy',
           'LALSimulation 6.2.0; 22 cases (3 inclinations per system/PSD plus two q=0.1 edge variants per PSD). '
           'All references are total_direct on the full PSDGrid.f_k sequence, cached by model and PSD. '
           'Every partition-of-unity check passed before coarse evaluation. SNR=25: |Delta ln Lambda|=312.5 relative error.',
           'Analytic aLIGO + TOY_LINES; real L1 Welch resolution 1/16 Hz interpolated in log PSD/log f. '
           'These are fixed cases, not the iteration-006 population. The q=0.1 variants keep m1 fixed; '
           'their chirp masses (5.67 and 14.79 Msun) lie below the nominal priors.',
           '| System | Method | n | median relative | worst relative | median abs Delta ln Lambda | worst abs Delta ln Lambda |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['system']} | {r['method']} | {r['n']} | {r['median_relative']:.3e} | {r['worst_relative']:.3e} | {r['median_lnLambda']:.3e} | {r['worst_lnLambda']:.3e} |")
    lines+=['','## Per-PSD summaries (median / worst relative; parentheses are SNR25 likelihood errors)',
            '| System | PSD | Method | median relative | worst relative | median likelihood | worst likelihood |',
            '|---|---|---|---:|---:|---:|---:|']
    for name in xhm_systems():
        for psd in ('analytic_lines','L1_real'):
            for method in METHODS:
                err=[r['relative_error'] for r in rows if r['system']==name and r['psd']==psd and r['method']==method]
                lines.append(f'| {name} | {psd} | {method} | {np.median(err):.3e} | {max(err):.3e} | {312.5*np.median(err):.3e} | {312.5*max(err):.3e} |')
    lines+=['','## Prior slack','Band 0 is a complete lag grid; its negative unwrapped slack is harmless.',
            '| System | K_MB | min slack, all (s) | min slack, truncated (s) |','|---|---:|---:|---:|']
    for r in slacks[::2]:
        lines.append(f"| {r['system']} | {r['K_MB']} | {r['minimum_slack']:.6f} | {r['minimum_truncated_slack']:.6f} |")
    lines+=['','## Real-XHM lag audit: (4,4) x (2,1), iota=1, nominal masses',
            'Outside fraction is sum outside |G| / sum |G|, not |G|^2. The exact folding error includes 4/T.',
            '| PSD | band | predicted interval (s) | block (s) | outside fraction | lag centroid (s) | Re folding HH |',
            '|---|---:|---|---|---:|---:|---:|']
    for r in audits:
        lines.append(f"| {r['psd']} | {r['band']} | {r['predicted']} | {r['block']} | {r['outside_fraction']:.6e} | {r['lag_centroid']:.6f} | {r['hh_folding_error_real']:.6e} |")
    lines+=['','## C5 and LAL convention findings',
            'XHM negative-m modes use AddMode at azimuth pi/2, with exp(+i |m| phi_ref). '
            'Raw modes are normalized to 1 Mpc and this nominal phase is removed; coefficients restore phase and distance. '
            'The mixed (3,2) series retains a small numerical phi_ref dependence in LAL 6.2.0. '
            'Consequently the strict requirement that matrices be identical to roundoff under phi_ref is unmet. '
            'Other extrinsics leave raw matrices identical. This is retained to preserve the direct-waveform identity.',
            'OneMode truncates HM tails at Mf=0.3; on BBH the adapter recovers them using single-mode ChooseFD calls '
            '(32 also activates 22 for consistent mixing). No such fallback is needed for timing systems BNS/GW190814.',
            '```json',json.dumps(matrix_report,indent=2),'```',
            f'Refused parameter sets: {refusals}',
            'Full per-case errors are in xhm_accuracy_iter005_full.json (git-ignored). No mode pairs were dropped.']
    (ROOT/'data/xhm_accuracy_iter005.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,name in zip(axes,xhm_systems()):
        rr=[r for r in aggregate if r['system']==name]
        x=np.arange(len(rr))
        ax.bar(x-.17,[r['median_lnLambda'] for r in rr],.34,label='Median')
        ax.bar(x+.17,[r['worst_lnLambda'] for r in rr],.34,label='Worst')
        ax.set_xticks(x,METHODS,rotation=35);ax.set_yscale('log');ax.set_title(name)
    axes[0].set_ylabel(r'$|\Delta\ln\Lambda|$, SNR 25');axes[-1].legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/iter005_xhm_errors.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
