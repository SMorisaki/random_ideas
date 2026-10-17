"""Iteration 003 margin scan, folding audit and prior-coverage tables.

Run: MPLCONFIGDIR=/tmp/mplconfig python scripts/margin_study.py
All error references are the discrete fine-grid Eq. (4), not IFFT-FFT.
"""

import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'data' / 'fontcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT))

from mbquad.bands import BandSetup
from mbquad.coarse import CoarseModes
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from mbquad.pairwise import PairwiseHH
from mbquad.psd import PSDGrid, aligo_analytic_psd
from mbquad.reference import hh_exact, lnlambda_error
from mbquad.testcases import toy_small


def wrong_placement(method, coarse, placement):
    """Deliberate ablations: centre every class at zero or reflect its block."""
    total = 0
    bs = method.bands
    for b, (H, grid) in enumerate(zip(coarse.h_b, bs.coarse_frequencies())):
        M = int(bs.M_b[b])
        rows = {r['lag_class']: r for r in method.summary() if r['b'] == b}
        grouped = {m: H[idx].sum(axis=0) for m, idx in method._groups.items()}
        total += sum(np.dot(method.class_weights[b][0], abs(h)**2).real
                     for h in grouped.values())
        for cl in method.classes[1:]:
            m0 = -M//2 if placement == 'zero' else -rows[cl]['m0']-M+1
            w = method.blockweights.weights(b, m0, M, grid.kappa)*grid.w
            if placement == 'zero':
                # Use the class-0 real weight for ALL entries, exactly as
                # a centred block applied to the real total |h|^2.
                w = w.real
            total += 2*np.dot(w, grouped[cl[0]].conj()*grouped[cl[1]]).real
    return float(4/bs.T*total)


def error_record(value, exact):
    return dict(relative_error=abs(value/exact-1),
                signed_relative_error=value/exact-1,
                abs_delta_lnLambda=float(abs(lnlambda_error(value,exact,25))))


def slack_report(case):
    """Full per-band/class summaries for the toy and specified prior setups."""
    cases = [('toy_small', case.bands, (4.85,4.85))]
    specifications = [
        ('BNS', 256, 2048, [256,128,64,32,16,8,4], 1.15, (1.15,1.25)),
        ('GW190814-like', 16, 1024, [16,8,4], 6.4, (6,7)),
        ('BBH 60+20', 16, 1024, [16,8,4], 25, (25,35))]
    for name,T,fhigh,durations,mc,prior in specifications:
        cases.append((name, BandSetup(T,1/4096,20,fhigh,durations,mc,4,5),prior))
    sections = ['# Prior-coverage slack — iteration 003',
                'All values are seconds. L_margin=5, sigma_pair; band geometry L=5, '
                'm_max=4. BNS dt=1/4096 s, 20–2048 Hz; GW190814-like and BBH '
                'dt=1/4096 s, 20–1024 Hz. Toy geometry is toy_small(). '
                'slack is requested coverage slack before clipping; block_end is exclusive. '
                'Anchors and margins use Mc_max at the low-lag end and Mc_min at the '
                'high-lag end. These are Newtonian prior-coverage diagnostics, not '
                'accuracy claims for real BNS/BBH waveforms. '
                '**Band 0 has M_b=N_b in every setup: all circular lags are retained '
                'and the discrete quadrature is exact for every block start. Its negative '
                'unwrapped SPA slack is reported and warned as requested, but does not '
                'indicate lost quadrature coverage.** All b>=1 classes below have positive slack.']
    for name,bands,prior in cases:
        psd = PSDGrid(aligo_analytic_psd,bands.T,bands.dt,bands.f_low,bands.f_high)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',RuntimeWarning)
            p = PairwiseHH(bands,psd,case.model.modes,mchirp_range=prior)
        minimum = min(r['slack'] for r in p.summary())
        sections.extend([f'## {name}: Mc_range={prior}; Mc_ref={bands.mchirp_ref}',
                         f'Minimum requested slack: {minimum:.6f} s; '
                         f'{len(caught)} clipping warnings. Cached block-weight memory: '
                         f'{p.blockweights.nbytes()/2**20:.3f} MiB.', p.format_summary()])
    (ROOT/'data/slack_iter003.md').write_text('\n\n'.join(sections)+'\n')


def main():
    case = toy_small(); bands = case.bands
    modes = CoarseModes.from_model(bands,case.model,case.model.modes)
    scans, baselines, audits = [], [], []
    for has_lines in (True,False):
        label = 'lines' if has_lines else 'smooth'
        psd = case.psdgrid if has_lines else PSDGrid(
            aligo_analytic_psd,bands.T,bands.dt,bands.f_low,bands.f_high)
        exact = hh_exact(case.model.total(psd.f_k),psd)
        default = PairwiseHH(bands,psd,modes.modes)
        methods = [('linear',LinearInterpolationHH(bands,psd)(modes.total())),
                   ('IFFT-FFT fast',IFFTFFTHH(bands,psd)(modes.total())),
                   ('all centred at zero',wrong_placement(default,modes,'zero')),
                   ('mirrored negative',wrong_placement(default,modes,'mirror'))]
        for name,value in methods:
            baselines.append(dict(PSD=label,method=name,**error_record(value,exact)))
        for units, margins in [('sigma_pair',[.5,1,2,3,5,8]),
                               ('sigma_mmax',[.5,1,2,3,5,8]),
                               ('T_b',[1/64,1/32,1/16,1/8])]:
            for L in margins:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',RuntimeWarning)
                    p = PairwiseHH(bands,psd,modes.modes,L_margin=L,margin_units=units)
                scans.append(dict(PSD=label,units=units,L_margin=L,
                                  min_slack=min(r['slack'] for r in p.summary()),
                                  warnings=[str(w.message) for w in caught],
                                  **error_record(p(modes),exact)))
        for b in (1,3,5):
            for slow,fast in [((4,4),(2,1)),((4,4),(2,2)),((3,3),(2,2)),((2,2),(2,1))]:
                cl = (slow[1],fast[1]); nh = int(bands.M_b[b])
                row = next(r for r in default.summary() if r['b']==b and r['lag_class']==cl)
                g = bands.window(psd.f_k,b)*case.model.mode(psd.f_k,slow).conj()*case.model.mode(psd.f_k,fast)
                for placement,m0 in [('class',row['m0']),('zero',-nh//2)]:
                    audit = default.blockweights.lag_audit(b,g,nh,m0)
                    for key in ('folding_error','direct_error','fine_sum','quadrature_sum'):
                        z = audit[key]; audit[key] = [z.real,z.imag]
                    audit['signed_relative_hh_error'] = 8/bands.T*audit['folding_error'][0]/exact
                    audits.append(dict(PSD=label,b=b,pair=[slow,fast],placement=placement,
                                       Lambda_min=row['Lambda_min'],Lambda_max=row['Lambda_max'],
                                       block_start=m0*bands.dt_b[b],
                                       block_end=(m0+nh)*bands.dt_b[b],**audit))
    result = dict(system='toy_small, five ToyModes defaults; SNR=25',
                  default=dict(L_margin=5,margin_units='sigma_pair',diag='block',impl='pairs'),
                  scan=scans,baselines=baselines,audits=audits)
    data = ROOT/'data'; data.mkdir(exist_ok=True)
    (data/'margin_study_iter003.json').write_text(json.dumps(result,indent=2)+'\n')
    sections = ['# Margin study — iteration 003',
                'toy_small(): 128 s, dt=1/1024 s, 20–256 Hz; five default SPA modes, '
                'Mc=Mc_ref=4.85 Msun, tc=126 s, band L=5. PSD is analytic aLIGO with '
                'TOY_LINES or without lines. Exact reference: fine-grid Eq. (4). '
                'Errors are magnitudes; |Delta ln Lambda| is at SNR 25.',
                '| PSD | baseline / wrong placement | relative error | abs(Delta ln Lambda) |\n'
                '| --- | --- | --- | --- |\n'+'\n'.join(
                    f"| {r['PSD']} | {r['method']} | {r['relative_error']:.6e} | {r['abs_delta_lnLambda']:.6e} |"
                    for r in baselines),
                '| PSD | units | margin | relative error | abs(Delta ln Lambda) | minimum slack (s) | warnings |\n'
                '| --- | --- | --- | --- | --- | --- | --- |\n'+'\n'.join(
                    f"| {r['PSD']} | {r['units']} | {r['L_margin']:g} | {r['relative_error']:.6e} | "
                    f"{r['abs_delta_lnLambda']:.6e} | {r['min_slack']:.6f} | {len(r['warnings'])} |" for r in scans),
                'Default: 5 sigma_pair. The 0.5–3 sigma plateau has slightly smaller toy errors; '
                '5 retains more conservative tail coverage without clipping in this system, '
                'still within twice the IFFT-FFT error. At 8 sigma the (4,1) class in band 3 '
                'requires clipping. Symmetric T_b margins cancel from the interval midpoint, '
                'so all four T_b fractions produce identical placements and errors here. '
                'This scan studies a centred fixed-length block, not a variable-length crop.',
                'Lag centroid and quantiles use |G|² on a lag axis unwrapped about each block centre. '
                'Outside fraction uses sum|G|; it need not track error because PSD folding matters. '
                'Folding error is complex quadrature minus fine sum, before 4/T. '
                'Relative HH error below is the signed contribution 2 Re(error) (4/T)/(h,h). '
                'The zero-block audit uses its complex W on an ordered pair; the all-centred '
                'total-waveform ablation uses real W for every matrix entry, as required '
                'for class 0 (the one-bin endpoint asymmetry otherwise matters).',
                '| PSD | b | pair | placement | outside fraction | centroid (s) | predicted interval (s) | '
                'folding error (Re, Im) | relative HH contribution |\n'
                '| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'+'\n'.join(
                    f"| {r['PSD']} | {r['b']} | {r['pair'][0]} x {r['pair'][1]} | {r['placement']} | "
                    f"{r['outside_fraction']:.6e} | {r['lag_centroid']:.6f} | "
                    f"[{r['Lambda_min']:.6f}, {r['Lambda_max']:.6f}] | "
                    f"({r['folding_error'][0]:.6e}, {r['folding_error'][1]:.6e}) | "
                    f"{r['signed_relative_hh_error']:.6e} |" for r in audits)]
    (data/'margin_study_iter003.md').write_text('\n\n'.join(sections)+'\n')
    make_figures(case, scans,baselines)
    slack_report(case)
    print('Wrote margin_study_iter003.json/.md, slack_iter003.md and both figures.')


def make_figures(case,scans,baselines):
    figures = ROOT/'figures'; figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,label in zip(axes,['lines','smooth']):
        for units,marker in [('sigma_pair','o'),('sigma_mmax','s'),('T_b','^')]:
            rows = [r for r in scans if r['PSD']==label and r['units']==units]
            # Separate labels preserve the meaning of the two different x units.
            ax.plot([r['L_margin'] for r in rows],[r['relative_error'] for r in rows],
                    marker=marker,label=units)
        baseline_styles = {
            'linear': ('0.4', '--'),
            'IFFT-FFT fast': ('purple', '-.'),
            'all centred at zero': ('brown', ':'),
            'mirrored negative': ('red', '--'),
        }
        for r in baselines:
            if r['PSD']==label:
                color, style = baseline_styles[r['method']]
                ax.axhline(r['relative_error'], color=color, ls=style,
                           alpha=.7, label=r['method'])
        ax.set(xscale='log',yscale='log',xlabel='margin multiplier (sigma or T_b)',
               ylabel='absolute relative (h,h) error',title=label)
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(figures/'iter003_margin_study.png',dpi=160); plt.close(fig)
    bs=case.bands; p=PairwiseHH(bs,case.psdgrid,case.model.modes)
    pairs=[((4,4),(2,1)),((4,4),(2,2)),((3,3),(2,2)),((2,2),(2,1))]
    fig,axes=plt.subplots(3,4,figsize=(15,9),layout='constrained')
    for row,b in enumerate([1,3,5]):
        n=int(bs.N_b[b]); f=np.arange(1,(n+1)//2)/bs.T
        for col,(slow,fast) in enumerate(pairs):
            g=np.zeros(n,complex)
            g[1:len(f)+1]=bs.window(f,b)*case.model.mode(f,slow).conj()*case.model.mode(f,fast)
            power=abs(n*np.fft.ifft(g))**2
            lag=(np.arange(n)+n//2)%n-n//2; lag=lag*bs.dt_b[b]
            # Show the positive lag near T via the equivalent unwrapped period.
            pred=next(r for r in p.summary() if r['b']==b and r['lag_class']==(slow[1],fast[1]))
            centre=(pred['block_start']+pred['block_end'])/2
            lag=centre+(lag-centre+bs.T/2)%bs.T-bs.T/2
            order=np.argsort(lag); ax=axes[row,col]
            ax.semilogy(lag[order],power[order]/power.max(),lw=.7)
            ax.axvspan(pred['Lambda_min'],pred['Lambda_max'],color='green',alpha=.18,label='SPA interval')
            for value in [pred['block_start'],pred['block_end']]:
                ax.axvline(value,color='green',ls='--',lw=.8)
            for value in [-bs.T_b[b]/2,bs.T_b[b]/2]:
                ax.axvline(value,color='red',ls=':',lw=.8)
            ax.set(xlim=(-.6*bs.T_b[b],1.6*bs.T_b[b]),ylim=(1e-12,2),
                   title=f'b={b}, {slow} x {fast}',xlabel='lag (s)',ylabel='|G|² / max')
            ax.grid(alpha=.2)
    fig.suptitle('Green: SPA interval and class block boundaries; red: centred block boundaries')
    fig.savefig(figures/'iter003_lag_audit.png',dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
