"""Iteration 008: XHM L scan, HOM band rules, route estimates, block displacement.

Run from multiband_quad/: python scripts/bandrule_iter008.py
Fine references reuse iteration 005/006 hashes. All cost estimates use warm
iteration-006 component medians, scaled with K_MB, not new measured timings.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/fontcache'))
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mbquad.bands import BandSetup
from mbquad.testcases import xhm_case, xhm_systems
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.reference import hh_exact
from mbquad.linear import LinearInterpolationHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.totalweights import CentredBlockHH
from iter006_common import ReferenceCache, quiet_lal, pairwise, stats, write_json

METHODS = ('linear','IFFT-FFT','pairwise','B1','B1_all','B0')
LS = (5,10,20,50)
PSDS = ('analytic_lines','L1_real')


def systems():
    result = xhm_systems()
    for p in result.values():
        p['inclination'] = 1.
    result['NSBH_strong'] = {**result['GW190814'], 'm2':2.3, 'inclination':1.5,
                             'mchirp_ref':4., 'mchirp_range':(4.,9.)}
    result['BBH_negative'] = {**result['BBH'], 'chi1z':-.5, 'chi2z':-.5}
    return result


def setup(p, L=5, **kw):
    return BandSetup(**{k:p[k] for k in ('T','dt','f_low','f_high','durations',
                                        'mchirp_ref','m_max')}, L=L, **kw)


def case(name, psd_name, params):
    base = 'GW190814' if name == 'NSBH_strong' else 'BBH' if name == 'BBH_negative' else name
    c = xhm_case(base, psd_name)
    for k in vars(c.model):
        if k in params:
            setattr(c.model,k,params[k])
    return c


def evaluate(model, psd, b, prior, exact, lh=5):
    adapter = XHMCoarseEvaluator(model,b)
    co, cm = adapter.coarse_total(), adapter.coarse_modes()
    ev = {'linear':LinearInterpolationHH(b,psd), 'IFFT-FFT':IFFTFFTHH(b,psd,impl='fast'),
          'pairwise':pairwise(b,psd,model.modes,mchirp_range=prior),
          'B1':BasebandIFFTFFTHH(b,psd,L_margin=lh),
          'B1_all':BasebandIFFTFFTHH(b,psd,L_margin=lh,all_half_bins=True),
          'B0':CentredBlockHH(b,psd)}
    errors = {k:float(abs(v(cm if k=='pairwise' else co)/exact-1)) for k,v in ev.items()}
    return errors, ev['B1'].modes


def route_costs(name, K_paper, K_half, timings):
    """Waveform once + D HH steps; linear K_MB extrapolation of warm medians."""
    base = 'GW190814' if name == 'NSBH_strong' else 'BBH' if name == 'BBH_negative' else name
    r = next(r for r in timings if r['system']==base)
    t, k0 = r['times_seconds'], r['K_MB']
    routes = {}
    for label, wave, method, k in [('IFFT-FFT paper','coarse_total','IFFT-FFT',K_paper),
                                  ('B0 hh_half','coarse_total','B0',K_half),
                                  ('pairwise paper','coarse_modes','pairwise',K_paper)]:
        wf, hh = 1000*t[wave]*k/k0, 1000*t[method+' HH']*k/k0
        routes[label] = dict(waveform_ms=wf, hh_ms_per_detector=hh,
                             one_detector_ms=wf+hh, three_detector_ms=wf+3*hh)
    return routes


def slack_study(cache):
    rows = []
    p = systems()['GW190814']
    for psd_name in PSDS:
        c = case('GW190814',psd_name,p)
        m, psd = c.model,c.psdgrid
        b = setup(p)
        fine = cache.get('GW190814',psd_name,m,psd)
        exact = hh_exact(fine,psd)
        cm = XHMCoarseEvaluator(m,b).coarse_modes()
        # Pair 44 x 21 is the unique extreme lag class. Its error reveals
        # loss of support even when the diagonal tail masks the total error.
        fine_pair = m.mode(psd.f_k,(4,4)).conj()*m.mode(psd.f_k,(2,1))
        for prior_name, prior in [('prior',p['mchirp_range']),('true_mass',(m.mchirp,m.mchirp))]:
            ev = pairwise(b,psd,m.modes,mchirp_range=prior)
            grouped = [{mm:sum(H[i] for i,lm in enumerate(m.modes) if lm[1]==mm)
                        for mm in (1,2,3,4)} for H in cm.h_b]
            base = ev(cm)
            for shift_sigma in (-16,-8,-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,8,16):
                value, slacks = base, []
                extreme_error = extreme_slack = None
                for row in ev.summary():
                    ib, cl = row['b'],row['lag_class']
                    if row['full_grid'] or cl == 0:
                        continue
                    flo = b.f_b[ib]-b.Df_b[ib]
                    sigma = float(np.sqrt((8/3)*b.lag_pair(flo,*cl,m.mchirp)/flo))
                    m0 = row['m0'] + int(np.rint(shift_sigma*sigma/b.dt_b[ib]))
                    grid = b.coarse_frequencies()[ib]
                    w = ev.blockweights.weights(ib,m0,int(b.M_b[ib]),grid.kappa)*grid.w
                    product = grouped[ib][cl[0]].conj()*grouped[ib][cl[1]]
                    delta = (8/b.T)*np.dot(w-ev.class_weights[ib][cl],product).real
                    value += delta
                    lo,hi = b.lag_interval_pair(ib,*cl,mchirp=m.mchirp)
                    slack = min(lo-m0*b.dt_b[ib],(m0+b.M_b[ib])*b.dt_b[ib]-hi)
                    slacks.append(float(slack))
                    if ib == 1 and cl == (4,1):
                        truth = (8/b.T)*np.dot(b.window(psd.f_k,ib)*fine_pair,psd.inv_S).real
                        approx = (8/b.T)*np.dot(w,product).real
                        extreme_error = float(abs(approx-truth)/exact)
                        extreme_slack = float(slack)
                rows.append(dict(psd=psd_name,prior=prior_name,shift_sigma=shift_sigma,
                                 slack_seconds=min(slacks),extreme_slack_seconds=extreme_slack,
                                 relative_error=float(abs(value/exact-1)),
                                 abs_delta_lnLambda=312.5*abs(value/exact-1),
                                 extreme_error_over_total=extreme_error))
    return rows


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(map(str,r))+' |' for r in rows]


def main():
    cache = ReferenceCache()
    timings = json.loads((ROOT/'data/xhm_timing_iter006.json').read_text())['results']
    raw, half_raw, geometry, full_checks = [], [], [], []
    for name,p in systems().items():
        for L in LS:
            b = setup(p,L)
            geometry.append(dict(system=name,L=L,rule='paper',K_MB=b.K_MB,bands=b.summary_table()))
        for lh in (3,5,8):
            b = setup(p,rule='hh_half',L_hh=lh)
            full = setup(p,rule='hh_full',L_hh=lh)
            full_checks.append(dict(system=name,L_hh=lh,K_MB=full.K_MB,
                                    binding=full.binding,edges=full.f_b.tolist()))
            geometry.append(dict(system=name,L=5,L_hh=lh,rule='hh_half',K_MB=b.K_MB,
                                 bands=b.summary_table()))
        for psd_name in PSDS:
            c = case(name,psd_name,p)
            # This strong-HOM system already exists under the iteration-005
            # GW190814 q=0.1 cache key; its new band prior does not affect HH.
            reference_name = 'GW190814' if name == 'NSBH_strong' else name
            fine = cache.get(reference_name,psd_name,c.model,c.psdgrid)
            exact = hh_exact(fine,c.psdgrid)
            for L in LS:
                b = setup(p,L)
                errors,modes = evaluate(c.model,c.psdgrid,b,p['mchirp_range'],exact)
                raw.append(dict(system=name,psd=psd_name,L=L,K_MB=b.K_MB,errors=errors,modes=modes))
            for lh in (3,5,8):
                b = setup(p,rule='hh_half',L_hh=lh)
                errors,modes = evaluate(c.model,c.psdgrid,b,p['mchirp_range'],exact,lh)
                default_modes = BasebandIFFTFFTHH(b,c.psdgrid).modes
                half_raw.append(dict(system=name,psd=psd_name,L_hh=lh,K_MB=b.K_MB,
                                     errors=errors,modes=modes,default_B1_modes=default_modes))
            print(name,psd_name,'complete',flush=True)
    aggregate, half_aggregate, slopes = [], [], []
    for name,p in systems().items():
        for L in LS:
            rr = [r for r in raw if r['system']==name and r['L']==L]
            aggregate.append(dict(system=name,L=L,K_MB=rr[0]['K_MB'],
                                  errors={m:stats([r['errors'][m] for r in rr]) for m in METHODS}))
        for method in ('pairwise','IFFT-FFT'):
            per_psd = {}
            for psd in PSDS:
                err = [next(r['errors'][method] for r in raw if r['system']==name and r['psd']==psd and r['L']==L) for L in LS]
                per_psd[psd] = float(np.polyfit(np.log(LS),np.log(np.maximum(err,1e-16)),1)[0])
            median_err = [next(r['errors'][method]['median_relative'] for r in aggregate if r['system']==name and r['L']==L) for L in LS]
            slopes.append(dict(system=name,method=method,median_psd_slope=float(np.polyfit(np.log(LS),np.log(median_err),1)[0]),per_psd=per_psd))
        kp = setup(p).K_MB
        for lh in (3,5,8):
            rr = [r for r in half_raw if r['system']==name and r['L_hh']==lh]
            kh = rr[0]['K_MB']
            half_aggregate.append(dict(system=name,L_hh=lh,K_MB=kh,increase_percent=100*(kh/kp-1),
                                       modes=rr[0]['modes'],default_B1_modes=rr[0]['default_B1_modes'],
                                       errors={m:stats([r['errors'][m] for r in rr]) for m in ('B0','B1','pairwise','IFFT-FFT')},
                                       routes=route_costs(name,kp,kh,timings)))
    slack = slack_study(cache)
    write_json('bandrule_iter008_full.json',dict(L_scan=raw,hh_half=half_raw,slack=slack,geometry=geometry))
    result = dict(L_scan=aggregate,slopes=slopes,hh_half=half_aggregate,hh_full=full_checks,
                  cache=dict(hits=cache.hits,misses=cache.misses),
                  slack_range=[min(r['slack_seconds'] for r in slack),max(r['slack_seconds'] for r in slack)])
    write_json('bandrule_iter008.json',result)
    report = ['# Iteration 008: HOM band rules', '',
              'XHM, OMP_NUM_THREADS=1. Inclination 1.0 except strong NSBH (23+2.3, 1.5); '
              'negative-spin BBH has chi1=chi2=-0.5. All five systems use the nominal durations. '
              'Strong NSBH uses the iteration-007 in-prior reference Mc=4 and range [4,9]; '
              'using nominal Mc_ref=6.4 would place it outside the band-design prior. '
              'Other references/priors are unchanged from testcases.py. Nominal GW190814 itself has Mc below 6.4 '
              '(the historical paper reference), explicitly retained for comparison.', '',
              'PSDs: analytic aLIGO + the five Gaussian TOY_LINES (unchanged from cached iteration 005), '
              'and measured L1 Welch16s log-log interpolated to the fine grid. Errors are against Eq. (4). '
              'Tables show median / worst over the two PSDs. SNR=25: |Delta ln Lambda|=312.5 times relative error. '
              'Full per-PSD results and band edges are in bandrule_iter008_full.json.', '', '## Paper-rule L scan']
    report += table(['System','L','K_MB']+list(METHODS),[
        [r['system'],r['L'],r['K_MB']]+[f"{r['errors'][m]['median_relative']:.2e} / {r['errors'][m]['worst_relative']:.2e}" for m in METHODS] for r in aggregate])
    report += ['', 'Likelihood errors at SNR 25 (median / worst):']
    report += table(['System','L']+list(METHODS),[
        [r['system'],r['L']]+[f"{r['errors'][m]['median_lnLambda']:.2e} / {r['errors'][m]['worst_lnLambda']:.2e}" for m in METHODS] for r in aggregate])
    report += ['', '## Fitted d log(error) / d log(L)',
               'Unweighted fits over all four L values. The expected -3 / -2 are tail-envelope exponents, '
               'not guaranteed exponents of the signed, PSD-weighted, band-summed error. Moving band edges '
               'changes cancellations and line positions; the dominant diagonal fold can set a different slope. '
               'Do not interpret four-point slopes as an asymptotic convergence proof.']
    report += table(['System','Method','median PSD error fit','analytic+lines','L1'],[
        [r['system'],r['method'],f"{r['median_psd_slope']:.2f}"]+[f"{r['per_psd'][p]:.2f}" for p in PSDS] for r in slopes])
    report += ['', '## hh_half at L=5: relative errors (median / worst)',
               'B1 uses L_margin=L_hh for the matched support criterion. Its modes include band 0="exact"; '
               'every other band is B0. The separate default-B1 column uses the unchanged L_margin=5, '
               'which need not select B0 when L_hh=3.']
    report += table(['System','L_hh','K_MB','increase %','matched B1 modes','default B1 modes','B0','B1','pairwise','IFFT-FFT'],[
        [r['system'],r['L_hh'],r['K_MB'],f"{r['increase_percent']:.2f}",','.join(r['modes']),','.join(r['default_B1_modes'])]+[
            f"{r['errors'][m]['median_relative']:.2e} / {r['errors'][m]['worst_relative']:.2e}" for m in ('B0','B1','pairwise','IFFT-FFT')] for r in half_aggregate])
    report += ['', 'hh_half likelihood errors at SNR 25:']
    report += table(['System','L_hh','B0','B1','pairwise','IFFT-FFT'],[
        [r['system'],r['L_hh']]+[f"{r['errors'][m]['median_lnLambda']:.2e} / {r['errors'][m]['worst_lnLambda']:.2e}" for m in ('B0','B1','pairwise','IFFT-FFT')] for r in half_aggregate])
    report += ['', '## Implied route costs, ms',
               'Estimated from iteration-006 warm wall-clock component medians: waveform_ms_per_point '
               '=1000*coarse_total_or_modes/K_MB; each HH cost scales linearly with K_MB as requested. '
               'One waveform + D identical-detector HH steps; extra projections/other likelihood work excluded. '
               'These are extrapolations, not fresh timings. Strong NSBH uses GW190814 costs; negative BBH '
               'uses BBH costs, including OneMode tail recovery. Waveform and HH columns remain separate.']
    report += table(['System','L_hh','route','waveform ms','HH ms/det','1 detector ms','3 detectors ms'],[
        [r['system'],r['L_hh'],label]+[f"{cost[k]:.3f}" for k in ('waveform_ms','hh_ms_per_detector','one_detector_ms','three_detector_ms')]
        for r in half_aggregate for label,cost in r['routes'].items()])
    report += ['', '## hh_full binding and resulting edges',
               'hh_full is not always weaker than Eq. 51: evaluating sigma at f_lo and adding the '
               'upper-edge margin can make it bind, particularly in long BNS bands. It uses conservative '
               'Lambda_max rather than subtracting the positive lower lag anchor. Across these cases '
               'K_MB(paper) <= K_MB(hh_full) <= K_MB(hh_half).']
    report += table(['System','L_hh','K_MB','binding per band','edges Hz'],[
        [r['system'],r['L_hh'],r['K_MB'],','.join(r['binding']),', '.join(f'{f:.3f}' for f in r['edges'])] for r in full_checks])
    report += ['', '## hh_half resulting edges (Hz; upper taper included)']
    report += table(['System','L_hh','edges Hz'],[
        [r['system'],r['L_hh'],', '.join(f"{x['f_lo']:.3f}" for x in r['bands'])+f", {r['bands'][-1]['f_hi']:.3f}"] for r in geometry if r['rule']=='hh_half'])
    report += ['', '## Block displacement: slack versus error',
               'GW190814-like, paper L=5, inclination 1.0. Diagonal blocks stay centred; cross-class blocks '
               'shift by units of sigma_pair(f_lo,Mc_true). Compare original prior corners [6,7] with '
               '[Mc_true,Mc_true], retaining L_margin=5 and fixed block length T_b. Narrowing the prior '
               'changes the interval and centre; it does not shorten that block. '
               'The requested +/-0.25,0.5,1 sigma generally retain coverage, so shifts extend to 16 sigma '
               'to force the lag interval outside. Slack is min(Lambda_min-block_start,block_end-Lambda_max) '
               'using true Newtonian anchors; negative means bulk support escapes. Extreme-class error '
               'is the band-1 (4,1) cross contribution error divided by total exact HH; full HH also '
               'contains the invariant diagonal-tail floor.']
    report += table(['PSD','prior','shift sigma','min slack s','extreme slack s','HH rel error','abs Delta lnLambda','extreme error / HH'],[
        [r['psd'],r['prior'],r['shift_sigma']]+[f"{r[k]:.3e}" for k in ('slack_seconds','extreme_slack_seconds','relative_error','abs_delta_lnLambda','extreme_error_over_total')] for r in slack])
    safe = [r['relative_error'] for r in slack if abs(r['shift_sigma'])<=1]
    worst = max(slack,key=lambda r:r['relative_error'])
    report += ['', f"For shifts through +/-1 sigma the total relative error stays in "
               f"[{min(safe):.2e}, {max(safe):.2e}]. The largest displaced error is "
               f"{worst['relative_error']:.2e} at slack {worst['slack_seconds']:.3f} s "
               f"({worst['psd']}, {worst['prior']}); |Delta ln Lambda|="
               f"{worst['abs_delta_lnLambda']:.3g} at SNR 25. Error is not monotone "
               'in signed slack because different class/PSD contributions can cancel.']
    report += ['', f"Reference cache: {cache.hits} hits, {cache.misses} misses (new spin variant only if no earlier run).",
               'Recommendation: use hh_half with a matched L_hh for B0 when total-waveform FFT-free '
               'evaluation is the objective; choose its accuracy setting from the error table, not merely '
               'from support. Keep Eq. 51 and prior-centred class blocks for method A, and audit model-phase '
               'slack at prior corners including spin. hh_full is an optional conservative geometric '
               'check, not evidence that every finite-window tail is covered. See prior_slack_iter008.md '
               'for the separate model-track audit and limitations.']
    (ROOT/'data/bandrule_iter008.md').write_text('\n'.join(report)+'\n')
    fig,axs = plt.subplots(2,3,figsize=(14,8))
    for ax,name in zip(axs.flat,systems()):
        rr = [r for r in aggregate if r['system']==name]
        for method in METHODS:
            ax.loglog(LS,[r['errors'][method]['median_relative'] for r in rr],'o-',label=method)
        ax.set_title(name); ax.set_xlabel('L'); ax.set_ylabel('relative HH error (median PSD)'); ax.grid(alpha=.2)
    axs.flat[-1].axis('off'); axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center')
    fig.tight_layout(); fig.savefig(ROOT/'figures/iter008_L_scan.png',dpi=160); plt.close(fig)
    fig,axs = plt.subplots(1,2,figsize=(12,4.5))
    for ax,psd in zip(axs,PSDS):
        for prior in ('prior','true_mass'):
            rr = [r for r in slack if r['psd']==psd and r['prior']==prior]
            ax.semilogy([r['slack_seconds'] for r in rr],[r['relative_error'] for r in rr],'o',label=prior+' total')
            ax.semilogy([r['extreme_slack_seconds'] for r in rr],[max(r['extreme_error_over_total'],1e-16) for r in rr],'.',label=prior+' extreme')
        ax.axvline(0,color='k',ls=':'); ax.set_title(psd); ax.set_xlabel('slack (s)'); ax.set_ylabel('absolute error / total exact HH'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT/'figures/iter008_slack.png',dpi=160); plt.close(fig)
    print('wrote data/bandrule_iter008.{md,json} and two figures',flush=True)


if __name__ == '__main__':
    with quiet_lal():
        main()
