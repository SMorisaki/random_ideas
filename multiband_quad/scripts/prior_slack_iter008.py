"""Iteration 008 prior grid: XHM phase tracks versus Newtonian block anchors.

Run: python scripts/prior_slack_iter008.py. No PSD or fine HH reference is
needed for stationary tracks; Eq. (4) errors live in bandrule_iter008.py.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
from pathlib import Path
import sys
import json
import itertools
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import lal
import lalsimulation as lalsim
from mbquad.bands import BandSetup
from mbquad.population import population_setup, chirp_mass, population_psd
from mbquad.lalwave import XHMModel, XHM_MODES
from mbquad.slack import phase_time_tracks, measured_slack
from mbquad.toywave import tau_0pn
from iter006_common import pairwise, quiet_lal, write_json


def prior_grid(name, n=5):
    """Grid directly in (Mc,q,chi_eff), clipped to iteration-007 mass boxes.

    Add a 3x3 component-mass mesh to include the transformed-box boundary.
    BNS/BBH use equal component spins; NSBH fixes chi2=0 as sample_population
    does. Effective-spin extrema then depend on q. This does not audit the
    extra differential-spin direction at fixed chi_eff.
    """
    boxes = {'bns':((1.2,2.),(1.2,2.),.05),
             'nsbh':((15.,35.),(1.5,4.),.5),
             'bbh':((30.,80.),(10.,40.),.8)}
    a,b,spin = boxes[name]
    cfg = population_setup(name)
    corners = [(m1,m2) for m1,m2 in itertools.product(a,b) if m1>=m2]
    if max(a[0],b[0]) <= min(a[1],b[1]):
        # The m1>=m2 boundary cuts through the BBH box; its q=1 corners
        # are not corners of the original rectangular component-mass box.
        corners.extend((mass,mass) for mass in (max(a[0],b[0]),min(a[1],b[1])))
    mcs = [chirp_mass(dict(m1=x,m2=y)) for x,y in corners]
    qs = [y/x for x,y in corners]
    masses = set(corners)
    for mc,q in itertools.product(np.linspace(min(mcs),max(mcs),n),np.linspace(min(qs),max(qs),n)):
        total = mc*((1+q)**2/q)**.6
        m1,m2 = total/(1+q),total*q/(1+q)
        if a[0]-1e-9 <= m1 <= a[1]+1e-9 and b[0]-1e-9 <= m2 <= b[1]+1e-9 and m1>=m2:
            masses.add((round(float(m1),10),round(float(m2),10)))
    for m1,m2 in itertools.product(np.linspace(*a,3),np.linspace(*b,3)):
        if m1>=m2 and chirp_mass(dict(m1=m1,m2=m2))>=cfg['mchirp_ref']:
            masses.add((round(float(m1),10),round(float(m2),10)))
    result = []
    for (m1,m2),s in itertools.product(sorted(masses),(-spin,0.,spin)):
        result.append(dict(m1=m1,m2=m2,chi1z=s,chi2z=0. if name=='nsbh' else s,
                           distance_mpc=100.,inclination=1.,phi_ref=0.,t_c=0.,
                           f_ref=20.,F_plus=.6,F_cross=.5))
    return result


def phase_grid(bands, oversampling=8):
    """Fine chirp-phase grid: Newtonian phase step <=2pi/oversampling.

    Uniform f^(-5/3) resolves the steep low-frequency phase; an additional
    8192-point log grid resolves merger/mixing features at high frequency.
    Return a strictly increasing grid spanning all window supports.
    """
    lo,hi = bands.f_low,bands.f_b[-1]
    tau = bands.tau(lo)
    cycles = .6*tau*lo*(1-(hi/lo)**(-5/3))
    n = max(4096,int(np.ceil(oversampling*cycles))+1)
    chirp = np.linspace(hi**(-5/3),lo**(-5/3),n)**(-3/5)
    f = np.sort(np.r_[chirp,np.geomspace(lo,hi,8192),bands.f_b,
                       bands.f_b-bands.Df_b])
    # Near-coincident insertions can spoil nonuniform finite differences.
    f = f[np.r_[True,np.diff(f)>1e-8]]
    return f[(f>=lo-1e-9)&(f<=hi+1e-9)]


def _probe_55_in_child():
    """Try an unsupported mode only in a disposable child (LAL may segfault)."""
    model = XHMModel(30,5,0,0,100,1,0,0)
    seq = lal.CreateREAL8Sequence(16)
    seq.data[:] = np.linspace(30,40,16)
    params = model._params(((2,2),(5,5)))
    try:
        h = lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
            seq,30*lal.MSUN_SI,5*lal.MSUN_SI,0,0,5,-5,1e6*lal.PC_SI,0,20,params)
        available = bool(np.any(abs(h.data.data)>0))
        return available, ('nonzero OneMode output' if available else 'OneMode returned zero 55')
    except (RuntimeError,ValueError) as exc:
        return False, str(exc)


def probe_55():
    """Isolate the installed OneMode 55 probe, including fatal native errors."""
    code = ("import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
            "from scripts.prior_slack_iter008 import _probe_55_in_child; "
            "import json; print(json.dumps(_probe_55_in_child()))")
    # The helper import resolves iter006_common through scripts on PYTHONPATH.
    env = {**os.environ,'PYTHONPATH':str(ROOT/'scripts')+os.pathsep+str(ROOT)}
    child = subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,
                           capture_output=True,text=True,timeout=30)
    if child.returncode:
        return False, f'OneMode 55 child exited {child.returncode}; unsupported native call'
    return tuple(json.loads(child.stdout.strip().splitlines()[-1]))


def raw_with_55(model,f):
    """Raw XHM modes plus 55 only after the capability probe succeeded."""
    raw = model.raw_modes(f)
    _,seq,inv = model._sequence(f)
    h = lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        seq,model.m1*lal.MSUN_SI,model.m2*lal.MSUN_SI,model.chi1z,model.chi2z,
        5,-5,1e6*lal.PC_SI,model.phi_ref,model.f_ref,model._params(((2,2),(5,5))))
    raw[5,5] = h.data.data[inv]
    return raw


def audit_config(name, duration_kind, mmax, available55):
    cfg = population_setup(name)
    params = {k:cfg[k] for k in ('T','dt','f_low','f_high','durations','mchirp_ref','m_max')}
    params['m_max'] = mmax
    if duration_kind=='skipped':
        params['durations'] = [cfg['T']/4**j for j in range(4)]
    bands = BandSetup(**params,L=5)
    # Pairwise uses no waveforms at construction. Include m=5 in its
    # geometry even when the installed waveform model cannot produce it.
    modes = XHM_MODES + (((5,5),) if mmax==5 else ())
    ev = pairwise(bands,population_psd('lines',cfg),modes,mchirp_range=cfg['mchirp_range'])
    placements = [dict(r,f_lo=float(bands.f_b[r['b']]-bands.Df_b[r['b']]),
                       f_hi=float(bands.f_b[r['b']+1])) for r in ev.summary()]
    f = phase_grid(bands)
    grid = prior_grid(name)
    detailed, ratio_rows, coverage = [], [], []
    convergence = []
    for index,p in enumerate(grid):
        model = XHMModel(**p)
        raw = raw_with_55(model,f) if mmax==5 and available55 else model.raw_modes(f)
        tracks = phase_time_tracks(f,raw)
        measured = measured_slack(f,tracks,placements)
        if mmax==5 and not available55:
            predicted = {lm:tau_0pn(2*f/lm[1],model.mchirp) for lm in modes}
            measured += [dict(r,source='Newtonian fallback') for r in measured_slack(f,predicted,placements)
                         if r['lag_class'] != 0 and 5 in r['lag_class']]
        for r in measured:
            r.setdefault('source','XHM phase')
            detailed.append(dict(index=index,Mc=model.mchirp,q=model.m2/model.m1,
                                 chi_eff=(model.m1*model.chi1z+model.m2*model.chi2z)/(model.m1+model.m2),**r))
        for ib in range(1,bands.B):
            edges = (bands.f_b[ib]-bands.Df_b[ib],bands.f_b[ib+1])
            for lm,track in tracks.items():
                vals = np.interp(edges,f,track)
                newton = tau_0pn(2*np.array(edges)/lm[1],model.mchirp)
                for edge,value,pred in zip(('lo','hi'),vals,newton):
                    if np.isfinite(value):
                        ratio_rows.append(dict(index=index,b=ib,mode=lm,edge=edge,ratio=float(value/pred),
                                               tau_measured=float(value),tau_newton=float(pred)))
            for slow,fast in ev.classes[1:]:
                if slow==5 and not available55:
                    continue
                for a,c in itertools.product(tracks,tracks):
                    if (a[1],c[1]) != (slow,fast):
                        continue
                    values = np.interp(edges,f,np.asarray(tracks[a])-np.asarray(tracks[c]))
                    pred = bands.lag_pair(np.array(edges),slow,fast,model.mchirp)
                    for edge,value,newton in zip(('lo','hi'),values,pred):
                        if np.isfinite(value):
                            ratio_rows.append(dict(index=index,b=ib,lag_class=(slow,fast),edge=edge,
                                                   ratio=float(value/newton)))
        # A resolution control at one grid point per configuration compares
        # the 8- and 16-sample-per-cycle derivative tracks on common f.
        if index == len(grid)//2:
            f2 = phase_grid(bands,16)
            raw2 = raw_with_55(model,f2) if mmax==5 and available55 else model.raw_modes(f2)
            tr2 = phase_time_tracks(f2,raw2)
            for lm,t in tracks.items():
                t2 = np.interp(f,f2,tr2[lm])
                newton = tau_0pn(2*f/lm[1],model.mchirp)
                mask = np.isfinite(t)&np.isfinite(t2)&(f<bands.f_b[-2])
                if np.any(mask):
                    convergence.append(dict(mode=lm,max_difference_over_newton=float(np.max(abs(t[mask]-t2[mask])/newton[mask]))))
        missing = len(placements)-len(measured)
        coverage.append(dict(index=index,missing_classes=missing,
                             Mc_above_prior_corner=model.mchirp>cfg['mchirp_range'][1]))
    aggregates = []
    for row in placements:
        rr = [r for r in detailed if r['b']==row['b'] and r['lag_class']==row['lag_class']]
        if not rr:
            continue
        worst = min(rr,key=lambda r:r['slack'])
        aggregates.append(dict(b=row['b'],lag_class=row['lag_class'],source=worst['source'],
                               full_grid=row['full_grid'],n=len(rr),negative=sum(r['negative'] for r in rr),
                               worst_slack=worst['slack'],worst_index=worst['index'],
                               measured_lo=min(r['measured_lo'] for r in rr),
                               measured_hi=max(r['measured_hi'] for r in rr),
                               block_start=row['block_start'],block_end=row['block_end'],
                               requested_slack=row['slack']))
    valid = [r for r in detailed if not r['full_grid'] and r['source']=='XHM phase']
    lagratios = [r['ratio'] for r in ratio_rows if 'lag_class' in r]
    tauratio = [r['ratio'] for r in ratio_rows if 'mode' in r]
    worst = min(valid,key=lambda r:r['slack'])
    result = dict(population=name,durations=duration_kind,m_max=mmax,n_grid=len(grid),n_f=len(f),K_MB=bands.K_MB,
                  prior=cfg['mchirp_range'],worst_measured_slack=worst['slack'],worst_point=worst,
                  negative_measured=sum(r['negative'] for r in valid),
                  lag_ratio_range=[min(lagratios),max(lagratios)],
                  tau_ratio_range=[min(tauratio),max(tauratio)],
                  lag_ratio_median=float(np.median(lagratios)),
                  Mc_above_corner=sum(r['Mc_above_prior_corner'] for r in coverage),
                  per_class=aggregates,convergence=convergence)
    print(name,duration_kind,mmax,'points',len(grid),'worst slack',worst['slack'],flush=True)
    return result, dict(grid=grid,slack=detailed,ratios=ratio_rows,coverage=coverage)


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(map(str,r))+' |' for r in rows]


def main():
    if '--reuse' in sys.argv:
        # Re-render an existing audit without repeating waveform generation.
        cached = json.loads((ROOT/'data/prior_slack_iter008.json').read_text())
        available55, reason55 = cached['mode55_available'],cached['mode55_probe']
        summaries = cached['results']
        full = json.loads((ROOT/'data/prior_slack_iter008_full.json').read_text())
    else:
        available55, reason55 = probe_55()
        summaries,full = [],[]
        for name in ('bns','nsbh','bbh'):
            for duration in (('dyadic','skipped') if name=='bns' else ('dyadic',)):
                for mmax in (4,5):
                    result,detail = audit_config(name,duration,mmax,available55)
                    summaries.append(result)
                    full.append(dict(population=name,durations=duration,m_max=mmax,**detail))
    for summary, detail in zip(summaries,full):
        ratios = [r['ratio'] for r in detail['ratios'] if 'lag_class' in r and r['edge']=='lo']
        summary['lower_edge_lag_ratio_range'] = [min(ratios),max(ratios)]
    write_json('prior_slack_iter008_full.json',full)
    # Keep the committed summary below ~100 KB; detailed dumps remain ignored.
    (ROOT/'data/prior_slack_iter008.json').write_text(json.dumps(
        dict(mode55_available=available55,mode55_probe=reason55,results=summaries),
        separators=(',',':'),allow_nan=False)+'\n')
    lines = ['# Iteration 008: model-phase prior slack audit','',
             'XHM, paper L=5, iteration-007 priors. A 5x5 (Mc,q) grid clipped to the component-mass '
             'boxes, augmented with their 3x3 boundary mesh; three effective-spin values at each mass point. '
             'BNS: masses [1.2,2] each, aligned equal spins +/-0.05; NSBH: [15,35] x [1.5,4], '
             'chi2=0 and chi1 in {-0.5,0,0.5}; BBH: [30,80] x [10,40], equal spins +/-0.8. '
             'Equal-spin slices span chi_eff but do not test differential-spin dependence at fixed chi_eff. '
             'NSBH points above the nominal upper Mc=9 anchor are retained and counted, as in sample_population. '
             'All parameter points and per-point/class slacks are in prior_slack_iter008_full.json.', '',
             '## Phase convention and resolution',
             'For h=exp(-i Psi), -(1/2pi)d arg(h)/df=t_lm=t_c-tau_lm, not positive time-to-merger. '
             'We differentiate raw t_c=0 modes; tau_lm=+(1/2pi)d arg(h_lm)/df. '
             'Positive class lag is tau_slow-tau_fast=t_fast-t_slow, matching PairwiseHH. '
             'Raw phases are unwrapped on a grid uniform in f^(-5/3) with at least 8 samples per '
             'Newtonian cycle plus 8192 logarithmic frequencies, differentiated using nonuniform '
             'central differences and smoothed with an 11-point cubic Savitzky-Golay filter. '
             'Samples below 1e-6 of each mode\'s peak amplitude are excluded before unwrapping. '
             'Exactly zero modes at equal mass have undefined phase; no measured track is assigned. '
             'The full nonzero supported frequency range is audited, including merger/mixing, '
             'so tiny Newtonian denominators at high frequency can give large or negative ratios. '
             'These ratios are diagnostics, not a claim that Newtonian inspiral tracks describe ringdown. '
             'ToyModes recovery within 2% is a unit test; one midpoint per configuration is also checked '
             'at twice the phase resolution. This audit measures stationary tracks, not window tails.', '',
             '## Mode (5,5) capability',
             f'Installed lalsimulation {lalsim.__version__}: available={available55}. Probe with explicit '
             f'lalParams mode array: {reason55}. '
             'When unavailable, every class involving m=5 uses the Newtonian prediction for BOTH members '
             'and is labelled fallback; these are excluded from measured extrema below. Existing modes '
             'remain XHM-measured on m_max=5 band geometry.', '',
             '## Population extrema (seconds; truncated bands only)']
    lines += table(['Population','durations','m_max','points','K_MB','worst measured slack','negative point/classes','lag ratio min / median / max','lower-edge lag ratio min / max','tau ratio min / max','Mc > upper corner'],[
        [r['population'],r['durations'],r['m_max'],r['n_grid'],r['K_MB'],f"{r['worst_measured_slack']:.6f}",r['negative_measured'],
         f"{r['lag_ratio_range'][0]:.3g} / {r['lag_ratio_median']:.3g} / {r['lag_ratio_range'][1]:.3g}",
         f"{r['lower_edge_lag_ratio_range'][0]:.3g} / {r['lower_edge_lag_ratio_range'][1]:.3g}",
         f"{r['tau_ratio_range'][0]:.3g} / {r['tau_ratio_range'][1]:.3g}",r['Mc_above_corner']] for r in summaries])
    for r,detail in zip(summaries,full):
        lines += ['', f"## {r['population']} / {r['durations']} / m_max={r['m_max']}",
                  f"Fine phase grid: {r['n_f']} points. Worst measured point: "
                  f"Mc={r['worst_point']['Mc']:.5f}, q={r['worst_point']['q']:.5f}, "
                  f"chi_eff={r['worst_point']['chi_eff']:.5f}, band={r['worst_point']['b']}, "
                  f"class={r['worst_point']['lag_class']}. "
                  'Full-grid (band 0) unwrapped slack does not constrain the circular exact quadrature.']
        class_ratios = {}
        for a in r['per_class']:
            values = [v['ratio'] for v in detail['ratios'] if 'lag_class' in v and v['b']==a['b']
                      and v['lag_class']==a['lag_class']]
            class_ratios[a['b'],str(a['lag_class'])] = (
                f'{min(values):.3g} / {max(values):.3g}' if values else 'n/a')
        lines += table(['b','class','source','n','block s','measured envelope s','worst clearance s','negative count','requested margin slack s','edge lag ratio min / max'],[
            [a['b'],a['lag_class'],a['source'],a['n'],f"[{a['block_start']:.5g}, {a['block_end']:.5g})",
             f"[{a['measured_lo']:.5g}, {a['measured_hi']:.5g}]",f"{a['worst_slack']:.6f}",
             ('full grid' if a['full_grid'] else ('NEGATIVE: '+str(a['negative']) if a['negative'] else 0)),
             f"{a['requested_slack']:.5g}",class_ratios[a['b'],str(a['lag_class'])]] for a in r['per_class']])
        lines += ['','Resolution control: max |tau_8-tau_16|/tau_Newton below the start of the final band: '+
                  ', '.join(f"{a['mode']}: {a['max_difference_over_newton']:.3g}" for a in r['convergence'])+'.']
    lines += ['', 'Newtonian versus measured ratios are measured / Newtonian at lower support starts and upper '
              'window edges, using each system\'s true chirp mass. Large endpoint ratios near merger are '
              'not lag-coverage failures by themselves; actual seconds of clearance decide coverage. '
              'A positive stationary-track clearance does not bound finite-window tails or guarantee an HH '
              'error at a chosen SNR. Use the separate fine-grid HH study for accuracy. '
              'Negative measured clearances, if present, must be addressed by model-aware prior envelopes '
              'or extra margins; band 0 is exempt. No waveform-phase claim is made for unavailable 55.']
    (ROOT/'data/prior_slack_iter008.md').write_text('\n'.join(lines)+'\n')
    print('wrote data/prior_slack_iter008.md',flush=True)


if __name__=='__main__':
    with quiet_lal():
        main()
