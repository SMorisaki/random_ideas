"""Scan the common time phase on two seeded strong-HOM XHM points."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
os.environ['MPLCONFIGDIR']='/tmp/mplconfig'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from mbquad.population import (METHODS,SEED,PopulationCache,make_bands,make_evaluators,
                               population_psd,population_setup,sample_population)
from iter006_common import quiet_lal


def main():
    rows=[]
    parameter_notes=[]
    lines=['# Iteration 007: common coalescence-time phase scan', '',
        'First seeded strong-HOM point in each NSBH/BBH stream (seed 20261017), '
        'both PSDs, L=5 and L=50. All intrinsic and other extrinsic parameters fixed. '
        '21 equally spaced t_c in [T-2.12,T-1.9]. Each row is HH(t_c)/HH(first)-1; '
        'the exact HH is time independent. Phase is recomputed by XHMCoarseEvaluator '
        'at every time. All mode tails are retained.', '',
        'The common phase cancels analytically in every mode product and in total '
        '|h|^2. Thus pairwise, B0 AND linear interpolation must be constant to '
        'roundoff. Only methods that interpolate the complex waveform (IFFT-FFT, '
        'B1 in bands actually using FFTs) can vary. At L=50 adaptive B1 uses B0 '
        'and is constant too. The expectation that linear varies is incompatible '
        'with its Eqs. (27)-(30) power-only formula.', '',
        '| Population | PSD | L | method | peak-to-peak / HH(first) | baseline signed relative error to exact |',
        '|---|---|---:|---|---:|---:|']
    table=['', '## Scan values: HH(t_c)/HH(first)-1', '',
           '| Population | PSD | L | t_c (s) | pairwise | IFFT-FFT | B1 | B1_all | linear | B0 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    colors=dict(zip(METHODS,plt.get_cmap('tab10').colors[:6]))
    for ipop,pop in enumerate(('nsbh','bbh')):
        cfg=population_setup(pop)
        params=sample_population(pop,1,SEED,strong_hom=True)[0]
        times=np.linspace(cfg['T']-2.12,cfg['T']-1.9,21)
        parameter_notes += [f"\n{pop} parameters: `{json.dumps(params,sort_keys=True)}`\n"]
        for ipsd,psd_name in enumerate(('lines','L1')):
            p=population_psd(psd_name,cfg)
            cache=PopulationCache(ROOT/f'data/population_cache_{pop}_{psd_name}.npz',p)
            with quiet_lal():
                exact=cache.get('tc_scan_strong0',params)
                for L in (5,50):
                    b=make_bands(cfg,L)
                    ev=make_evaluators(params,cfg,p,b)
                    norms={m:[] for m in METHODS}
                    for tc in times:
                        pars={**params,'t_c':float(tc)}
                        co,cm=ev['adapter'].coarse_total(pars),ev['adapter'].coarse_modes(pars)
                        for m in METHODS:
                            norms[m].append(ev[m](cm if m=='pairwise' else co))
                    changes={m:np.array(v)/v[0]-1 for m,v in norms.items()}
                    for m in METHODS:
                        span=float(np.ptp(norms[m])/norms[m][0])
                        baseline=float(norms[m][0]/exact-1)
                        lines.append(f'| {pop} | {psd_name} | {L} | {m} | {span:.6e} | {baseline:+.6e} |')
                        rows.append(dict(population=pop,psd=psd_name,L=L,method=m,params=params,
                                         tc=times.tolist(),hh=norms[m],exact_hh=exact,
                                         peak_to_peak_relative=span,relative_change=changes[m].tolist()))
                        if m=='pairwise' and span>1e-12:
                            raise AssertionError('Pairwise time invariance failed')
                    for i,tc in enumerate(times):
                        table.append(f'| {pop} | {psd_name} | {L} | {tc:.6f} | '
                            +' | '.join(f'{changes[m][i]:+.6e}' for m in
                                       ('pairwise','IFFT-FFT','B1','B1_all','linear','B0'))+' |')
                    if L==5:
                        ax=axes[ipop,ipsd]
                        for m in METHODS:
                            ax.plot(times-cfg['T'],changes[m],label=m,color=colors[m],lw=1.2)
                        ax.set(title=f'{pop.upper()}, {psd_name}, L=5',xlabel=r'$t_c-T$ (s)',
                               ylabel=r'$(HH(t_c)-HH(t_{c,0}))/HH(t_{c,0})$')
                        ax.grid(alpha=.25)
    lines += ['', '## Fixed parameters']+parameter_notes+table
    (ROOT/'data/tc_scan_iter007.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'data/tc_scan_iter007_full.json').write_text(json.dumps(rows,allow_nan=False))
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=6,loc='lower center',frameon=False)
    fig.suptitle('Common time phase: pairwise and power-only methods are invariant')
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.savefig(ROOT/'figures/iter007_tc_scan.png',dpi=170)
    plt.close(fig)
    print('Time scan completed; pairwise invariance passed for both points/PSDs/L values.')


if __name__=='__main__':
    main()
