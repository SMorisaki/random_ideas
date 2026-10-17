"""Iteration 007 population evidence (artifact names iter007)."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from mbquad.population import (METHODS, SEED, PopulationCache, chirp_mass, evaluate_point,
    make_bands, make_evaluators, population_lines, population_psd, population_setup, sample_population)
from iter006_common import quiet_lal


def compact(obj):
    """Six significant digits in committed summaries; full precision stays ignored."""
    if isinstance(obj,float):
        return float(f'{obj:.6g}')
    if isinstance(obj,dict):
        return {k:compact(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):
        return [compact(v) for v in obj]
    return obj


def summarize(points):
    """Percentiles of absolute quadratic-term lnLambda error at SNR25."""
    result = {}
    subsets = ['regular','strong_hom','combined']
    if any(p['above_upper_corner'] for p in points):
        subsets.append('regular_inside_corners')
    for subset in subsets:
        if subset == 'regular_inside_corners':
            pp = [p for p in points if p['subset']=='regular' and not p['above_upper_corner']]
        else:
            pp = points if subset == 'combined' else [p for p in points if p['subset']==subset]
        if not pp:
            continue
        result[subset] = {}
        for L in ('5','50'):
            rows = {}
            for method in METHODS:
                values = [p['errors'][L][method] for p in pp]
                q = np.percentile(values,[50,90,99,100])
                rows[method] = dict(zip(('median','p90','p99','max'),map(float,q)))
                rows[method]['median_relative'] = float(q[0]/312.5)
                rows[method]['max_relative'] = float(q[3]/312.5)
            for method in ('pairwise','B1','B1_all'):
                rows[method]['ratio_median'] = rows[method]['median']/rows['IFFT-FFT']['median']
                rows[method]['ratio_p99'] = rows[method]['p99']/rows['IFFT-FFT']['p99']
            result[subset][L] = rows
    return result


def run_population(name, psd_name, n):
    start = perf_counter()
    setup = population_setup(name)
    regular = sample_population(name,n,SEED)
    strong = sample_population(name,20,SEED,strong_hom=True) if name!='bns' else []
    params = regular+strong
    p = population_psd(psd_name,setup)
    cache = PopulationCache(ROOT/f'data/population_cache_{name}_{psd_name}.npz',p)
    bands = {L:make_bands(setup,L) for L in (5,50)}
    with quiet_lal():
        ev = {L:make_evaluators(params[0],setup,p,b) for L,b in bands.items()}
    geometry = {}
    for L,b in bands.items():
        slack = ev[L]['pairwise'].summary()
        geometry[str(L)] = dict(K_MB=b.K_MB, bands=b.summary_table(),
            B1_modes=ev[L]['B1'].modes, B1_all_modes=ev[L]['B1_all'].modes,
            min_slack_all=min(r['slack'] for r in slack),
            min_slack_truncated=min(r['slack'] for r in slack if not r['full_grid']),
            negative_slack=[r for r in slack if r['slack']<0])
    timings = {L:{} for L in bands}
    points, full, exact_times = [], [], []
    # Warm the adapters and all HH kernels before collecting timings.
    with quiet_lal():
        for L in bands:
            evaluate_point(params[0],{**setup,'exact_hh':1.},p,bands[L],ev[L],L)
    for i,pars in enumerate(params):
        tick = perf_counter()
        with quiet_lal():
            exact = cache.get(i,pars)
        exact_times.append(perf_counter()-tick)
        row = dict(index=i,subset='regular' if i<n else 'strong_hom',params=pars,
                   mchirp=chirp_mass(pars), above_upper_corner=chirp_mass(pars)>setup['mchirp_range'][1],
                   errors={}, min_slack={})
        relative = {}
        with quiet_lal():
            for L,b in bands.items():
                rel = evaluate_point(pars,{**setup,'exact_hh':exact,'timings':timings[L]},p,b,ev[L],L)
                relative[str(L)] = rel
                row['errors'][str(L)] = {m:312.5*abs(x) for m,x in rel.items()}
                row['min_slack'][str(L)] = geometry[str(L)]['min_slack_truncated']
        points.append(row)
        full.append(dict(**row,exact_hh=exact,signed_relative=relative))
        if (i+1)%10==0 or i+1==len(params):
            print(f'{name} {psd_name}: {i+1}/{len(params)} ({perf_counter()-start:.1f} s)',flush=True)
    result = dict(population=name,psd=psd_name,seed=SEED,n=n,n_strong=len(strong),setup=setup,
                  rejections=dict(regular=regular.rejections,strong_hom=getattr(strong,'rejections',{})),
                  proposals=dict(regular=regular.proposals,strong_hom=getattr(strong,'proposals',0)),
                  above_upper_corner_indices=[r['index'] for r in points if r['above_upper_corner']],
                  geometry=geometry,points=points,statistics=summarize(points),
                  SNR25_factor=312.5,SNR50_multiplier=4,SNR100_multiplier=16,
                  cache_hits=cache.hits,cache_misses=cache.misses,
                  wall_seconds=perf_counter()-start,refusals=[],
                  timing_median_seconds={str(L):{k:float(np.median(v)) for k,v in tt.items()}
                                         for L,tt in timings.items()},
                  exact_lookup_median_seconds=float(np.median(exact_times)))
    target = ROOT/f'data/population_iter007_{name}_{psd_name}.json'
    target.write_text(json.dumps(compact(result),separators=(',',':'),allow_nan=False)+'\n')
    target.with_name(target.stem+'_full.json').write_text(json.dumps(dict(result=result,points=full),allow_nan=False))
    print(f'Wrote {target.name}: {target.stat().st_size} bytes',flush=True)
    return result


def write_summary():
    files = [ROOT/f'data/population_iter007_{pop}_{psd}.json'
             for pop in ('bns','nsbh','bbh') for psd in ('lines','L1')]
    results = [json.loads(f.read_text()) for f in files if f.exists()]
    lines = ['# Iteration 007: XHM population accuracy', '',
        'Seed 20261017, independent SeedSequence streams by population/subset, prefix-stable. '
        'Regular priors are component-mass boxes conditioned on Mc >= Mc_ref and m1 >= m2; '
        'BBH also q >= 0.1. Strong-HOM draws condition the same mass/spin prior on q <= 0.15 '
        'and Mc <= Mc_max, and the isotropic inclination prior on 1.3 < iota < 1.8 '
        '(20 additional NSBH/BBH points). '
        'All angles, spins and t_c vary. Distance = 100 Mpc; antenna norm = 1.', '',
        'Exact reference: XHMModel.total_direct, full positive-frequency sequence, Eq. (4), '
        'DC/Nyquist excluded. Scalar exact norms cached per population/index/PSD independently '
        'of L; only one fine waveform is held at a time. Coarse modes retain iteration-005 '
        'Mf>0.3 tail recovery. All six methods use unchanged defaults; pairwise uses specified '
        'prior corners; B1_all is p=1, all_half_bins=True with adaptive shortcuts.', '',
        'Numbers below are |Delta ln Lambda| = 312.5 |Delta HH|/HH at SNR25, with the '
        'linear term held fixed; this is the quadratic-term Fig. 4 diagnostic. '
        'At SNR50 multiply by 4; at SNR100 by 16. Divide by 312.5 for relative HH error. '
        'Percentiles use NumPy linear interpolation. Separate regular, strong-HOM, and combined '
        'tables prevent the 20 stress points from changing the regular population quantiles. '
        'Where regular draws exceed the upper pairwise corner, regular_inside_corners '
        'provides a sensitivity check without changing the primary draws.', '',
        'L1 file: data/psd_L1_GW190814_welch16s.txt, skip_dc=True, log PSD/log frequency '
        'interpolation by PSDGrid. Its 1/16 Hz resolution is interpolated to 1/256 Hz for BNS; '
        'unresolved measured lines are not recovered. Synthetic PSD: S_analytic * '
        '[1 + sum h/(1+((f-f0)/gamma)^2)], gamma is HWHM. The same line list is used for '
        'both L values and every population. One line lies in each BNS L=5 band and one '
        'within Df of each internal BNS edge at each L.', '',
        '| f0 (Hz) | HWHM (Hz) | height / baseline |', '|---:|---:|---:|']
    lines += [f'| {f:.9f} | {w:g} | {h:g} |' for f,w,h in population_lines()]
    lines += ['', '## Setup checks', '',
              '| Population / PSD | n + strong | rejections regular / strong (Mc low, Mc high, q) | above Mc upper corner | wall s |',
              '|---|---:|---|---|---:|']
    for r in results:
        rej = r['rejections']
        lines.append(f"| {r['population']} / {r['psd']} | {r['n']} + {r['n_strong']} | "
                     f"{rej['regular']} / {rej['strong_hom']} | {r['above_upper_corner_indices']} | {r['wall_seconds']:.1f} |")
    lines += ['', 'The NSBH mass box reaches Mc=9.32103 Msun whereas the requested pairwise '
              'upper anchor is 9.0. Draws are retained and flagged above; these test the specified '
              'corners but are outside their upper coverage assumption. No Eq. (51) root failed '
              'and no reference mass or duration was adjusted. Negative requested slack occurs '
              'only in the full-grid band (exact for any circular block); no truncated-band '
              'slack was clipped. Full flags are in each JSON geometry; flagged point '
              'parameters and chirp masses are in the per-point JSON records.', '',
              '| Population | L | K_MB | min slack all / truncated (s) | adaptive B1 band modes |',
              '|---|---:|---:|---:|---|']
    seen_populations=set()
    for r in results:
        if r['population'] in seen_populations:
            continue
        seen_populations.add(r['population'])
        for L,g in r['geometry'].items():
            lines.append(f"| {r['population']} | {L} | {g['K_MB']} | "
                         f"{g['min_slack_all']:.6g} / {g['min_slack_truncated']:.6g} | {g['B1_modes']} |")
    lines += ['', '## Error distributions', '',
              '| Population | PSD | subset | L | method | median | p90 | p99 | max | median relative HH | max relative HH |',
              '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for subset, ls in r['statistics'].items():
            for L,methods in ls.items():
                for m,s in methods.items():
                    lines.append(f"| {r['population']} | {r['psd']} | {subset} | {L} | {m} | "
                                 + ' | '.join(f'{s[k]:.3e}' for k in ('median','p90','p99','max','median_relative','max_relative'))+' |')
    lines += ['', '## Ratios to IFFT-FFT and criteria', '',
              'C1: pairwise median <= 1 and p99 <= 2. C1\': B1 median and p99 <= 1.5. '
              'Ratios compare distribution statistics, not per-point ratios. Each row must '
              'pass for a population-wide claim. Strong-HOM is also assessed separately.', '',
              '| Population | PSD | subset | L | method | median ratio | p99 ratio | criterion met |',
              '|---|---|---|---:|---|---:|---:|---|']
    for r in results:
        for subset,ls in r['statistics'].items():
            for L,methods in ls.items():
                for m in ('pairwise','B1','B1_all'):
                    s=methods[m]
                    ok=s['ratio_median'] <= (1 if m=='pairwise' else 1.5) and s['ratio_p99'] <= (2 if m=='pairwise' else 1.5)
                    lines.append(f"| {r['population']} | {r['psd']} | {subset} | {L} | {m} | "
                                 f"{s['ratio_median']:.3f} | {s['ratio_p99']:.3f} | {'yes' if ok else 'NO'} |")
    lines += ['', 'Empirical failures on the regular and strong-HOM subsets:']
    for method in ('pairwise','B1','B1_all'):
        failures=[]
        for r in results:
            for subset in ('regular','strong_hom'):
                for L, methods in r['statistics'].get(subset,{}).items():
                    s=methods[method]
                    if s['ratio_median'] > (1 if method=='pairwise' else 1.5) or s['ratio_p99'] > (2 if method=='pairwise' else 1.5):
                        failures.append(f"{r['population']}/{r['psd']}/{subset}/L={L}")
        lines += ['', f"{method}: "+(', '.join(failures) if failures else 'none')+'.']
    lines += ['', '## Observed warm per-evaluation timings', '',
              'OMP_NUM_THREADS=1; OMP_WAIT_POLICY=PASSIVE; scipy FFT workers=1. '
              'Medians across population evaluations after one complete untimed warm-up; '
              'waveform and HH steps timed separately (no reference or setup cost). '
              'These are throughput diagnostics with varying parameters, not a repeated fixed-system '
              'microbenchmark. Exact lookup times include cold reference calls on cache misses.', '',
              '| Population | PSD | L | waveform total / modes (ms) | linear / IFFT-FFT / pair / B1 / B1_all / B0 HH (ms) |',
              '|---|---|---:|---|---|']
    for r in results:
        for L,tt in r['timing_median_seconds'].items():
            lines.append(f"| {r['population']} | {r['psd']} | {L} | "
                         f"{1e3*tt['coarse_total']:.4f} / {1e3*tt['coarse_modes']:.4f} | "
                         + ' / '.join(f'{1e3*tt[m]:.4f}' for m in METHODS)+' |')
    lines += ['', 'No LAL-refused parameter sets. All populations used the full requested n=100 '
              'unless a command-line override is shown above. Reported run wall times are from '
              'the final run; cache hit/miss counts are recorded in the JSON. '
              'See tc_scan_iter007.md and figures/iter007_tc_scan.png for the 21-point time scan. '
              'Pairwise peak-to-peak relative variation is at most 3.94e-16; linear is '
              'also invariant because it uses power. Per-point signed relative errors '
              'and exact norms are retained in *_full.json; committed JSON uses six significant '
              'digits and remains below 100 KB. Figures show index scatters and empirical CDFs.']
    (ROOT/'data/population_iter007.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pop',choices=['bns','nsbh','bbh','all'],default='all')
    parser.add_argument('--psd',choices=['lines','L1','both'],default='both')
    parser.add_argument('--n',type=int,default=100)
    args=parser.parse_args()
    if args.n<1:
        parser.error('--n must be positive')
    for pop in ('bns','nsbh','bbh') if args.pop=='all' else [args.pop]:
        for psd in ('lines','L1') if args.psd=='both' else [args.psd]:
            run_population(pop,psd,args.n)
            write_summary()


if __name__=='__main__':
    main()
