#!/usr/bin/env python
"""Validate analytic rules on cached references, including pruning and cost tails.

Timings use three warmed compiled loops per point. Per-point records and
historical snapshots live in ignored data/*_full.json; no cached reference
is recomputed. Grid curves include former fallback cases.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final_benchmark import parameter_sets,references,KEYS,cost,write
from check_geometry import repaired_cases
from dpmarg.table1d import load_table
from dpmarg.params import reduce_params
from dpmarg.fast import fast_diagnostics
from dpmarg.method2 import (lnL_marg_m2_table,method2_analytic_diagnostics,
    _analytic_compute,_geometry,_analytic_plan,method2_table_diagnostics,
    method2_analytic_error,ANALYTIC_GH_ORDER,ANALYTIC_WIDTH_SWITCH,
    ANALYTIC_SKIP_MARGIN,ANALYTIC_GRID_SPACING,ANALYTIC_SINH_WIDTH)


@njit(cache=False)
def repeated(p,table,rule,repeats):
    total=0.
    for _ in range(repeats):
        total+=lnL_marg_m2_table(p[0],p[1],p[2],p[3],p[4],p[5],p[6],table,64,rule)
    return total


@njit(cache=False)
def repeated_geometry(p,analytic,repeats):
    total=0.
    pars=reduce_params(p[0],p[1],p[2],p[3],p[4])
    for _ in range(repeats):
        if analytic:
            c,w,b,m,n=_analytic_plan(*pars,p[5],p[6])
        else:
            c,w,b=_geometry(*pars,p[5],p[6])
        total+=c[0]+w[0]+b[0]
    return total


def stats(errors,times):
    return dict(max_error=float(np.max(errors)),p99_error=float(np.quantile(errors,.99)),
        above_1e3=int(np.sum(np.array(errors)>1e-3)),above_1e4=int(np.sum(np.array(errors)>1e-4)),
        fraction_above_1e4=float(np.mean(np.array(errors)>1e-4)),
        median_us=float(np.median(times)),mean_us=float(np.mean(times)),
        p99_us=float(np.quantile(times,.99)))


def aggregate(rows,quick=False):
    result={rule:stats([r['error' if rule=='analytic' else 'scan_error'] for r in rows],
        [r['timing_us'][rule] if not quick else 0. for r in rows]) for rule in ('scan','analytic')}
    result['branches']={str(m):float(np.mean([r['branch']==m for r in rows])) for m in (3,4,5,6)}
    result['two_interval_fraction']=float(np.mean([r['branch']==6 and r['N']==64 for r in rows]))
    result['three_interval_fraction']=float(np.mean([r['branch']==6 and r['N']==96 for r in rows]))
    result['radial_fallbacks_mean']=float(np.mean([r['radial_fallbacks'] for r in rows]))
    result['radial_fallbacks_no_skip_mean']=float(np.mean([r['no_skip_fallbacks'] for r in rows]))
    result['scan_radial_fallbacks_mean']=float(np.mean([r['scan_fallbacks'] for r in rows]))
    result['nodes_skipped_fraction']=sum(r['skipped'] for r in rows)/sum(r['N'] for r in rows)
    result['skip_max_difference']=max(r['skip_difference'] for r in rows)
    result['failures']=[r['name'] for r in rows if r['error']>1e-4]
    return result


def degenerate_report_lines():
    """Include the separately reproducible corner audit in the shared report."""
    path=ROOT/'data/degenerate_check.json'
    if not path.exists():return []
    d=json.loads(path.read_text())
    lines=['','## Degenerate corner (300 grid + 300 random points)','',
        'Random seed 90092; log10(lam1) uniform [3,7], rho=R/sqrt(lam1) log-uniform [1,300], independent 20% atoms at q=0 and eta=0. Prior (0.1,100); grid xi=0.3, random xi uniform [0,pi]. References: lnL_marg_ref, cached in ignored `data/degenerate_reference_full.json`.',
        '', '| Set | Rule | Max error before / after | >1e-4 before / after | >1e-3 before / after |',
        '|---|---|---:|---:|---:|']
    for label,g in d['sets'].items():
        for rule,row in g.items():
            old=d.get('before',{}).get(label,{}).get(rule)
            if old:
                lines.append(f"| {label} | {rule} | {old['max_error']:.6g} / {row['max_error']:.6g} | {old['above_1e4']} / {row['above_1e4']} | {old['above_1e3']} / {row['above_1e3']} |")
    lines+=['','| Set | Rule | Median / mean / p99 us | Branch fractions |',
        '|---|---|---:|---|']
    for label,g in d['sets'].items():
        for rule,row in g.items():
            branches=', '.join(f"{k}: {v:.2%}" for k,v in row['branches'].items() if v)
            lines.append(f"| {label} | {rule} | {row['median_us']:.4g} / {row['mean_us']:.4g} / {row['p99_us']:.4g} | {branches} |")
    lines+=['','Branches 0/1/2: legacy uniform/sinh/split GL; 3/4/5/6: GH/grid/corrected sinh/split GL. The corner scan uses its seeded search with the same fixed quadrature; outside it the legacy rule is retained.',
        '',f"Analytic >1e-4: {d['failures']['analytic'] or 'none'}.",
        '', '| Remaining scan point | lam1 | q | eta | R/lam1 | Error |',
        '|---|---:|---:|---:|---:|---:|']
    for r in d['failures']['scan']:
        lines.append(f"| {r['name']} | {r['lam1']:.6g} | {r['q']:.6g} | {r['eta']:.6g} | {r['R_over_lam1']:.6g} | {r['scan_error']:.6g} |")
    lines+=['','All remaining scan failures are near-degenerate weak-axis / clipped-prior regimes; none exceeds 1e-3.',
        '', '![Degenerate corner accuracy and cost](../figures/degenerate_check.png)']
    return lines


def audit_geometry(tables):
    path=ROOT/'data/geometry_check_full.json'
    if not path.exists():return None
    candidates=[r for r in json.loads(path.read_text()) if r['missed_scan_peaks'] or r['width_ratio']>1.5]
    rows=[];to_reference=[]
    # Reuse every available independent reference, even after a repair
    # moves its scan256 difference below the request threshold. Otherwise
    # successive audits would silently discard their strongest checks.
    cachepath=ROOT/'data/final_reference_full.json'
    known={k:v['value'] for k,v in json.loads(cachepath.read_text())['points'].items()} if cachepath.exists() else {}
    for name in ('regression_points.json','method2_random_reference.json'):
        for c in json.loads((ROOT/'data'/name).read_text()):
            key=','.join(float(c[k]).hex() for k in KEYS)
            known[key]=c['lnL_marg_ref']
    for c in candidates:
        p=np.array(c['parameters']);tab=tables[p[6]]
        a=lnL_marg_m2_table(*p,tab,rule='analytic');b=lnL_marg_m2_table(*p,tab,256,'scan')
        r=dict(name=c['name'],parameters=p.tolist(),analytic=a,scan256=b,difference=abs(a-b))
        key=','.join(float(v).hex() for v in p)
        if key in known:
            r.update(reference=known[key],analytic_error=abs(a-known[key]),scan256_error=abs(b-known[key]))
        rows.append(r)
        if abs(a-b)>8e-5 and key not in known:
            to_reference.append(dict(name=c['name'],**dict(zip(KEYS,p))))
    if to_reference:
        references({'geometry_audit':to_reference})
        byname={c['name']:c['lnL_marg_ref'] for c in to_reference}
        for r in rows:
            if r['name'] in byname:
                r['reference']=byname[r['name']]
                r['analytic_error']=abs(r['analytic']-r['reference'])
                r['scan256_error']=abs(r['scan256']-r['reference'])
    write(ROOT/'data/method2b_geometry_audit_full.json',rows)
    return dict(count=len(rows),max_scan256_difference=max((r['difference'] for r in rows),default=0.),
        reference_checks=sum('reference' in r for r in rows),new_reference_requests=len(to_reference),
        max_reference_error=max((r.get('analytic_error',0.) for r in rows),default=0.),
        above_1e4=sum(r.get('analytic_error',r['difference'])>1e-4 for r in rows),
        worst=sorted(rows,key=lambda r:r.get('analytic_error',r['difference']),reverse=True)[:12])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='accuracy and skip calibration only')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    tables={hi:load_table(ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz') for hi in (100.,10.)}
    if args.audit_only:
        result=audit_geometry(tables);write(ROOT/'data/method2b_geometry_audit.json',result);print(result);return
    sets=parameter_sets();sets['repaired_cases']=repaired_cases();references(sets)
    oldpath=ROOT/'data/method2b_round1_full.json'
    if not oldpath.exists():oldpath=ROOT/'data/method2b_before_full.json'
    old=json.loads(oldpath.read_text()) if oldpath.exists() else {}
    baseline={}
    for label,rows in old.items():
        baseline[label]={rule:stats([r['error' if rule=='analytic' else 'scan_error'] for r in rows],
            [r['timing_us'][rule] for r in rows]) for rule in ('scan','analytic')}
        baseline[label]['radial_fallbacks_mean']=float(np.mean([r['radial_fallbacks'] for r in rows]))
        baseline[label]['branches']={str(m):float(np.mean([r['branch']==m for r in rows])) for m in (3,4,5,6)}
    full={};summary=dict(sets={},before=baseline,configuration=dict(w_c=ANALYTIC_WIDTH_SWITCH,
        N_GH=ANALYTIC_GH_ORDER,M=ANALYTIC_SKIP_MARGIN,grid_spacing_factor=ANALYTIC_GRID_SPACING,
        sinh_width_threshold=ANALYTIC_SINH_WIDTH,grid_sizes=[48,64,96,128],sinh_nodes=64,
        split_nodes_per_interval=32,skip_certificate_relative_bound=float(np.exp(-25)),pedestal_probes=16,
        conservative_width_factor=.8,weak_axis_resolution='include radial RMS and distant-radius cores',
        guarded_quadrature='precomputed log-radius GL12, panels <=.5; analytic rule only'))
    summary['configuration'].update(corner_GH_order=32,corner_width_factor=.25,
        corner_domain='lam2/lam1 < .01 and eta < .1',
        kink_rule='endpoint-corrected sinh64 per significant patch; fixed grid when resolved',
        corner_sinh_stretch='2 for single patch, 3 for multiple patches',
        unequal_width_cut_power=.25,endpoint_stencil_suppression_bound_gap=12,
        prior_dominated_min_grid=64,
        scan_corner_rule='seeded scan search with shared fixed quadrature at default 64-node budget')
    for label,cases in sets.items():
        records=[];oldrows={r['name']:r for r in old.get(label,[])}
        for c in cases:
            p=np.array([c[k] for k in KEYS]);tab=tables[p[6]];ref=c['lnL_marg_ref']
            d=method2_analytic_diagnostics(*p,tab)
            noskip=_analytic_compute(*p,tab,ANALYTIC_GH_ORDER,0,False,np.inf)
            scan=method2_table_diagnostics(*p,tab,64,'scan')
            fast,fastbranch=fast_diagnostics(*p,tab)
            r=dict(name=c['name'],parameters=p.tolist(),reference=ref,analytic=d[0],scan=scan[0],
                error=abs(d[0]-ref),scan_error=abs(scan[0]-ref),branch=int(d[1]),N=int(d[2]),w=d[3],
                radial_calls=int(d[4]),radial_fallbacks=int(d[5]),skipped=int(d[6]),
                no_skip_fallbacks=int(noskip[5]),no_skip_calls=int(noskip[1]),
                scan_fallbacks=int(scan[5]),skip_difference=abs(d[0]-noskip[0]),
                fast_error=abs(fast-ref),fast_branch=int(fastbranch))
            previous=oldrows.get(c['name'],{})
            r['former_fallback']=previous.get('branch')==5
            if not args.quick:
                r['timing_us']={rule:cost(repeated,(p,tab,rule),16) for rule in ('scan','analytic')}
                r['geometry_us']={rule:cost(repeated_geometry,(p,rule=='analytic'),64) for rule in ('scan','analytic')}
                r['gh_errors']={}
                for n in (8,12,16,20,24,32):
                    output=_analytic_compute(*p,tab,n)
                    plan=_analytic_plan(*reduce_params(*p[:5]),p[5],p[6],n)
                    if output[2]==3 and plan[4]==n:r['gh_errors'][str(n)]=abs(output[0]-ref)
                r['grid_errors']={}
                if d[1]==4 or r['former_fallback']:
                    for n in (16,24,32,48,64,96,128):
                        r['grid_errors'][str(n)]=abs(_analytic_compute(*p,tab,12,n)[0]-ref)
                r['doubled_node_difference']=method2_analytic_error(*p,tab)[1]
            records.append(r)
        full[label]=records;summary['sets'][label]=aggregate(records,args.quick)
        g=summary['sets'][label]
        print(label,'max',g['analytic']['max_error'],'failures',g['failures'],
            'skip_delta',g['skip_max_difference'],'skip_fraction',g['nodes_skipped_fraction'],
            'fallbacks',g['radial_fallbacks_no_skip_mean'],'->',g['radial_fallbacks_mean'],flush=True)
    allrows=[r for group in full.values() for r in group]
    summary['overall']=aggregate(allrows,args.quick)
    summary['dispatcher_max_error']=max(r['fast_error'] for r in allrows)
    summary['all_below_1e4']=all(r['error']<=1e-4 for r in allrows)
    summary['all_below_1e3']=all(r['error']<=1e-3 for r in allrows)
    summary['overall_branches']=summary['overall']['branches']
    geompath=ROOT/'data/geometry_check.json'
    if geompath.exists():summary['geometry_agreement']=json.loads(geompath.read_text())
    if not args.quick:
        summary['gh_curve']={str(n):dict(count=sum(str(n) in r['gh_errors'] for r in allrows),
            max_error=max((r['gh_errors'].get(str(n),0.) for r in allrows),default=0.)) for n in (8,12,16,20,24,32)}
        summary['grid_curves']={}
        for domain,predicate in [('all_grid',lambda r:r['branch']==4),
                ('former_fallback',lambda r:r['former_fallback']),
                ('former_fallback_width_ge_threshold',lambda r:r['former_fallback'] and r['w']>=ANALYTIC_SINH_WIDTH)]:
            rows=[r for r in allrows if predicate(r)]
            summary['grid_curves'][domain]={str(n):dict(count=len(rows),
                max_error=max((r['grid_errors'][str(n)] for r in rows),default=0.),
                above_1e4=sum(r['grid_errors'][str(n)]>1e-4 for r in rows)) for n in (16,24,32,48,64,96,128)}
        summary['spacing_curve']={}
        for factor in (.5,.7,1.,1.1,1.2,1.3,1.4):
            pairs=[(r['grid_errors'][str(n)],r['name'],n) for r in allrows if r['former_fallback']
                for n in (48,64,96,128) if np.pi/n<=factor*r['w']]
            summary['spacing_curve'][str(factor)]=dict(count=len(pairs),
                max_error=max((e[0] for e in pairs),default=0.),above_1e4=sum(e[0]>1e-4 for e in pairs))
        summary['geometry_us']={rule:dict(median=float(np.median([r['geometry_us'][rule] for r in allrows])),
            p99=float(np.quantile([r['geometry_us'][rule] for r in allrows],.99))) for rule in ('scan','analytic')}
        summary['max_doubled_node_difference']=max(abs(r['doubled_node_difference']) for r in allrows)
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        for label,rule in [('scan','scan'),('analytic before','before'),('analytic after','analytic')]:
            values=([r['timing_us']['analytic'] for group in old.values() for r in group] if rule=='before'
                else [r['timing_us'][rule] for r in allrows])
            if values:
                values=np.sort(values);ax.semilogx(values,np.arange(1,len(values)+1)/len(values),label=label)
        ax.set(xlabel='Warmed cost (us)',ylabel='Cumulative fraction');ax.legend()
        fig.savefig(ROOT/'figures/method2b_cost.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        for domain,curve in summary['grid_curves'].items():
            ax.semilogy([int(n) for n in curve],[v['max_error'] for v in curve.values()],'o-',label={'all_grid':'Selected grid cases','former_fallback':'All former sinh cases',
                    'former_fallback_width_ge_threshold':'Former sinh cases, w >= 0.017'}[domain])
        ax.axhline(1e-4,color='gray',ls=':');ax.legend();ax.set(xlabel='Periodic grid nodes',ylabel='Worst absolute log error')
        fig.savefig(ROOT/'figures/method2b_error_vs_nodes.png',dpi=160);plt.close(fig)
    summary['geometry_audit']=audit_geometry(tables)
    guardpath=ROOT/'data/method2b_guard_check.json'
    if guardpath.exists():summary['guard_quadrature_validation']=json.loads(guardpath.read_text())
    summary['cost_targets_met']={label:all(g['analytic'][k]<g['scan'][k] for k in ('mean_us','p99_us')) for label,g in summary['sets'].items()}
    summary['elapsed_s']=time.perf_counter()-start
    write(ROOT/'data/method2b_check_full.json',full)
    write(ROOT/'data/method2b_check.json',summary)
    lines=['# Method 2b validation: degenerate-corner and clipping-cusp repair','',
        f'M={ANALYTIC_SKIP_MARGIN:g}; grid spacing factor {ANALYTIC_GRID_SPACING:g}; residual sinh w<{ANALYTIC_SINH_WIDTH:g}; GH12 (GH32 in the corner), grid48/64/96/128, corrected sinh64, split GL32 per interval.',
        'Cached brute-force references; absolute errors in ln L. Costs: median / mean / p99 us across per-point medians of three warmed compiled loops of 16 repetitions.',
        '', '| Set | Rule | Max error | Fraction >1e-4 | Median / mean / p99 us |', '|---|---|---:|---:|---:|']
    for label,g in summary['sets'].items():
        for rule,row in [('scan',g['scan']),('analytic before',baseline.get(label,{}).get('analytic')),('analytic after',g['analytic'])]:
            if row:lines.append(f"| {label} | {rule} | {row['max_error']:.6g} | {row['fraction_above_1e4']:.3%} | {row['median_us']:.4g} / {row['mean_us']:.4g} / {row['p99_us']:.4g} |")
    lines+=['','Before branches (GH / grid / sinh / split GL):','',*['- '+label+': '+' / '.join(f"{g['branches'][str(m)]:.3%}" for m in (3,4,5,6)) for label,g in baseline.items()],
        '', '| Set | GH | Grid | Sinh | Split GL32 | Mean radial fallbacks: scan / old / no skip / skip | Nodes skipped | Max skip delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in summary['sets'].items():
        lines.append('| '+label+' | '+' | '.join(f"{g['branches'][str(m)]:.3%}" for m in (3,4,5,6))+
            f" | {g['scan_radial_fallbacks_mean']:.4g} / {baseline.get(label,{}).get('radial_fallbacks_mean',0):.4g} / {g['radial_fallbacks_no_skip_mean']:.4g} / {g['radial_fallbacks_mean']:.4g} | {g['nodes_skipped_fraction']:.3%} | {g['skip_max_difference']:.3g} |")
    o=summary['overall']
    lines+=['',f"Overall analytic median / mean / p99: {o['analytic']['median_us']:.4g} / {o['analytic']['mean_us']:.4g} / {o['analytic']['p99_us']:.4g} us; max skip delta {o['skip_max_difference']:.3g}; skipped {o['nodes_skipped_fraction']:.3%} of nodes.",
        'Proxy pruning first requires a weighted gap > M. A separate upper/lower integral bound certifies a summed relative loss <= exp(-25)=1.39e-11, regardless of proxy width-factor error. The maximum retained weighted node supplies the lower bound; every skipped node gets 1/N of the error budget. Both signs are included in the upper bound.',
        'Widths for non-GH rules have a calibrated 0.8 safety factor (0.25 for q<0.01, eta<0.1); weak-axis resolution includes the prior RMS radius and distant-radius cores. Kinks use corrected sinh64 patches, with cuts weighted by the fourth root of widths when their ratio exceeds 10. Corner sinh scales are twice the conservative grid widths (three times for multiple patches). Smooth overlapping peaks outside the corner use GL32 intervals.',
        'Kink width is min(one-sided curvature widths, sqrt(b)/abs(a_prime-b_prime*x0)). GH is forbidden for kinks. Candidate mass includes finite radial width; a=b=0 has pair mass 2*(xmin^-3-xmax^-3)/3 and supplies no singular endpoint candidate. Significant xmax shoulders are resolved even when they are not maxima.',
        'For lambda2=eta=0, R^2<=lambda1, the exact null-axis maximum uses finite pair curvature (lambda1-R^2)*<r^2>_prior. Four additional independent reference regressions cover this sub-unit-SNR limit.',
        'Outside the corner, r*sqrt(b)<6 invalidates the isolated radial-saddle width. For these formal proxy cusps, quadrature retains the broad prior/one-sided scale, forbids GH, and uses at least 64 grid nodes. This removes artificial narrow patches and resolves the audited prior-tail features.',
        'Corner GH requires the peak to lie at least 20 widths from the weak axis. A full-Laplace refinement of the closed-form stationary seed retains radial-volume islands missing from the continuous proxy maxima. The default scan comparison retains its seeded scan search and shares fixed corner quadrature to avoid false nested convergence.',
        'Real pedestal mass cannot be pruned. An optional precomputed composite GL12 rule in log radius accelerates unsafe-cell quadrature for the analytic rule on the validated priors; other domains retain the general direct kernel. Fallback counts still include these direct quadrature calls.',
        'The no-skip comparison uses identical plans/nodes. Historical before costs are from the saved iteration-009 run, so scan/after on this run provide the direct cost comparison.']
    if not args.quick:
        for domain,curve in summary['grid_curves'].items():
            lines+=['',f'Grid error curves: {domain}','', '| N | Count | Max error | >1e-4 |','|---:|---:|---:|---:|']
            for n,row in curve.items():lines.append(f"| {n} | {row['count']} | {row['max_error']:.6g} | {row['above_1e4']} |")
        lines+=['','| Spacing / predicted w | Former-fallback (point,N) pairs | Max error | >1e-4 |','|---:|---:|---:|---:|']
        for factor,row in summary['spacing_curve'].items():lines.append(f"| {factor} | {row['count']} | {row['max_error']:.6g} | {row['above_1e4']} |")
        lines+=['',f"Maximum doubled-node difference: {summary['max_doubled_node_difference']:.6g}."]
    a=summary['geometry_audit']
    lines+=['',f"Geometry integral audit: {a['count']} points, max scan256 difference {a['max_scan256_difference']:.6g}; {a['reference_checks']} independent reference checks, max reference error {a['max_reference_error']:.6g}, {a['above_1e4']} failures above 1e-4.",
        f"Guard quadrature validation: {summary.get('guard_quadrature_validation')}.",
        f"Mean and p99 below scan per set: {summary['cost_targets_met']}.",
        '',f"Points above 1e-4: {o['failures'] or 'none'}. Full point records: `data/method2b_check_full.json`."]
    lines+=degenerate_report_lines()
    (ROOT/'data/method2b_summary.md').write_text('\n'.join(lines)+'\n')
    print('Overall',o,flush=True)
    if o['analytic']['above_1e4'] or o['skip_max_difference']>=1e-7 or (summary['geometry_audit'] and summary['geometry_audit']['above_1e4']):
        raise SystemExit('accuracy target failed; inspect saved per-point results')

if __name__=='__main__':main()
