#!/usr/bin/env python
"""Iteration 007: independent coarse/fine slices, with isolated design changes.

220 original seed-501 bases, >=1025 fine nodes, all 8--512 coarse grids.
Counts must pass on every finer tested grid. Null means unresolved through
512, not an extrapolation. Absolute-v2 slices use only the physical triangle;
a grid on that rectangle would also waste half its cells. Fine sampling is
an empirical diagnostic, not a bound between fine nodes.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg import table1d, table4d, params, regimes
from dpmarg.reference import lnL_marg_ref
from dpmarg.method2 import lnL_marg_m2_table
from slice_study_method1 import interp, summary_required, COUNTS

AXES=['v1','w','radial','eta']
# normalization, radial, weak, angle, geometry; one change at a time.
DESIGNS={
    'baseline':(0,0,0,0,0),
    'norm_integral_raw':(3,0,0,0,0),
    'norm_softmax_raw':(4,0,0,0,0),
    'norm_integral':(1,0,0,0,0),
    'norm_softmax':(2,0,0,0,0),
    'radial3':(0,1,0,0,0),
    'radial3_exact':(0,2,0,0,0),
    'weak_absolute':(0,0,1,0,0),
    'weak_q':(0,0,2,0,0),
    'symmetric':(0,0,0,0,1),
    'angle_cos':(0,0,0,1,0),
    'combined':(1,1,2,1,1),
    'combined_exact':(1,2,2,1,1),
}


@njit(cache=True)
def sample(points,radial,options,norms):
    """Compute method 2 once per point, then each requested proxy remainder."""
    out=np.empty((len(norms),len(points)))
    for i in range(len(points)):
        l1,l2,R,eta=table4d.from_unit(points[i],1e8,24.,.1,100.,*options)
        value=lnL_marg_m2_table(R*np.cos(eta),R*np.sin(eta),.5*(l1-l2),0.,.5*(l1+l2),.1,100.,radial)
        for j in range(len(norms)):
            out[j,i]=value-table4d.normalization_value(l1,l2,R,eta,.1,100.,norms[j])
    return out


def generator_cases():
    """20 documented extremes: 18 named plus both prior narrow-null failures."""
    return regimes.named_cases()[:18]+[
        dict(zip(['A','B','C','D','E'],params.expand_params(l1,0.,19.2047382462,1.24240566613)),
             name=f'narrow_null_axis_{l1:g}',xmin=.1,xmax=100.)
        for l1 in [651154.6134045325,633390.2841571145]]


def generator_checks(radial):
    checks=[]
    for c in generator_cases():
        p=[c[k] for k in ['A','B','C','D','E','xmin','xmax']]
        value=lnL_marg_m2_table(*p,radial);truth=lnL_marg_ref(*p)
        checks.append(dict(name=c['name'],value=value,reference=truth,error=abs(value-truth)))
    if max(c['error'] for c in checks)>5e-5:raise AssertionError(checks)
    return checks


def diagnostics(radial):
    """Cusp slopes, fixed-physical q=1 limits, and warmed proxy timings."""
    cusp={}
    epsilons=np.logspace(-3,-8,6)
    for mode in range(5):
        f0=table4d.reduced_value(1e4,0.,1.,0.,.1,100.,radial,mode)
        slopes=[]
        for e in epsilons:
            plus=table4d.reduced_value(1e4,0.,1.,e,.1,100.,radial,mode)
            minus=table4d.reduced_value(1e4,0.,1.,-e,.1,100.,radial,mode)
            slopes.append(dict(epsilon=e,right=(plus-f0)/e,left=(f0-minus)/e))
        cusp[str(mode)]=slopes
    qrows=[]
    for q in [1.,1-1e-12,1-1e-9,1-1e-6,1-1e-3]:
        for geom in [0,1]:
            l1=1e8;l2=q*l1;R=5.45e6
            values=[table4d.reduced_value(l1,l2,R,e,.1,100.,radial) for e in np.linspace(0,np.pi/2,33)]
            coordinates=[table4d.to_unit(l1,l2,R,e,1e8,24.,.1,100.,1,2,0,geom)[2] for e in np.linspace(0,np.pi/2,33)]
            qrows.append(dict(q=q,geometry=geom,F_range=float(np.ptp(values)),u_range=float(np.ptp(coordinates))))
    timings={}
    test=np.array([params.reduce_params(*(c[k] for k in ['A','B','C','D','E'])) for c in regimes.random_params(220,501)])
    for mode in range(3):
        _time_norm(test,mode)
        times=[]
        for _ in range(7):
            t=time.perf_counter_ns();_time_norm(test,mode);times.append((time.perf_counter_ns()-t)/(len(test)*10))
        timings[str(mode)]=float(np.median(times))
    isotropic={}
    for mode in range(3):
        isotropic[str(mode)]=float(np.ptp([table4d.reduced_value(1e8,1e8,5.45e6,e,.1,100.,radial,mode) for e in np.linspace(0,np.pi/2,1025)]))
    return dict(cusp_slopes=cusp,q1_limit=qrows,q1_transition=q1_transition(radial),
                normalization_median_ns=timings,isotropic_eta_remainder_range=isotropic)


def q1_transition(radial):
    """Resolve the physical angular-width layer delta_lambda*xmin²~1.

    With R=0, both coordinate choices fix the radial coordinate at zero:
    averaging angles cannot remove this weak-axis layer. Compare the
    zero-signal and a narrow single-peak case at fixed physical parameters.
    """
    rows=[]
    for R in [0.,5.45e6]:
        f1=table4d.reduced_value(1e8,1e8,R,.3,.1,100.,radial)
        for gap in np.r_[0.,np.logspace(-12,-2,11)]:
            q=1-gap;l2=1e8*q
            value=table4d.reduced_value(1e8,l2,R,.3,.1,100.,radial)
            u=[table4d.to_unit(1e8,l2,R,.3,1e8,24.,.1,100.,0,0,0,g)[2] for g in [0,1]]
            rows.append(dict(R=R,q=q,delta_lambda_xmin2=1e6*gap,F=value,
                             F_minus_q1=value-f1,radial_u=u))
    return rows


@njit(cache=True)
def _time_norm(points,mode):
    total=0.
    for _ in range(10):
        for p in points:total+=table4d.normalization_value(p[0],p[1],p[2],p[3],.1,100.,mode)
    return total


def describe(case,base,axis,options,error):
    physical=table4d.from_unit(base,1e8,24.,.1,100.,*options)
    l1,l2,R,eta=physical;peak,r,_=table4d.lnL_max_numba(*physical,.1,100.)
    regime='null weak eigenvalue' if l2==0 and l1 else 'isotropic' if l1==l2 else 'anisotropic'
    regime+= '; no linear signal' if R==0 else '; lower radial boundary' if r<.100001 else '; upper radial boundary' if r>99.99999 else '; enclosed radial peak'
    return dict(name=case['name'],physical=list(physical),q=l2/l1 if l1 else 0.,
                snr=float(np.sqrt(max(0.,2*peak))),peak_radius=r,regime=regime,error64=float(error))


def worst_generator_checks(summary,full,radial):
    """Independently check three sharp radial features at their actual worst nodes."""
    checks=[];fine=np.linspace(0,1,summary['fine_count']);coarse=np.linspace(0,1,64)
    for name in ['radial3','radial3_exact','symmetric']:
        norm,*opts=DESIGNS[name];opts=tuple(opts);cached=full['designs'][name]
        i=cached['axes']['radial']['worst_indices'][0];base=np.array(cached['bases'][i])
        z=np.tile(base,(len(fine)+64,1));z[:,2]=np.r_[fine,coarse]
        y=sample(z,radial,opts,np.array([norm],dtype=np.int64))[0]
        delta=interp(coarse,y[len(fine):],fine,'catmull')-y[:len(fine)]
        j=int(np.argmax(abs(delta)));physical=table4d.from_unit(z[j],1e8,24.,.1,100.,*opts)
        raw=params.expand_params(*physical)
        value=lnL_marg_m2_table(*raw,.1,100.,radial);truth=lnL_marg_ref(*raw,.1,100.)
        maximum_difference=table4d.lnL_max_numba(*physical,.1,100.)[0]-params.lnL_max(*raw,.1,100.)[0]
        checks.append(dict(design=name,name=full['cases'][i]['name'],fine_index=j,physical=list(physical),
                           value=value,reference=truth,error=abs(value-truth),
                           error64=float(abs(delta[j])),maximum_difference=maximum_difference))
    if max(c['error'] for c in checks)>5e-5:raise AssertionError(checks)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fine',type=int,default=1025)
    parser.add_argument('--reuse',action='store_true',help='reuse matching completed designs; regenerate worst-slice plots')
    args=parser.parse_args()
    if args.fine<1025:parser.error('--fine must be >=1025')
    start=time.perf_counter();radial=table1d.load_table(ROOT/'data/table1d_xmin0.1_xmax100.npz')
    cases=regimes.named_cases()+regimes.random_params(200,501)
    physical=np.array([params.reduce_params(*(c[k] for k in ['A','B','C','D','E'])) for c in cases])
    fine=np.linspace(0,1,args.fine);counts=np.array(COUNTS)
    checks=generator_checks(radial)
    summary=dict(base_count=220,random_seed=501,fine_count=args.fine,counts=COUNTS,bmax=1e8,smax=24.,
                 tolerances=[1e-3,1e-4],design_options={k:list(v) for k,v in DESIGNS.items()},designs={},
                 generator_checks=dict(n=20,max_abs_error=max(c['error'] for c in checks)),
                 notes=[
                     'Null = no resolution <=512 passes all finer tested grids. Sampled diagnostics, not certificates.',
                     'Old physical radial envelope is retained for every design. Exterior bases are projected.',
                     'Weak absolute uses valid triangular intervals, rescaled to [0,1] separately on each slice; counts are optimistic for a global rectangle.',
                     'Weak absolute and baseline weak slices are mathematically identical at fixed lam1; their v1 slices differ.',
                     's3 uses c=co/(b+sigma0^2), k_a=1, sigma0=1/(xmax-xmin), to include b=0 continuously.',
                     'radial3_exact uses literal c=co/b for b>0 and the regularized convention only at b=0; its inverse uses bracketed Newton in asinh(R/scale).',
                     'Two-ended q stretch has eps=1/(1+lam1*xmax^2); symmetric averages ridge and linear-direction coordinates.',
                     'Proxy normalizations pair signed angles, use b+2*T/xmax^2 and add T to remove radial degeneracy. Fixed angular scans are not certified upper envelopes.',
                     'Integral proxy is 64 paired angles on [0,pi); softmax is 64 full-circle angles, T=1.',
                     'Uniform eta uses even-reflected Catmull tangents. Cos(2eta) uses ordinary edge tangents; even symmetry does not imply zero derivative there.'
                 ])
    full=dict(cases=cases,physical=physical.tolist(),generator_checks=checks,designs={})
    prior_summary=prior_full=None
    if args.reuse:
        prior_summary=json.loads((ROOT/'data/method1b_slices.json').read_text())
        prior_full=json.loads((ROOT/'data/method1b_slices_full.json').read_text())
        if prior_summary['fine_count']!=args.fine or prior_summary['counts']!=COUNTS or prior_full['cases']!=cases:
            raise ValueError('cached study protocol does not match')
        summary['reused_previous_elapsed_s']=prior_summary['elapsed_s']
        summary['reused_designs']=[]
    groups={}
    for name,design in DESIGNS.items():groups.setdefault(design[1:],[]).append((name,design[0]))
    plot_curves={axis:[] for axis in AXES}
    for options,designs in groups.items():
        if prior_summary is not None and all(name in prior_summary['designs'] and
                prior_summary['design_options'][name]==list(DESIGNS[name]) for name,_ in designs):
            for name,norm in designs:
                summary['reused_designs'].append(name)
                summary['designs'][name]=prior_summary['designs'][name]
                full['designs'][name]=prior_full['designs'][name]
                cached=full['designs'][name]
                for axis,axisname in enumerate(AXES):
                    i=cached['axes'][axisname]['worst_indices'][0]
                    base=np.array(cached['bases'][i]);left,right=cached['axes'][axisname]['intervals'][i]
                    xall=np.r_[fine,np.linspace(0,1,64)]
                    points=np.tile(base,(len(xall),1));points[:,axis]=left+(right-left)*xall
                    y=sample(points,radial,options,np.array([norm],dtype=np.int64))[0]
                    pred=interp(np.linspace(0,1,64),y[len(fine):],fine,'catmull',axis==3 and not options[2])
                    plot_curves[axisname].append((name,summary['designs'][name]['axes'][axisname]['worst_slices'][0],y[:len(fine)],pred))
            print(json.dumps(dict(reused=[name for name,_ in designs])),flush=True)
            continue
        bases=np.clip(np.array([table4d.to_unit(*p,1e8,24.,.1,100.,*options) for p in physical]),0,1)
        norms=np.array([n for _,n in designs],dtype=np.int64)
        sample(bases[:1],radial,options,norms)
        for name,_ in designs:
            summary['designs'][name]={'axes':{}}
            full['designs'][name]=dict(bases=bases.tolist(),axes={})
        for axis,axisname in enumerate(AXES):
            errors={name:{kind:np.empty((220,len(counts))) for kind in ['catmull','natural','linear']} for name,_ in designs}
            curves={name:[] for name,_ in designs};intervals=[]
            for i,base in enumerate(bases):
                left,right=0.,1.
                if options[1]==1:
                    if axis==0:left=base[1]
                    if axis==1:right=base[0]
                intervals.append([left,right])
                # Independent physical generator at every coarse node, including
                # nodes not present in the fine grid; no fine-grid subsampling.
                xall=np.concatenate([fine]+[np.linspace(0,1,n) for n in counts])
                points=np.tile(base,(len(xall),1));points[:,axis]=left+(right-left)*xall
                yall=sample(points,radial,options,norms)
                for d,(name,_) in enumerate(designs):
                    truth=yall[d,:len(fine)];offset=len(fine)
                    for j,n in enumerate(counts):
                        x=np.linspace(0,1,n);y=yall[d,offset:offset+n];offset+=n
                        for kind in errors[name]:
                            pred=interp(x,y,fine,kind,axis==3 and not options[2])
                            errors[name][kind][i,j]=np.max(abs(pred-truth))
                        if n==64:representative=interp(x,y,fine,'catmull',axis==3 and not options[2])
                    curves[name].append((truth,representative))
            for name,_ in designs:
                result={};raw={}
                for kind,err in errors[name].items():
                    result[kind]={};required={}
                    for tol in [1e-3,1e-4]:
                        result[kind][str(tol)],required[str(tol)]=summary_required(err,counts,tol)
                    result[kind]['max_error512']=float(err[:,-1].max())
                    raw[kind]=dict(errors=err.tolist(),required=required)
                ranked=np.argsort(errors[name]['catmull'][:,COUNTS.index(64)])[::-1]
                worst=[];seen=set()
                for i in ranked:
                    key=tuple(np.round(np.delete(bases[i],axis),12))
                    if key in seen:continue
                    seen.add(key);worst.append(int(i))
                    if len(worst)==3:break
                rows=[describe(cases[i],bases[i],axis,options,errors[name]['catmull'][i,COUNTS.index(64)]) for i in worst]
                result['worst_slices']=rows[:1]
                summary['designs'][name]['axes'][axisname]=result
                full['designs'][name]['axes'][axisname]=dict(interpolation=raw,intervals=intervals,worst_indices=worst,worst_slices=rows)
                i=worst[0];plot_curves[axisname].append((name,rows[0],*curves[name][i]))
            compact={name:{k:summary['designs'][name]['axes'][axisname][k]['0.001'] for k in ['catmull','natural']} for name,_ in designs}
            print(json.dumps(dict(axis=axisname,results=compact,elapsed_s=time.perf_counter()-start)),flush=True)
    summary['diagnostics']=diagnostics(radial)
    worst_checks=worst_generator_checks(summary,full,radial)
    full['worst_slice_generator_checks']=worst_checks
    summary['worst_slice_generator_checks']=dict(n=len(worst_checks),max_abs_error=max(c['error'] for c in worst_checks))
    requirements=[summary['designs']['combined']['axes'][a]['catmull']['0.001']['p99'] for a in AXES]
    exact_requirements=[summary['designs']['combined_exact']['axes'][a]['catmull']['0.001']['p99'] for a in AXES]
    plausible=any(all(n is not None and n<=70 for n in row) for row in [requirements,exact_requirements])
    lower=[n or 513 for n in requirements]
    summary['prototype']=dict(eligible=plausible,p99_counts=requirements,p99_entry_lower_bound=int(np.prod(lower)),
                              literal_p99_counts=exact_requirements,
                              reason='Build justified' if plausible else 'Combined 99th-percentile axis counts exceed ~70; skip prototype.')
    if plausible:raise RuntimeError('Measured prototype trigger reached: build and validate before reporting completion.')
    for axisname,rows in plot_curves.items():
        fig,axs=plt.subplots(2,len(rows),figsize=(3.3*len(rows),6),sharex=True)
        for j,(name,row,truth,pred) in enumerate(rows):
            axs[0,j].plot(fine,truth);axs[0,j].plot(fine,pred,'--')
            axs[1,j].plot(fine,pred-truth);axs[1,j].axhline(1e-3,color='k',ls=':')
            axs[0,j].set_title(f"{name}\n{row['name']}\nq={row['q']:.2g}, SNR={row['snr']:.2g}",fontsize=9)
            axs[1,j].set_xlabel(axisname+' (unit interval)')
        axs[0,0].set_ylabel('lnL marginal - N');axs[1,0].set_ylabel('64-node Catmull error')
        fig.tight_layout();fig.savefig(ROOT/f'figures/method1b_slices_{axisname}.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(17,8),sharey=True)
    for row,tol in enumerate([1e-3,1e-4]):
        for col,axis in enumerate(AXES):
            ax=axs[row,col]
            for j,name in enumerate(DESIGNS):
                r=summary['designs'][name]['axes'][axis]['catmull'][str(tol)]
                ax.scatter(j,r['median'] or 700,marker='o',color='C0')
                ax.scatter(j,r['p99'] or 700,marker='^',color='C1')
                ax.scatter(j,r['max'] or 700,marker='x',color='C2')
            ax.axhline(70,color='green',ls=':');ax.axhline(512,color='grey',ls=':')
            ax.set_xticks(range(len(DESIGNS)),DESIGNS,rotation=75,fontsize=8)
            ax.set_title(f'{axis}, tol={tol:g}');ax.set_yscale('log');ax.set_ylim(6,900)
    axs[0,0].set_ylabel('Required nodes: median o, 99% triangle, max x\n700 denotes >512')
    fig.tight_layout();fig.savefig(ROOT/'figures/method1b_points_per_axis.png',dpi=150);plt.close(fig)
    summary['elapsed_s']=time.perf_counter()-start
    (ROOT/'data/method1b_slices_full.json').write_text(json.dumps(full))
    encoded=json.dumps(summary,indent=2)
    if len(encoded)>100000:encoded=json.dumps(summary,indent=1)
    (ROOT/'data/method1b_slices.json').write_text(encoded+'\n')
    print(json.dumps(dict(prototype=summary['prototype'],elapsed_s=summary['elapsed_s'])),flush=True)


if __name__=='__main__':main()
