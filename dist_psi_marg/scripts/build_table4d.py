#!/usr/bin/env python
"""Build the capped method-1 prototype and validate against saved/fresh values."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dpmarg import table1d,table4d,regimes
from dpmarg.method2 import lnL_marg_m2_table_array


@njit(cache=True)
def time_lookup(parameters,table,radial,order,repeats):
    total=0.
    for k in range(repeats):
        for p in parameters:
            total+=table4d.lnL_marg_m1(p[0],p[1],p[2],p[3],p[4],p[5],p[6],table,radial,order)
    return total


@njit(cache=True)
def time_interpolation(points,values,order,repeats):
    total=0.
    for k in range(repeats):
        for z in points:total+=table4d.interpolate(values,z,order)
    return total


def stats(error):
    return dict(max=float(np.max(error)),p99=float(np.quantile(error,.99)),median=float(np.median(error)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',type=int,nargs=4,default=None)
    p.add_argument('--bmax',type=float,default=1e8)
    p.add_argument('--smax',type=float,default=24.)
    p.add_argument('--reuse',action='store_true')
    args=p.parse_args()
    radial=table1d.load_table(ROOT/'data/table1d_xmin0.1_xmax100.npz')
    slice_path=ROOT/'data/method1_slices.json'
    slices=json.loads(slice_path.read_text()) if slice_path.exists() else None
    # All four worst-case axis requirements exceed the measured range.
    # Use a balanced near-cap grid: 66^3*69 = 19,837,224 entries.
    shape=args.shape or [66,66,66,69]
    if slices and args.shape is None:
        req=[slices['axes'][a]['catmull']['0.001']['max'] for a in ['v1','w','radial','eta']]
        if all(n is not None for n in req) and np.prod(req)<=20000000:shape=req
    path=ROOT/'data/table4d_xmin0.1_xmax100.npz'
    start=time.perf_counter();table4d.build_table(radial,(4,4,4,4),args.bmax,args.smax)
    warmup=time.perf_counter()-start
    if args.reuse:
        table=table4d.load_table(path)
        prior_path=ROOT/'data/method1_check.json'
        previous=json.loads(prior_path.read_text()) if prior_path.exists() else {}
        build_s=previous.get('build_s')
        warmup=previous.get('warmup_s',warmup)
    else:
        start=time.perf_counter();table=table4d.build_table(radial,shape,args.bmax,args.smax)
        build_s=time.perf_counter()-start;table4d.save_table(path,table)
    result=dict(shape=list(table[0].shape),entries=int(table[0].size),bytes=table[0].nbytes,build_s=build_s,warmup_s=warmup,
                bmax=table[1],smax=table[2],xmin=table[3],xmax=table[4],sets={},
                guard='None: interpolation errors are deliberately exposed; only exterior points fall back.',
                interpolation='tensor Catmull-Rom with linear edge extension; even eta reflection')
    result['stored_Ftilde_range']=[float(table[0].min()),float(table[0].max())]
    (ROOT/'data/method1_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    full={}
    sources=[('regression',json.loads((ROOT/'data/regression_points.json').read_text())),
             ('stored_random',json.loads((ROOT/'data/method2_random_reference.json').read_text())),
             ('fresh_random',regimes.random_params(2000,505))]
    for name,cases in sources:
        original_count=len(cases)
        cases=[c for c in cases if c['xmin']==table[3] and c['xmax']==table[4]]
        parameters=np.array([[c[k] for k in ['A','B','C','D','E','xmin','xmax']] for c in cases])
        generator=lnL_marg_m2_table_array(parameters,radial)
        truth=np.array([c['lnL_marg_ref'] for c in cases]) if name!='fresh_random' else generator
        row=dict(n=len(cases),excluded_other_prior=original_count-len(cases),truth='stored brute-force reference' if name!='fresh_random' else 'method2 table generator',
                 generator_reference_error=stats(abs(generator-truth)))
        points=[]
        full[name]=dict(cases=cases,truth=truth.tolist(),generator=generator.tolist())
        for order,label in [(3,'cubic'),(1,'linear')]:
            values,flags=table4d.evaluate_array(parameters,table,radial,order)
            error=abs(values-truth);inside=flags==0
            time_lookup(parameters[:1],table,radial,order,1)
            times=[]
            for k in range(7):
                start=time.perf_counter();time_lookup(parameters,table,radial,order,5)
                times.append((time.perf_counter()-start)*1e9/(5*len(parameters)))
            inside_times=[]
            if inside.any():
                for k in range(7):
                    start=time.perf_counter();time_lookup(parameters[inside],table,radial,order,5)
                    inside_times.append((time.perf_counter()-start)*1e9/(5*inside.sum()))
            row[label]=dict(error=stats(error),inside_error=stats(error[inside]) if inside.any() else None,
                            fallback_rate=float(flags.mean()),lookup_ns=float(np.median(times)),
                            inside_lookup_ns=float(np.median(inside_times)) if inside_times else None,
                            worst_name=cases[int(error.argmax())]['name'],fail_1e3=int((error>1e-3).sum()))
            full[name][label]=dict(values=values.tolist(),errors=error.tolist(),fallbacks=flags.tolist())
        result['sets'][name]=row;print(json.dumps({name:row}),flush=True)
    # Memory-only interpolation timing excludes reduction, coordinate inversion,
    # max reconstruction and fallback. Full API timing above includes all of them.
    z=np.random.default_rng(506).uniform(size=(2000,4));result['interpolation_only_ns']={}
    for order in [1,3]:
        time_interpolation(z[:1],table[0],order,1);times=[]
        for k in range(7):
            start=time.perf_counter();time_interpolation(z,table[0],order,10)
            times.append((time.perf_counter()-start)*1e9/(10*len(z)))
        result['interpolation_only_ns'][str(order)]=float(np.median(times))
    slices=json.loads(slice_path.read_text()) if slice_path.exists() else None
    # Uncontended cost model, measured after validation on a reproducible
    # uniform sample of actual grid nodes. It is not another full build.
    rng=np.random.default_rng(507)
    probes=np.column_stack([rng.integers(n,size=10000)/(n-1) for n in table[0].shape])
    table4d.sample_unit(probes[:1],table[1],table[2],table[3],table[4],radial)
    costs=[]
    for _ in range(7):
        start=time.perf_counter()
        table4d.sample_unit(probes,table[1],table[2],table[3],table[4],radial)
        costs.append((time.perf_counter()-start)/len(probes))
    result['generator_ns_per_grid_node']=float(np.median(costs)*1e9)
    result['estimated_serial_build_s']=float(np.median(costs)*table[0].size)
    if slices:
        result['slice_required_nodes']={a:{k:slices['axes'][a][k] for k in ['catmull','natural','linear']} for a in slices['axes']}
        required=[slices['axes'][a]['catmull']['0.001']['max'] for a in ['v1','w','radial','eta']]
        result['slice_implied_entries']=int(np.prod(required)) if all(n is not None for n in required) else None
        result['generator_spot_checks']=slices['generator_checks']
        result['slice_entry_lower_bound']=int(np.prod([n or 513 for n in required]))
        if build_s is not None:
            result['slice_build_s_lower_bound']=build_s*result['slice_entry_lower_bound']/table[0].size
    result['verdict']='Feasible' if max(row['cubic']['error']['max'] for row in result['sets'].values())<=1e-3 else 'Not feasible on the measured whole domain with this regular grid and coordinates'
    (ROOT/'data/method1_check.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'data/method1_check_full.json').write_text(json.dumps(full))
    print(json.dumps(dict(verdict=result['verdict'],interpolation_only_ns=result['interpolation_only_ns'])),flush=True)


if __name__=='__main__':main()
