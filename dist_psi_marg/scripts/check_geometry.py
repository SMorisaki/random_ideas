#!/usr/bin/env python
"""Geometry-only audit: 100000 random points, named and repaired cases.

The comparator is method2._geometry (64 proxy samples plus refinements,
now with analytic seeds in the degenerate corner); it is not an oracle at
clipping cusps or for a hidden island. Corrected cusp transition widths
can deliberately differ from the legacy one-sided curvature widths.
Every significant comparator peak must have a candidate within .2 widths
or 1e-6 in continuous-proxy height, and a width ratio in [2/3,1.5].
No distance kernel is called in this audit.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.regimes import random_params, named_cases
from dpmarg.params import reduce_params, expand_params
from dpmarg.geometry import analytic_geometry, proxy_derivatives, interior_theta
from dpmarg.method2 import _geometry, _proxy
KEYS=('A','B','C','D','E','xmin','xmax')


def repaired_cases():
    """Five iteration-008 and two iteration-005 geometry regressions."""
    raw=[(-.5984956658182099,.05940349086388996,-2.291486743920588,-2.064735554121742,3.0900921157124963),
         expand_params(.5982561314887299,0.,.2631884412553095,1.1737617861221479),
         (16.948711265673715,27.607321753023136,-2.1348128164665057,4.483601570832346,6.633182378713948),
         (21.395814952434804,18.805908983120858,1.244753479681153,5.817197177739085,16.896495387635404),
         (-.4050033597832454,3.732158650815139,-.08419409158783564,.0031788339344747253,.32508833883254384)]
    names=['seed808_44','seed808_915','posterior_124','posterior_876','clipping_regression']
    for l1 in (651154.6134045325,633390.2841571145):
        raw.append(expand_params(l1,0.,19.2047382462,1.24240566613));names.append(f'iter005_{l1:g}')
    return [dict(name=n,**dict(zip(KEYS,(*p,.1,100.)))) for n,p in zip(names,raw)]


@njit(cache=False)
def compare(p):
    l1,l2,R,eta=reduce_params(p[0],p[1],p[2],p[3],p[4])
    a,w,b=analytic_geometry(l1,l2,R,eta,p[5],p[6])
    c,v,d=_geometry(l1,l2,R,eta,p[5],p[6])
    missing=0;location=0.;ratio=1.;height=0.
    smooth_width=0;merged=0;cusps=0
    scan_heights=np.array([_proxy(t,l1,l2,R,eta,p[5],p[6]) for t in c])
    jglobal=np.argmax(scan_heights)
    global_location=0.;global_height=0.;global_ratio=1.
    for j in range(len(c)):
        best=1e300; kbest=0
        for k in range(len(a)):
            dist=abs((a[k]-c[j]+np.pi/2)%np.pi-np.pi/2)
            if dist<best:
                best,kbest=dist,k
        delta=abs(_proxy(a[kbest],l1,l2,R,eta,p[5],p[6])-_proxy(c[j],l1,l2,R,eta,p[5],p[6]))
        loc=best/v[j]
        wr=max(w[kbest]/v[j],v[j]/w[kbest])
        location=max(location,loc);ratio=max(ratio,wr);height=max(height,delta)
        aa=abs(R*np.cos(c[j]-eta));bb=l2+(l1-l2)*np.cos(c[j])**2
        cusp=min(abs(aa-bb*p[5]),abs(aa-bb*p[6]))<1e-5*max(aa,bb*p[5],1e-30)
        cusps+=int(cusp)
        if wr>1.5 and not cusp:smooth_width+=1
        if loc>.2 and delta>1e-6:
            missing+=1
            if best<min(w[kbest],v[j]):merged+=1
        if j==jglobal:
            global_location,global_height,global_ratio=loc,delta,wr
    return (location,ratio,height,missing,len(a),len(c),global_location,global_height,
            global_ratio,smooth_width,merged,cusps)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--n',type=int,default=100000)
    args=parser.parse_args();start=time.perf_counter()
    cases=[]
    for seed in (901,902,903,904):
        group=random_params(args.n//4,seed)
        for c in group:c['name']=f'seed{seed}_{c["name"]}'
        cases+=group
    cases+=named_cases()+repaired_cases()
    rows=[]
    for c in cases:
        p=np.array([c[k] for k in KEYS]);m=compare(p)
        rows.append(dict(name=c['name'],parameters=p.tolist(),location_widths=m[0],width_ratio=m[1],
            height_difference=m[2],missed_scan_peaks=int(m[3]),analytic_peaks=int(m[4]),scan_peaks=int(m[5]),
            global_location_widths=m[6],global_height_difference=m[7],global_width_ratio=m[8],
            smooth_width_mismatches=int(m[9]),merged_scan_peaks=int(m[10]),scan_cusps=int(m[11])))
    mismatches=[r for r in rows if r['missed_scan_peaks'] or r['width_ratio']>1.5]
    summary=dict(count=len(rows),mismatches=len(mismatches),mismatch_rate=len(mismatches)/len(rows),
        location_mismatches=sum(r['missed_scan_peaks']>0 for r in rows),
        width_mismatches=sum(r['width_ratio']>1.5 for r in rows),
        global_location_mismatches=sum(r['global_location_widths']>.2 and r['global_height_difference']>1e-6 for r in rows),
        smooth_width_mismatches=sum(r['smooth_width_mismatches']>0 for r in rows),
        missed_unmerged_peaks=sum(r['missed_scan_peaks']-r['merged_scan_peaks'] for r in rows),
        scan_cusps=sum(r['scan_cusps'] for r in rows),
        worst=sorted(mismatches,key=lambda r:r['height_difference'],reverse=True)[:12],
        elapsed_s=time.perf_counter()-start)
    (ROOT/'data/geometry_check_full.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'data/geometry_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
