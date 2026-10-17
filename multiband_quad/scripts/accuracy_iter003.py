"""Fine-grid Eq. (4) accuracy of method A and iteration-002 baselines."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from mbquad.coarse import CoarseModes
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from mbquad.pairwise import PairwiseHH
from mbquad.psd import PSDGrid, aligo_analytic_psd
from mbquad.reference import hh_exact, lnlambda_error
from mbquad.testcases import toy_small
from mbquad.toywave import ToyModes


def main():
    case=toy_small(); bs=case.bands
    rows=[]; errors={name:[] for name in ['linear','IFFT-FFT fast','pairwise block','pairwise linear']}
    for lines in [True,False]:
        psd=case.psdgrid if lines else PSDGrid(aligo_analytic_psd,bs.T,bs.dt,bs.f_low,bs.f_high)
        for hom in [False,True]:
            model=case.model if hom else ToyModes(4.85,t_c=126,modes={(2,2):(1,0,0)})
            coarse=CoarseModes.from_model(bs,model,model.modes); total=coarse.total()
            exact=hh_exact(model.total(psd.f_k),psd)
            values=[LinearInterpolationHH(bs,psd)(total),IFFTFFTHH(bs,psd)(total)]
            values.extend(PairwiseHH(bs,psd,coarse.modes,diag=diag)(coarse) for diag in ['block','linear'])
            for name,value in zip(errors,values):
                rel=abs(value/exact-1); dll=float(abs(lnlambda_error(value,exact,25)))
                errors[name].append([rel,dll])
                rows.append(f"| {'lines' if lines else 'smooth'} | {'five modes' if hom else '(2,2)'} | "
                            f'{name} | {rel:.6e} | {dll:.6e} |')
    sections=['# Accuracy — iteration 003',
              'toy_small(): T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s; '
              'Mc=Mc_ref=4.85 Msun, tc=126 s, m_max=4, band L=5. Five modes use ToyModes defaults; '
              'quadrupole uses (ratio,phase,growth)=(1,0,0). Analytic aLIGO PSD with TOY_LINES or smooth. '
              'Reference is brute-force fine-grid Eq. (4). Pairwise: L_margin=5, sigma_pair, impl=pairs. '
              'Each row is one deterministic case. |Delta ln Lambda| = 25² relative error / 2.',
              '| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |\n'
              '| --- | --- | --- | --- | --- |\n'+'\n'.join(rows)]
    summary=['| method | median relative error | worst relative error | median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |',
             '| --- | --- | --- | --- | --- |']
    for name,err in errors.items():
        med=np.median(err,axis=0); worst=np.max(err,axis=0)
        summary.append(f'| {name} | {med[0]:.6e} | {worst[0]:.6e} | {med[1]:.6e} | {worst[1]:.6e} |')
    sections.extend(['Summary across four deterministic cases (not a population):','\n'.join(summary)])
    report='\n\n'.join(sections)+'\n'
    (ROOT/'data/accuracy_iter003.md').write_text(report); print(report)


if __name__=='__main__':
    main()
