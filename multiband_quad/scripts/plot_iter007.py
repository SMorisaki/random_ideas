"""Index scatter and empirical CDFs for the requested Fig. 4 protocol."""
import os
os.environ['OMP_NUM_THREADS']='1'
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
from mbquad.population import METHODS

COLORS=dict(zip(METHODS,plt.get_cmap('tab10').colors[:6]))


def load(pop,psd):
    return json.loads((ROOT/f'data/population_iter007_{pop}_{psd}.json').read_text())


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    for pop in ('bns','nsbh','bbh'):
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
        for row,psd in enumerate(('lines','L1')):
            result=load(pop,psd)
            for col,L in enumerate(('5','50')):
                ax=axes[row,col]
                for m in METHODS:
                    vals=np.maximum([p['errors'][L][m] for p in result['points']],1e-14)
                    ax.scatter(range(len(vals)),vals,s=12,alpha=.65,color=COLORS[m],label=m)
                if result['n_strong']:
                    ax.axvline(result['n']-.5,color='grey',lw=.8,ls='--')
                    ax.text(.99,.97,'strong-HOM at right',ha='right',va='top',transform=ax.transAxes,fontsize=8)
                ax.set(yscale='log',title=f"{psd}, L={L}, K_MB={result['geometry'][L]['K_MB']}")
                ax.grid(alpha=.2)
                if row==1:
                    ax.set_xlabel('draw index')
                if col==0:
                    ax.set_ylabel(r'$|\Delta\ln\Lambda|$, SNR 25')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=6,loc='lower center',frameon=False)
        fig.suptitle(f'IMRPhenomXHM {pop.upper()} — population quadratic-term errors')
        fig.tight_layout(rect=(0,.045,1,.95))
        fig.savefig(ROOT/f'figures/iter007_errors_{pop}.png',dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for row,psd in enumerate(('lines','L1')):
        for col,L in enumerate(('5','50')):
            ax=axes[row,col]
            for m in METHODS:
                for pop,style in zip(('bns','nsbh','bbh'),('-','--',':')):
                    r=load(pop,psd)
                    values=np.sort([p['errors'][L][m] for p in r['points']])
                    ax.step(np.maximum(values,1e-14),np.arange(1,len(values)+1)/len(values),
                            where='post',color=COLORS[m],ls=style,lw=1.2)
            ax.set(xscale='log',title=f'{psd}, L={L}',ylim=(0,1.02))
            ax.grid(alpha=.2)
            if row==1:
                ax.set_xlabel(r'$|\Delta\ln\Lambda|$, SNR 25')
            if col==0:
                ax.set_ylabel('empirical CDF')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS[m],label=m) for m in METHODS]
    handles += [Line2D([],[],color='black',ls=style,label=pop.upper())
                for pop,style in zip(('bns','nsbh','bbh'),('-','--',':'))]
    fig.legend(handles=handles,ncol=5,loc='lower center',frameon=False)
    fig.suptitle('XHM error CDFs — regular + strong-HOM draws')
    fig.tight_layout(rect=(0,.07,1,.96))
    fig.savefig(ROOT/'figures/iter007_cdf.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
