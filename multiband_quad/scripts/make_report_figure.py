#!/usr/bin/env python
"""Plot recorded Eq. (26) errors and route costs; no waveform evaluations.

Accuracy uses regular-population SNR-25 statistics from iteration 007,
separately for each PSD and L. Costs use iteration-009 L=5 component sums
(one waveform plus D quadratic steps); the hh_half waveform cost is scaled
by its point count. Neither panel represents a complete likelihood timing.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep Matplotlib and Fontconfig writes inside this research directory.
os.environ['MPLCONFIGDIR'] = str(ROOT / 'data' / 'fontcache' / 'matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'data' / 'fontcache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ('linear', 'IFFT-FFT', 'pairwise', 'B1', 'B1_all', 'B0')
COLORS = dict(zip(METHODS, ('#7f7f7f', '#e69f00', '#009e73', '#d55e00', '#9b59b6', '#0072b2')))
LABELS = {'linear': 'Linear', 'IFFT-FFT': 'IFFT-FFT', 'pairwise': 'A (pairwise)',
          'B1': 'B1', 'B1_all': 'B1 all half bins', 'B0': 'B0'}
ROUTES = {'IFFT-FFT': 'IFFT-FFT paper bands', 'pairwise': 'pairwise paper bands',
          'B1': 'B1 paper bands', 'B0': 'B0 hh_half bands'}


def read_json(name):
    """Read a local committed JSON summary without recomputing statistics."""
    return json.loads((ROOT / 'data' / name).read_text())


def main():
    """Write figures/report_summary.png with accuracy and cost panels."""
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, (accuracy, cost) = plt.subplots(1, 2, figsize=(14, 7.5),
                                       gridspec_kw={'width_ratios': [1.2, 1]})
    groups = [(pop, psd) for pop in ('bns', 'nsbh', 'bbh') for psd in ('lines', 'L1')]
    offsets = np.linspace(-.30, .30, len(METHODS))
    for group, (pop, psd) in enumerate(groups):
        data = read_json(f'population_iter007_{pop}_{psd}.json')
        stats = data['statistics']['regular']
        for method, offset in zip(METHODS, offsets):
            values = [stats[str(L)][method]['median'] for L in (5, 50)]
            if not np.all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError(f'Invalid medians: {pop}/{psd}/{method}')
            y = group + offset
            accuracy.plot(values, [y, y], color=COLORS[method], alpha=.35, lw=1)
            accuracy.plot(values[0], y, 'o', color=COLORS[method], ms=5)
            accuracy.plot(values[1], y, 'D', mec=COLORS[method], mfc='white', ms=5)
    accuracy.set(xscale='log', yticks=range(len(groups)),
                 yticklabels=[f'{pop.upper()} / {psd}' for pop, psd in groups],
                 xlabel=r'Median $|\Delta\ln\Lambda|$ at SNR 25',
                 title='(a) Accuracy: paper bands, 100 regular draws')
    accuracy.invert_yaxis()
    accuracy.grid(axis='x', which='major', alpha=.2)
    accuracy.legend(handles=[Line2D([], [], marker='o', color='black', ls='', label='L = 5'),
                             Line2D([], [], marker='D', mfc='white', color='black', ls='', label='L = 50')],
                    loc='lower right', frameon=False)

    records = [r for r in read_json('perf_iter009.json')['results'] if r['L'] == 5]
    for group, record in enumerate(records):
        for offset, (method, route) in zip(np.linspace(-.27, .27, len(ROUTES)), ROUTES.items()):
            row = record['routes'][route]
            values = [1000 * row[f'sum_{D}_seconds'] for D in (1, 3)]
            y = group + offset
            cost.plot(values, [y, y], color=COLORS[method], lw=1.5, alpha=.6)
            cost.plot(values[0], y, 'o', color=COLORS[method], ms=6)
            cost.plot(values[1], y, '^', mec=COLORS[method], mfc='white', ms=7)
    cost.set(yticks=range(len(records)), yticklabels=['BNS', 'GW190814-like', 'BBH 60+20'],
             xlabel='Waveform + quadratic term (ms)', xlim=(0, 18),
             title='(b) Cost: L = 5, warm component sums')
    cost.invert_yaxis()
    cost.grid(axis='x', alpha=.2)
    cost.legend(handles=[Line2D([], [], marker='o', color='black', ls='', label='1 detector'),
                         Line2D([], [], marker='^', mfc='white', color='black', ls='', label='3 detectors (estimate)')],
                loc='lower right', frameon=False)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(.5, .07))
    fig.suptitle('Multiband quadratic term with higher modes', fontsize=16)
    fig.text(.5, .025, 'Accuracy: population_iter007_* JSON; cost: perf_iter009.json.  '
             'Cost-panel B0 uses hh_half; accuracy-panel B0 uses paper bands.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .94), w_pad=3)
    output = ROOT / 'figures' / 'report_summary.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
