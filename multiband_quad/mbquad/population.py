"""Seeded XHM populations and fine-grid Eq. (4) accuracy comparisons.

The distributions are component-mass priors conditioned on Mc >= Mc_ref,
plus q <= .15, Mc <= Mc_max and 1.3 < inclination < 1.8 for strong-HOM samples. All
waveforms use the sequence driver, with iteration-005 OneMode tail recovery.
"""
import hashlib
import json
from pathlib import Path
import warnings

import lalsimulation
import numpy as np

from .bands import BandSetup
from .baseband import BasebandIFFTFFTHH
from .ifftfft import IFFTFFTHH
from .lalwave import XHMCoarseEvaluator, XHMModel, XHM_MODES
from .linear import LinearInterpolationHH
from .pairwise import PairwiseHH
from .psd import (PSDGrid, add_lorentzian_lines, aligo_analytic_psd,
                  load_asd_or_psd)
from .reference import hh_exact
from .totalweights import CentredBlockHH

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('linear', 'IFFT-FFT', 'pairwise', 'B1', 'B1_all', 'B0')
SEED = 20261017


def population_setup(name):
    """Return the fixed segment and prior for Eqs. (47), (51), (55)."""
    common = dict(dt=1/4096, f_low=20., m_max=4)
    choices = {
        'bns': dict(T=256., f_high=2048., durations=[256,128,64,32,16,8,4],
                    mchirp_ref=1., mchirp_range=(1.,1.8)),
        'nsbh': dict(T=16., f_high=1024., durations=[16,8,4],
                     mchirp_ref=4., mchirp_range=(4.,9.)),
        'bbh': dict(T=16., f_high=1024., durations=[16,8,4],
                    mchirp_ref=12., mchirp_range=(12.,50.))}
    if name.lower() not in choices:
        raise ValueError('Population must be bns, nsbh or bbh')
    return {**common, **choices[name.lower()]}


def chirp_mass(params):
    """Mc=(m1 m2)^(3/5)/(m1+m2)^(1/5), in solar masses."""
    a, b = params['m1'], params['m2']
    return (a*b)**.6/(a+b)**.2


class PopulationSamples(list):
    """Parameter dictionaries with rejection counts, without model metadata."""
    def __init__(self):
        super().__init__()
        self.rejections = dict(chirp_mass=0, chirp_mass_upper=0, mass_ratio=0)
        self.proposals = 0


def sample_population(name, n, seed, strong_hom=False):
    """Draw reproducible conditional priors; see module docstring.

    Regular angles have cos(iota) uniform in [-1,1], phi in [0,2pi),
    psi in [0,pi); F+=(cos 2psi), Fx=(sin 2psi). Strong-HOM angles use
    the same isotropic prior restricted to (1.3,1.8). Independent streams
    by population/subset preserve prefixes when n changes. The specified
    NSBH mass box can exceed the pairwise upper anchor Mc=9; regular
    draws are retained and flagged, while strong-HOM draws stay inside it.
    """
    name = name.lower()
    cfg = population_setup(name)
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if strong_hom and name == 'bns':
        raise ValueError('No strong-HOM BNS population is specified')
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, ('bns','nsbh','bbh').index(name), int(strong_hom)]))
    result = PopulationSamples()
    while len(result) < n:
        result.proposals += 1
        if name == 'bns':
            m1, m2 = sorted(rng.uniform(1.2,2.,2), reverse=True)
            s1, s2 = rng.uniform(-.05,.05,2)
        elif name == 'nsbh':
            m1, m2 = rng.uniform(15,35), rng.uniform(1.5,4)
            s1, s2 = rng.uniform(-.5,.5), 0.
        else:
            m1, m2 = rng.uniform(30,80), rng.uniform(10,40)
            s1, s2 = rng.uniform(-.8,.8,2)
        if (name == 'bbh' and m2/m1 < .1) or m2 > m1 or (strong_hom and m2/m1 > .15):
            result.rejections['mass_ratio'] += 1
            continue
        if chirp_mass(dict(m1=m1,m2=m2)) < cfg['mchirp_ref']:
            result.rejections['chirp_mass'] += 1
            continue
        if strong_hom and chirp_mass(dict(m1=m1,m2=m2)) > cfg['mchirp_range'][1]:
            result.rejections['chirp_mass_upper'] += 1
            continue
        cosi = rng.uniform(np.cos(1.8),np.cos(1.3)) if strong_hom else rng.uniform(-1,1)
        psi = rng.uniform(0,np.pi)
        result.append(dict(m1=float(m1), m2=float(m2), chi1z=float(s1), chi2z=float(s2),
                           distance_mpc=100., inclination=float(np.arccos(cosi)),
                           phi_ref=float(rng.uniform(0,2*np.pi)),
                           t_c=float(rng.uniform(cfg['T']-2.12,cfg['T']-1.9)),
                           f_ref=20., F_plus=float(np.cos(2*psi)), F_cross=float(np.sin(2*psi))))
    return result


def make_bands(setup, L):
    """Eq. (51) bands at fixed durations, with the requested margin L."""
    return BandSetup(**{k: setup[k] for k in ('T','dt','f_low','f_high','durations',
                                             'mchirp_ref','m_max')}, L=L)


def population_lines():
    """Fixed Lorentzian list tied to the BNS L=5 and L=50 transitions.

    One line in each L=5 band, and a line at f_b - Df_b/2 for each
    internal edge at both L values. HWHM cycles .01,.03,.1,.5 Hz and
    heights 1e2,1e3,1e4. This SAME PSD is used for every population/L.
    """
    bands = [make_bands(population_setup('bns'), L) for L in (5,50)]
    centres = [float((lo+hi)/2) for lo,hi in zip(bands[0].f_b[:-1],
                                               np.minimum(bands[0].f_b[1:],2048.))]
    for b in bands:
        centres.extend(float(f-df/2) for f,df in zip(b.f_b[1:-1],b.Df_b[1:-1]))
    widths, heights = (.01,.03,.1,.5), (100.,1000.,10000.)
    return [(f,widths[i%4],heights[i%3]) for i,f in enumerate(centres)]


def population_psd(name, setup):
    """Eq. (6) analytic+Lorentzians or measured L1 log-log interpolation.

    L1 Welch16s has 1/16 Hz resolution, interpolated to 1/256 Hz for BNS;
    finer sampling does not restore unresolved measured spectral features.
    """
    if name == 'lines':
        fn = add_lorentzian_lines(aligo_analytic_psd, population_lines())
    elif name == 'L1':
        fn = load_asd_or_psd(ROOT/'data/psd_L1_GW190814_welch16s.txt', skip_dc=True)
    else:
        raise ValueError('PSD must be lines or L1')
    return PSDGrid(fn, **{k:setup[k] for k in ('T','dt','f_low','f_high')})


class PopulationCache:
    """Store scalar exact Eq. (4) norms per point; never retain fine waveforms.

    The key includes point index, all model parameters, grid, actual inverse
    PSD bytes and LAL version. L and coarse methods are deliberately absent.
    Each miss writes atomically in the cache directory, so interrupted runs
    can resume. Files are small, git-ignored data/population_cache_*.npz.
    """
    def __init__(self, path, psdgrid):
        self.path, self.psdgrid = Path(path), psdgrid
        self.signature = hashlib.sha256(psdgrid.inv_S.tobytes()).hexdigest()
        self.values = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as z:
                self.values = dict(zip(z['keys'].tolist(), z['hh'].tolist()))
        self.hits = self.misses = 0

    def key(self, index, params):
        spec = dict(index=index, params=params, T=self.psdgrid.T, dt=self.psdgrid.dt,
                    psd=self.signature, lal=lalsimulation.__version__,
                    reference='XHM-total_direct-sequence-v1')
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def get(self, index, params):
        """Compute total_direct on the full positive fine sequence on a miss."""
        key = self.key(index, params)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        model = XHMModel(**params)
        exact = hh_exact(model.total_direct(self.psdgrid.f_k), self.psdgrid)
        if not np.isfinite(exact) or exact <= 0:
            raise ValueError('Invalid exact norm')
        self.values[key] = exact
        self.misses += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix('.pending.npz')
        np.savez(pending, keys=np.array(list(self.values)), hh=np.array(list(self.values.values())))
        pending.replace(self.path)
        return exact


def make_evaluators(params, setup, psdgrid, bands):
    """Prepare fixed grids and all six methods; retain requested slack flags."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        pair = PairwiseHH(bands, psdgrid, XHM_MODES, mchirp_range=setup['mchirp_range'])
    # Full-grid negative unwrapped slack is harmless; no truncated-band
    # failure may be hidden by PairwiseHH's historical margin clipping.
    bad = [r for r in pair.summary() if not r['full_grid'] and r['slack'] < 0]
    if bad:
        raise ValueError(f'Negative requested truncated-band slack: {bad}')
    return dict(adapter=XHMCoarseEvaluator(params,bands),
                linear=LinearInterpolationHH(bands,psdgrid),
                **{'IFFT-FFT':IFFTFFTHH(bands,psdgrid,impl='fast')},
                pairwise=pair, B1=BasebandIFFTFFTHH(bands,psdgrid),
                B1_all=BasebandIFFTFFTHH(bands,psdgrid,all_half_bins=True),
                B0=CentredBlockHH(bands,psdgrid))


def evaluate_point(params, setup, psdgrid, bands, evaluators, L):
    """Signed relative (HH_method-HH_exact)/HH_exact for all six methods.

    Eq. (4) exact uses XHMModel.total_direct on the fine sequence, independent
    of L. setup may supply exact_hh or cache plus index; otherwise compute it
    without caching. Prepared XHMCoarseEvaluator supplies both polarization
    and projected mode samples, preserving the iteration-005 BBH tails.
    Optional setup['timings'] collects warm-call seconds by component.
    """
    from time import perf_counter
    if bands.L != L:
        raise ValueError('L must match bands.L')
    if 'exact_hh' in setup:
        exact = setup['exact_hh']
    elif 'cache' in setup:
        exact = setup['cache'].get(setup['index'],params)
    else:
        exact = hh_exact(XHMModel(**params).total_direct(psdgrid.f_k),psdgrid)
    if not np.isfinite(exact) or exact <= 0:
        raise ValueError('Invalid exact norm')
    timings = setup.get('timings')
    start = perf_counter()
    co = evaluators['adapter'].coarse_total(params)
    total_time = perf_counter()-start
    start = perf_counter()
    cm = evaluators['adapter'].coarse_modes(params)
    mode_time = perf_counter()-start
    if timings is not None:
        timings.setdefault('coarse_total',[]).append(total_time)
        timings.setdefault('coarse_modes',[]).append(mode_time)
    result = {}
    for method in METHODS:
        start = perf_counter()
        value = evaluators[method](cm if method == 'pairwise' else co)
        elapsed = perf_counter()-start
        result[method] = float((value-exact)/exact)
        if not np.isfinite(result[method]):
            raise ValueError(f'Nonfinite {method} norm')
        if timings is not None:
            timings.setdefault(method,[]).append(elapsed)
    return result
