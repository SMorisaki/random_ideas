"""Documented SPA and XHM systems for Morisaki Eq. (26) comparisons.

Each helper states its geometry and prior. Fixed cases differ from the
random populations; see REPORT.md §1 before comparing result tables.
"""

from dataclasses import dataclass

from .bands import BandSetup
from .psd import PSDGrid, add_lines, aligo_analytic_psd
from .toywave import ToyModes


TOY_LINES = ((26.5, 0.01, 3000), (36.0, 0.01, 2000),
             (48.5, 0.01, 1000), (60.0, 0.02, 500), (120.0, 0.05, 100))


@dataclass(frozen=True)
class ToyCase:
    """Model, fine PSDGrid and BandSetup for a reproducible comparison."""

    model: ToyModes
    psdgrid: PSDGrid
    bands: BandSetup


def toy_small():
    """Return the iteration-001 128-s, 20–256 Hz five-mode test system.

    dt = 1/1024 s; durations = (128, 64, 32, 16, 8, 4) s; chirp mass and
    reference mass 4.85 Msun; t_c = 126 s; m_max = 4; L = 5. The analytic
    aLIGO PSD has the five TOY_LINES Gaussian lines, resolving narrow
    inverse-noise features on the fine grid. Geometry follows Eqs. (47),
    (51), (53), (55), (56); the exact reference uses Eq. (4).
    """
    T, dt, f_low, f_high = 128.0, 1 / 1024, 20.0, 256.0
    bands = BandSetup(T, dt, f_low, f_high, [128, 64, 32, 16, 8, 4],
                      mchirp_ref=4.85, m_max=4, L=5)
    psdgrid = PSDGrid(add_lines(aligo_analytic_psd, TOY_LINES), T, dt, f_low, f_high)
    return ToyCase(ToyModes(mchirp=4.85, t_c=T - 2.0), psdgrid, bands)


def xhm_systems():
    """Documented iteration-005 systems; dt=1/4096 s and L=5 throughout.

    q=0.1 variants in the accuracy study keep m1 fixed and set m2=0.1*m1;
    their original band geometry/prior is retained as a stress test.
    """
    common = dict(chi1z=0., chi2z=0., dt=1/4096, f_low=20., m_max=4,
                  L=5., phi_ref=.7, F_plus=.6, F_cross=.5, f_ref=20.)
    systems = {}
    for name, p in [
        ('BNS', dict(m1=1.4, m2=1.4, T=256., f_high=2048.,
                     durations=[256,128,64,32,16,8,4], mchirp_ref=1.15,
                     mchirp_range=(1.15,1.25), inclination=1., distance_mpc=40.)),
        ('GW190814', dict(m1=23., m2=2.6, chi1z=.1, T=16., f_high=1024.,
                     durations=[16,8,4], mchirp_ref=6.4, mchirp_range=(6.,7.),
                     inclination=.8, distance_mpc=240.)),
        ('BBH', dict(m1=60., m2=20., chi1z=.3, chi2z=-.2, T=16., f_high=1024.,
                     durations=[16,8,4], mchirp_ref=25., mchirp_range=(25.,35.),
                     inclination=1.3, distance_mpc=800.))]:
        systems[name] = {**common, **p}
        systems[name]['t_c'] = p['T']-2.
    return systems


def xhm_case(name, psd='analytic_lines'):
    """Return XHMModel, PSDGrid and bands for one xhm_systems name.

    Real L1/H1 Welch files have resolution 1/16 Hz. load_asd_or_psd
    interpolates log PSD in log frequency onto PSDGrid's k/T grid; this
    interpolation cannot recover unresolved narrow noise lines.
    """
    from pathlib import Path
    from .lalwave import XHMModel
    from .psd import load_asd_or_psd
    p = xhm_systems()[name]
    model = XHMModel(**{k:p[k] for k in ('m1','m2','chi1z','chi2z','distance_mpc',
                       'inclination','phi_ref','t_c','f_ref','F_plus','F_cross')})
    bands = BandSetup(**{k:p[k] for k in ('T','dt','f_low','f_high','durations',
                                        'mchirp_ref','m_max','L')})
    if psd == 'analytic_lines':
        fn = add_lines(aligo_analytic_psd, TOY_LINES)
    elif psd in ('L1_real', 'H1_real'):
        fn = load_asd_or_psd(Path(__file__).resolve().parents[1] /
                            f'data/psd_{psd[:2]}_GW190814_welch16s.txt', skip_dc=True)
    else:
        raise ValueError('Unknown PSD')
    return ToyCase(model, PSDGrid(fn, p['T'], p['dt'], p['f_low'], p['f_high']), bands)
