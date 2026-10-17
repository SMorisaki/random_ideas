"""Reusable documented systems for regression checks and later iterations."""

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
