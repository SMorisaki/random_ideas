"""Multiband gravitational-wave quadrature research tools.

Equations refer to Morisaki, Phys. Rev. D 104, 044062 (2021), arXiv:2104.07813.
"""

from .bands import AllCoarseGrid, BandSetup, CoarseGrid
from .psd import PSDGrid, add_lines, aligo_analytic_psd, load_asd_or_psd
from .reference import hh_exact, hh_exact_banded, lnlambda_error
from .toywave import ToyModes, dtau_0pn_df, psi_22, tau_0pn

__all__ = [
    "AllCoarseGrid", "BandSetup", "CoarseGrid", "PSDGrid", "ToyModes", "add_lines",
    "aligo_analytic_psd", "dtau_0pn_df", "hh_exact", "hh_exact_banded",
    "lnlambda_error", "load_asd_or_psd", "psi_22", "tau_0pn",
]
