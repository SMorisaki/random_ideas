"""Exact norm partition and the specified signed log-likelihood error scale."""

import numpy as np
import pytest

from mbquad import PSDGrid, hh_exact, hh_exact_banded, lnlambda_error


def test_exact_banded_matches_fine_grid(toy_case):
    case = toy_case
    h = case.model.total(case.psdgrid.f_k)
    exact = hh_exact(h, case.psdgrid)
    assert exact > 0
    assert hh_exact_banded(case.model.total, case.psdgrid, case.bands) == pytest.approx(exact, rel=1e-12)
    assert hh_exact(3 * h, case.psdgrid) == pytest.approx(9 * exact, rel=1e-14)


def test_constant_waveform_normalization_and_unused_values():
    grid = PSDGrid(lambda f: 2, T=8, dt=1 / 16, f_low=2, f_high=4)
    h = np.full_like(grid.f_k, 3, dtype=complex)
    # Inclusive frequencies 2, 2.125, ..., 4 give 17 bins.
    assert hh_exact(h, grid) == 4 / 8 * 17 * 9 / 2
    h[grid.inv_S == 0] = np.nan
    assert hh_exact(h, grid) == 4 / 8 * 17 * 9 / 2
    with pytest.raises(ValueError, match="shape"):
        hh_exact(h[:-1], grid)
    h[grid.inv_S > 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        hh_exact(h, grid)


def test_banded_checks_grid_compatibility(toy_case, bbh_bands):
    with pytest.raises(ValueError, match="matching"):
        hh_exact_banded(toy_case.model.total, toy_case.psdgrid, bbh_bands)


def test_lnlambda_error_sign_and_snr_scaling():
    assert lnlambda_error(101, 100, 20) == 2
    assert lnlambda_error(99, 100, 20) == -2
    assert lnlambda_error(100, 100, 20) == 0
    assert lnlambda_error(101, 100, 40) == 8
    assert lnlambda_error(101, 100, 0) == 0
    assert lnlambda_error(202, 200, 20) == 2
    np.testing.assert_array_equal(lnlambda_error([99, 100, 101], 100, 20), [-2, 0, 2])


@pytest.mark.parametrize("approx,exact,snr", [(1, 0, 25), (1, -1, 25), (1, 1, -1), (np.nan, 1, 25)])
def test_invalid_lnlambda_inputs(approx, exact, snr):
    with pytest.raises(ValueError):
        lnlambda_error(approx, exact, snr)
