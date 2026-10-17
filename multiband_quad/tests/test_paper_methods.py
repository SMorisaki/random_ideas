"""Regression checks for Morisaki Eqs. (27)–(46), on iteration-001 grids."""

from copy import copy

import numpy as np
import pytest

from mbquad.bands import BandSetup
from mbquad.coarse import CoarseWaveform
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from mbquad.psd import PSDGrid, aligo_analytic_psd
from mbquad.reference import hh_exact
from mbquad.toywave import ToyModes


@pytest.fixture(scope="module")
def methods(toy_case):
    return (LinearInterpolationHH(toy_case.bands, toy_case.psdgrid),
            IFFTFFTHH(toy_case.bands, toy_case.psdgrid, impl="paper"),
            IFFTFFTHH(toy_case.bands, toy_case.psdgrid, impl="fast"))


def test_coarse_constructors(toy_case):
    bands = toy_case.bands
    f = bands.all_coarse_frequencies().f
    h = toy_case.model.total(f)
    from_array = CoarseWaveform.from_array(bands, h)
    from_model = CoarseWaveform.from_model(bands, toy_case.model.total)
    for b, grid in enumerate(bands.coarse_frequencies()):
        np.testing.assert_array_equal(from_array.h_b[b], from_model.h_b[b])
        np.testing.assert_allclose(from_array.h_b[b], toy_case.model.total(grid.f))
    with pytest.raises(ValueError, match="K_MB"):
        CoarseWaveform.from_array(bands, h[:-1])
    with pytest.raises(ValueError, match="finite"):
        CoarseWaveform.from_array(bands, np.full_like(h, np.nan))
    with pytest.raises(ValueError, match="per band"):
        CoarseWaveform(bands, [h])


def test_linear_constant_power_identity():
    """Eq. (29) partitions unity: sum c = (4/T) sum 1/S for w=1.

    Each fine point's two linear basis weights sum to one, even in an
    extrapolated endpoint interval. Use a single band with its final taper
    above the analysis interval, hence no window transitions on PSD support.
    """
    bands = BandSetup(16, 1 / 128, 5, 40, [16], 6.4, 4, 5)
    psd = PSDGrid(lambda f: 2.0, bands.T, bands.dt, bands.f_low, bands.f_high)
    linear = LinearInterpolationHH(bands, psd)
    active = psd.inv_S > 0
    np.testing.assert_array_equal(bands.window(psd.f_k[active], 0), 1)
    expected = (4 / bands.T) * np.sum(psd.inv_S)
    assert np.sum(linear.c_b[0]) == pytest.approx(expected, rel=1e-14)
    coarse = CoarseWaveform.from_array(bands, np.ones(bands.K_MB, dtype=complex))
    assert linear(coarse) == pytest.approx(expected, rel=1e-14)


def test_linear_endpoint_ranges_use_true_weight_frequencies(toy_case, methods):
    """Compare Eq. (29) directly, including its extrapolated end bins."""
    bands, psd = toy_case.bands, toy_case.psdgrid
    linear = methods[0]
    for b, grid in enumerate(bands.coarse_frequencies()):
        f_tilde = grid.f.copy()
        f_tilde[0], f_tilde[-1] = 0, np.inf
        a = bands.window(psd.f_k, b) * psd.inv_S
        for j in (0, 1, len(grid.f) // 2, len(grid.f) - 2, len(grid.f) - 1):
            c = 0.0
            if j < len(grid.f) - 1:
                mask = (psd.f_k >= f_tilde[j]) & (psd.f_k < f_tilde[j + 1])
                c += np.sum((grid.f[j + 1] - psd.f_k[mask]) * a[mask])
            if j > 0:
                mask = (psd.f_k >= f_tilde[j - 1]) & (psd.f_k < f_tilde[j])
                c += np.sum((psd.f_k[mask] - grid.f[j - 1]) * a[mask])
            c *= 4 * bands.T_b[b] / bands.T
            assert linear.c_b[b][j] == pytest.approx(c, rel=2e-14, abs=0)


@pytest.mark.parametrize("hom", [False, True])
@pytest.mark.parametrize("lines", [False, True])
def test_accuracy_and_fft_agreement(toy_case, methods, hom, lines):
    case = toy_case
    model = case.model if hom else ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
    if lines:
        psd = case.psdgrid
        linear, paper, fast = methods
    else:
        psd = PSDGrid(aligo_analytic_psd, case.bands.T, case.bands.dt,
                      case.bands.f_low, case.bands.f_high)
        linear = LinearInterpolationHH(case.bands, psd)
        paper = IFFTFFTHH(case.bands, psd, impl="paper")
        fast = IFFTFFTHH(case.bands, psd, impl="fast")
    coarse = CoarseWaveform.from_model(case.bands, model.total)
    exact = hh_exact(model.total(psd.f_k), psd)
    lin, fft_p, fft_f = linear(coarse), paper(coarse), fast(coarse)
    assert abs(lin / exact - 1) < 5e-6
    assert abs(fft_p / exact - 1) < 1e-6
    assert fft_f == pytest.approx(fft_p, rel=1e-12)
    np.testing.assert_allclose(paper.band_contributions(coarse),
                               fast.band_contributions(coarse), rtol=1e-12, atol=0)
    if hom and lines:
        assert 3 * abs(fft_p - exact) < abs(lin - exact)
    # Band 0 has M_b=N_b: neither waveform nor inverse-PSD lags are cropped.
    assert case.bands.M_b[0] == case.bands.N_b[0]
    exact0 = (4 / psd.T) * np.sum(case.bands.window(psd.f_k, 0)
                                * np.abs(model.total(psd.f_k))**2 * psd.inv_S)
    assert paper.band_contributions(coarse)[0] == pytest.approx(exact0, rel=1e-10)


def test_half_point_waveform_interior(toy_case, methods):
    """Eq. (46), pure (2,2): max half-point error 4.213e-5 in band 5.

    Interior means the central 60% of the final band's unit-window plateau.
    Compare actual half points only, with pointwise |delta h|/|h|. Lower
    bands retain larger tails (up to 2.5e-4 under this interior definition),
    so this is a specified interior regression, not a global error bound.
    """
    bands = toy_case.bands
    model = ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
    coarse = CoarseWaveform.from_model(bands, model.total)
    b = bands.B - 1
    paper, fast = methods[1:]
    f = paper.half_point_frequencies(b)
    h = paper.half_point_waveform(coarse, b)
    np.testing.assert_allclose(fast.half_point_waveform(coarse, b), h,
                               rtol=2e-12, atol=1e-38)
    span = bands.f_b[b + 1] - bands.Df_b[b + 1] - bands.f_b[b]
    interior = ((f > bands.f_b[b] + 0.2 * span)
                & (f < bands.f_b[b + 1] - bands.Df_b[b + 1] - 0.2 * span))
    half = np.isclose(np.mod(f * bands.T_b[b], 1), 0.5)
    mask = interior & half
    assert np.count_nonzero(mask) > 100
    np.testing.assert_array_equal(bands.window(f[mask], b), 1)
    exact = model.total(f[mask])
    assert np.max(np.abs((h[mask] - exact) / exact)) < 1e-4


def test_fft_zero_extension_beyond_original_nyquist():
    """Eq. (32) uses zero inverse PSD beyond the original positive grid."""
    bands = BandSetup(8, 1 / 128, 5, 64, [8], 20, 4, 5)
    assert bands.N_b[0] > bands.N
    psd = PSDGrid(lambda f: 2, bands.T, bands.dt, bands.f_low, bands.f_high)
    model = ToyModes(20, t_c=6)
    coarse = CoarseWaveform.from_model(bands, model.total)
    exact = hh_exact(model.total(psd.f_k), psd)
    for impl in ("paper", "fast"):
        fft = IFFTFFTHH(bands, psd, impl=impl)
        assert fft(coarse) == pytest.approx(exact, rel=1e-12, abs=0)


def test_setup_and_impl_validation(toy_case, methods):
    bands = toy_case.bands
    other = PSDGrid(aligo_analytic_psd, bands.T, bands.dt, 21, bands.f_high)
    for cls in (LinearInterpolationHH, IFFTFFTHH):
        with pytest.raises(ValueError, match="matching"):
            cls(bands, other)
    with pytest.raises(ValueError, match="impl"):
        IFFTFFTHH(bands, toy_case.psdgrid, impl="unknown")
    copied_bands = copy(bands)
    coarse = CoarseWaveform.from_model(copied_bands, toy_case.model.total)
    for method in methods:
        with pytest.raises(ValueError, match="BandSetup"):
            method(coarse)
