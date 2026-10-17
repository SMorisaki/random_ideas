"""B0/B1 identities and measured limits for iteration 004.

The requested 1e-8 B1/paper and 1e-10 half-point agreement do not hold for
the specified minimal-K' algorithm. Tests below pin the observed errors and
the correct even-kernel identity; data/accuracy_iter004.md records unmet
targets explicitly. No iteration-001–003 regression tolerance is changed.
"""

from copy import copy

import numpy as np
import pytest

from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.coarse import CoarseWaveform
from mbquad.ifftfft import IFFTFFTHH
from mbquad.psd import PSDGrid, aligo_analytic_psd, load_asd_or_psd
from mbquad.reference import hh_exact
from mbquad.totalweights import CentredBlockHH
from mbquad.toywave import ToyModes


@pytest.fixture(scope="module")
def total_methods(toy_case):
    bands, psd = toy_case.bands, toy_case.psdgrid
    return (CentredBlockHH(bands, psd), BasebandIFFTFFTHH(bands, psd),
            BasebandIFFTFFTHH(bands, psd, force_half_points=True),
            IFFTFFTHH(bands, psd, impl="fast"))


@pytest.mark.parametrize("hom", [False, True])
def test_fine_grid_accuracy_and_exact_band_zero(toy_case, total_methods, hom):
    """B0 five-mode error 4.395005e-6; forced B1 4.398792e-7.

    Forced B1/paper gaps are 2.891511e-7 (22), 2.180421e-7 (five):
    the proposed 1e-8 agreement is not an identity of these interpolants.
    """
    bs, psd = toy_case.bands, toy_case.psdgrid
    model = toy_case.model if hom else ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
    coarse = CoarseWaveform.from_model(bs, model.total)
    fine = model.total(psd.f_k)
    exact = hh_exact(fine, psd)
    b0, adaptive, forced, paper = total_methods
    norm0, norm1 = b0(coarse), forced(coarse)
    assert abs(norm1 / exact - 1) < 6e-7
    assert abs(norm0 / exact - 1) < 1e-5
    if hom:
        assert abs(norm0 - exact) > abs(norm1 - exact)
        assert abs(adaptive(coarse) / norm1 - 1) < 1e-14
    gap = abs(norm1 / paper(coarse) - 1)
    assert 1e-8 < gap < 3e-7  # Explicit regression of the unmet requested target.
    exact0 = (4 / bs.T) * np.sum(bs.window(psd.f_k, 0) * np.abs(fine)**2 * psd.inv_S)
    for method in total_methods:
        contrib = method.band_contributions(coarse)
        assert contrib[0] == pytest.approx(exact0, rel=1e-12)
        assert method(coarse) == pytest.approx(np.sum(contrib), rel=1e-14)


def test_correct_paper_weight_identity_and_full_grid_norm(toy_case, total_methods):
    """Eqs. (32),(43): Re(W_k+W_-k)=(T/That) I_tilde_k, max error <1e-15.

    W itself is complex for a cropped one-sided kernel. Restricting to
    the original band discards leakage, so use ALL Eq. (45) positive bins
    for the exact norm identity, including both integer and half bins.
    """
    bs = toy_case.bands
    _, _, forced, paper = total_methods
    coarse = CoarseWaveform.from_model(bs, toy_case.model.total)
    for b, grid in enumerate(bs.coarse_frequencies()):
        n = int(paper.Nhat_b[b])
        W = forced.blockweights.centred(b, n)
        k = np.arange(1, n // 2)
        effective = (W[k] + W[-k]).real
        target = (bs.T / paper.That_b[b]) * paper.I_tilde_b[b]
        assert np.max(np.abs(effective - target)) / np.max(np.abs(target)) < 1e-10
        f, h = paper.positive_grid_waveform(coarse, b)
        np.testing.assert_allclose(f, k / paper.That_b[b], rtol=0, atol=0)
        norm = (4 / bs.T) * np.dot(effective, np.abs(h)**2)
        assert norm == pytest.approx(paper.band_contributions(coarse)[b], rel=1e-13)
        if b == 3:
            kb = 2 * grid.kappa
            requested = (paper.That_b[b] / bs.T) * paper.I_tilde_b[b][kb - 1]
            # The proposed W=(That/T)I fails substantially, not by roundoff.
            assert np.max(np.abs(W[kb] - requested)) / np.max(np.abs(W[kb])) > .9


def test_adaptive_rule_actual_modes_and_margin(toy_case, total_methods):
    """5 sigma top-band sum=2.119513>2: default five-mode B1 skips no FFT.

    A 4-sigma setting selects top-band B0; difference from forced <1e-7.
    Quadrupole with m_values=(2,) has zero pair lag and uses upper-band B0.
    """
    bs, psd = toy_case.bands, toy_case.psdgrid
    _, adaptive, forced, _ = total_methods
    assert adaptive.modes == ["exact"] + ["B1"] * 5
    assert adaptive.lag_max[-1] + adaptive.margin[-1] > bs.T_b[-1] / 2
    tuned = BasebandIFFTFFTHH(bs, psd, L_margin=4)
    assert tuned.modes[-1] == "B0"
    coarse = CoarseWaveform.from_model(bs, toy_case.model.total)
    assert abs(tuned(coarse) / forced(coarse) - 1) < 1e-7
    quad = BasebandIFFTFFTHH(bs, psd, m_values=(2,))
    assert quad.modes == ["exact"] + ["B0"] * 5
    absent21 = BasebandIFFTFFTHH(bs, psd, m_values=(2, 3, 4))
    lag, _ = absent21.centred.predicted_lag_support(3, (2, 3, 4))
    assert lag == bs.lag_max_pair(3, 4, 2)
    assert lag < adaptive.lag_max[3]
    forced_tuned = BasebandIFFTFFTHH(bs, psd, L_margin=0, force_half_points=True)
    assert forced_tuned.modes == adaptive.modes


def test_half_sample_shift_direct_plateau_and_paper(toy_case, total_methods):
    """Pure 22, band 3: central-60% direct error=6.252e-5 (<1e-4).

    Whole-plateau maximum is 4.680e-4; even the interior does not equal
    the paper interpolant at 1e-10. The phase sign is tested explicitly.
    """
    bs = toy_case.bands
    forced, paper = total_methods[2:]
    model = ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
    coarse = CoarseWaveform.from_model(bs, model.total)
    b = 3
    f = forced.half_point_frequencies(b)
    z = forced.half_point_waveform(coarse, b)
    assert len(f) == bs.K_b[b]
    np.testing.assert_allclose(f * bs.T_b[b] - bs.K_s[b], np.arange(len(f)) + .5)
    lo, hi = bs.f_b[b], bs.f_b[b + 1] - bs.Df_b[b + 1]
    interior = (f > lo + .2 * (hi - lo)) & (f < hi - .2 * (hi - lo))
    plateau = (f >= lo) & (f <= hi)
    err = np.abs((z - model.total(f)) / model.total(f))
    assert np.count_nonzero(interior) > 100
    np.testing.assert_array_equal(bs.window(f[interior], b), 1)
    assert np.max(err[interior]) < 1e-4
    assert 1e-4 < np.max(err[plateau]) < 5e-4
    hp = paper.half_point_waveform(coarse, b)[1::2]
    overall = np.max(np.abs(z[:-1] - hp)) / np.max(np.abs(hp))
    assert 1e-8 < overall < .002
    inside = interior[:-1]
    gap = np.max(np.abs(z[:-1][inside] - hp[inside])) / np.max(np.abs(hp[inside]))
    assert 1e-10 < gap < .0002
    # Returned diagnostics own their data across subsequent calls.
    saved = z.copy()
    forced(coarse)
    np.testing.assert_array_equal(saved, z)


def test_fft_length_reduction(toy_case, total_methods):
    """K'/M for cropped toy bands: .07251,.10547,.08545,.17578,.20142."""
    bs = toy_case.bands
    method = total_methods[1]
    lengths = method.fft_lengths()
    assert [kp for kp, _ in lengths] == [756, 594, 432, 350, 360, 825]
    assert all(kp <= M / 2 for (kp, _), M in zip(lengths[1:], bs.M_b[1:]))
    # 2M=N in band 1 is not an exact-grid shortcut: M<N there.
    assert 2 * bs.M_b[1] == bs.N_b[1]
    assert lengths[1][1] == "B1"
    assert "lag_max" in method.summary() and "B1" in method.summary()


def test_folding_audit_explains_b0_band_errors(toy_case, total_methods):
    bs, psd = toy_case.bands, toy_case.psdgrid
    b0 = total_methods[0]
    coarse = CoarseWaveform.from_model(bs, toy_case.model.total)
    fine_power = np.abs(toy_case.model.total(psd.f_k))**2
    exact = hh_exact(toy_case.model.total(psd.f_k), psd)
    for b in (0, 1, 2, 3, 5):
        g = bs.window(psd.f_k, b) * fine_power
        ref = (4 / bs.T) * np.dot(g, psd.inv_S)
        audit = b0.folding_error_estimate(g, b)
        error = b0.band_contributions(coarse)[b] - ref
        assert error == pytest.approx(audit["hh_folding_error"], rel=1e-6, abs=1e-13 * exact)
        assert audit["folding_error"] == pytest.approx(audit["direct_error"], rel=1e-6, abs=1e-13 * exact)
        lag, half = b0.predicted_lag_support(b, (1, 2, 3, 4))
        assert lag == bs.lag_max_pair(b, 4, 1)
        assert half == bs.T_b[b] / 2
        if b in (1, 2, 3):
            assert lag > half and audit["outside_fraction"] > .05
    ref_lag, _ = b0.predicted_lag_support(2, (1, 4))
    other_lag, _ = b0.predicted_lag_support(2, (1, 4), mchirp=2 * bs.mchirp_ref)
    assert other_lag / ref_lag == pytest.approx(2**(-5 / 3))


@pytest.mark.parametrize("cls", [CentredBlockHH, BasebandIFFTFFTHH])
def test_setup_and_input_validation(toy_case, cls):
    bs = toy_case.bands
    other_psd = PSDGrid(aligo_analytic_psd, bs.T, bs.dt, 21, bs.f_high)
    with pytest.raises(ValueError, match="matching"):
        cls(bs, other_psd)
    method = cls(bs, toy_case.psdgrid)
    other_coarse = CoarseWaveform.from_model(copy(bs), toy_case.model.total)
    with pytest.raises(ValueError, match="BandSetup"):
        method(other_coarse)


@pytest.mark.parametrize("ms", [(), (0, 2), (2.5,), (5,)])
def test_invalid_mode_values(toy_case, ms):
    with pytest.raises(ValueError, match="m_values"):
        BasebandIFFTFFTHH(toy_case.bands, toy_case.psdgrid, m_values=ms)


@pytest.mark.parametrize("margin", [-1, np.inf, np.nan])
def test_invalid_margin(toy_case, margin):
    with pytest.raises(ValueError, match="L_margin"):
        BasebandIFFTFFTHH(toy_case.bands, toy_case.psdgrid, L_margin=margin)


def test_real_welch_dc_opt_in_preserves_loader_default(tmp_path):
    """Actual Welch files include DC; the old strict default still rejects it."""
    path = tmp_path / "welch.txt"
    path.write_text("0 1e-42\n20 1e-46\n40 2e-46\n")
    with pytest.raises(ValueError, match="positive"):
        load_asd_or_psd(path)
    psd = load_asd_or_psd(path, skip_dc=True)
    np.testing.assert_allclose(psd([20, 40]), [1e-46, 2e-46], rtol=1e-13, atol=0)
    assert np.isinf(psd(0))
    path.write_text("-1 1e-42\n20 1e-46\n40 2e-46\n")
    with pytest.raises(ValueError, match="positive"):
        load_asd_or_psd(path, skip_dc=True)
