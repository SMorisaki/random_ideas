"""SPA time/phase relation, arbitrary frequency ordering and optional tapers."""

import numpy as np
import pytest

from mbquad import ToyModes, dtau_0pn_df, psi_22, tau_0pn


def test_newtonian_time_normalization():
    # Eq. (53), with MSUN_S = 4.925491025543576e-6: about 160 s at 20 Hz.
    assert tau_0pn(20, 1.21) == pytest.approx(159.781154, rel=1e-8)
    assert tau_0pn(40, 1.21) / tau_0pn(20, 1.21) == pytest.approx(2**(-8 / 3))
    assert tau_0pn(20, 2.42) / tau_0pn(20, 1.21) == pytest.approx(2**(-5 / 3))
    f = np.array([20.0, 50, 100])
    step = f * 1e-5
    numeric = (tau_0pn(f + step, 4.85) - tau_0pn(f - step, 4.85)) / (2 * step)
    np.testing.assert_allclose(numeric, dtau_0pn_df(f, 4.85), rtol=1e-8)


def test_mode_phase_derivative_matches_stationary_time(toy_case):
    model = toy_case.model
    f = np.array([25.0, 35, 60, 120, 240])
    step = f * 1e-4
    for lm in model.modes:
        np.testing.assert_allclose(model.tau_lm(f, lm), tau_0pn(2 * f / lm[1], model.mchirp), rtol=1e-14)
        # A five-point derivative avoids phase wrapping and resolves the
        # Newtonian chirp correction on top of the linear coalescence phase.
        derivative = (model.phase(f - 2 * step, lm) - 8 * model.phase(f - step, lm)
                      + 8 * model.phase(f + step, lm) - model.phase(f + 2 * step, lm)) / (12 * step)
        np.testing.assert_allclose(derivative / (2 * np.pi), model.t_c - model.tau_lm(f, lm), rtol=1e-6)


def test_sum_and_unsorted_frequencies(toy_case):
    model = toy_case.model
    f = np.array([100.0, 20, 250, 31.25, 20, 72])
    order = np.argsort(f)
    np.testing.assert_array_equal(model.total(f), sum(model.mode(f, lm) for lm in model.modes))
    for lm in model.modes:
        np.testing.assert_array_equal(model.mode(f, lm)[order], model.mode(f[order], lm))
        assert model.mode(100, lm) == model.mode(np.array([100]), lm)[0]
    assert model.total([]).shape == (0,)


def test_amplitude_growth_phase_and_time_conventions():
    model = ToyModes(4.85, t_c=100, phi_c=0.4, dist_scale=2,
                     modes={(3, 3): (0.5, 0.7, 1.0)})
    f = np.array([50.0, 100, 200])
    amplitude = (f / 100) * f**(-7 / 6)
    expected_phase = 1.5 * psi_22(2 * f / 3, 4.85, 100, 0.4) + 0.7
    np.testing.assert_allclose(model.mode(f, (3, 3)), amplitude * np.exp(-1j * expected_phase), rtol=1e-14)
    shifted = ToyModes(4.85, t_c=100.123, phi_c=0.4, dist_scale=2,
                       modes=model.modes)
    np.testing.assert_allclose(shifted.mode(f, (3, 3)) / model.mode(f, (3, 3)),
                               np.exp(-2j * np.pi * f * 0.123), rtol=1e-10)


def test_optional_cosine_taper():
    lm = (2, 2)
    untapered = ToyModes(4.85, modes={lm: (1, 0, 0)})
    tapered = ToyModes(4.85, modes=untapered.modes, f_taper_lm={lm: (100, 200)})
    f = np.array([50.0, 100, 150, 200, 250])
    np.testing.assert_allclose(tapered.mode(f, lm) / untapered.mode(f, lm), [1, 1, 0.5, 0, 0], atol=1e-15)


@pytest.mark.parametrize("f", [0, -1, np.nan, [1, -1]])
def test_spa_requires_positive_frequencies(f):
    with pytest.raises(ValueError, match="frequencies"):
        ToyModes(4.85).total(f)


def test_mode_defaults_are_independent():
    a, b = ToyModes(4.85), ToyModes(4.85)
    a.modes.pop((4, 4))
    assert (4, 4) in b.modes
