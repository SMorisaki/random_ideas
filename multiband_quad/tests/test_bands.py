"""Paper geometry, frequency compression, coarse-grid metadata and failure modes."""

import numpy as np
import pytest

from mbquad import BandSetup, tau_0pn


def check_geometry(bands):
    assert np.all(np.diff(bands.f_b) > 0)
    assert np.all(bands.f_b[:-1] < bands.f_b[1:] - bands.Df_b[1:])
    assert np.all(bands.M_b == bands.T_b / bands.dt_b)
    assert np.all(bands.K_s <= bands.K_e)
    assert np.all((bands.N_b & (bands.N_b - 1)) == 0)
    assert np.all((bands.N_b - 1) // 2 >= bands.f_b[1:] * bands.T)
    # Halving N_b must fail Eq. (13): it is the smallest allowed power of two.
    assert np.all((bands.N_b // 2 - 1) // 2 < bands.f_b[1:] * bands.T)
    f = bands.f_b[1:-1]
    np.testing.assert_allclose(bands.tau(f) + bands.L * np.sqrt(-bands.dtau_df(f)),
                               bands.T_b[1:] + bands.tc_min_minus_T, rtol=0, atol=1e-9)
    np.testing.assert_allclose(bands.Df_b[1:-1], 1 / np.sqrt(-bands.dtau_df(f)), rtol=1e-14)
    assert bands.Df_b[0] == 0
    assert bands.Df_b[-1] == 100 / bands.T_minus_tc_max
    assert bands.f_b[-1] == bands.f_high + bands.Df_b[-1]


def test_bns_geometry_and_table_i(bns_bands):
    check_geometry(bns_bands)
    table_i = {2: 45, 4: 27}[bns_bands.m_max]
    assert bns_bands.K_orig / bns_bands.K_MB == pytest.approx(table_i, rel=0.15)
    assert bns_bands.K_orig == 519169


def test_bbh_geometry(bbh_bands):
    check_geometry(bbh_bands)


def test_toy_geometry_and_coarse_arrays(toy_case):
    bands = toy_case.bands
    check_geometry(bands)
    assert bands.K_orig == 30209
    assert bands.K_MB == 3294
    grids = bands.coarse_frequencies()
    all_grid = bands.all_coarse_frequencies()
    assert len(all_grid.f) == bands.K_MB
    for b, grid in enumerate(grids):
        assert len(grid.f) == bands.K_b[b]
        np.testing.assert_array_equal(grid.kappa, np.arange(bands.K_s[b], bands.K_e[b] + 1))
        np.testing.assert_array_equal(grid.f, grid.kappa / bands.T_b[b])
        np.testing.assert_array_equal(grid.w, bands.window(grid.f, b))
        selected = all_grid.band_index == b
        np.testing.assert_array_equal(all_grid.f[selected], grid.f)
        np.testing.assert_array_equal(all_grid.kappa[selected], grid.kappa)
        np.testing.assert_array_equal(all_grid.w[selected], grid.w)


def test_lag_uses_support_start(toy_case):
    bands = toy_case.bands
    for b in range(bands.B):
        f_lo = bands.f_b[b] - bands.Df_b[b]
        expected = tau_0pn(f_lo / 2, bands.mchirp_ref) - tau_0pn(2 * f_lo, bands.mchirp_ref)
        assert bands.lag_max_pair(b, 4, 1) == pytest.approx(expected, rel=1e-14)
        assert bands.lag_max_pair(b, 2, 2) == 0
    with pytest.raises(ValueError, match="m_slow"):
        bands.lag_max_pair(0, 1, 4)


def test_summary_and_print(bbh_bands, capsys):
    rows = bbh_bands.summary_table()
    assert len(rows) == bbh_bands.B
    assert set(rows[0]) == {"T_b", "f_lo", "f_hi", "Df_lo", "Df_hi", "N_b", "M_b", "K_s", "K_e", "K_b"}
    assert rows[0]["T_b"] == 8
    bbh_bands.print_summary()
    assert capsys.readouterr().out == bbh_bands.format_summary() + "\n"


@pytest.mark.parametrize("overrides,match", [
    ({"durations": [8, 3]}, "durations"),
    ({"durations": [4, 2]}, "durations"),
    ({"durations": [8, 8]}, "durations"),
    ({"durations": [8, 4, 2]}, "no positive root"),
    ({"f_low": 100}, "root is at or below"),
    ({"f_high": 30}, "overlapping transitions"),
    ({"mchirp_ref": 0}, "mchirp_ref"),
    ({"m_max": 0}, "m_max"),
    ({"L": 0}, "L"),
    ({"T_minus_tc_max": 0}, "T_minus_tc_max"),
])
def test_invalid_geometry(overrides, match):
    parameters = dict(T=8, dt=1 / 4096, f_low=20, f_high=1024,
                      durations=[8, 4], mchirp_ref=20, m_max=4, L=5)
    parameters.update(overrides)
    with pytest.raises(ValueError, match=match):
        BandSetup(**parameters)


def test_adaptive_root_bracket():
    # The internal edge is above the nominal high frequency here, so the
    # bracket must expand before reporting incompatible transition geometry.
    with pytest.raises(ValueError, match="overlapping transitions"):
        BandSetup(8, 1 / 4096, 20, 30, [8, 4], 20, 4, 5)
