"""PSD normalization, text-file interpretation, and fine-grid conventions."""

import numpy as np
import pytest

from mbquad import PSDGrid, add_lines, aligo_analytic_psd, load_asd_or_psd


def test_analytic_fit_and_nonpositive_frequencies():
    # At x=1 the bracket reduces to 1 - 5 + 111/3 = 33.
    assert aligo_analytic_psd(215) == pytest.approx(33e-49)
    assert np.all(np.isinf(aligo_analytic_psd([-10, 0])))
    assert np.all(aligo_analytic_psd(np.geomspace(1, 10000, 100)) > 0)


def test_lines_multiply_and_do_not_mutate_input():
    original = np.array([2.0, 3.0, 4.0])
    psd_fn = lambda f: original
    modified = add_lines(psd_fn, [(50, 1, 9), (50, 2, 4)])
    np.testing.assert_allclose(modified(np.full(3, 50.0)), original * 50)
    np.testing.assert_array_equal(original, [2, 3, 4])
    assert add_lines(lambda f: 2, [(50, 1, 9)])(50) == 20
    assert add_lines(aligo_analytic_psd, [])(50) == aligo_analytic_psd(50)


@pytest.mark.parametrize("line", [(50, 0, 1), (50, -1, 1), (50, 1, -1), (np.nan, 1, 1)])
def test_invalid_lines(line):
    with pytest.raises(ValueError, match="Lines"):
        add_lines(aligo_analytic_psd, [line])


@pytest.mark.parametrize("as_asd", [False, True])
def test_log_log_file_interpolation(tmp_path, as_asd):
    # A power law must be reproduced exactly by log-log interpolation.
    f = np.array([100.0, 10.0, 1000.0])  # deliberately unordered
    S = 1e-46 * (f / 10)**2
    path = tmp_path / "noise.txt"
    np.savetxt(path, np.column_stack([f, np.sqrt(S) if as_asd else S]))
    psd_fn = load_asd_or_psd(path)
    query = np.array([10, np.sqrt(1000), 100, 1000])
    np.testing.assert_allclose(psd_fn(query), 1e-46 * (query / 10)**2, rtol=3e-14, atol=0)
    assert np.isinf(psd_fn(9))
    assert np.all(np.isinf(psd_fn([-1, 0, 1001])))
    assert psd_fn(100) == pytest.approx(1e-44, rel=1e-14, abs=0)


@pytest.mark.parametrize("rows", [
    [[10, 1e-46]], [[10, 1e-46], [10, 2e-46]],
    [[0, 1e-46], [10, 1e-46]], [[10, 0], [20, 1e-46]],
    [[10, np.nan], [20, 1e-46]], [[10, 1e-46, 1], [20, 1e-46, 1]],
])
def test_invalid_noise_files(tmp_path, rows):
    path = tmp_path / "bad.txt"
    np.savetxt(path, rows)
    with pytest.raises(ValueError):
        load_asd_or_psd(path)


def test_psd_grid_indices_mask_and_nyquist():
    seen = []

    def psd_fn(f):
        seen.append(f.copy())
        return f**2

    grid = PSDGrid(psd_fn, T=8, dt=1 / 16, f_low=2, f_high=8)
    np.testing.assert_array_equal(grid.f_k, np.arange(1, 64) / 8)
    assert grid.N == 128
    assert grid.f_k[-1] < 1 / (2 * grid.dt)
    active = grid.f_k >= 2
    assert np.all(seen[0] >= 2)
    np.testing.assert_allclose(grid.S_k[active], grid.f_k[active]**2)
    assert np.all(np.isinf(grid.S_k[~active]))
    assert np.all(grid.inv_S[~active] == 0)
    np.testing.assert_allclose(grid.inv_S[active] * grid.S_k[active], 1)
    assert grid.f is grid.f_k


@pytest.mark.parametrize("T,dt", [(3, 1), (8.1, 1), (0, 1), (8, 0)])
def test_invalid_sampling(T, dt):
    with pytest.raises(ValueError):
        PSDGrid(aligo_analytic_psd, T, dt, 0.1, 0.4)


@pytest.mark.parametrize("value", [0, -1, np.nan, -np.inf])
def test_invalid_psd_values(value):
    with pytest.raises(ValueError, match="PSD must be positive"):
        PSDGrid(lambda f: value, 8, 1 / 16, 2, 8)


def test_zero_weight_infinite_psd():
    grid = PSDGrid(lambda f: np.inf, 8, 1 / 16, 2, 8)
    assert np.all(grid.inv_S == 0)


def test_bns_grid_memory_bound():
    grid = PSDGrid(aligo_analytic_psd, 256, 1 / 4096, 20, 2048)
    assert sum(a.nbytes for a in (grid.f_k, grid.S_k, grid.inv_S)) < 13 * 1024**2
