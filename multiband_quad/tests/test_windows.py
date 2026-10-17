"""Eq. (9) supports, transition continuity, and Eq. (11) partition of unity."""

import numpy as np
import pytest


def test_bbh_partition_on_fine_grid(bbh_bands):
    f = np.arange(1, bbh_bands.N // 2) / bbh_bands.T
    total = sum(bbh_bands.window(f, b) for b in range(bbh_bands.B))
    active = (f >= bbh_bands.f_low) & (f <= bbh_bands.f_high)
    np.testing.assert_allclose(total[active], 1, rtol=0, atol=1e-12)


def test_support_and_continuity(bbh_bands):
    bands = bbh_bands
    for b in range(bands.B):
        lo, hi = bands.f_b[b:b + 2]
        dlo, dhi = bands.Df_b[b:b + 2]
        outside = np.array([-10, lo - dlo - 1, hi, hi + 1, 2 * hi])
        np.testing.assert_array_equal(bands.window(outside, b), 0)
        edges = [(lo, 1), (hi - dhi, 1), (hi, 0)]
        if dlo:
            edges.append((lo - dlo, 0))
        # The global f_low boundary is a deliberate hard edge, not a taper.
        for edge, value in edges:
            if b == 0 and edge == lo:
                continue
            samples = edge + np.array([-1e-7, 0, 1e-7])
            np.testing.assert_allclose(bands.window(samples, b), value, rtol=0, atol=1e-12)
        f = np.linspace(lo - dlo - 1, hi + 1, 2000)
        w = bands.window(f, b)
        assert np.all((w >= 0) & (w <= 1))
        np.testing.assert_allclose(bands.sqrt_window(f, b)**2, w, rtol=0, atol=2e-16)
        assert bands.window((lo + hi - dhi) / 2, b) == 1
        if dlo:
            assert bands.window(lo - dlo / 2, b) == pytest.approx(0.5)
        assert bands.window(hi - dhi / 2, b) == pytest.approx(0.5)


@pytest.mark.parametrize("b", [-1, 2, 0.5])
def test_invalid_band(bbh_bands, b):
    with pytest.raises(IndexError, match="Band index"):
        bbh_bands.window(100, b)
