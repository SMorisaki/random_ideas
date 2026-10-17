"""Shared regression fixtures; temporary products remain inside this repository."""

import pytest

from mbquad import BandSetup
from mbquad.testcases import toy_small


@pytest.fixture(scope="session")
def toy_case():
    return toy_small()


@pytest.fixture(scope="session")
def bbh_bands():
    return BandSetup(T=8, dt=1 / 4096, f_low=20, f_high=1024,
                     durations=[8, 4], mchirp_ref=20, m_max=4, L=5)


@pytest.fixture(scope="session", params=[2, 4])
def bns_bands(request):
    return BandSetup(T=256, dt=1 / 4096, f_low=20, f_high=2048,
                     durations=[256, 128, 64, 32, 16, 8, 4],
                     mchirp_ref=1.15, m_max=request.param, L=5)
