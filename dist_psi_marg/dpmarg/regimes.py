"""Reproducible synthetic cases; rho labels distinguish actual SNR from R/sqrt(lam1)."""

import math

import numpy as np

from .params import expand_params


def named_cases(xmin=0.1, xmax=100.0):
    """Return documented extremes for the finite normalized annular integral.

    The large-lam1 cases have prescribed *actual* annular SNR, with an
    isotropic distance peak below xmin. Random cases instead sample the
    proxy R/sqrt(lam1) specified in TASK.md; these need not have that SNR.
    """
    cases = []

    def add(name, l1, l2, R, eta, note):
        A, B, C, D, E = expand_params(l1, l2, R, eta)
        cases.append(dict(name=name, A=A, B=B, C=C, D=D, E=E,
                          xmin=xmin, xmax=xmax, note=note))

    add("zero", 0, 0, 0, 0, "Exactly unit likelihood, including prior normalization.")
    add("no_linear_signal", 20, 3, 0, 0, "A=B=0 with nonzero anisotropic quadratic penalty; two equal angular lobes.")
    add("isotropic_low", 1, 1, 0.3, 0.37, "C=D=0; independent Bessel check at low SNR.")
    add("isotropic_high", 4, 4, 160, 0.37, "C=D=0, actual SNR=80 when the distance peak is enclosed.")
    for eta, label in [(0, "0"), (math.pi/4, "pi4"), (math.pi/2, "pi2")]:
        add("degenerate_eta_"+label, 10, 0, 1, eta,
            "Exactly lam2=0; tests a flat quadratic direction and arbitrary linear angle.")
    for rho in (0.5, 10, 100, 300):
        l1 = 1e8
        R = (rho*rho + l1*xmin*xmin)/(2*xmin)
        add(f"large_lam1_rho_{rho:g}", l1, l1, R, 0.3,
            f"lam1=1e8; actual annular SNR={rho:g}; narrow angular and radial endpoint layers.")
    for rho in (10, 100):
        # x0=xmin/2, f(xmin,theta=0)=0 would give zero SNR. Choose
        # x0=3*xmin/4 to have a positive endpoint maximum.
        l1 = 2*rho*rho/(xmin*xmin)
        add(f"peak_below_rho_{rho}", l1, l1, 0.75*xmin*l1, 0.2,
            "x0=0.75*xmin; prescribed actual SNR, lower-boundary radial maximum.")
        l1 = rho*rho/(3*xmax*xmax)
        add(f"peak_above_rho_{rho}", l1, l1, 2*xmax*l1, 0.2,
            "x0=2*xmax; prescribed actual SNR, upper-boundary radial maximum.")
    for rho in (0.3, 1, 3):
        add(f"low_snr_{rho:g}", 1, 1, rho, 0.6, "Isotropic enclosed peak with prescribed low SNR.")
    add("two_comparable_lobes", 100, 1, 0.2, 0,
        "Linear term along stiff axis; reflection gives two equal g(theta) lobes near +/-pi/2.")
    add("strong_anisotropy", 1e4, 10, 30, math.pi/2-0.01,
        "lam2/lam1=1e-3; linear direction near the weak eigenvector.")
    return cases


def random_params(n, seed, xmin=0.1, xmax=100.0):
    """Sample log-uniform lam1 and proxy rho=R/sqrt(lam1), then rotate.

    lam1 is in [1e-4,1e8], rho in [0.1,300] with a 10% atom at 0;
    q is uniform in [0,1] with 10% atoms at each of 0 and 1. Eta is
    uniform in [0,pi/2] and xi in [0,pi). Returns raw coefficient dicts.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        lam1 = 10**rng.uniform(-4, 8)
        selector = rng.random()
        q = 0.0 if selector < 0.1 else (1.0 if selector < 0.2 else rng.random())
        rho = math.exp(rng.uniform(math.log(0.1), math.log(300)))
        if rng.random() < 0.1:
            rho = 0.0
        eta, xi = rng.uniform(0, math.pi/2), rng.uniform(0, math.pi)
        A, B, C, D, E = expand_params(lam1, lam1*q, rho*math.sqrt(lam1), eta, xi)
        cases.append(dict(name=f"random_{i:02d}", A=A, B=B, C=C, D=D, E=E,
                          xmin=xmin, xmax=xmax,
                          note=f"seed={seed}; sampled proxy rho=R/sqrt(lam1)={rho:.12g}; xi={xi:.12g}"))
    return cases
