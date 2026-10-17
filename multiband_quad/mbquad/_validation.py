"""Shared validation of the uniformly sampled data segment."""

import numpy as np


def sample_count(T, dt):
    """Return N = T/dt for a finite segment with a power-of-two N >= 4."""
    if not np.isfinite(T) or not np.isfinite(dt) or T <= 0 or dt <= 0:
        raise ValueError("T and dt must be finite and positive")
    count = T / dt
    if not np.isfinite(count) or not np.isclose(count, round(count), rtol=0, atol=1e-8):
        raise ValueError("N = T/dt must be an integer")
    N = int(round(count))
    if N < 4 or N & (N - 1):
        raise ValueError("N = T/dt must be a power of two >= 4")
    return N


def frequency_limits(f_low, f_high, dt):
    """Validate a positive analysis interval up to and including Nyquist."""
    if not (np.isfinite(f_low) and np.isfinite(f_high)
            and 0 < f_low < f_high <= 0.5 / dt):
        raise ValueError("Require 0 < f_low < f_high <= 1/(2 dt)")
