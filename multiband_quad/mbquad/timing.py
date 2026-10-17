"""Wall-clock timing helpers for separately measuring waveform and HH costs."""

from time import perf_counter
from statistics import median


def median_time(fn, *args, number=100, repeat=7):
    """Return median seconds per call over repeat batches, after one warm-up.

    Each batch calls fn(*args) number times. Setup/precomputation belongs
    outside fn when measuring only the per-evaluation quadratic-term step.
    """
    if (not isinstance(number, int) or not isinstance(repeat, int)
            or number < 1 or repeat < 1):
        raise ValueError("number and repeat must be positive integers")
    fn(*args)
    samples = []
    for _ in range(repeat):
        start = perf_counter()
        for _ in range(number):
            fn(*args)
        samples.append((perf_counter() - start) / number)
    return median(samples)
