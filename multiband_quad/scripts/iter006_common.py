"""Shared reproducible cache, timing and report helpers for iteration 006."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import warnings

import numpy as np
from mbquad.pairwise import PairwiseHH
from mbquad.timing import median_time

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def quiet_lal():
    """Capture C stderr in an unlinked local temporary file; retain Python errors."""
    saved = os.dup(2)
    with tempfile.TemporaryFile(dir=ROOT/'data') as captured:
        try:
            os.dup2(captured.fileno(), 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def measure(fn, target=.15, repeat=5):
    """Median warm batch seconds per evaluation; target seconds per batch."""
    fn()
    start = perf_counter()
    for _ in range(3):
        fn()
    estimate = (perf_counter()-start)/3
    number = max(1, min(10000, int(target/max(estimate, 1e-7))))
    return median_time(fn, number=number, repeat=repeat)


def pairwise(bands, psd, modes, **kw):
    """Suppress known full-grid slack warnings; report requested slack in JSON."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return PairwiseHH(bands, psd, modes, **kw)


class ReferenceCache:
    """Reuse the exact iteration-005 hash and full positive-frequency sequences."""
    def __init__(self):
        self.path = ROOT/'data/xhm_reference_cache.npz'
        with np.load(self.path) as archive:
            self.arrays = dict(archive)
        self.hits = self.misses = 0

    def get(self, name, psd_name, model, psd):
        spec = dict(system=name, psd=psd_name, model=vars(model), N=psd.N, T=psd.T,
                    lal_version='6.2.0', reference='ChooseFDWaveformSequence-v1')
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if key not in self.arrays:
            self.arrays[key] = model.total_direct(psd.f_k)
            self.misses += 1
            np.savez(self.path, **self.arrays)
        else:
            self.hits += 1
        return self.arrays[key]


def exact_bands(fine, psd, bands):
    """Eq. (26), independently weighted fine-grid contribution of every band."""
    return np.array([(4/bands.T)*np.sum(bands.window(psd.f_k, b)
                    * abs(fine)**2 * psd.inv_S) for b in range(bands.B)])


def stats(errors):
    """Median/worst relative HH and absolute likelihood error at SNR=25."""
    e = np.abs(errors)
    return dict(median_relative=float(np.median(e)), worst_relative=float(np.max(e)),
                median_lnLambda=float(312.5*np.median(e)), worst_lnLambda=float(312.5*np.max(e)))


def write_json(name, obj):
    (ROOT/'data'/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
