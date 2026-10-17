"""Total-waveform samples on the per-band coarse grids of Eq. (24)."""

import numpy as np


def check_setup(bands, psdgrid):
    """Require matching Eq. (6) analysis limits and Eq. (4) fine grids."""
    if any(getattr(bands, key) != getattr(psdgrid, key)
           for key in ("T", "dt", "f_low", "f_high")):
        raise ValueError("PSDGrid and BandSetup must have matching segment and frequency limits")


class CoarseWaveform:
    """Unwindowed TOTAL h(f_kappa) on each band's Eq. (24) grid.

    ``h_b[b]`` has indices K_s^(b)..K_e^(b), including overlap and zero-window
    boundary points. Arrays must be finite complex samples. Construction
    from a concatenated array retains views, so callers must keep that array
    unchanged during evaluation. Windows are applied by the HH evaluator.
    """

    def __init__(self, bands, h_b):
        self.bands = bands
        self.h_b = tuple(np.asarray(h, dtype=np.complex128) for h in h_b)
        if (len(self.h_b) != bands.B
                or any(h.shape != (int(n),) for h, n in zip(self.h_b, bands.K_b))):
            raise ValueError("Expected one waveform array of shape (K_b[b],) per band")
        if any(np.any(~np.isfinite(h)) for h in self.h_b):
            raise ValueError("Coarse waveform samples must be finite")

    @classmethod
    def from_array(cls, bands, h):
        """Split samples ordered like all_coarse_frequencies() (Eqs. 24–25)."""
        h = np.asarray(h, dtype=np.complex128)
        if h.shape != (bands.K_MB,):
            raise ValueError("Concatenated waveform must have shape (K_MB,)")
        return cls(bands, np.split(h, np.cumsum(bands.K_b)[:-1]))

    @classmethod
    def from_model(cls, bands, model):
        """Evaluate callable model(f) once on the concatenated Eq. (24) grid."""
        return cls.from_array(bands, model(bands.all_coarse_frequencies().f))

    def check_bands(self, bands):
        """Reject accidental use with a different coarse-grid setup."""
        if self.bands is not bands:
            raise ValueError("CoarseWaveform must use the evaluator's BandSetup instance")


class CoarseModes:
    """Unwindowed mode samples H_b of shape (n_modes,K_b), method_idea §3.

    Positive-m contributions must already combine ±m and the detector
    response. This container neither projects nor rescales those samples.
    For PairwiseHH.mode_matrix, it may instead hold raw intrinsic modes;
    coefficients are then supplied separately to hh_from_matrix.
    """

    def __init__(self, bands, modes, h_b):
        self.bands = bands
        self.modes = list(map(tuple, modes))
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must be nonempty and unique")
        self.h_b = tuple(np.asarray(h, dtype=complex) for h in h_b)
        if (len(self.h_b) != bands.B or any(
                h.shape != (len(self.modes), int(k))
                for h, k in zip(self.h_b, bands.K_b))):
            raise ValueError("Expected mode arrays with shape (n_modes, K_b[b])")
        if any(np.any(~np.isfinite(h)) for h in self.h_b):
            raise ValueError("Coarse mode samples must be finite")

    @classmethod
    def from_model(cls, bands, model, modes):
        """Evaluate each model.mode(f,lm) once on Eq. (24)'s concatenated grid."""
        modes = list(modes)
        f = bands.all_coarse_frequencies().f
        h = np.asarray([model.mode(f, lm) for lm in modes])
        return cls(bands, modes, np.split(h, np.cumsum(bands.K_b)[:-1], axis=1))

    def total(self):
        """Sum mode contributions, returning Eq. (24)'s CoarseWaveform."""
        return CoarseWaveform(self.bands, [h.sum(axis=0) for h in self.h_b])

    def check_bands(self, bands):
        """Require the evaluator's BandSetup instance, as for CoarseWaveform."""
        if self.bands is not bands:
            raise ValueError("CoarseModes must use the evaluator's BandSetup instance")
