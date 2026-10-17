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
