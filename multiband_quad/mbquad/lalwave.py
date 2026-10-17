"""IMRPhenomXHM arbitrary-sequence adapter (LALSimulation 6.2 conventions)."""

import numpy as np
import lal
import lalsimulation as lalsim

from .toywave import _positive_f, tau_0pn

XHM_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))


class XHMModel:
    """Nonprecessing detector strain on arbitrary positive Hz sequences.

    Masses are solar masses, distance is Mpc, angles radians, t_c seconds.
    LAL's negative-m OneMode output combines with AddMode harmonics at
    azimuth pi/2. The Fourier convention is exp(-2 pi i f t_c): its inverse
    FFT places coalescence near t_c, with LAL's small merger time offset.
    No samples or LAL results are cached between calls.

    Raw modes use distance=1 Mpc and have exp(+i |m| phi_ref) removed.
    C=(F_plus c_plus+F_cross c_cross) exp(+i |m| phi_ref)/distance_mpc,
    c_plus=(Y_-m+(-1)^l conj(Y_m))/2,
    c_cross=i(Y_-m-(-1)^l conj(Y_m))/2, Y at azimuth pi/2.
    This normalization enables extrinsic factorization. LAL's mixed (3,2)
    mode has a small residual phi_ref dependence in version 6.2.0; retaining
    it is necessary to reproduce ChooseFDWaveformSequence exactly. Thus
    exact phase-independent matrices are not guaranteed by this adapter.
    For Mf>0.3, OneMode's omitted higher-mode tails are recovered with
    single-mode ChooseFD sequence calls at fixed projection geometry.
    The (3,2) recovery also activates (2,2) to retain the driver's mixing.
    """

    def __init__(self, m1, m2, chi1z, chi2z, distance_mpc, inclination,
                 phi_ref, t_c, f_ref=20.0, F_plus=1.0, F_cross=0.0,
                 modes=XHM_MODES):
        values = (m1, m2, chi1z, chi2z, distance_mpc, inclination, phi_ref,
                  t_c, f_ref, F_plus, F_cross)
        if not np.all(np.isfinite(values)) or min(m1, m2, distance_mpc) <= 0:
            raise ValueError('Require finite parameters and positive masses/distance')
        if max(abs(chi1z), abs(chi2z)) > 1 or f_ref < 0:
            raise ValueError('Require |chi_z| <= 1 and f_ref >= 0')
        for key, value in zip(('m1', 'm2', 'chi1z', 'chi2z', 'distance_mpc',
                               'inclination', 'phi_ref', 't_c', 'f_ref',
                               'F_plus', 'F_cross'), values):
            setattr(self, key, float(value))
        self.modes = tuple(map(tuple, modes))
        if not self.modes or len(set(self.modes)) != len(self.modes) or any(
                lm not in XHM_MODES for lm in self.modes):
            raise ValueError('Choose unique modes from XHM_MODES (no (4,3))')

    @property
    def mchirp(self):
        """Chirp mass (m1*m2)^(3/5)/(m1+m2)^(1/5), in solar masses."""
        return (self.m1 * self.m2)**.6 / (self.m1 + self.m2)**.2

    def _sequence(self, f):
        f = _positive_f(f)
        # Unique also handles exact duplicates in overlapping coarse bands.
        unique, inverse = np.unique(f.ravel(), return_inverse=True)
        seq = lal.CreateREAL8Sequence(len(unique))
        seq.data[:] = unique
        return f, seq, inverse

    def _params(self, modes=None):
        p = lal.CreateDict()
        ma = lalsim.SimInspiralCreateModeArray()
        for l, m in (self.modes if modes is None else modes):
            lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
            lalsim.SimInspiralModeArrayActivateMode(ma, l, -m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(p, ma)
        return p

    def _one_mode(self, seq, lm, sign=-1):
        """Unscaled negative-m LAL series at 1 Mpc and the actual phi_ref."""
        l, m = lm
        h = lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
            seq, self.m1 * lal.MSUN_SI, self.m2 * lal.MSUN_SI,
            self.chi1z, self.chi2z, l, sign * m, 1e6 * lal.PC_SI,
            self.phi_ref, self.f_ref, self._params()).data.data.copy()
        # OneMode truncates ALL modes at Mf=0.3, whereas ChooseFD's
        # sequence driver retains HM ringdown tails. Recover those tiny
        # tails through a single-mode polarization call at fixed geometry.
        # This is needed for the requested 1e-10 identity on massive BBH.
        tail = seq.data > .3 / ((self.m1+self.m2)*lal.MTSUN_SI)
        if lm != (2, 2) and np.any(tail):
            hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
                self.phi_ref, self.m1*lal.MSUN_SI, self.m2*lal.MSUN_SI,
                0, 0, self.chi1z, 0, 0, self.chi2z, self.f_ref,
                1e6*lal.PC_SI, 1., self._params(
                    ((2, 2), lm) if lm == (3, 2) and (2, 2) in self.modes else (lm,)),
                lalsim.IMRPhenomXHM, seq)
            ym = lal.SpinWeightedSphericalHarmonic(1., np.pi/2, -2, l, -m)
            yp = (-1)**l*np.conj(lal.SpinWeightedSphericalHarmonic(1., np.pi/2, -2, l, m))
            h[tail] = hp.data.data[tail] / (.5*(ym+yp))
            if sign == 1:
                h[tail] = (-1)**l*h[tail].conj()
        return h

    def raw_modes(self, f):
        """Unit-distance LAL -m modes with nominal orbital phase removed.

        No t_c or projection is included. The (3,2) LAL mixing residual
        described in the class docstring remains; all other extrinsics
        are absent. Arrays have the input shape and ordering.
        """
        f, seq, inv = self._sequence(f)
        return {lm: (self._one_mode(seq, lm)[inv].reshape(f.shape)
                     * np.exp(-1j * lm[1] * self.phi_ref)) for lm in self.modes}

    def coefficients(self):
        """Return C for mode = C * raw * exp(-2 pi i f t_c); see class formula."""
        result = {}
        for l, m in self.modes:
            ym = lal.SpinWeightedSphericalHarmonic(self.inclination, np.pi/2, -2, l, -m)
            yp = (-1)**l * np.conj(lal.SpinWeightedSphericalHarmonic(
                self.inclination, np.pi/2, -2, l, m))
            result[l, m] = (.5 * self.F_plus * (ym + yp)
                            + .5j * self.F_cross * (ym - yp)) * np.exp(
                                1j * m * self.phi_ref) / self.distance_mpc
        return result

    def mode(self, f, lm):
        """Detector contribution of one |m| pair, including the t_c shift."""
        if lm not in self.modes:
            raise KeyError(lm)
        f, seq, inv = self._sequence(f)
        raw = self._one_mode(seq, lm)[inv].reshape(f.shape) * np.exp(-1j*lm[1]*self.phi_ref)
        return self.coefficients()[lm] * raw * np.exp(-2j*np.pi*f*self.t_c)

    def total(self, f):
        """Sum all detector-frame |m| contributions on f > 0."""
        raw = self.raw_modes(f)
        c = self.coefficients()
        return sum(c[lm] * h for lm, h in raw.items()) * np.exp(-2j*np.pi*np.asarray(f)*self.t_c)

    def polarizations(self, f):
        """Return ChooseFDWaveformSequence (h_plus,h_cross), including t_c."""
        f, seq, inv = self._sequence(f)
        hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
            self.phi_ref, self.m1*lal.MSUN_SI, self.m2*lal.MSUN_SI,
            0, 0, self.chi1z, 0, 0, self.chi2z, self.f_ref,
            self.distance_mpc*1e6*lal.PC_SI, self.inclination,
            self._params(), lalsim.IMRPhenomXHM, seq)
        phase = np.exp(-2j*np.pi*f*self.t_c)
        return tuple(h.data.data[inv].reshape(f.shape)*phase for h in (hp,hc))

    def total_direct(self, f):
        """Detector projection F_plus h_plus+F_cross h_cross, including t_c."""
        hp, hc = self.polarizations(f)
        return self.F_plus*hp + self.F_cross*hc

    def tau_lm(self, f, lm):
        """Eq. (55): tau_0pn(2f/|m|,Mc), only a leading-order inspiral track."""
        if lm not in self.modes:
            raise KeyError(lm)
        return tau_0pn(2*_positive_f(f)/lm[1], self.mchirp)
