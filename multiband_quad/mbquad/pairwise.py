"""Mode-pair coarse-grid quadratic term, method_idea §§3 and 8."""

from collections.abc import Mapping
import warnings

import numpy as np

from .blockweights import BlockWeights
from .coarse import CoarseModes, check_setup
from .linear import LinearInterpolationHH
from .toywave import tau_0pn


class PairwiseHH:
    """(4/T) sum_b conj(H_i) W_ij H_j on Eq. (24)'s coarse grid.

    Positive-m, non-precessing mode contributions are supplied by the caller.
    Lag classes depend only on m (Newtonian Eqs. 53,55); class 0 comprises
    diagonals and same-m pairs. Windows multiply products, never the PSD
    kernel. Prior endpoints cover Newtonian chirp-mass variation only.

    Margins are L_margin times sqrt(|tau'_slow-tau'_fast|), the conservative
    sqrt(-tau'_mmax), or T_b. Negative requested slack warns; equal amounts
    are removed from both margins (down to zero). An interval wider than
    T_b cannot be covered even with zero margins and remains flagged.
    summary() retains requested slack and margins as well as clipped ones.
    If M_b=N_b, every block contains the complete circular lag grid; the
    quadrature is exact even when the unwrapped SPA interval exceeds T_b.
    pairs is the faster default in the iteration-003 benchmark.
    """

    def __init__(self, bands, psdgrid, modes, L_margin=5.0,
                 margin_units="sigma_pair", mchirp_range=None, diag="block",
                 impl="pairs"):
        check_setup(bands, psdgrid)
        self.modes = list(map(tuple, modes))
        if (not self.modes or len(set(self.modes)) != len(self.modes)
                or any(len(lm) != 2 or not all(isinstance(x, (int, np.integer)) for x in lm)
                       or not lm[0] >= lm[1] >= 1 for lm in self.modes)):
            raise ValueError("modes must be unique positive integer (l,m), l >= m >= 1")
        if max(m for _, m in self.modes) > bands.m_max:
            raise ValueError("Mode m exceeds the BandSetup m_max")
        if margin_units not in ("sigma_pair", "sigma_mmax", "T_b"):
            raise ValueError("Unknown margin_units")
        if diag not in ("block", "linear") or impl not in ("einsum", "pairs"):
            raise ValueError("Require diag='block'|'linear', impl='einsum'|'pairs'")
        if not np.isfinite(L_margin) or L_margin < 0:
            raise ValueError("L_margin must be finite and nonnegative")
        prior = (bands.mchirp_ref, bands.mchirp_ref) if mchirp_range is None else mchirp_range
        if (len(prior) != 2 or not np.all(np.isfinite(prior))
                or not 0 < prior[0] <= prior[1]):
            raise ValueError("Require 0 < Mc_min <= Mc_max")
        self.bands, self.impl, self.diag = bands, impl, diag
        self.L_margin, self.margin_units = float(L_margin), margin_units
        self.mchirp_range = tuple(map(float, prior))
        self.blockweights = BlockWeights(bands, psdgrid)
        ms = sorted({m for _, m in self.modes})
        self.classes = [0] + [(s, f) for s in ms for f in ms if s > f]
        self._groups = {m: [i for i, lm in enumerate(self.modes) if lm[1] == m] for m in ms}
        self._summary, self.class_weights, self.W = [], [], []
        linear = LinearInterpolationHH(bands, psdgrid) if diag == "linear" else None
        for b, grid in enumerate(bands.coarse_frequencies()):
            M = int(bands.M_b[b])
            weights = {}
            for cl in self.classes:
                row = self._placement(b, cl)
                self._summary.append(row)
                if cl == 0 and linear is not None:
                    w = linear.c_b[b] * bands.T / 4
                else:
                    w = self.blockweights.weights(b, row["m0"], M, grid.kappa) * grid.w
                    if cl == 0:
                        # {-M/2,...,M/2-1} differs from its reflection at one bin.
                        w = w.real
                weights[cl] = w
            W = np.empty((len(self.modes), len(self.modes), len(grid.f)), complex)
            for i, (_, mi) in enumerate(self.modes):
                for j, (_, mj) in enumerate(self.modes):
                    W[i, j] = (weights[0] if mi == mj else
                               weights[(mi, mj)] if mi > mj else weights[(mj, mi)].conj())
            self.class_weights.append(weights)
            self.W.append(W)
        self.W = tuple(self.W)
        self.class_weights = tuple(self.class_weights)

    def _placement(self, b, cl):
        bands = self.bands
        T_b, dt = float(bands.T_b[b]), bands.dt_b[b]
        M = int(bands.M_b[b])
        lo = hi = dlo = dhi = 0.0
        if cl != 0:
            slow, fast = cl
            mcmin, mcmax = self.mchirp_range
            flo, fhi = bands.f_b[b] - bands.Df_b[b], bands.f_b[b + 1]
            lo = float(bands.lag_pair(fhi, slow, fast, mcmax))
            hi = float(bands.lag_pair(flo, slow, fast, mcmin))
            if self.margin_units == "T_b":
                dlo = dhi = self.L_margin * T_b
            else:
                def sigma(f, mc):
                    tau = (bands.lag_pair(f, slow, fast, mc)
                           if self.margin_units == "sigma_pair"
                           else tau_0pn(2 * f / bands.m_max, mc))
                    return np.sqrt((8 / 3) * tau / f)
                dlo = float(self.L_margin * sigma(fhi, mcmax))
                dhi = float(self.L_margin * sigma(flo, mcmin))
        requested = (dlo, dhi)
        slack = T_b - (hi - lo) - dlo - dhi
        if slack < 0:
            warnings.warn(f"Band {b}, class {cl}: negative slack {slack:.6g} s; "
                          "clipping margins symmetrically", RuntimeWarning, stacklevel=3)
            available = max(0.0, T_b - (hi - lo))
            # Equal absolute reductions, capped at each margin's zero boundary.
            excess = dlo + dhi - available
            cut = min(min(dlo, dhi), excess / 2)
            dlo, dhi = dlo - cut, dhi - cut
            excess = max(0.0, dlo + dhi - available)
            if dlo > dhi:
                dlo = max(0.0, dlo - excess)
            else:
                dhi = max(0.0, dhi - excess)
        m0 = -M // 2 if cl == 0 else int(np.rint((lo - dlo + hi + dhi) / (2 * dt))) - M // 2
        return dict(b=b, lag_class=cl, Lambda_min=lo, Lambda_max=hi,
                    delta_lo=dlo, delta_hi=dhi, requested_delta_lo=requested[0],
                    requested_delta_hi=requested[1], slack=slack,
                    full_grid=bool(M == bands.N_b[b]),
                    effective_slack=T_b - (hi - lo) - dlo - dhi,
                    m0=m0, block_start=float(m0 * dt), block_end=float((m0 + M) * dt))

    def _samples(self, coarse_modes):
        if isinstance(coarse_modes, Mapping):
            if set(coarse_modes) != set(self.modes):
                raise ValueError("Mapping must contain exactly the evaluator's modes")
            for value in coarse_modes.values():
                value.check_bands(self.bands)
            return tuple(np.asarray([coarse_modes[lm].h_b[b] for lm in self.modes])
                         for b in range(self.bands.B))
        if not isinstance(coarse_modes, CoarseModes):
            raise TypeError("Expected CoarseModes or mapping of mode to CoarseWaveform")
        coarse_modes.check_bands(self.bands)
        if set(coarse_modes.modes) != set(self.modes):
            raise ValueError("CoarseModes must contain exactly the evaluator's modes")
        if coarse_modes.modes == self.modes:
            return coarse_modes.h_b
        order = [coarse_modes.modes.index(lm) for lm in self.modes]
        return tuple(h[order] for h in coarse_modes.h_b)

    def __call__(self, coarse_modes, impl=None):
        """Evaluate the §3 quadratic form; no FFT or waveform call at run time.

        einsum uses the Hermitian tensor. pairs combines same-m modes first,
        then sums each distinct lag class once. impl overrides self.impl.
        """
        impl = self.impl if impl is None else impl
        if impl not in ("einsum", "pairs"):
            raise ValueError("impl must be 'einsum' or 'pairs'")
        total = 0.0
        for b, (H, W, weights) in enumerate(zip(
                self._samples(coarse_modes), self.W, self.class_weights)):
            if impl == "einsum":
                total += np.einsum("ik,ijk,jk->", H.conj(), W, H).real
            elif self.bands.M_b[b] == self.bands.N_b[b] and self.diag == "block":
                # All full-grid blocks have the identical W=a, irrespective
                # of their starts: combine every pair before the one dot.
                total += np.dot(weights[0], np.abs(H.sum(axis=0))**2).real
            else:
                grouped = {m: H[idx[0]] if len(idx) == 1 else H[idx].sum(axis=0)
                           for m, idx in self._groups.items()}
                total += sum(np.dot(weights[0], np.abs(h)**2).real for h in grouped.values())
                for slow, fast in self.classes[1:]:
                    total += 2 * np.dot(weights[(slow, fast)],
                                        grouped[slow].conj() * grouped[fast]).real
        return float((4 / self.bands.T) * total)

    def band_contributions(self, coarse_modes):
        """Return each §3 quadratic-form band norm, including 4/T.

        This diagnostic uses the Hermitian tensor (also for linear class-0
        weights); the default runtime pairs kernel is unchanged.
        """
        return (4 / self.bands.T) * np.asarray([
            np.einsum('ik,ijk,jk->', H.conj(), W, H).real
            for H, W in zip(self._samples(coarse_modes), self.W)])

    def mode_matrix(self, coarse_modes):
        """Return Hermitian M_ij=sum_b,k conj(H_i) W_ij H_j (§8.13).

        Input raw modes omit coefficients and t_c. W already contains w_b.
        M excludes 4/T: use hh_from_matrix(M,C,T=bands.T). Ordering is
        self.modes, even if the input uses another order. Extrinsic
        independence requires the supplied raw modes to have that property.
        """
        M = sum(np.einsum('ik,ijk,jk->ij', H.conj(), W, H)
                for H, W in zip(self._samples(coarse_modes), self.W))
        return (M + M.conj().T) / 2

    def hh_from_matrix(self, M, C):
        """Return (4/T) C^dagger M C for this setup's segment duration."""
        return hh_from_matrix(M, C, T=self.bands.T)

    def summary(self):
        """Per-band/class prior anchors, margins, block and slack, all in seconds."""
        return [row.copy() for row in self._summary]

    def format_summary(self):
        """Markdown table of summary(); slack is before margin clipping."""
        keys = ["b", "lag_class", "Lambda_min", "Lambda_max", "delta_lo", "delta_hi",
                "block_start", "block_end", "slack"]
        lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
        for row in self.summary():
            lines.append("| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float)
                                          else str(row[k]) for k in keys) + " |")
        return "\n".join(lines)


def hh_from_matrix(M, C, T):
    """Return (4/T) Re(C^dagger M C), M excluding the Eq. (4) factor.

    C must follow the evaluator's mode ordering. T is explicit because
    an ordinary ndarray M carries no segment-duration metadata.
    """
    M, C = np.asarray(M), np.asarray(C)
    if C.ndim != 1 or M.shape != (len(C), len(C)) or not np.isfinite(T) or T <= 0:
        raise ValueError('Require square M, matching C and positive T')
    return float((4/T) * np.vdot(C, M @ C).real)
