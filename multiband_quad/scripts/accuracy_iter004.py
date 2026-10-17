"""Fine-grid accuracy and folding audit for total-waveform B0/B1 (§§3,7,8.8)."""

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbquad.bands import BandSetup  # noqa: E402
from mbquad.baseband import BasebandIFFTFFTHH  # noqa: E402
from mbquad.coarse import CoarseModes  # noqa: E402
from mbquad.ifftfft import IFFTFFTHH  # noqa: E402
from mbquad.linear import LinearInterpolationHH  # noqa: E402
from mbquad.pairwise import PairwiseHH  # noqa: E402
from mbquad.psd import PSDGrid, aligo_analytic_psd, load_asd_or_psd  # noqa: E402
from mbquad.reference import hh_exact  # noqa: E402
from mbquad.testcases import toy_small  # noqa: E402
from mbquad.totalweights import CentredBlockHH  # noqa: E402
from mbquad.toywave import DEFAULT_MODES, ToyModes  # noqa: E402


def experiment_cases():
    """Fourteen fixed cases, including matched/synthetic smooth-PSD controls."""
    toy = toy_small()
    bs = toy.bands
    smooth = PSDGrid(aligo_analytic_psd, bs.T, bs.dt, bs.f_low, bs.f_high)
    strong = {lm: (ratio, DEFAULT_MODES[lm][1], 0.0)
              for lm, ratio in zip(DEFAULT_MODES, (1, 0.3, 0.6, 0.1, 0.4))}
    gw = BandSetup(16, 1 / 4096, 20, 1024, [16, 8, 4], 6.4, 4, 5)
    real = PSDGrid(load_asd_or_psd(ROOT / "data/psd_L1_GW190814_welch16s.txt", skip_dc=True),
                   gw.T, gw.dt, gw.f_low, gw.f_high)
    gw_smooth = PSDGrid(aligo_analytic_psd, gw.T, gw.dt, gw.f_low, gw.f_high)
    # Same broadband sensitivity, with narrow features removed by a 16-Hz
    # running median on the original 1/16-Hz grid (257 bins, inclusive).
    measured = np.loadtxt(ROOT / "data/psd_L1_GW190814_welch16s.txt")
    measured = measured[measured[:, 0] > 0]
    median_log_S = median_filter(np.log(measured[:, 1]), size=257, mode="nearest")

    def smooth_measured(f):
        return np.exp(np.interp(f, measured[:, 0], median_log_S))

    gw_median = PSDGrid(smooth_measured, gw.T, gw.dt, gw.f_low, gw.f_high)
    geometries = [("toy_small", bs, [("lines", toy.psdgrid), ("smooth", smooth)], DEFAULT_MODES),
                  ("strong-HOM toy", bs, [("lines", toy.psdgrid), ("smooth", smooth)], strong),
                  ("GW190814-like", gw, [("real L1", real), ("smooth control", gw_smooth),
                                         ("L1 median16Hz", gw_median)], DEFAULT_MODES)]
    for name, bands, psds, modes in geometries:
        for psd_name, psd in psds:
            for hom in (False, True):
                model = ToyModes(bands.mchirp_ref, t_c=bands.T - 2,
                                 modes=modes if hom else {(2, 2): (1, 0, 0)})
                yield name, psd_name, "five" if hom else "22", bands, psd, model


def convention_checks(case):
    """Measure the requested identities separately from fine-grid norm accuracy."""
    bs, psd = case.bands, case.psdgrid
    forced = BasebandIFFTFFTHH(bs, psd, force_half_points=True)
    adaptive = BasebandIFFTFFTHH(bs, psd)
    paper = IFFTFFTHH(bs, psd)
    rows = ["| b | W vs requested (That/T) I, max-scaled | correct even-kernel identity | half points vs paper, interior | half points vs paper, overall | K'/M |",
            "| --- | --- | --- | --- | --- | --- |"]
    coarse = CoarseModes.from_model(bs, case.model, case.model.modes).total()
    for b, grid in enumerate(bs.coarse_frequencies()):
        n = int(paper.Nhat_b[b])
        W = forced.blockweights.centred(b, n)
        k = np.arange(1, n // 2)
        I = paper.I_tilde_b[b]
        even = (W[k] + W[-k]).real
        correct = np.max(np.abs(even - bs.T / paper.That_b[b] * I)) / np.max(np.abs(even))
        kb = (n // bs.M_b[b]) * grid.kappa
        target = paper.That_b[b] / bs.T * I[kb - 1]
        wrong = np.max(np.abs(W[kb] - target)) / np.max(np.abs(W[kb]))
        interior_error = overall_error = float("nan")
        if b:
            z = forced.half_point_waveform(coarse, b)[:-1]
            hp = paper.half_point_waveform(coarse, b)[1::2]
            f = forced.half_point_frequencies(b)[:-1]
            lo, hi = bs.f_b[b], bs.f_b[b + 1] - bs.Df_b[b + 1]
            mask = (f > lo + .2 * (hi - lo)) & (f < hi - .2 * (hi - lo))
            interior_error = np.max(np.abs(z[mask] - hp[mask])) / np.max(np.abs(hp[mask]))
            overall_error = np.max(np.abs(z - hp)) / np.max(np.abs(hp))
        rows.append(f"| {b} | {wrong:.6e} | {correct:.6e} | {interior_error:.6e} | "
                    f"{overall_error:.6e} | {forced.Kprime_b[b]/bs.M_b[b]:.6f} |")
    quad = ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
    co = CoarseModes.from_model(bs, quad, quad.modes).total()
    f = forced.half_point_frequencies(3)
    z = forced.half_point_waveform(co, 3)
    lo, hi = bs.f_b[3], bs.f_b[4] - bs.Df_b[4]
    direct = quad.total(f)
    plateau = (f >= lo) & (f <= hi)
    interior = (f > lo + .2 * (hi - lo)) & (f < hi - .2 * (hi - lo))
    err = np.abs((z - direct) / direct)
    agreement = []
    adaptive_gap = []
    for model in (quad, case.model):
        co = CoarseModes.from_model(bs, model, model.modes).total()
        actual = BasebandIFFTFFTHH(bs, psd, m_values=[m for _, m in model.modes])
        agreement.append(abs(forced(co) / paper(co) - 1))
        adaptive_gap.append(abs(actual(co) / forced(co) - 1))
    return ["## Convention checks and unmet requested thresholds",
            "The specified algorithms are retained. These checks expose differences in the requested "
            "identities rather than changing iteration-001–003 behavior or silently tuning B1. "
            "For the one-sided complex A_m, the exact paper-weight identity is "
            "Re[W(k)+W(-k)] = (T/That) I_tilde(k), NOT W(k)=(That/T) I_tilde(k). "
            "It follows by symmetrizing A_m before the DFT in Eq. (32). The missing negative-frequency "
            "weight and the inverted normalization matter. The table checks the correct identity on "
            "ALL positive bins and the requested identity on the band bins.",
            "\n".join(rows),
            "The minimal K' periodic baseband interpolant and the paper's real-signal full-spectrum "
            "interpolant have different tails and image contributions. Eq. (36) is approximate for these "
            "finite-window signals; it does not make the two discrete interpolants identical. Public "
            "half-point samples include a global minus sign relative to raw fft(ifft(x)*phase), from "
            "the negative-time origin n-K'; this aligns with the paper and direct ToyModes phases. "
            "The norm uses |z|² and is unaffected. Restricted-band quadrature also discards the paper "
            "interpolant's out-of-band power; exact weight identities must use its entire positive grid.",
            f"Pure (2,2), band 3: max pointwise half-point error against direct evaluation is "
            f"{np.max(err[plateau]):.6e} on the whole plateau and {np.max(err[interior]):.6e} "
            "in its central 60%. Only the latter meets 1e-4.",
            "| requested check | achieved | result |\n"
            "| --- | --- | --- |\n"
            f"| forced B1 vs paper <1e-8 relative | 22: {agreement[0]:.6e}; five: {agreement[1]:.6e} | unmet |\n"
            "| complex W=(That/T) I within 1e-10 | up to 0.996 max-scaled discrepancy; correct symmetrized identity <7e-16 | incorrect identity |\n"
            f"| adaptive vs forced <1e-7, actual m values | 22: {adaptive_gap[0]:.6e}; five: {adaptive_gap[1]:.6e} | unmet for 22 |\n"
            "| default five-mode adaptive uses upper-band B0 | no B0 with prescribed 5 sigma margin | unmet |\n"
            "| B0 five-mode relative error <1e-5 and worse than B1 (lines) | 4.395005e-6 vs B1 4.398792e-7 | met |\n"
            f"| pure 22 half points, whole plateau band 3 <1e-4 | {np.max(err[plateau]):.6e}; central 60% {np.max(err[interior]):.6e} | met only in interior |\n"
            "| complex half-point agreement with paper, interior 1e-10 / overall ~1e-8 | interior 1.59e-5 to 3.70e-4 / overall 6.78e-4 to 4.03e-3, five modes | unmet |\n"
            "| K' <= M/2, all toy b>=1 | K'/M=0.07251,0.10547,0.08545,0.17578,0.20142 | met |",
            f"Default five-mode adaptive modes: {adaptive.modes}. In the top band, "
            f"lag_max+5 sigma={adaptive.lag_max[-1]+adaptive.margin[-1]:.6f} s > "
            f"T_b/2={bs.T_b[-1]/2:.6f} s: the requested rule does NOT select B0 there. "
            "With L_margin=4 it does; that optional setting is tested separately. Quadrupole-only "
            "evaluators use m_values=(2,) and do select B0 in upper bands. Finally, M=N needs no "
            "interpolation; 2M=N still needs half points (toy band 1)."]


def main():
    """Write errors and exact B0 folding decompositions, all at SNR 25."""
    sections = ["# Accuracy — iteration 004",
                "All references use brute-force Eq. (4) on the PSDGrid fine grid; |Delta ln Lambda| "
                "=25² |Delta(h,h)|/(2(h,h)_exact). Each table entry is one deterministic system, "
                "not a population. Pairwise uses iteration-003 defaults (pairs, block, 5 sigma_pair). "
                "B1 uses the m values actually present, L_margin=5 sigma_mmax, K'=next_fast_len(K_b). "
                "No waveform calls enter the HH step.",
                "Toy geometry: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s; "
                "Mc=Mc_ref=4.85 Msun, tc=126 s. Default ratios=(1,.25,.35,.1,.15); strong-HOM "
                "ratios=(1,.3,.6,.1,.4), same phases, no amplitude growth. Both use smooth analytic "
                "aLIGO and TOY_LINES PSDs. Quadrupole controls are identical between those two toys. "
                "GW190814-like: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, Mc=Mc_ref=6.4, "
                "tc=14 s, default ratios; real L1 PSD loaded with load_asd_or_psd from "
                "data/psd_L1_GW190814_welch16s.txt with skip_dc=True (the file includes f=0, "
                "unused by Eq. 4), a smooth analytic control, and a matched L1 median16Hz "
                "control (running median of log PSD, 257 bins on the original 1/16-Hz grid; "
                "this removes narrow features, not just individual known lines). "
                "All geometries use m_max=4, L=5."]
    errors = {name: [] for name in ("linear", "IFFT-FFT fast", "pairwise", "B0", "B1 adaptive", "B1 forced")}
    by_case, band_rows, measurements = {}, [], {}
    for system, noise, modes, bs, psd, model in experiment_cases():
        cm = CoarseModes.from_model(bs, model, model.modes)
        co = cm.total()
        h = model.total(psd.f_k)
        exact = hh_exact(h, psd)
        b0 = CentredBlockHH(bs, psd)
        b1 = BasebandIFFTFFTHH(bs, psd, m_values=[m for _, m in model.modes])
        forced = BasebandIFFTFFTHH(bs, psd, m_values=b1.m_values, force_half_points=True)
        methods = [LinearInterpolationHH(bs, psd), IFFTFFTHH(bs, psd),
                   PairwiseHH(bs, psd, cm.modes), b0, b1, forced]
        values = [method(cm if name == "pairwise" else co)
                  for name, method in zip(errors, methods)]
        measurements[system, noise, modes] = dict(zip(errors, (abs(v / exact - 1) for v in values)))
        rows = by_case.setdefault(system, [])
        for name, value in zip(errors, values):
            rel = abs(value / exact - 1)
            dll = 25**2 * rel / 2
            errors[name].append((rel, dll))
            rows.append(f"| {noise} | {modes} | {name} | {rel:.6e} | {dll:.6e} |")
        v0, v1 = b0.band_contributions(co), b1.band_contributions(co)
        for b in range(bs.B):
            g = bs.window(psd.f_k, b) * np.abs(h)**2
            ref = (4 / bs.T) * np.dot(g, psd.inv_S)
            audit = b0.folding_error_estimate(g, b)
            # This is an independent fine-grid audit of the implemented B0 error.
            if not np.isclose(v0[b] - ref, audit["hh_folding_error"],
                              rtol=1e-6, atol=1e-13 * exact):
                raise AssertionError("B0 folding decomposition disagrees with its band norm")
            band_rows.append(f"| {system}, {noise}, {modes} | {b} | "
                             f"{(v0[b]-ref)/exact:+.6e} | {(v1[b]-ref)/exact:+.6e} | "
                             f"{audit['hh_folding_error']/exact:+.6e} | "
                             f"{audit['outside_fraction']:.4e} | {b1.lag_max[b]:.6f} | "
                             f"{bs.T_b[b]/2:.6f} | {b1.margin[b]:.6f} | {b1.modes[b]} |")
    for name, rows in by_case.items():
        sections.extend([f"## {name}",
                         "| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |\n"
                         "| --- | --- | --- | --- | --- |\n" + "\n".join(rows)])
    stats = ["| method | median relative error | worst relative error | median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |",
             "| --- | --- | --- | --- | --- |"]
    for name, values in errors.items():
        med, worst = np.median(values, axis=0), np.max(values, axis=0)
        stats.append(f"| {name} | {med[0]:.6e} | {worst[0]:.6e} | {med[1]:.6e} | {worst[1]:.6e} |")
    real_errors = measurements["GW190814-like", "real L1", "five"]
    smooth_errors = measurements["GW190814-like", "L1 median16Hz", "five"]
    sections.extend(["## Real-PSD narrow-feature comparison",
                     f"For five modes, B0 relative error is {real_errors['B0']:.6e} on real L1 "
                     f"versus {smooth_errors['B0']:.6e} on its matched running-median control "
                     f"({real_errors['B0']/smooth_errors['B0']:.2f}x). B1 adaptive gives "
                     f"{real_errors['B1 adaptive']:.6e} versus {smooth_errors['B1 adaptive']:.6e}. "
                     "Narrow PSD features therefore matter much more for B0 in this case. "
                     "The per-band folding audit below explains the resulting signed error; "
                     "smoothing changes other local structure too, so this is not a single-line isolation test.",
                     "## Summary over fourteen fixed cases", "\n".join(stats),
                     "## Per-band error and fine-grid folding audit",
                     "Signed band errors below are divided by the TOTAL fine-grid norm, so columns "
                     "sum to the signed global relative error. Multiply their magnitudes by 312.5 "
                     "for SNR-25 log-likelihood contributions. The folding column independently "
                     "evaluates (4/TN) sum_out G_m (conj A_alias - conj A_m). outside_fraction uses "
                     "|G|, not |G|²; it alone does not determine accuracy. B0 cuts the predicted "
                     "extreme-pair bulk where lag_max>T_b/2, especially toy bands 1–4. Band 0 "
                     "retains all circular lags and is exact even if its unwrapped lag exceeds the limit.",
                     "| case | b | B0 signed relative contribution | B1 adaptive signed relative contribution | B0 folding contribution | outside fraction | lag_max (s) | T_b/2 (s) | margin (s) | B1 mode |\n"
                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(band_rows)])
    sections.extend(convention_checks(toy_small()))
    report = "\n\n".join(sections) + "\n"
    (ROOT / "data/accuracy_iter004.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
