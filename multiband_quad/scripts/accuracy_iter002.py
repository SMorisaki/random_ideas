"""Check Eqs. (28), (45) against Eq. (4); write data/accuracy_iter002.md."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbquad.coarse import CoarseWaveform  # noqa: E402
from mbquad.ifftfft import IFFTFFTHH  # noqa: E402
from mbquad.linear import LinearInterpolationHH  # noqa: E402
from mbquad.psd import PSDGrid, aligo_analytic_psd  # noqa: E402
from mbquad.reference import hh_exact, lnlambda_error  # noqa: E402
from mbquad.testcases import toy_small  # noqa: E402
from mbquad.toywave import ToyModes  # noqa: E402


def main():
    """Report absolute relative norm and log-likelihood errors at SNR 25."""
    case = toy_small()
    bands = case.bands
    rows, by_method = [], {"linear": [], "IFFT-FFT paper": [], "IFFT-FFT fast": []}
    for lines in (True, False):
        psd = case.psdgrid if lines else PSDGrid(
            aligo_analytic_psd, bands.T, bands.dt, bands.f_low, bands.f_high)
        methods = (LinearInterpolationHH(bands, psd),
                   IFFTFFTHH(bands, psd, impl="paper"),
                   IFFTFFTHH(bands, psd, impl="fast"))
        for hom in (False, True):
            model = case.model if hom else ToyModes(4.85, t_c=126, modes={(2, 2): (1, 0, 0)})
            coarse = CoarseWaveform.from_model(bands, model.total)
            exact = hh_exact(model.total(psd.f_k), psd)
            for name, method in zip(by_method, methods):
                value = method(coarse)
                rel = abs((value - exact) / exact)
                dll = abs(lnlambda_error(value, exact, 25))
                by_method[name].append((rel, dll))
                rows.append(f"| {'lines' if lines else 'smooth'} | "
                            f"{'five modes' if hom else '(2,2) only'} | {name} | "
                            f"{exact:.9e} | {rel:.6e} | {dll:.6e} |")
    sections = ["# Paper-method accuracy — iteration 002",
                "Fine-grid Eq. (4) reference on toy_small(): T=128 s, dt=1/1024 s, "
                "20–256 Hz; T_b=(128,64,32,16,8,4) s; mchirp=mchirp_ref=4.85 Msun, "
                "t_c=126 s, m_max=4, L=5. Five modes use ToyModes defaults; quadrupole "
                "uses only (2,2) with (ratio,phase,growth)=(1,0,0). PSD: analytic aLIGO "
                "with TOY_LINES, or the same PSD without lines. Windows and coarse grids "
                "are identical for both waveform choices.",
                "Errors are magnitudes. |Delta ln Lambda| = SNR² |Delta(h,h)/(h,h)|/2 "
                "at SNR=25, holding (d,h) fixed. Each row is one deterministic system.",
                "| PSD | waveform | method | exact (h,h) | relative error | "
                "abs(Delta ln Lambda), SNR 25 |\n"
                "| --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows)]
    summary = ["| method | median relative error | worst relative error | "
               "median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |",
               "| --- | --- | --- | --- | --- |"]
    for name, values in by_method.items():
        values = np.asarray(values)
        med, worst = np.median(values, axis=0), np.max(values, axis=0)
        summary.append(f"| {name} | {med[0]:.6e} | {worst[0]:.6e} | "
                       f"{med[1]:.6e} | {worst[1]:.6e} |")
    sections.extend(["Summary over the four waveform/PSD combinations (not a population):",
                     "\n".join(summary),
                     "No changes to the paper's formulas. The real-FFT path changes only "
                     "the transform implementation. half_point_waveform returns sqrt(w)h; "
                     "the pure-(2,2) half-point test covers the central 60% of band 5's "
                     "unit plateau, with maximum pointwise relative error 4.213e-5. "
                     "This is not a global interpolation bound: the same interior definition "
                     "gives errors up to 2.5e-4 in lower bands from the Eq. (36) tail approximation."])
    report = "\n\n".join(sections) + "\n"
    output = ROOT / "data" / "accuracy_iter002.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
