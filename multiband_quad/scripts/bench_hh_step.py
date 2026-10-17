"""Time only the HH step after coarse-waveform evaluation, Eqs. (28), (45)."""

from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbquad.bands import BandSetup  # noqa: E402
from mbquad.coarse import CoarseWaveform  # noqa: E402
from mbquad.ifftfft import IFFTFFTHH  # noqa: E402
from mbquad.linear import LinearInterpolationHH  # noqa: E402
from mbquad.psd import PSDGrid, add_lines, aligo_analytic_psd  # noqa: E402
from mbquad.testcases import TOY_LINES, toy_small  # noqa: E402
from mbquad.timing import median_time  # noqa: E402
from mbquad.toywave import ToyModes  # noqa: E402


def main():
    """Write median seconds/call as milliseconds, waveform and HH separately."""
    toy = toy_small()
    gw_bands = BandSetup(T=16, dt=1 / 4096, f_low=20, f_high=1024,
                         durations=[16, 8, 4], mchirp_ref=6.4, m_max=4, L=5)
    gw_psd = PSDGrid(add_lines(aligo_analytic_psd, TOY_LINES),
                     gw_bands.T, gw_bands.dt, gw_bands.f_low, gw_bands.f_high)
    cases = [("toy_small", toy.bands, toy.psdgrid, toy.model),
             ("GW190814-like", gw_bands, gw_psd, ToyModes(6.4, t_c=14))]
    sections = ["# HH-step timing — iteration 002",
                f"Machine: {platform.machine()}, Python {platform.python_version()}, "
                f"numpy {np.__version__}, scipy {scipy.__version__}. Wall-clock median "
                "per call after one warm-up; 7 batches of 100 calls. scipy real FFTs use "
                "one worker. Both models use all five default toy modes and analytic "
                "aLIGO PSD + TOY_LINES. Setup/precomputation is outside all timings.",
                "toy_small: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s, "
                "mchirp=mchirp_ref=4.85 Msun, t_c=126 s. GW190814-like uses iteration-001 "
                "geometry: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, "
                "mchirp=mchirp_ref=6.4 Msun, t_c=14 s. Both use m_max=4, L=5.",
                "The waveform column times model.total(f) on the precomputed concatenated "
                "coarse grid. Each HH column reuses those total-waveform samples and excludes "
                "waveform evaluation, container construction, and precomputation. The fast "
                "implementation reuses input/work buffers; scipy allocates FFT outputs."]
    rows = ["| system | K_MB | waveform (ms) | linear HH (ms) | "
            "paper IFFT-FFT HH (ms) | fast IFFT-FFT HH (ms) |",
            "| --- | --- | --- | --- | --- | --- |"]
    for label, bands, psd, model in cases:
        f = bands.all_coarse_frequencies().f
        coarse = CoarseWaveform.from_array(bands, model.total(f))
        methods = (LinearInterpolationHH(bands, psd),
                   IFFTFFTHH(bands, psd, impl="paper"),
                   IFFTFFTHH(bands, psd, impl="fast"))
        times = [median_time(model.total, f)]
        times.extend(median_time(method, coarse) for method in methods)
        rows.append(f"| {label} | {bands.K_MB} | "
                    + " | ".join(f"{1e3 * value:.6f}" for value in times) + " |")
    sections.append("\n".join(rows))
    report = "\n\n".join(sections) + "\n"
    output = ROOT / "data" / "bench_hh_step_iter002.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
