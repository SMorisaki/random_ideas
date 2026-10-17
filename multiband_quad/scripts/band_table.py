"""Print the iteration-001 band/lag tables and save data/band_tables.md.

Run from the repository directory with ``python scripts/band_table.py``.
Geometry follows Morisaki (2021), Eqs. (13), (24), (47), (51), (53), (55),
(56); the sample-count ratio follows Eqs. (8), (25). No waveform is evaluated.
"""

from pathlib import Path
import sys

# Keep the standalone script runnable before any package installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbquad import BandSetup  # noqa: E402


def configurations():
    """Yield the specified BNS (m_max=2,4), GW190814-like and BBH geometries."""
    for m_max in (2, 4):
        yield f"BNS, m_max = {m_max}", BandSetup(
            T=256, dt=1 / 4096, f_low=20, f_high=2048,
            durations=[256, 128, 64, 32, 16, 8, 4], mchirp_ref=1.15, m_max=m_max, L=5)
    yield "GW190814-like, m_max = 4", BandSetup(
        T=16, dt=1 / 4096, f_low=20, f_high=1024,
        durations=[16, 8, 4], mchirp_ref=6.4, m_max=4, L=5)
    yield "BBH, m_max = 4", BandSetup(
        T=8, dt=1 / 4096, f_low=20, f_high=1024,
        durations=[8, 4], mchirp_ref=20, m_max=4, L=5)


def format_lags(bands):
    """Format the Appendix-A predicted (m_slow,m_fast)=(4,1) lag in seconds."""
    lines = ["| b | support start (Hz) | lag_max(4,1) (s) | T_b/2 (s) | T_b (s) |",
             "| --- | --- | --- | --- | --- |"]
    for b, T_b in enumerate(bands.T_b):
        lines.append(f"| {b} | {bands.f_b[b] - bands.Df_b[b]:.6f} | "
                     f"{bands.lag_max_pair(b, 4, 1):.6f} | {T_b / 2:g} | {T_b:g} |")
    return "\n".join(lines)


def main():
    """Generate tables from the specified geometries, then print and save them."""
    sections = ["# Multibanding geometry — iteration 001",
                "Times are seconds; frequencies and transition widths are Hz. "
                "f_lo and f_hi are nominal edges; support starts at f_lo - Df_lo. "
                "K_orig uses Eq. (8), K_MB uses Eq. (25). All setups use L = 5, "
                "tc_min_minus_T = -2.12 s and T_minus_tc_max = 1.9 s.",
                "The upper taper extends beyond f_high. In BNS, f_high is Nyquist: "
                "Eq. (8) counts that endpoint, while the Eq. (4) fine grid excludes it "
                "(one-bin difference). Lag predictions omit window tails. "
                "The (4,1) lag for the m_max = 2 setup is diagnostic only: "
                "that setup was designed for m <= 2."]
    for label, bands in configurations():
        sections.extend([f"## {label}",
                         f"K_orig = {bands.K_orig}; K_MB = {bands.K_MB}; "
                         f"K_orig / K_MB = {bands.K_orig / bands.K_MB:.6f}.",
                         bands.format_summary(), "Predicted largest (4,1) lag:",
                         format_lags(bands)])
    report = "\n\n".join(sections) + "\n"
    output = ROOT / "data" / "band_tables.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
