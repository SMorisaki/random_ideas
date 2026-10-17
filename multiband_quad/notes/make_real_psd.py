"""Orchestrator script (run outside the repository, next to the downloaded GWOSC files).
Welch (median-average) PSD of the public GWOSC 4 kHz strain around GW190814 (GPS 1249852257.0),
16 s Tukey-windowed segments, 50% overlap, excluding +-32 s around the event and any segment with NaNs.
Output: two-column text (frequency [Hz], one-sided PSD [1/Hz]) at df = 1/16 Hz, written next to the inputs."""
import numpy as np, h5py, sys
from scipy.signal import welch
from scipy.signal.windows import tukey
GPS_EVENT = 1249852257.0; SEG = 16.0; FS = 4096
for det in ("L1", "H1"):
    with h5py.File(f"{det}_4096_4k.hdf5") as h:
        x = h["strain/Strain"][:]; t0 = h["meta/GPSstart"][()]
    n_seg = int(SEG*FS); nan = np.isnan(x)
    i_ev = int((GPS_EVENT - t0)*FS); mask = np.ones_like(x, bool); mask[i_ev - 32*FS: i_ev + 32*FS] = False
    win = tukey(n_seg, alpha=0.1)
    psds = []
    for start in range(0, len(x) - n_seg + 1, n_seg//2):
        sl = slice(start, start + n_seg)
        if nan[sl].any() or not mask[sl].all(): continue
        f, p = welch(x[sl], fs=FS, window=win, nperseg=n_seg, noverlap=0, detrend="constant", return_onesided=True, scaling="density")
        psds.append(p)
    psds = np.array(psds)
    # median average with the bias correction for the median of chi^2_2 variables (as in LVK/bilby)
    med = np.median(psds, axis=0)/np.log(2)
    out = f"psd_{det}_GW190814_welch16s.txt"
    np.savetxt(out, np.column_stack([f, med]), fmt="%.6f %.8e", header=f"GWOSC {det} 4 kHz strain around GW190814, Welch median PSD, {len(psds)} segments of {SEG:.0f} s (Tukey alpha=0.1, 50% overlap), event +-32 s excluded, median/ln2 bias-corrected. columns: f[Hz] PSD[1/Hz]")
    print(det, "segments:", len(psds), "nan samples:", int(nan.sum()), "->", out)
    for ff in (20, 30, 60, 100, 300, 1000): print(f"   S({ff} Hz) = {med[int(ff*SEG)]:.3e}")
