"""B1 variants: (i) band-only half bins (current), (ii) all K'' half bins of the baseband period, for zero-padding p."""
import numpy as np, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("toy", "toy_blockweights.py")
with contextlib.redirect_stdout(io.StringIO()):
    toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
from scipy.fft import next_fast_len, fft, ifft

def hh_B1(modes, p=1, allbins=False, real_signal=False):
    tot = 0.0
    for bd in toy.bands:
        b = bd['b']; Mb, Nb, kap, fc = bd['Mb'], bd['Nb'], bd['kap'], bd['fc']
        h = sum(toy.hmode(fc, lm) for lm in modes); g_int = bd['wfc']*np.abs(h)**2
        if Mb == Nb:
            W = toy.block_weight(bd, -Mb//2, Mb)[kap]; tot += (4/toy.T)*np.real(np.sum(W*g_int)); continue
        x = np.sqrt(bd['wfc'])*h; K = len(kap); Kp = next_fast_len(p*K)
        W2 = toy.block_weight(bd, -Mb, 2*Mb)          # weights on the 2T_b grid, periodic in index mod 2M
        if not real_signal:
            y = ifft(x, Kp); n = np.arange(Kp)
            z = fft(y*np.exp(-1j*np.pi*n/Kp))          # half bins j=0..Kp-1 at frequencies f_Ks + (j+1/2)/T_b
            if allbins:
                jj = np.arange(Kp); idx = (2*(kap[0] + jj) + 1) % (2*Mb)
                tot += (4/toy.T)*np.real(np.sum(W2[2*kap]*g_int) + np.sum(W2[idx]*np.abs(z)**2))
            else:
                tot += (4/toy.T)*np.real(np.sum(W2[2*kap]*g_int) + np.sum(W2[2*kap+1]*np.abs(z[:K])**2))
        else:
            # real-signal variant: build the real time series over the band Nyquist like the paper but with the
            # baseband trick is impossible; instead emulate the paper with M points to confirm equality with hh_ifftfft
            pass
    return tot

for label, modes in [("quadrupole", [(2,2)]), ("five modes", list(toy.MODES))]:
    ex = toy.hh_exact(modes)
    print(f"== {label}: paper IFFT-FFT rel.err = {(toy.hh_ifftfft(modes)-ex)/ex:+.3e}")
    for p in (1, 2, 4, 8, 16):
        print(f"   p={p:2d}: band-only {(hh_B1(modes, p, False)-ex)/ex:+.3e} | all half bins {(hh_B1(modes, p, True)-ex)/ex:+.3e}")
