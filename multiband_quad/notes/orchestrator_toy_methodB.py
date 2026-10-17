"""Method B (total waveform only): baseband IFFT-FFT with half-sample shift; compare with the paper's IFFT-FFT."""
import numpy as np, timeit
import importlib.util, sys
spec = importlib.util.spec_from_file_location("toy", "toy_blockweights.py")
# run the toy module (prints its own results) and reuse its objects
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
from scipy.fft import next_fast_len, fft, ifft

def hh_methodB(modes, adaptive=True, verbose=False):
    tot = 0.0; fftlens = []
    for bd in toy.bands:
        b = bd['b']; Mb, Nb, dtb, kap, fc = bd['Mb'], bd['Nb'], bd['dtb'], bd['kap'], bd['fc']
        Tb = toy.Tb[b]
        h = sum(toy.hmode(fc, lm) for lm in modes)
        g_int = bd['wfc']*np.abs(h)**2                       # w |h|^2 on the integer grid
        # predicted two-sided lag support of |h|^2 in this band: lag_max = tau_mmax - tau_mmin at the band start
        flo = toy.fedge[b] - toy.dF[b] if b > 0 else toy.fedge[b]
        lag_max = toy.tau0(2*flo/4) - toy.tau0(2*flo/1)
        need_half = (not adaptive) or (lag_max + 0.125*Tb > 0.5*Tb)
        if (Mb == Nb) or not need_half:
            # integer grid suffices: centred block of length M (band 0: full block == exact)
            Nhat = Mb
            W = toy.block_weight(bd, -Mb//2, Mb)[kap]
            tot += (4/toy.T)*np.real(np.sum(W*g_int)); fftlens.append(0); continue
        # half points via baseband IFFT-FFT: K' >= K samples of sqrt(w) h starting at kappa = Ks
        x = np.sqrt(bd['wfc'])*h; K = len(kap); Kp = next_fast_len(K)
        y = ifft(x, Kp)                                       # demodulated time series, step Tb/Kp, support assumed [T-Tb, T] -> n' = n
        n = np.arange(Kp)
        z = fft(y*np.exp(-1j*np.pi*n/Kp))                     # spectrum at kappa + 1/2 (kappa = Ks + j), j = 0..Kp-1
        g_half = np.abs(z[:K])**2                             # w |h|^2 at half points inside the band (j < K)
        Nhat = 2*Mb
        W2 = toy.block_weight(bd, -Mb, 2*Mb)                  # symmetric block of length 2M on the 2T grid
        tot += (4/toy.T)*np.real(np.sum(W2[2*kap]*g_int) + np.sum(W2[2*kap+1]*g_half))
        fftlens.append(Kp)
    if verbose: print("   FFT lengths per band (method B):", fftlens, " paper IFFT/FFT lengths:", [(bd['Mb'], bd['Nhat']) for bd in toy.bands])
    return tot

for label, modes in [("quadrupole only", [(2,2)]), ("with HOM", list(toy.MODES))]:
    ex = toy.hh_exact(modes)
    print(f"== {label}: exact = {ex:.6e}")
    print(f"   paper IFFT-FFT          rel.err = {(toy.hh_ifftfft(modes)-ex)/ex:+.3e}")
    print(f"   method B (adaptive)     rel.err = {(hh_methodB(modes, True, verbose=True)-ex)/ex:+.3e}")
    print(f"   method B (always half)  rel.err = {(hh_methodB(modes, False)-ex)/ex:+.3e}")
# timing of the (h,h) step only (modes precomputed on the coarse grids)
modes = list(toy.MODES)
pre = {bd['b']: sum(toy.hmode(bd['fc'], lm) for lm in modes) for bd in toy.bands}
def step_paper():
    tot = 0.0
    for bd in toy.bands:
        Mb, Nhat, dtb = bd['Mb'], bd['Nhat'], bd['dtb']; Tb_ = toy.Tb[bd['b']]
        x = np.zeros(Mb, complex); x[bd['kap']] = pre[bd['b']]*np.sqrt(bd['wfc'])
        hhat = (2/Tb_)*np.real(Mb*np.fft.ifft(x)); hc = np.zeros(Nhat); hc[Nhat-Mb:] = hhat
        htil = dtb*np.fft.fft(hc); kk = np.arange(1, (Nhat-1)//2+1)
        tot += (4/(Nhat*dtb))*np.sum(np.real(bd['Itil'][kk])*np.abs(htil[kk])**2)
    return tot
Wcache = {}
for bd in toy.bands:
    b = bd['b']; Mb, kap = bd['Mb'], bd['kap']
    Wcache[b] = (toy.block_weight(bd, -Mb//2, Mb)[kap], toy.block_weight(bd, -Mb, 2*Mb))
def step_B():
    tot = 0.0
    for bd in toy.bands:
        b = bd['b']; Mb, Nb, kap = bd['Mb'], bd['Nb'], bd['kap']; Tb = toy.Tb[b]
        h = pre[b]; g_int = bd['wfc']*np.abs(h)**2
        flo = toy.fedge[b] - toy.dF[b] if b > 0 else toy.fedge[b]
        lag_max = toy.tau0(2*flo/4) - toy.tau0(2*flo/1)
        if (Mb == Nb) or not (lag_max + 0.125*Tb > 0.5*Tb):
            tot += (4/toy.T)*np.real(np.sum(Wcache[b][0]*g_int)); continue
        x = np.sqrt(bd['wfc'])*h; K = len(kap); Kp = next_fast_len(K)
        y = ifft(x, Kp); z = fft(y*np.exp(-1j*np.pi*np.arange(Kp)/Kp))
        W2 = Wcache[b][1]
        tot += (4/toy.T)*np.real(np.sum(W2[2*kap]*g_int) + np.sum(W2[2*kap+1]*np.abs(z[:K])**2))
    return tot
for name, fn in [("paper IFFT-FFT step", step_paper), ("method B step", step_B)]:
    t = min(timeit.repeat(fn, number=20, repeat=5))/20
    print(f"   time {name:22s}: {1e3*t:6.3f} ms   value rel.err = {(fn()-toy.hh_exact(modes))/toy.hh_exact(modes):+.2e}")
print("   K per band:", [len(bd['kap']) for bd in toy.bands])
