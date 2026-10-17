"""Toy check of the pairwise block-weight quadrature for (h,h) (orchestrator's own test).
Synthetic Newtonian multi-mode SPA waveform, paper-style bands, analytic aLIGO-like PSD with lines."""
import numpy as np, time
from scipy.optimize import brentq

MSUN_S = 4.925491025543576e-06
T, dt = 128.0, 1/1024.
N = int(round(T/dt)); assert N & (N-1) == 0
f_low, f_high = 20.0, 256.0
Mc = 4.85  # Msun; tau_44(20 Hz) ~ 100 s < T
tc = T - 2.0
L = 5.0; tc_min_minus_T = -2.12; tc_max = T - 1.9
MODES = {(2,2): 1.0, (2,1): 0.25, (3,3): 0.35, (3,2): 0.10, (4,4): 0.15}
PHI0 = {(2,2): 0.0, (2,1): 0.7, (3,3): 1.3, (3,2): 2.1, (4,4): 0.4}
m_max = 4

GMc = Mc*MSUN_S
def tau0(f):  return (5/256)*GMc*(np.pi*GMc*f)**(-8/3)       # Eq. 53
def dtau0(f): return -(8/3)*tau0(f)/f
def tau_hom(f):  return tau0(2*f/m_max)                        # Eq. 55
def dtau_hom(f): return dtau0(2*f/m_max)*(2/m_max)
def Psi22(f): return 2*np.pi*f*tc - np.pi/4 + (3/128)*(np.pi*GMc*f)**(-5/3)
def hmode(f, lm):
    l, m = lm; fo = 2*f/m
    return MODES[lm]*1e-22*np.where(f > 0, f, 1.0)**(-7/6)*np.exp(-1j*((m/2)*Psi22(np.maximum(fo,1e-3)) + PHI0[lm]))

LINES = [(26.5,0.01,3000.), (36.,0.01,2000.), (48.5,0.01,1000.), (60.,0.02,500.), (120.,0.05,100.)]
def psd(f):
    x = f/215.0
    s = 1e-49*(x**-4.14 - 5*x**-2 + 111*(1 - x**2 + 0.5*x**4)/(1 + 0.5*x**2))
    for fl, wl, hl in LINES:
        s *= 1 + hl*np.exp(-0.5*((f-fl)/wl)**2)
    return s

# ---- bands (Sec. II.E/F) ----
Tb = [T/2**b for b in range(6)]; B = len(Tb)
fedge = [f_low]
for b in range(1, B):
    g = lambda f: tau_hom(f) + L*np.sqrt(-dtau_hom(f)) - (Tb[b] + tc_min_minus_T)   # Eq. 51
    fedge.append(brentq(g, f_low, 5000.))
dF = [0.0] + [1/np.sqrt(-dtau_hom(f)) for f in fedge[1:]]                              # Eq. 47
dF_B = 100/(T - tc_max); fedge.append(f_high + dF_B); dF.append(dF_B)                 # Eq. 56
for b in range(B): assert fedge[b] < fedge[b+1] - dF[b+1], (b, fedge[b], fedge[b+1], dF[b+1])
print("band edges f^(b):", np.round(fedge, 2)); print("Df^(b):", np.round(dF, 3))

def window(f, b):   # Eq. 9
    lo, hi, dlo, dhi = fedge[b], fedge[b+1], dF[b], dF[b+1]
    w = np.zeros_like(f)
    if dlo > 0:
        s = (f > lo-dlo) & (f < lo); w[s] = 0.5*(1 + np.cos(np.pi*(f[s]-lo)/dlo))
    s = (f >= lo) & (f <= hi-dhi); w[s] = 1.0
    s = (f > hi-dhi) & (f < hi);   w[s] = 0.5*(1 - np.cos(np.pi*(f[s]-hi)/dhi))
    return w

# ---- fine grid reference ----
k = np.arange(1, N//2); fk = k/T
invS = np.where((fk >= f_low) & (fk <= f_high), 1/psd(fk), 0.0)
wsum = sum(window(fk, b) for b in range(B)); assert np.allclose(wsum[(fk>=f_low)&(fk<=f_high)], 1.0)

def hh_exact(modes):
    h = sum(hmode(fk, lm) for lm in modes)
    return (4/T)*np.sum(np.abs(h)**2*invS)

# ---- per-band precomputation ----
bands = []
for b in range(B):
    Nb = 2; 
    while (Nb-1)//2 < fedge[b+1]*T: Nb *= 2                                           # Eq. 13
    dtb = T/Nb; Mb = int(round(Tb[b]/dtb)); assert abs(Mb*dtb - Tb[b]) < 1e-12
    Ks, Ke = int(np.ceil((fedge[b]-dF[b])*Tb[b])), int(np.floor(fedge[b+1]*Tb[b]))
    kap = np.arange(Ks, Ke+1); fc = kap/Tb[b]
    # A_m = sum_k a_k e^{2 pi i k m / Nb}, a_k = 1/S_k on the fine grid (one-sided), via ifft
    a = np.zeros(Nb, complex); a[k[k < Nb//2]] = invS[k < Nb//2]
    A = Nb*np.fft.ifft(a)                        # A_m, m = 0..Nb-1
    # IFFT-FFT precompute (Eqs. 32, 38, 43): I_m = (2/T) Re sum_k e^{2 pi i k m/Nb}/S_k
    I = (2/T)*np.real(A); Nhat = min(2*Mb, Nb)
    Ic = np.concatenate([I[:(Nhat+1)//2], I[Nb-Nhat//2:]]) if Nhat < Nb else I
    Itil = dtb*np.fft.fft(Ic)                     # Itilde_c,k
    # linear interpolation weights c_k (Eqs. 29-30): ranges use f~ (0 / inf at the ends), weights use the true f
    wb = window(fk, b); ab = wb*invS
    ftil = fc.copy(); ftil_m1 = np.concatenate([[0.0], fc[:-1]]); ftil_p1 = np.concatenate([fc[1:], [np.inf]])
    ftil[0] = 0.0; ftil[-1] = np.inf; ftil_m1[1] = 0.0; ftil_p1[-2] = np.inf   # f~_{Ks}=0, f~_{Ke}=inf
    c = np.zeros(len(kap))
    for j in range(len(kap)):
        s1 = (fk >= ftil[j]) & (fk < ftil_p1[j]); s2 = (fk >= ftil_m1[j]) & (fk < ftil[j])
        right = np.sum((fc[j+1] - fk[s1])*ab[s1]) if j < len(kap)-1 else 0.0
        left  = np.sum((fk[s2] - fc[j-1])*ab[s2]) if j > 0 else 0.0
        c[j] = (4*Tb[b]/T)*(right + left)
    bands.append(dict(b=b, Nb=Nb, dtb=dtb, Mb=Mb, Ks=Ks, Ke=Ke, kap=kap, fc=fc, A=A, Itil=Itil, Nhat=Nhat, c=c, wfc=window(fc, b)))

def block_weight(bd, m0, Nhat):
    """W_kappa(B) = (1/Nhat) sum_{m in B} A_m^* e^{2 pi i m kappa/Nhat}, B = {m0, ..., m0+Nhat-1} mod Nb."""
    Nb, A = bd['Nb'], bd['A']; idx = (m0 + np.arange(Nhat)) % Nb
    cvec = np.conj(A[idx]); kk = np.arange(Nhat)
    return np.exp(2j*np.pi*m0*kk/Nhat)*np.fft.ifft(cvec)   # ifft includes 1/Nhat

def hh_linear(modes):
    tot = 0.0
    for bd in bands:
        h = sum(hmode(bd['fc'], lm) for lm in modes); tot += np.sum(bd['c']*np.abs(h)**2)
    return tot

def hh_ifftfft(modes):
    tot = 0.0
    for bd in bands:
        Mb, Nhat, dtb, Tb_ = bd['Mb'], bd['Nhat'], bd['dtb'], Tb[bd['b']]
        h = sum(hmode(bd['fc'], lm) for lm in modes)*np.sqrt(bd['wfc'])
        x = np.zeros(Mb, complex); x[bd['kap']] = h
        hhat = (2/Tb_)*np.real(Mb*np.fft.ifft(x))          # Eq. 46 (real time series, length Mb)
        hc = np.zeros(Nhat); hc[Nhat-Mb:] = hhat             # last Mb samples
        htil = dtb*np.fft.fft(hc)
        kk = np.arange(1, (Nhat-1)//2 + 1)
        tot += (4/(Nhat*dtb))*np.sum(np.real(bd['Itil'][kk])*np.abs(htil[kk])**2)   # Eq. 45
    return tot

def hh_pairwise(modes, margin_frac=1/16, centred_all=False):
    """Block weights on the coarse grid: diagonal/centred block [-M/2, M/2); ordered cross pairs (slower first):
    one-sided block covering positive lags [-delta, M-delta) (and the mirrored one tried too)."""
    tot = 0.0
    tau_of = lambda lm: lambda f: tau0(2*f/lm[1])
    for bd in bands:
        Mb, Nb, kap, fc = bd['Mb'], bd['Nb'], bd['kap'], bd['fc']
        delta = int(round(margin_frac*Mb))
        Wc = block_weight(bd, -Mb//2, Mb)[kap]                 # centred
        Wp = block_weight(bd, -delta, Mb)[kap]                 # positive lags
        Wm = block_weight(bd, -(Mb-delta), Mb)[kap]            # negative lags
        hv = {lm: hmode(fc, lm)*bd['wfc'] for lm in modes}     # window attached to the product once below
        for i, A_ in enumerate(modes):
            for j, A2 in enumerate(modes):
                g = np.conj(hmode(fc, A_))*hmode(fc, A2)*bd['wfc']
                if i == j or centred_all: W = Wc
                else:
                    slower_first = tau_of(A_)(100.) > tau_of(A2)(100.)
                    W = Wp if slower_first else Wm
                tot += (4/T)*np.real(np.sum(W*g))
    return tot

print("\nmax lag (4,4)-(2,1) at band start vs T_b/2 and T_b:")
for b in range(B):
    flo = fedge[b] - dF[b] if b > 0 else fedge[b]
    lag = tau0(2*flo/4) - tau0(2*flo/1)
    print(f"  band {b}: T_b={Tb[b]:6.1f} s  lag_max={lag:7.2f} s  (T_b/2={Tb[b]/2:6.1f})  lines in band: {[l[0] for l in LINES if fedge[b] <= l[0] < fedge[b+1]]}")
for label, modes in [("quadrupole only", [(2,2)]), ("with HOM", list(MODES))]:
    ex = hh_exact(modes)
    print(f"\n== {label}: exact (h,h) = {ex:.6e}")
    for name, val in [("linear interp", hh_linear(modes)), ("IFFT-FFT", hh_ifftfft(modes)),
                      ("pairwise, all centred (wrong on purpose)", hh_pairwise(modes, centred_all=True)),
                      ("pairwise one-sided, margin M/16", hh_pairwise(modes, 1/16)),
                      ("pairwise one-sided, margin M/8", hh_pairwise(modes, 1/8)),
                      ("pairwise one-sided, margin M/32", hh_pairwise(modes, 1/32))]:
        print(f"  {name:45s} rel.err = {(val-ex)/ex:+.3e}")
import timeit
modes = list(MODES)
for name, fn in [("linear", hh_linear), ("IFFT-FFT", hh_ifftfft), ("pairwise", hh_pairwise)]:
    t = min(timeit.repeat(lambda: fn(modes), number=3, repeat=3))/3
    print(f"  time {name:10s} (incl. toy waveform eval): {1e3*t:7.2f} ms")
K_MB = sum(len(bd['kap']) for bd in bands); K_orig = int(np.sum((fk>=f_low)&(fk<=f_high)))
print(f"\nK_orig = {K_orig}, K_MB = {K_MB}, ratio = {K_orig/K_MB:.1f};  M_b = {[bd['Mb'] for bd in bands]}")
