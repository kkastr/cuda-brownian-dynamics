import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


def acf(x):

    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size // 2] / np.sum(xp**2)


def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    psd = F * F.conjugate()
    res = np.fft.ifft(psd)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    return res / n


def msd_fft(r):
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2


def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds


def fitfunc(x, a, b):
    return a * x + b


fname = './data/trajectory-N10000-frames1000-lbox1000-seed1234.csv'

N, nframes, lbox, seed = (int(num) for num in re.findall(r'\d+', fname))

outfreq = 100
dt = 0.1

df = pd.read_csv(fname)

rdf = df.groupby('id')

msdx = rdf.x.apply(lambda k: msd_fft(k.values.reshape(nframes, 1)))
msdy = rdf.y.apply(lambda k: msd_fft(k.values.reshape(nframes, 1)))
msdz = rdf.z.apply(lambda k: msd_fft(k.values.reshape(nframes, 1)))

msdr = msdx + msdy + msdz

msd = msdr.mean(axis=0)[1:]

time = np.arange(1, nframes)

halfway = nframes // 2

px = np.log10(time)
py = np.log10(msd)

out, cov = optimization.curve_fit(fitfunc, px, py, [0, 0])

a = out[0]
b = out[1]

# print(f"a = {out[0]}, b = {out[1]}")

Dsim = (10**b / 6) / (outfreq * dt)  # approriate time units

yfit = (10**b) * time**a

plt.loglog(time, yfit, linestyle='--', color='k', label='Theory', zorder=5)
plt.loglog(time, msd, color='r', label='Simulation', zorder=0)
plt.ylabel('Mean Square Displacement')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()
