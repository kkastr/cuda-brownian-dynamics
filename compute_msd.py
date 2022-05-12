import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import time


def acf_fft(x):

    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size // 2] / np.sum(xp**2)


def acf_psd(x):
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    psd = F * F.conjugate()
    res = np.fft.ifft(psd)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    return res / n


def msd_fft(r):
    N = r.shape[0]
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    Q = 2 * D.sum()
    S2 = np.sum([acf_psd(r[:, i]) for i in range(r.shape[1])], axis=0)
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2

def msd_standard(r):
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

scols = ['x', 'y', 'z']

dfn = df.astype({'frame': np.int16, 'id': np.int16,
                 'x': np.float32, 'y': np.float32, 'z': np.float32})

rdf = dfn.groupby(['id'])

before = time.perf_counter()

msds = np.zeros((N, nframes))

for key, grp in rdf:

    msds[key, :] = msd_fft(grp[scols].values)


msd = msds.mean(axis=0)[1:]

after = time.perf_counter()

elapsed = after - before

print(f"elapsed: {elapsed} seconds")

t = np.arange(1, nframes)

halfway = nframes // 2

fx = np.log10(t)
fy = np.log10(msd)

out, cov = optimization.curve_fit(fitfunc, fx, fy, [0, 0])

a = out[0]
b = out[1]

print(f"a = {out[0]}, b = {out[1]}")

Dth = 1

Dsim = (10**b / 6) / (outfreq * dt)  # approriate time units

Derror = abs(Dth - Dsim) / Dth

yfit = (10**b) * t**a

plt.rc('font', family='serif', size=20)
plt.rc('lines', linewidth=4, aa=True)

text_props = dict(boxstyle='round', facecolor='white', alpha=0.25)

fig = plt.figure(figsize=(10, 6))

plt.loglog(t, yfit, linestyle='--', color='k', label='Theory', zorder=5)
plt.loglog(t, msd, color='r', label='Simulation', zorder=0)

plt.text(100, 1000, f'$D_\\mathrm{{error}} = {(100 * Derror):.3f} \\%$', bbox=text_props)

plt.ylabel('Mean Square Displacement')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/msd.png', dpi=600)
plt.show()
