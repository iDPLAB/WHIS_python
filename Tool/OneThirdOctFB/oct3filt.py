# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, decimate
from oct3dsgn import oct3dsgn 

def oct3filt(B, A, x, fs=44100, plot=False):

    F = np.array([
        100,125,160,200,250,315,400,500,630,800,
        1000,1250,1600,2000,2500,3150,4000,5000
    ])
    ff = 1000.0 * ((2 ** (1/3.0)) ** np.arange(-10, 8))


    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    m = len(x)

    P = np.zeros(18)

    for i in range(17, 12, -1):  # MATLAB 18:-1:13
        y = lfilter(B[i], A[i], x)
        P[i] = np.sum(y ** 2) / m


    N = 3
    Bu, Au = oct3dsgn(ff[14], fs, N)
    Bc, Ac = oct3dsgn(ff[13], fs, N)
    Bl, Al = oct3dsgn(ff[12], fs, N)

    x_dec = x.copy()
    for j in range(3, -1, -1):

        x_dec = decimate(x_dec, 2, zero_phase=True)
        m_dec = len(x_dec)

        indu = j * 3 + 2
        indc = j * 3 + 1
        indl = j * 3 + 0

        y = lfilter(B[indu], A[indu], x_dec)
        P[indu] = np.sum(y ** 2) / m_dec

        y = lfilter(B[indc], A[indc], x_dec)
        P[indc] = np.sum(y ** 2) / m_dec

        y = lfilter(B[indl], A[indl], x_dec)
        P[indl] = np.sum(y ** 2) / m_dec

    Pref = 1.0
    idx = P > 0
    P_db = np.full_like(P, np.nan)
    P_db[idx] = 10 * np.log10(P[idx] / Pref)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(18), P_db, color="skyblue", edgecolor="k")
        plt.xticks(np.arange(1, 18, 3), F[1::3])
        plt.xlabel("Frequency band [Hz]")
        plt.ylabel("Power [dB]")
        plt.title("One-third-octave spectrum")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.show()

    return P_db, F
