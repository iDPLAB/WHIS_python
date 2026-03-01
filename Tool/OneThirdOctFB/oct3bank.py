# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
from scipy.signal import butter, lfilter, decimate
from oct3dsgn import oct3dsgn

def oct3bank(x=None, fs=44100):
    
    Fs = fs
    N = 3
    F = np.array([100,125,160,200,250,315,400,500,630,800,1000,1250,
                  1600,2000,2500,3150,4000,5000])
    ff = 1000.0 * ((2 ** (1/3.0)) ** np.arange(-10, 8))

    P = np.zeros(18)
    has_input = x is not None

    if has_input:
        x = np.asarray(x)
        # if stereo, convert to mono
        if x.ndim > 1:
            x = x.mean(axis=1)
        m = len(x)


    BB = []
    AA = []
    for i in range(17, 12, -1):
        B, A = oct3dsgn(ff[i], Fs, N)
        if has_input:
            y = lfilter(B, A, x)
            P[i] = np.sum(y**2) / m
        else:
            BB.append(B)
            AA.append(A)


    if has_input:
        Bu, Au = oct3dsgn(ff[15], Fs, N)
        Bc, Ac = oct3dsgn(ff[14], Fs, N)
        Bl, Al = oct3dsgn(ff[13], Fs, N)

        x_dec = x
        for j in range(3, -1, -1):

            x_dec = decimate(x_dec, 2, zero_phase=True)
            m_dec = len(x_dec)


            y = lfilter(Bu, Au, x_dec)
            P[j*3 + 2] = np.sum(y**2) / m_dec

            y = lfilter(Bc, Ac, x_dec)
            P[j*3 + 1] = np.sum(y**2) / m_dec

            y = lfilter(Bl, Al, x_dec)
            P[j*3 + 0] = np.sum(y**2) / m_dec

        # convert to dB, reference = 1
        Pref = 1.0
        idx = P > 0
        P[idx] = 10.0 * np.log10(P[idx] / Pref)
        P[~idx] = np.nan
        return P, F

    else:

        BB_full = []
        AA_full = []
        for i in range(18):
            B, A = oct3dsgn(ff[i], Fs, N)
            BB_full.append(B)
            AA_full.append(A)

        return np.array(BB_full, dtype=object), np.array(AA_full, dtype=object)
