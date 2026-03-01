# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
from scipy.signal import lfilter
from oct3dsgn import oct3dsgn

def OneThirdOctFB(Snd, ParamOct3=None):

    print("### OneThirdOctFB ###")

    if ParamOct3 is None:
        ParamOct3 = {}

    ParamOct3.setdefault("OrderFilter", 3)
    ParamOct3.setdefault("fs", 48000)
    ParamOct3.setdefault("FreqRange", [100, 13000])
    ParamOct3.setdefault("FilterDelay1kHz", 0.003)   # 3ms delay baseline
    ParamOct3.setdefault("NrmlzPwrdB", 50)

    FcLabel = np.array([
        16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000, 12500, 16000, 20000
    ])

    FcList = 1000 * (2 ** (1/3)) ** np.arange(-18, 14)
    FboundaryList = np.vstack((FcList * 2 ** (-1/6), FcList * 2 ** (1/6)))

    fr = ParamOct3["FreqRange"]
    num_range = np.where((FcLabel >= min(fr)) & (FcLabel <= max(fr)))[0]

    FcLabel = FcLabel[num_range]
    FcList = FcList[num_range]
    FboundaryList = FboundaryList[:, num_range]

    ParamOct3["FcLabel"] = FcLabel
    ParamOct3["FcList"] = FcList
    ParamOct3["FboundaryList"] = FboundaryList
    ParamOct3["NumRange"] = num_range

    fs = ParamOct3["fs"]
    NumDelay1kHz = int(ParamOct3["FilterDelay1kHz"] * fs)

    LenOct3 = len(FcList)
    LenSnd = len(Snd)
    FBoct3 = np.zeros((LenOct3, LenSnd))
    FBoct3DlyCmp = np.zeros_like(FBoct3)
    NumDelay = np.zeros(LenOct3, dtype=int)

    for nf, Fc in enumerate(FcList):
        bz, ap = oct3dsgn(Fc, fs, ParamOct3["OrderFilter"])
        nDelay = int(np.fix(1000 / Fc * NumDelay1kHz))
        Snd1 = np.concatenate((Snd, np.zeros(nDelay)))
        SndFilt = lfilter(bz, ap, Snd1)
        FBoct3[nf, :] = SndFilt[:LenSnd]
        FBoct3DlyCmp[nf, :] = SndFilt[nDelay:nDelay + LenSnd]
        NumDelay[nf] = nDelay

    ParamOct3["NumDelay"] = NumDelay

    SndSyn = np.mean(FBoct3DlyCmp, axis=0)
    GainAnaSyn = np.sqrt(np.mean(Snd ** 2)) / np.sqrt(np.mean(SndSyn ** 2))
    ParamOct3["GainAnaSyn"] = GainAnaSyn

    PwrdB = 10 * np.log10(np.mean(FBoct3 ** 2, axis=1)) + ParamOct3["NrmlzPwrdB"]

    return FBoct3, FBoct3DlyCmp, PwrdB, ParamOct3
