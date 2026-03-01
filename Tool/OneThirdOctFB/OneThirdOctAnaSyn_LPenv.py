# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.


import numpy as np
from scipy.signal import butter, lfilter, resample_poly, hilbert


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x**2))


def OneThirdOctAnaSyn_LPenv(Snd, ParamOct3=None):

    if ParamOct3 is None:
        ParamOct3 = {}


    ParamOct3.setdefault('fs', 48000)
    ParamOct3['fsEnv'] = 2000
    ParamOct3.setdefault('FreqRange', [100, 13000])
    ParamOct3.setdefault('StrNorm', 'Normlize2InputSnd')
    ParamOct3.setdefault('LPFfc', 16)
    ParamOct3.setdefault('LPForder', 2)


    bzLP, apLP = butter(ParamOct3['LPForder'], ParamOct3['LPFfc']/(ParamOct3['fsEnv']/2))
    print(f"### OneThirdOctAnaSyn_LPenv ###\n ---  Modification: Lowpass filter, fcMod = {ParamOct3['LPFfc']} (Hz) ---")


    FBoct3, FBoct3DlyCmp, PwrdB, ParamOct3 = OneThirdOctFB(Snd, ParamOct3)

    # Prepare output
    LenOct3, LenSnd = FBoct3DlyCmp.shape
    FBoct3Mod = np.zeros_like(FBoct3DlyCmp)

    fs = int(ParamOct3['fs'])
    fsEnv = int(ParamOct3['fsEnv'])

    for nf in range(LenOct3):

        Fout = FBoct3DlyCmp[nf, :]


        analytic = hilbert(Fout)
        FoutAmp = np.abs(analytic)
        FoutPhs = np.angle(analytic)


        ModEnv = resample_poly(FoutAmp, up=fsEnv, down=fs)

        ModEnvLP = lfilter(bzLP, apLP, ModEnv)

        FoutAmpMod1 = resample_poly(ModEnvLP, up=fs, down=fsEnv)

        LenMod = len(FoutAmpMod1)
        if LenMod >= LenSnd:
            FoutAmpMod = FoutAmpMod1[:LenSnd]
        else:
            FoutAmpMod = np.concatenate([FoutAmpMod1, np.zeros(LenSnd - LenMod)])


        FoutSyn = np.real(FoutAmpMod * np.exp(1j * FoutPhs))
        FBoct3Mod[nf, :] = FoutSyn


    SndSyn0 = np.mean(FBoct3Mod, axis=0)


    AmpNorm = ParamOct3.get('GainAnaSyn', None)
    if ParamOct3.get('StrNorm', '') == 'Normlize2InputSnd' or AmpNorm is None:
        AmpNorm = rms(Snd) / (rms(SndSyn0) + 1e-12)

    ParamOct3['AmpNorm'] = AmpNorm
    SndSyn = AmpNorm * SndSyn0

    return SndSyn, FBoct3Mod, ParamOct3
