import numpy as np
from math import ceil
from scipy.signal import lfilter
from scipy.interpolate import interp1d

from Tool.SimTimeVaryFilter_AnaSyn import SimTimeVaryFilter_AnaSyn
from Tool.SetFrame4TimeSequence import SetFrame4TimeSequence
from Tool.PwrSpec2MinPhaseFilter import pwr_spec_to_min_phase_filter as PwrSpec2MinPhaseFilter



def WHISv30_DirectTVF(SndIn, whisparam):
    """
    Python 版 WHISv30_DirectTVF
    输入:
      SndIn: 1D numpy array (输入信号)
      WHISparam: 参数字典，需包含 'GainReductdB' 和 'GCparamHL' 等
    输出:
      SndOut, WHISparam (更新后的)
      
      for key, value in vars(whisparam).items():
        print(f"{key}: {type(value)}")
    """
    

    
    SwPlot = whisparam.swplot
    GainReductdB = whisparam.GainReductdB  # shape: (NumCh, LenFrameGC)
    GCparam = whisparam.GCparamHL
    NumCh, LenFrameGC = GainReductdB.shape

    TVFparam = whisparam.TVFparam
    fs = whisparam.fs
    TVFparam['fs'] = fs
    TVFparam['Ctrl'] = 'ana'
    _, WinFrame, TVFparam = SimTimeVaryFilter_AnaSyn(SndIn, None, TVFparam)
    
    fs_gcframe = GCparam.dyn_hpaf.fs
    fs_shift = 1/TVFparam['Tshift']
    fsRatio = fs_gcframe / fs_shift


    TVF_len = TVFparam['LenFrame']
    GainReductdBTVF = np.zeros((NumCh, TVF_len))


    for nch in range(NumCh):
        seq = GainReductdB[nch, :]
        frame_len = int(round(fsRatio * 2))
        frame_shift = int(round(fsRatio))
        
        FrameMtrx, _ = SetFrame4TimeSequence(seq, frame_len, frame_shift)

        WinWeight = np.hanning(frame_len)

        WinWeight = WinWeight / (np.mean(WinWeight) * frame_len)

        FrameWin = WinWeight.reshape(1, -1) @ FrameMtrx
        FrameWin = FrameWin.ravel()

        nput = min(len(FrameWin), TVF_len)
        GainReductdBTVF[nch, :nput] = FrameWin[:nput]


    pwrAFG1 = 10.0 ** (GainReductdBTVF / 10.0)


    TVFparam['TresponseLength'] = 0.010
    TVFparam['Nfft'] = 1024
    freqBin = np.linspace(0.0, fs / 2.0, TVFparam['Nfft'] // 2 + 1)


    Fr1 = np.array(GCparam.fr1).flatten()

    low_max = np.min(Fr1) * 0.99
    if low_max > 0:
        Fr1ExtL = np.arange(0.0, low_max + 1e-9, 20.0)
    else:
        Fr1ExtL = np.array([0.0])

    if Fr1ExtL.size > 1:
        pwrExtL = (2.0 ** (Fr1ExtL / np.max(Fr1ExtL) - 1.0))[:, None] * pwrAFG1[0:1, :]
    else:
        pwrExtL = (2.0 ** (Fr1ExtL * 0.0 - 1.0))[:, None] * pwrAFG1[0:1, :]


    if (np.max(Fr1) + 500) < (fs / 2.0):
        Fr1ExtU = np.arange(np.max(Fr1) + 500.0, fs / 2.0 + 1e-9, 500.0)
        pwrExtU = (2.0 ** (-(Fr1ExtU / np.min(Fr1)) + 1.0))[:, None] * pwrAFG1[-1:, :]
    else:
        Fr1ExtU = np.array([], dtype=float)
        pwrExtU = np.zeros((0, pwrAFG1.shape[1]))

    FrAll = np.concatenate([Fr1ExtL, Fr1, Fr1ExtU])
    pwrAFG = np.vstack([pwrExtL, pwrAFG1, pwrExtU])
    frame_len_win = WinFrame.shape[0]
    WinFrameMod = np.zeros_like(WinFrame)

    for nf in range(TVFparam['LenFrame']):

        log_pwr = np.log(pwrAFG[:, nf] + 1e-300)  # 防止 log(0)
        interp_fn = interp1d(FrAll, log_pwr, kind='linear', fill_value='extrapolate', bounds_error=False)
        pwrSpec = np.exp(interp_fn(freqBin))
        AmpSpec = np.sqrt(np.maximum(pwrSpec, 0.0))


        FilterMinPhsFull = PwrSpec2MinPhaseFilter(freqBin, AmpSpec, fs)
        nhalf = int(ceil(TVFparam['TresponseLength'] * fs))
        if nhalf < 1:
            nhalf = 1
        FilterMinPhsHalf = FilterMinPhsFull[:nhalf]

        frame_vec = WinFrame[:, nf]
        tmpRsp = lfilter(FilterMinPhsHalf, [1.0], frame_vec)
        outRsp = lfilter(FilterMinPhsHalf, [1.0], tmpRsp)
        Lmin = min(len(outRsp), WinFrameMod.shape[0])
        WinFrameMod[:Lmin, nf] = outRsp[:Lmin]

        if nf == 0 or (nf + 1) % 50 == 0:
            print(f"Frame #{nf+1} / #{TVFparam['LenFrame']}")

    TVFparam['Ctrl'] = 'syn'
    SndOut, WinFrame_used, TVFparam = SimTimeVaryFilter_AnaSyn(None, WinFrameMod, TVFparam)

    whisparam.TVFparam = TVFparam
    
    return SndOut, whisparam



