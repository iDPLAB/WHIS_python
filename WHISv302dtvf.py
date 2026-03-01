import numpy as np
import time
from scipy.signal import resample
import matplotlib.pyplot as plt

from gcfb_v234.gcfb_v234 import gcfb_v23_asym_func_in_out as GCFBv23_AsymFuncInOut
from gcfb_v234.gcfb_v234 import gcfb_v23_asym_func_in_out_inv_io_func as GCFBv23_AsymFuncInOut_InvIOfunc
from gcfb_v234.GCFBv23_DelayCmpnst import GCFBv23_DelayCmpnst

from WHISv30_DirectTVF import WHISv30_DirectTVF
from Tool.OneThirdOctFB.OneThirdOctAnaSyn_LPenv import OneThirdOctAnaSyn_LPenv

import Param_Init as Param

def WHISv302dtvf(dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL,
                 SrcSnd, Snd4GCFB, whisparam, fs, Tsnd=None, t0=None):
    """
    Python 版 WHISv302 的 DTVF 分支
    输入:
      dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL :  GCFB 的结果
      SrcSnd : 原始归一化后的源信号 (1D numpy array)
      WHISparam : 类字典，会被更新
      fs : 采样率 (int)
      Tsnd : 信号时长(秒)，可选，用于日志计算
      t0 : 计时起点(由上层传入 time.time() )
    返回:
      SndOut, WHISparam
    """

    # 计时起点
    if t0 is None:
        t0 = time.time()

    NumCh, LenFrame = dcGCframeHL.shape

    GainReductdB_ACT = np.zeros((NumCh, LenFrame))

    for nch in range(NumCh):
        Fr1query = GCparamHL.fr1[nch] if isinstance(GCparamHL.fr1, (list, np.ndarray)) else GCparamHL.fr1[nch]
        CompressionHealthNH = 1.0
        CompressionHealthHL = np.array(GCparamHL.hloss.fb_compression_health)[nch]

        PindB_HL = GCrespHL.lvl_db_frame[nch, :]

        _, IOfuncdB_HL, _ = GCFBv23_AsymFuncInOut(GCparamHL, GCrespHL, Fr1query, CompressionHealthHL, PindB_HL)

        PindB_NH = GCFBv23_AsymFuncInOut_InvIOfunc(GCparamHL, GCrespHL, Fr1query, CompressionHealthNH, IOfuncdB_HL)

        GainReductdB_ACT[nch, :] = -(PindB_HL - PindB_NH)
        

    FB_PinLossdB_PAS = GCparamHL.hloss.fb_pin_loss_db_pas

    GainReductdB_PAS = -FB_PinLossdB_PAS * np.ones((NumCh, LenFrame))

    

    GainReductdB = GainReductdB_ACT + GainReductdB_PAS


    DCparam = Param.GCparam()

    DCparam.fs = GCparamHL.dyn_hpaf.fs
    GainReductdB_Dcmpnst, DCparam = GCFBv23_DelayCmpnst(GainReductdB, GCparamHL, DCparam)
    GainReductdB = GainReductdB_Dcmpnst


    
    # 绘图
    if whisparam.swplot == 1:
        plt.figure(10, figsize=(10, 8))
        nchAll = np.arange(1, NumCh + 1)
        GainRdB = np.mean(GainReductdB_ACT, axis=0)

        tFrame = np.arange(len(GainRdB)) / 2000.0
        plt.subplot(4, 1, 1)
        plt.plot(np.arange(len(SrcSnd)) / fs, SrcSnd * 100 + np.mean(GainRdB))
        plt.plot(tFrame, GainRdB)
        plt.title('SrcSnd & GainRdB')

        plt.subplot(4, 1, 2)
        plt.imshow(GainReductdB_ACT * (-1), aspect='auto', origin='lower')
        plt.title('GainReductdB_ACT (-1)')
        
        plt.subplot(4, 1, 3)
        plt.plot(nchAll, np.mean(GCrespHL.lvl_db_frame, axis=1))
        plt.title('Mean LvldBframe')

        plt.subplot(4, 1, 4)
        mean_gain = np.mean(GainReductdB, axis=1)
        mean_act = np.mean(GainReductdB_ACT, axis=1)
        mean_pas = np.mean(GainReductdB_PAS, axis=1)
        plt.plot(nchAll, mean_gain, label='total')
        plt.plot(nchAll, mean_act, '--', label='ACT')
        plt.plot(nchAll, mean_pas, '-.', label='PAS')
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    

    if whisparam.allow_down_sampling == 1:
        fsOrig = whisparam.fsorig
        rate = whisparam.rate_down_sampling
        print(f"Up-sampling for sound output: {whisparam.fs} --> {fsOrig} Hz")
        # 目标长度 = LenFrame * rate
        target_len = LenFrame * rate
        GainReductdBUp = np.zeros((NumCh, target_len))
        for nch in range(NumCh):
            GainReductdBUp[nch, :] = resample(GainReductdB[nch, :], target_len)
        GainReductdB = GainReductdBUp
        whisparam.fs = fsOrig

    whisparam.GCparamHL = GCparamHL
    whisparam.GainReductdB = GainReductdB
    
    SndHLoss, whisparam = WHISv30_DirectTVF(SrcSnd, whisparam)
    SndOut = SndHLoss


    if Tsnd is None:
        Tsnd = len(SrcSnd) / float(fs)
    elapsed = time.time() - t0
    print(f"Elapsed time is {elapsed:.4f} (sec) = {elapsed/Tsnd:.4f} times RealTime.")

    # Envelope modulation loss (Snd-level 处理)
    if whisparam.EMLoss != None and whisparam.EMLoss.LPFfc != None:
        print('----  Envelope modulation loss  ----')
        ParamOct3 = {}
        ParamOct3['LPFfc'] = whisparam['EMLoss']['LPFfc']
        ParamOct3['LPForder'] = whisparam['EMLoss'].get('LPForder', None)
        SndEMLoss, FBoct3Mod, ParamOct3 = OneThirdOctAnaSyn_LPenv(SndHLoss, ParamOct3)
        SndOut = SndEMLoss

    return SndOut, whisparam
