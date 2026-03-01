import numpy as np
import time
from scipy.signal import butter, lfilter, resample, hilbert
import matplotlib.pyplot as plt
import Param_Init as Param

from gcfb_v234.gcfb_v234 import gcfb_v23_asym_func_in_out as GCFBv23_AsymFuncInOut
from gcfb_v234.gcfb_v234 import gcfb_v23_asym_func_in_out_inv_io_func as GCFBv23_AsymFuncInOut_InvIOfunc
from gcfb_v234.GCFBv23_DelayCmpnst import GCFBv23_DelayCmpnst
from gcfb_v234.gcfb_v234 import gcfb_v23_synth_snd as GCFBv23_SynthSnd


def WHISv302fbas(dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL,
                 SrcSnd, Snd4GCFB, WHISparam, fs, GCparam, AmpdB, Tsnd=None):
    """


    args:
      - dcGCframeHL     : (NumCh, LenFrame) 直流/包络(?) frame（未直接使用但保留）
      - scGCsmplHL      : (NumCh, N_samples_per_channel) GCFB sample-level signals (complex or real)
      - GCparamHL       : 字典，包含 fs, DynHPAF.fs 等
      - GCrespHL        : 字典，包含 LvldBframe, pGCframe, scGCframe 等
      - GCparam         : 原始 GCparam（可能包含子字段）
      - WHISparam       : 类参数（会被更新）
      - AmpdB           : 从 Eqlz2MeddisHCLevel 返回的幅度校正数组（至少2个元素）
      - SrcSnd          : 源声音数组（行向量）
      - LenSnd          : 源声音的样本长度 (int)
      - Tsnd            : 声音时长（秒）
    返回:
      - SndOut : 合成后的音频（numpy 1D array）
      - WHISparam : 更新后的参数字典（包含 GCparamHL, GainReductdB 等）
    """
    LenSnd = len(SrcSnd) 
    t_start = time.time()

    apLP = 1
    bzLP = 1
    SwEnvModLoss = 0
    StrEMLoss = ''

    if WHISparam.EMLoss != None and WHISparam.EMLoss.LPFfc != None:
        SwEnvModLoss = 1
        lp_order = WHISparam.EMLoss.get('LPForder', 2)
        fc = WHISparam.EMLoss.LPFfc
        dyn_fs = GCparamHL.DynHPAF.fs
        Wn = fc / (dyn_fs / 2.0)
        bzLP, apLP = butter(lp_order, Wn)
        StrEMLoss = f'Envelop modulation LPF, fcMod = {int(fc)} (Hz)'
        print(StrEMLoss)

    NumCh, LenFrame = dcGCframeHL.shape
    scGCmod = np.zeros((NumCh, LenSnd), dtype=np.complex128)
    GainReductdB = np.zeros((NumCh, LenFrame))
    GainReductdB_ACT = np.zeros((NumCh, LenFrame))
    GainReductdB_PAS = np.zeros((NumCh, LenFrame))

    for nch in range(NumCh):
        Fr1query = GCparamHL.fr1[nch]  # 中心频率
        CompressionHealthNH = 1.0
        CompressionHealthHL = GCparamHL.hloss.fb_compression_health[nch]

        PindB_HL = GCrespHL.lvl_db_frame[nch, :]

        _, IOfuncdB_HL, _ = GCFBv23_AsymFuncInOut(GCparamHL, GCrespHL, Fr1query, CompressionHealthHL, PindB_HL)

        PindB_NH = GCFBv23_AsymFuncInOut_InvIOfunc(GCparamHL, GCrespHL, Fr1query, CompressionHealthNH, IOfuncdB_HL)


        GainReductdB_ACT[nch, :] = -(PindB_HL - PindB_NH)


        GainReductdB_PAS[nch, :] = -GCparamHL.hloss.fb_pin_loss_db_pas[nch] * np.ones(LenFrame)


        GainReductdB[nch, :] = GainReductdB_ACT[nch, :] + GainReductdB_PAS[nch, :]

        newfs = GCparamHL.fs
        oldfs = GCparamHL.dyn_hpaf.fs

        len_out = int(np.round(LenFrame * (newfs / oldfs)))
        if len_out < 1:
            len_out = 1

        GainReductdB_smpl = resample(GainReductdB[nch, :], num=len_out)

        LenGRR = len(GainReductdB_smpl)
        if LenGRR >= LenSnd:
            GainReductdB_smpl = GainReductdB_smpl[:LenSnd]
        else:
            GainReductdB_smpl = np.concatenate([GainReductdB_smpl, np.zeros(LenSnd - LenGRR)])


        GainReductRatio_smpl = 10.0 ** (GainReductdB_smpl / 20.0)

        scGC_smpl = GainReductRatio_smpl * scGCsmplHL[nch, :LenSnd]

        if SwEnvModLoss == 0:
            scGCmod[nch, :] = scGC_smpl
        else:
            # Envelope Modulation Loss 处理
            scGC_amp = np.abs(hilbert(scGC_smpl))
            scGC_phase = np.angle(hilbert(scGC_smpl))

            frame_fs = GCparamHL.DynHPAF.fs

            len_frame_samples = LenFrame
            len_frame_out = int(np.round(len(scGC_amp) * (frame_fs / newfs)))
            if len_frame_out < 1:
                len_frame_out = 1
            scGC_frame = resample(scGC_amp, num=len_frame_out)

            scGC_frame_mod = lfilter(bzLP, apLP, np.abs(scGC_frame))


            len_back = len(scGC_amp)
            scGC_amp_mod = resample(scGC_frame_mod, num=len_back)

            LenSFM = len(scGC_amp_mod)
            if LenSFM >= LenSnd:
                scGC_amp_mod = scGC_amp_mod[:LenSnd]
            else:
                scGC_amp_mod = np.concatenate([scGC_amp_mod, np.zeros(LenSnd - LenSFM)])

            scGCmod[nch, :] = np.real(scGC_amp_mod * np.exp(1j * scGC_phase))

    DCparam = Param.GCparam()

    DCparam.fs = GCparamHL.fs
    
    scGCmodDC, _ = GCFBv23_DelayCmpnst(scGCmod, GCparamHL, DCparam)

    SndHLoss = GCFBv23_SynthSnd(scGCmodDC, GCparamHL)

    SndOut = (10.0 ** (-AmpdB[1] / 20.0)) * SndHLoss

    if WHISparam.swplot == 1:
        plt.figure(10); plt.clf()
        nchAll = np.arange(1, 101)
        GainRdB = np.mean(GainReductdB_ACT, axis=0)
        tFrame = np.arange(len(GainRdB)) / 2000.0
        plt.subplot(4, 1, 1)
        plt.plot(np.arange(len(SrcSnd)) / GCparamHL.fs, SrcSnd * 100 + np.mean(GainRdB), tFrame, GainRdB)
        plt.subplot(4, 1, 2)
        plt.imshow(GainReductdB_ACT * (-1), aspect='auto', origin='lower')
        plt.subplot(4, 1, 3)
        plt.plot(np.arange(1, len(GCrespHL.lvl_db_frame[:, 0]) + 1), np.mean(GCrespHL.lvl_db_frame, axis=1))
        plt.subplot(4, 1, 4)
        mean_GR = np.mean(GainReductdB, axis=1)
        mean_ACT = np.mean(GainReductdB_ACT, axis=1)
        mean_PAS = np.mean(GainReductdB_PAS, axis=1)
        plt.plot(nchAll[:len(mean_GR)], mean_GR, nchAll[:len(mean_ACT)], mean_ACT, '--', nchAll[:len(mean_PAS)], mean_PAS, '-.')
        plt.show()

    if WHISparam.allow_down_sampling == 1:
        print(f"Up-sampling for sound output: {WHISparam.fs} --> {WHISparam.fsOrig} Hz")
        Rate = WHISparam.RateDownSampling
        GainReductdBUp = np.zeros((NumCh, LenFrame * Rate))
        for nch in range(NumCh):

            len_up = LenFrame * Rate
            GainReductdBUp[nch, :] = resample(GainReductdB[nch, :], num=len_up)
        GainReductdB = GainReductdBUp
        WHISparam.fs = WHISparam.fsOrig

    WHISparam.GCparamHL = GCparamHL
    WHISparam.GainReductdB = GainReductdB

    t_elapsed = time.time() - t_start

    if Tsnd is None:
        Tsnd = len(SrcSnd) / float(fs)
    
    print(f'Elapsed time is {t_elapsed:.4f} (sec) = {t_elapsed / Tsnd:.4f} times RealTime.')

    return SndOut, WHISparam
