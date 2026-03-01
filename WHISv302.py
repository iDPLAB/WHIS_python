import numpy as np
import scipy.signal as signal
import time


from gcfb_v234.gcfb_v234 import gcfb_v234 as GCFBv234
from gcfb_v234.utils import eqlz2meddis_hc_level

from WHISv302dtvf import WHISv302dtvf
from WHISv302fbas import WHISv302fbas

def WHISv302(SrcSnd, whisparam):
    """
    args：
    SrcSnd: 输入音频（源音频）
    WHISparam: 类参数

    return：
    SndOut: 处理后的音频
    WHISparam: 更新后的参数
    """
    
    
    print('\n------------------ WHISv302 --------------------')

    # 获取采样率，默认 48000
    if whisparam.fs == None:
        whisparam.fs = 48000
    fs = whisparam.fs
    # 检查采样率是否为 48000 或 24000
    if fs not in [48000, 24000]:
        raise ValueError("Sampling rate should be 48000 or 24000. No 44100 Hz or other fs supported.")
    if whisparam.hloss == None or whisparam.hloss.type == None:
        raise ValueError("Specify HLoss.Type (e.g., WHISparam.HLoss.Type = 'HL3').")
    if whisparam.hloss.compression_health == None:
        raise ValueError("Specify HLoss.CompressionHealth (e.g., WHISparam.HLoss.CompressionHealth = 0.5).")
    if whisparam.calibtone == None or whisparam.calibtone.SPLdB == None:
        raise ValueError("Specify CalibTone.SPLdB (e.g., WHISparam.CalibTone.SPLdB = 80).")
    if whisparam.srcsnd == None or whisparam.srcsnd.SPLdB == None:
        raise ValueError("Specify SrcSnd.SPLdB (e.g., WHISparam.SrcSnd.SPLdB = 65).")

    if whisparam.swplot == None:
        whisparam.swplot = 0
    if whisparam.allow_down_sampling == None:
        whisparam.allow_down_sampling = 0
    
    # 处理采样率和下采样
    if whisparam.allow_down_sampling == 1:
        whisparam.rate_down_sampling = 2
        whisparam.fsorig = whisparam.fs
        whisparam.fs = whisparam.fsorig // whisparam.rate_down_sampling
        Snd4Ana = signal.resample(SrcSnd, int(len(SrcSnd) * whisparam.fs / whisparam.fsorig))
        print(f"Down-sampling for calculation: {whisparam.fsorig} --> {whisparam.fs} Hz")
    else:
        Snd4Ana = SrcSnd


    gcparam = whisparam.gcparam
    if gcparam.fs == None:
        gcparam.fs = fs
    if gcparam.fs != whisparam.fs:
        raise ValueError('GCparam.fs must be equal to WHISparam.fs')
    if gcparam.num_ch == None:
        gcparam.num_ch = 100
    if gcparam.f_range == None:
        gcparam.f_range = np.array([100, 12000]) if not whisparam.allow_down_sampling else np.array([100, 8000])
    if gcparam.out_mid_crct == None:
        gcparam.out_mid_crct = 'FreeField'

    gcparam.ctrl = 'dynamic'
    # GCparam['DynHPAF'] = {'StrPrc': 'frame-base'}
    gcparam.dyn_hpaf_str_prc = 'frame-base'
    

    Snd4GCFB, AmpdB = eqlz2meddis_hc_level(Snd4Ana, whisparam.srcsnd.SPLdB)
    whisparam.Eqlz2MeddisHCLevel_AmpdB = AmpdB
    start_time = time.time()

    gcparam.hloss = whisparam.hloss
    gcparam.hloss_type = whisparam.hloss_type
    gcparam.hloss_compression_health = whisparam.hloss.compression_health
    dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL = GCFBv234(Snd4GCFB, gcparam)

    whisparam.hloss = GCparamHL.hloss
    whisparam.hloss_type = gcparam.hloss_type


    if whisparam.synth_method == 'DTVF':
        StrSynthMethod = 'dtvf'
        SndOut, _ = WHISv302dtvf(dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL, SrcSnd, Snd4GCFB, whisparam, fs) 
    elif whisparam.synth_method == 'FBAnaSyn':
        StrSynthMethod = 'fbas'
        SndOut, _ = WHISv302fbas(dcGCframeHL, scGCsmplHL, GCparamHL, GCrespHL, SrcSnd, Snd4GCFB, whisparam, fs, gcparam, AmpdB)
    else:
        raise ValueError('Specify WHISparam.SynthMethod: "DTVF" or "FBAnaSyn".')


    whisparam.version = f"{__name__} {StrSynthMethod}"


    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds.")
    
    
    return SndOut, whisparam
