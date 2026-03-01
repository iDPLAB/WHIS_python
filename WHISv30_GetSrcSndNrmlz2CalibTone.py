# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
from typing import Tuple, Any
import Param

def WHISv30_GetSrcSndNrmlz2CalibTone(SndLoad: np.ndarray,
                                     RecCalibTone: np.ndarray,
                                     whisparam) -> Tuple[np.ndarray, Any]:
    """
    转译自 Matlab: [SrcSnd, WHISparam] = WHISv30_GetSrcSndNrmlz2CalibTone(SndLoad,RecCalibTone,WHISparam)
    Parameters
    ----------
    SndLoad : 1D np.ndarray
        待缩放的信号（原始声音）
    RecCalibTone : 1D np.ndarray
        录音的校准音（含 taper），从中截取中间部分计算 RMS
    WHISparam : object
        需要包含 WHISparam.SrcSnd.SPLdB 和 WHISparam.CalibTone.SPLdB 之类字段
    Returns
    -------
    SrcSnd : np.ndarray
        经过放大/归一化后的声音
    WHISparam : 原传入对象（在其上写入或更新若干字段）
    """
    # ensure inputs are numpy arrays (float)
    SndLoad = np.asarray(SndLoad, dtype=float).ravel()
    RecCalibTone = np.asarray(RecCalibTone, dtype=float).ravel()

    # RMSlevel_SndLoad
    RMSlevel_SndLoad = np.sqrt(np.mean(SndLoad ** 2))

    # LenTruncate = 0.1 * WHISparam.fs
    fs = whisparam.fs
    if fs is None:
        # 有些实现把 fs 放在 WHISparam['WHISparam'] 或其他地方，常见是 WHISparam['fs']
        raise ValueError("WHISparam must contain sampling rate 'fs'")

    LenTruncate = int(round(0.1 * fs))

    n_total = len(RecCalibTone)
    start = LenTruncate
    end = n_total - LenTruncate  # python slice is exclusive on end

    if end <= start:
        raise ValueError("RecCalibTone is too short (after truncation length < 1)")

    rec_segment = RecCalibTone[start:end]
    RMSlevel_RecCalibTone = np.sqrt(np.mean(rec_segment ** 2))

    # AmpdB1 = (WHISparam.SrcSnd.SPLdB - WHISparam.CalibTone.SPLdB)
    SrcSnd_struct = whisparam.srcsnd
    CalibTone_struct = whisparam.calibtone

    SrcSnd_SPLdB = SrcSnd_struct.SPLdB
    CalibTone_SPLdB = CalibTone_struct.SPLdB
    if SrcSnd_SPLdB is None or CalibTone_SPLdB is None:
        raise ValueError("WHISparam must contain SrcSnd.SPLdB and CalibTone.SPLdB")

    AmpdB1 = float(SrcSnd_SPLdB) - float(CalibTone_SPLdB)
    Amp1 = 10.0 ** (AmpdB1 / 20.0) * (RMSlevel_RecCalibTone / (RMSlevel_SndLoad if RMSlevel_SndLoad != 0 else 1e-20))

    SrcSnd = Amp1 * SndLoad

    # 更新 WHISparam 中的字段（按 Matlab 名称）
    # WHISparam.SrcSnd.RMSDigitalLeveldB = 20*log10(sqrt(mean(SrcSnd.^2)));
    RMS_digital_level_srcsnd = 20.0 * np.log10(np.sqrt(np.mean(SrcSnd ** 2)) + 1e-20)  # 加 small eps 防 log10(0)

    # 确保 SrcSnd 子结构存在并写入字段
    SrcSnd_struct.RMSDigitalLeveldB = float(RMS_digital_level_srcsnd)
    SrcSnd_struct.StrNormalizeWeight = 'RMS'
    SrcSnd_struct.SndLoad_RMSDigitalLeveldB = float(20.0 * np.log10(RMSlevel_SndLoad + 1e-20))
    SrcSnd_struct.RecordedCalibTone_RMSDigitalLeveldB = float(20.0 * np.log10(RMSlevel_RecCalibTone + 1e-20))
    # 如果原 WHISparam['SrcSnd'] 为空，写回去
    whisparam.SrcSnd = SrcSnd_struct

    return SrcSnd, whisparam
