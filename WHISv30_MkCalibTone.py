# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from Tool.TaperWindow import taper_window

def WHISv30_MkCalibTone(whisparam):
    """
    使用用户的 taper_window 实现的 WHISv30_MkCalibTone.
    返回: CalibTone (numpy array), 更新后的 WHISparam
    """
    fs = whisparam.fs

    # 与 MATLAB 版本对应的默认参数
    Tsnd = 5.0
    Freq = 1000.0
    RMSDigitalLeveldB = -26.0
    Ttaper = 0.005

    # 把参数写入 WHISparam（保持结构化）
    whisparam.calibtone.Tsnd = Tsnd
    whisparam.calibtone.Freq = Freq
    whisparam.calibtone.RMSDigitalLeveldB = RMSDigitalLeveldB
    whisparam.calibtone.Ttaper = Ttaper

    # Source sound info
    whisparam.srcsnd.RMSDigitalLevelStrWeight = 'RMS'
    whisparam.srcsnd.RMSDigitalLeveldB = RMSDigitalLeveldB

    # 生成声音
    LenCalib = int(round(Tsnd * fs))
    LenTaper = int(round(Ttaper * fs))

    # AmpCalib = 10^(dB/20) * sqrt(2)
    AmpCalib = (10.0 ** (RMSDigitalLeveldB / 20.0)) * np.sqrt(2.0)

    # 使用你提供的 taper_window（type 用 'han' 与原 MATLAB 对应）
    TaperWin, type_taper_full = taper_window(LenCalib, 'han', len_taper=LenTaper, range_sigma=3, sw_plot=0)

    t = np.arange(LenCalib) / fs
    CalibTone = AmpCalib * TaperWin * np.sin(2.0 * np.pi * Freq * t)

    # 生成 Name 字符串，模仿 MATLAB int2str 行为
    freq_khz = Freq / 1000.0
    if float(freq_khz).is_integer():
        freq_khz_str = str(int(freq_khz))
    else:
        freq_khz_str = ("{:.3f}".format(freq_khz)).rstrip('0').rstrip('.')

    name = f"Snd_CalibTone_{freq_khz_str}kHz_RMS{int(RMSDigitalLeveldB)}dB"
    whisparam.calibtone.Name = name
    whisparam.calibtone.TaperTypeFull = type_taper_full

    return CalibTone, whisparam