# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

from WHISv30_MkCalibTone import WHISv30_MkCalibTone
from WHISv30_GetSrcSndNrmlz2CalibTone import WHISv30_GetSrcSndNrmlz2CalibTone
from WHISv302 import WHISv302
import copy

def WHISv30_Batch(SndLoad, WHISparam):
    whisparam = copy.deepcopy(WHISparam)
    """
    WHISv30 批处理执行函数
    
    args：
    SndLoad : 输入声音（音频）
    WHISparam : 类参数
    """
    
    # 检查输入音频是否为单声道
    mm, nn = SndLoad.shape if len(SndLoad.shape) > 1 else (1, len(SndLoad))  # 处理一维数组
    if min(mm, nn) > 1:
        raise ValueError("输入音频必须为单声道（行向量）。")
    
    SndLoad = SndLoad.flatten()  # 转换为行向量

    # 初始化参数
    whisparam.sw_gui_batch = 'Batch'
    whisparam.allow_down_sampling = 0  # 不进行下采样
    # 设置默认的合成方法
    if whisparam.synth_method == None:
        whisparam.synth_method = 'DTVF'  # 默认 DTVF 合成方法
    # 音压级校准
    print(f"CalibTone SPLdB = {whisparam.calibtone.SPLdB} (dB)")
    print(f"SrcSnd SPLdB = {whisparam.srcsnd.SPLdB} (dB)")

    # 创建校准音
    CalibTone, WHISparam = WHISv30_MkCalibTone(whisparam)
    RecordedCalibTone = CalibTone


    SrcSnd, WHISparam = WHISv30_GetSrcSndNrmlz2CalibTone(SndLoad, RecordedCalibTone, whisparam)

    
    SndWHIS, WHISparam = WHISv302(SrcSnd, whisparam)

    return SndWHIS, SrcSnd, RecordedCalibTone, whisparam
