# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np

def SetFrame4TimeSequence(Snd, LenWin, LenShift=None):
    # 默认 LenShift = LenWin/2
    if LenShift is None:
        LenShift = LenWin // 2

    IntDivFrame = LenWin / LenShift

    # 条件检查：LenWin 必须是偶数，LenWin / LenShift 必须是整数
    if (IntDivFrame % 1 != 0) or (LenWin % 2 != 0):
        print(f"LenWin = {LenWin}, LenShift = {LenShift}, Ratio = {IntDivFrame:.4f} <-- should be integer value")
        print("LenWin must be even number")
        raise ValueError("LenShift must be LenWin/Integer value")

    IntDivFrame = int(IntDivFrame)

    # 变成 1D 数组
    Snd = np.asarray(Snd).flatten()

    # === Zero padding: 前后补 LenWin/2 ===
    pad = LenWin // 2
    Snd1 = np.concatenate([np.zeros(pad), Snd, np.zeros(pad)])
    LenSnd1 = len(Snd1)

    # === 总共需要多少帧（非重叠帧）===
    NumFrame1 = int(np.ceil(LenSnd1 / LenWin))
    nlim = LenWin * NumFrame1

    # 末尾补零到正好 nlim
    Snd1 = np.concatenate([Snd1[:min(nlim, LenSnd1)], np.zeros(max(0, nlim - LenSnd1))])
    LenSnd1 = len(Snd1)

    # === 总帧数（包含重叠 shift）===
    NumFrameAll = (NumFrame1 - 1) * IntDivFrame + 1

    SndFrame = np.zeros((LenWin, NumFrameAll))
    NumSmplPnt = np.zeros(NumFrameAll, dtype=int)

    # 主循环：构造多种起点偏移的帧
    for nid in range(IntDivFrame):
        NumFrame2 = NumFrame1 - (1 if nid > 0 else 0)

        # 每次偏移 nid * LenShift
        nSnd = nid * LenShift + np.arange(NumFrame2 * LenWin)

        Snd2 = Snd1[nSnd]
        Mtrx = Snd2.reshape(LenWin, NumFrame2)

        # 对应到最终输出的位置
        num = np.arange(nid, NumFrameAll, IntDivFrame)
        SndFrame[:, num] = Mtrx

        # 帧中心 index（MATLAB: (num-1)*LenShift）
        NumSmplPnt[num] = (num) * LenShift

    # === 去除超出原始 Snd 长度的帧 ===
    valid = np.where(NumSmplPnt <= len(Snd))[0]

    SndFrame = SndFrame[:, valid]
    NumSmplPnt = NumSmplPnt[valid]

    return SndFrame, NumSmplPnt
