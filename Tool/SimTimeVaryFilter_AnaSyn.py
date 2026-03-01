import numpy as np
import math

def SimTimeVaryFilter_AnaSyn(Snd, WinFrame=None, TVFparam=None):
    """

    Args:
        Snd (1D array): 输入声音（row-vector in matlab）
        WinFrame (2D array or None): (LenWin x LenFrame) 窗口帧（用于合成时提供）
        TVFparam (dict): 参数，必须包含或会被设置的字段：
            - Ctrl : 'ana' 或 'syn'（若 WinFrame 未提供，则默认 'ana'）
            - fs (Hz)
            - Twin (sec)
            - Tshift (sec)
            - NameWin : 'hanning' 或 'full-ones'
            - SwWinAnaSyn : 'sqrt-sqrt' 等（保留，不直接影响实现）
            结果函数会在 TVFparam 中写入许多字段（LenWin, LenShift, WinAna, WinSyn, LenFrame ...）

    Returns:
        SndMod: 若 Ctrl == 'syn' 则返回合成后的信号（1D numpy array），若 Ctrl == 'ana' 返回 None
        WinFrame: 若 Ctrl == 'ana' 返回计算得到的 WinFrame (LenWin x LenFrame numpy array)
        TVFparam: 更新后的参数字典
    """

    if TVFparam is None:
        raise ValueError("TVFparam must be provided (dict).")


    if WinFrame is None or (isinstance(WinFrame, (list, np.ndarray)) and len(WinFrame) == 0):
        TVFparam.setdefault('Ctrl', 'ana')

    if 'Ctrl' not in TVFparam:
        raise ValueError("Specify TVFparam.Ctrl = 'ana' or 'syn'")

    ctrl = TVFparam['Ctrl']


    if str(ctrl).lower().startswith('ana'):

        TVFparam.setdefault('fs', 48000)
        fs = TVFparam['fs']

        if 'Twin' not in TVFparam:
            TVFparam['Twin'] = 0.020
        if 'Tshift' not in TVFparam:
            TVFparam['Tshift'] = TVFparam['Twin'] / 2.0
        if 'NameWin' not in TVFparam:
            TVFparam['NameWin'] = 'hanning'
        if 'SwWinAnaSyn' not in TVFparam:
            TVFparam['SwWinAnaSyn'] = 'sqrt-sqrt'


        if not math.isclose(TVFparam['Tshift'], TVFparam['Twin'] / 2.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('Tshift should be Twin/2.')


        LenShift = int(round(TVFparam['Tshift'] * fs))
        LenWin = int(round(TVFparam['Twin'] * fs))

        if LenShift <= 0 or LenWin <= 0:
            raise ValueError('Computed LenShift/LenWin must be positive integers.')

        namewin = TVFparam['NameWin']
        if namewin == 'hanning':

            WinAna = np.sqrt(np.hanning(LenWin))
            WinSyn = WinAna.copy()
        elif namewin == 'full-ones':
            WinAna = np.hanning(LenWin)
            WinSyn = np.ones(LenWin)
        else:
            raise ValueError("TVFparam.NameWin should be 'hanning' or 'full-ones'")

        LenSnd = int(len(Snd))

        LenFrame = int(math.ceil(LenSnd / LenShift)) + 1

        ZpadPre = np.zeros(LenShift, dtype=float)
        ZpadPost_len = int(LenFrame * LenShift - LenSnd)
        if ZpadPost_len < 0:
            ZpadPost_len = 0
        ZpadPost = np.zeros(ZpadPost_len, dtype=float)

        SndZp = np.concatenate([ZpadPre, np.asarray(Snd).flatten(), ZpadPost])


        WinFrame_out = np.zeros((LenWin, LenFrame), dtype=float)
        NsmplSnd = np.zeros(LenFrame, dtype=int)

        for nf in range(1, LenFrame + 1):
            start = (nf - 1) * LenShift
            nRange = np.arange(start, start + LenWin)

            WinFrame_out[:, nf - 1] = SndZp[nRange] * WinAna

            NsmplSnd[nf - 1] = (nf - 1) * LenShift + 1


        TVFparam['Nshift'] = LenShift
        TVFparam['LenShift'] = LenShift
        TVFparam['N'] = LenShift
        TVFparam['LenWin'] = LenWin
        TVFparam['WinAna'] = WinAna
        TVFparam['WinSyn'] = WinSyn
        TVFparam['LenFrame'] = LenFrame
        TVFparam['NumFrame'] = LenFrame
        TVFparam['LenSnd'] = LenSnd
        TVFparam['LenSndZp'] = len(SndZp)
        TVFparam['NsmplSnd'] = NsmplSnd.tolist()
        TVFparam['ZpadPre'] = ZpadPre
        TVFparam['ZpadPost'] = ZpadPost


        return None, WinFrame_out, TVFparam


    elif str(ctrl).lower().startswith('syn'):
        # restore params
        if 'LenWin' not in TVFparam:
            raise ValueError('TVFparam missing LenWin for synthesis')

        LenWin = int(TVFparam['LenWin'])

        if 'LenShift' in TVFparam:
            LenShift = int(TVFparam['LenShift'])
        elif 'N' in TVFparam:
            LenShift = int(TVFparam['N'])
        else:
            raise ValueError('TVFparam missing LenShift / N for synthesis')

        if 'LenFrame' in TVFparam:
            LenFrame = int(TVFparam['LenFrame'])
        elif 'NumFrame' in TVFparam:
            LenFrame = int(TVFparam['NumFrame'])
        else:
            raise ValueError('TVFparam missing LenFrame / NumFrame for synthesis')

        if 'LenSnd' not in TVFparam or 'LenSndZp' not in TVFparam:
            raise ValueError('TVFparam missing LenSnd / LenSndZp for synthesis')

        LenSnd = int(TVFparam['LenSnd'])
        LenSndZp = int(TVFparam['LenSndZp'])


        WinFrame_arr = np.asarray(WinFrame)
        if WinFrame_arr.ndim != 2:
            raise ValueError('WinFrame should be 2D array (LenWin x LenFrame)')
        mm, nn = WinFrame_arr.shape
        if mm != LenWin or nn != LenFrame:
            raise ValueError('Check WinFrame dimensions: expected LenWin x LenFrame')

        SndSyn = np.zeros(LenSndZp, dtype=float)
        WinSyn = np.asarray(TVFparam.get('WinSyn', np.ones(LenWin))).flatten()

        for nf in range(1, LenFrame + 1):
            start = (nf - 1) * LenShift
            nRange = np.arange(start, start + LenWin)
            SndSyn[nRange] += WinFrame_arr[:, nf - 1] * WinSyn

        start_extract = LenShift
        end_extract = LenShift + LenSnd
        SndMod = SndSyn[start_extract:end_extract].copy()

        return SndMod, WinFrame_arr, TVFparam

    else:
        raise ValueError("Specify TVFparam.Ctrl = 'ana' or 'syn'")

# end of function
