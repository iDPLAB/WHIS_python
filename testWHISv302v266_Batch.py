import os
import numpy as np
import soundfile as sf
import subprocess
import matplotlib.pyplot as plt
from WHISv30_Batch import WHISv30_Batch
import Param_Init as Param


def main():
    current_dir = os.getcwd() + "/Sound"


    NameSrcSnd = 'Snd_Hello123'

    if not os.path.exists(f'{current_dir}/{NameSrcSnd}.wav'):
        str_cmd = f'cp -p -f "{current_dir}/{NameSrcSnd}.wav" {current_dir}'
        print(str_cmd)
        subprocess.run(str_cmd, shell=True)

    SndIn, fs = sf.read(f'{current_dir}/{NameSrcSnd}.wav')
    SndIn = SndIn.flatten()

    plt.plot(SndIn)
    plt.show()

    # 初始化 WHIS 参数
    whisparam = Param.WHISparam()
    whisparam.fs = fs
    whisparam.hloss.type = 'HL4'
    whisparam.hloss_type = 'HL4'
    whisparam.calibtone.SPLdB = 65
    whisparam.srcsnd.SPLdB = 65
    whisparam.gcparam.out_mid_crct = 'FreeField'
    whisparam.swplot = 1

    # WHIS 版本选择列表
    SwWHISversionList = [1, 2]

    StrEMLoss = ''
    if whisparam.EMLoss != None and whisparam.EMLoss.LPFfc != None:
        StrEMLoss = f'_EmLpf{int(whisparam.EMLoss.LPFfc)}'

    # 压缩健康度（compression health）列表
    CmprsHlthList = [1, 0.5, 0]

    for nCmprsHlth in range(len(CmprsHlthList)):
        whisparam.hloss.compression_health = CmprsHlthList[nCmprsHlth]
        StrCmprsHlth = f'_Cmprs{int(whisparam.hloss.compression_health * 100)}'

        for SwWHISversion in SwWHISversionList:

            if SwWHISversion == 1:
                StrWHIS = '_WHISv302dtvf'
                whisparam.synth_method = 'DTVF'

                SndWHIS, SrcSnd, RecCalibTone, WHISparam1 = WHISv30_Batch(
                    SndIn, whisparam
                )

            elif SwWHISversion == 2:
                StrWHIS = '_WHISv302fabs'
                whisparam.synth_method = 'FBAnaSyn'

                SndWHIS, SrcSnd, RecCalibTone, WHISparam1 = WHISv30_Batch(
                    SndIn, whisparam
                )

            NameSrcSnd1 = f'{current_dir}/{NameSrcSnd}{StrWHIS}_Src.wav'
            sf.write(NameSrcSnd1, SrcSnd, fs)
            print(NameSrcSnd1)

            NameSrcSnd_Rdct20 = f'{current_dir}/{NameSrcSnd}_Rdct-20dB.wav'
            SrcSndRdct20dB = SrcSnd * (10 ** (-20 / 20))
            sf.write(NameSrcSnd_Rdct20, SrcSndRdct20dB, fs)
            print(NameSrcSnd_Rdct20)

            NameSndWHIS1 = (
                f'{current_dir}/{NameSrcSnd}_'
                f'{whisparam.hloss.type}{StrCmprsHlth}{StrEMLoss}{StrWHIS}.wav'
            )
            sf.write(NameSndWHIS1, SndWHIS, fs)
            print(NameSndWHIS1)

            # 计算 RMS 比例（WHIS 输出相对于源信号）
            DiffRMS = (
                np.sqrt(np.mean(SndWHIS ** 2)) /
                np.sqrt(np.mean(SrcSnd ** 2))
            )

            RMSleveldB = 20 * np.log10([
                np.sqrt(np.mean(SrcSnd ** 2)),
                np.sqrt(np.mean(SrcSndRdct20dB ** 2)),
                np.sqrt(np.mean(SndWHIS ** 2)),
                DiffRMS
            ])

            DistortionRMS = 20 * np.log10(
                np.sqrt(
                    np.mean((SrcSnd - SndWHIS[:len(SrcSnd)]) ** 2)
                ) / np.sqrt(np.mean(SndWHIS ** 2))
            )

            StrCond = [f'{StrWHIS} {StrCmprsHlth}']

            # 打印结果
            print(f"RMS Level (dB): {RMSleveldB}")
            print(f"Distortion RMS (dB): {DistortionRMS}")
            print(f"Condition: {StrCond}")

    return 0


if __name__ == "__main__":
    main()
