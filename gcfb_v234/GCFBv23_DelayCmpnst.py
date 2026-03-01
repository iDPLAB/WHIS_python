import numpy as np

def GCFBv23_DelayCmpnst(GCval, GCparam, DCparam):
    """
    GC filter delay compensation function.

    输入：
    GCval: 输入的 GC 信号，大小为 (NumCh, LenVal)
    GCparam: GCFB 的参数字典
    DCparam: 延迟补偿的参数字典，包含 fs、TdelayFilt1kHz 等
    
    输出：
    GCcmpnst: 补偿后的 GC 信号
    DCparam: 更新后的延迟补偿参数字典，包含每个通道的补偿长度
    """
    print("*** GC filter delay compensation ***")
    
    # 确保采样频率已指定
    if DCparam.fs == None:
        raise ValueError("Specify sampling frequency (fs) of input GCval.")
    
    # 延迟参数：设置默认值（通常不修改，但可以外部控制）
    if DCparam.TdelayFilt1kHz == None:
        DCparam.TdelayFilt1kHz = 0.002  # 默认值：2 ms @ 1 kHz
    if DCparam.TdelayFB == None:
        DCparam.TdelayFB = 0  # 默认 GCFB 延迟为 0 ms
    
    if DCparam.TdelayFilt1kHz < 0 or DCparam.TdelayFB < 0:
        raise ValueError("Negative delay compensation is not allowed.")
    
    NumCh, LenVal = GCval.shape
    GCcmpnst = np.zeros_like(GCval)
    
    for nch in range(NumCh):
        # 计算每个通道的补偿长度
        NumCmpnst = int((DCparam.TdelayFilt1kHz * 1000 / GCparam.fr1[nch] + DCparam.TdelayFB) * DCparam.fs)
        
        # 打印当前补偿进度
        if nch % 50 == 0 or nch == NumCh - 1 or nch == 0:
            print(f"Compensating delay: ch #{nch+1} / #{NumCh}. [Delay = {NumCmpnst / DCparam.fs * 1000:.2f} (ms)]")
        
        # 如果补偿长度大于信号长度，抛出错误
        if abs(NumCmpnst) > LenVal:
            raise ValueError("Sampling point for Compensation is greater than the signal length.")
        
        # 进行延迟补偿
        GCcmpnst[nch, :] = np.concatenate((GCval[nch, NumCmpnst:], np.zeros(NumCmpnst)))
        
        # 记录每个通道的补偿长度
        DCparam.NumCmpnst.append(NumCmpnst)

    return GCcmpnst, DCparam
