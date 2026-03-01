import numpy as np
from scipy.fft import fft, ifft

def pwr_spec_to_min_phase_filter(freq, power_spectrum, fs):
    # 检查输入长度是否匹配
    if len(freq) != len(power_spectrum):
        raise ValueError('Lengths of freq and powerSpectrum are different.')
    
    # 检查频率范围
    if freq[0] != 0 or freq[-1] != fs / 2:
        raise ValueError('freq[0] != 0 or freq[-1] != fs/2.  0 <= freq <= fs/2')
    
    # 检查频率间隔是否均匀
    if np.abs(np.mean(np.diff(freq)) - np.mean(np.diff(freq[:10]))) > 10 * np.finfo(float).eps:
        raise ValueError('Frequency spacing is not uniform. Uniformly sampled in 0 <= freq <= fs/2')
    
    # 构造双倍频谱
    double_spectrum = np.concatenate([power_spectrum, power_spectrum[-2:0:-1]])
    
    fftl = len(double_spectrum)
    
    # 计算凯普斯特姆
    cepstrum = np.fft.ifft(np.log(double_spectrum) / 2)
    
    # 计算最小相位滤波器
    filter_min_phs = np.real(np.fft.ifft(np.exp(np.fft.fft(np.concatenate([
        [cepstrum[0]], 
        2 * cepstrum[1:fftl//2], 
        np.zeros(fftl//2 + 1)])))))
    
    return filter_min_phs
