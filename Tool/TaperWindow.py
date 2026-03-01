# Copyright (c)the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

import numpy as np
from scipy.signal import windows

def taper_window(len_win, type_taper, len_taper=None, range_sigma=3, sw_plot=0):
    if len_taper is None:
        len_taper = len_win // 2
    if range_sigma is None or (isinstance(range_sigma, (list, tuple, np.ndarray)) and len(range_sigma) == 0):
        range_sigma = 3
    if (len_taper * 2 + 1) >= len_win:
        len_taper = len_win // 2
    key = type_taper[:3].upper()
    if key == 'HAM':
        taper = windows.hamming(len_taper * 2 + 1, sym=True)
        type_taper_full = 'Hamming'
    elif key == 'HAN' or key == 'COS':
        taper = windows.hann(len_taper * 2 + 1, sym=True)
        type_taper_full = 'Hanning/Cosine'
    elif key == 'BLA':
        taper = windows.blackman(len_taper * 2 + 1, sym=True)
        type_taper_full = 'Blackman'
    elif key == 'GAU':
        nn = np.arange(-len_taper, len_taper + 1)
        if len_taper == 0:
            taper = np.array([1.0])
        else:
            taper = np.exp(-((range_sigma * nn / len_taper) ** 2) / 2.0)
        type_taper_full = 'Gauss'
    else:
        left = np.arange(1, len_taper + 1)
        right = left[::-1]
        center = len_taper + 1
        taper = np.concatenate([left, np.array([center]), right]) / float(len_taper + 1)
        type_taper_full = 'Line'
    len_taper = int(len_taper)
    left_part = taper[0:len_taper]
    right_part = taper[len_taper + 1:]
    middle = np.ones(len_win - len_taper * 2)
    taper_win = np.concatenate([left_part, middle, right_part])
    if sw_plot == 1:
        plt.plot(taper_win)
        plt.title(f'TypeTaper = {type_taper_full}')
        plt.xlabel('Points'); plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    return taper_win, type_taper_full