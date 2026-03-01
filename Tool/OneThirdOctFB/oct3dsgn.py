import numpy as np
from scipy.signal import butter

def oct3dsgn(Fc, Fs, N=3):

    if Fc <= 0 or Fs <= 0:
        raise ValueError("Fc and Fs must be positive numbers.")
    if Fc > 0.88 * (Fs / 2):
        raise ValueError("Design not possible. Check frequencies (Fc too high).")

    f1 = Fc / (2 ** (1 / 6))
    f2 = Fc * (2 ** (1 / 6))

    Qr = Fc / (f2 - f1)
    Qd = (np.pi / 2 / N) / np.sin(np.pi / 2 / N) * Qr
    alpha = (1 + np.sqrt(1 + 4 * Qd ** 2)) / (2 * Qd)

    W1 = Fc / (Fs / 2) / alpha
    W2 = Fc / (Fs / 2) * alpha

    if W1 <= 0 or W2 >= 1:
        raise ValueError(f"Invalid normalized band edges: W1={W1:.4f}, W2={W2:.4f}")

    B, A = butter(N, [W1, W2], btype="bandpass")

    return B, A
