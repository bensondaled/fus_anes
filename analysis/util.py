import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, tf2zpk, zpk2sos, sosfiltfilt
from scipy.ndimage import median_filter

from multitaper import multitaper_spectrogram as mts

def filter_eeg(data, fs, lo=50, hi=0.5, notch=60):

    b, a = butter(2, [hi, lo], fs=fs, btype='bandpass', analog=False)
    b0, a0 = iirnotch(notch, 20.0, fs=fs)
    
    b = np.convolve(b, b0)
    a = np.convolve(a, a0)

    Z, P, K = tf2zpk(b, a)
    sos = zpk2sos(Z, P, K)

    out = sosfiltfilt(sos, data, axis=0)
    return out

def detect_switch(signal, threshold=1.0, min_duration=3, baseline_window=5000):
    baseline = median_filter(signal, size=baseline_window)
    elevated = signal > (baseline + threshold)

    changes = np.diff(elevated.astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # if still elevated at end of signal
    if len(ends) < len(starts):
        ends = np.append(ends, len(signal))

    # filter by duration
    switch_starts = [s for s, e in zip(starts, ends) if (e - s) >= min_duration]

    return np.array(switch_starts)

