import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, tf2zpk, zpk2sos, sosfiltfilt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from mne.time_frequency import psd_array_multitaper

from multitaper import multitaper_spectrogram as mts, nanpow2db

def filter_eeg(data, fs, lo=50, hi=0.5, notch=60):
    data = np.nan_to_num(data)

    b, a = butter(2, [hi, lo], fs=fs, btype='bandpass', analog=False)
    b0, a0 = iirnotch(notch, 20.0, fs=fs)
    
    b = np.convolve(b, b0)
    a = np.convolve(a, a0)

    Z, P, K = tf2zpk(b, a)
    sos = zpk2sos(Z, P, K)

    out = sosfiltfilt(sos, data, axis=0)
    return out

def detect_switch(signal, threshold=1.0, min_duration=3, baseline_window=5000):
    #baseline = median_filter(signal, size=baseline_window)
    baseline = np.ones_like(signal) * np.percentile(signal, 1)
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

def fit_sigmoid(x, y, return_ec50=False, b0=0.5):
    def sigmoid(x, A, B, C, D):
        '''
        A: bottom plateau
        B: Hill slope
        C: EC50
        D: Top plateau
        '''
        return A + (D - A) / (1 + (x / C)**B)
    
    params, covariance = curve_fit(sigmoid,
                                   x,
                                   y,
                                   p0=(np.min(y), b0, np.median(y), np.max(y)),
                                   maxfev=100000,
                                   method='trf')
    xvals = np.linspace(np.min(x), np.max(x), 50)
    yvals = sigmoid(xvals, *params)
    if return_ec50:
        return xvals, yvals, params[2]
    return xvals, yvals

def mts_mne(eeg, window_size=30.0):
    sfreq = eeg.info['sfreq']

    fmin, fmax = 1, 40
    win_len = window_size # seconds per PSD window
    step = win_len   # seconds between windows
    bandwidth = 2.0  # multitaper bandwidth (Hz)

    nperseg = int(win_len * sfreq)
    nstep = int(step * sfreq)

    data = eeg.get_data()
    n_samples = data.shape[1]

    freqs = None
    psd_time = []
    times = []

    print(len(list(range(0,n_samples-nperseg+1,nstep))))

    for start in range(0, n_samples - nperseg + 1, nstep):
        seg = data[:, start:start+nperseg]
        psd, freqs = psd_array_multitaper(
            seg,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            bandwidth=bandwidth,
            adaptive=True,
            normalization='full',
            verbose=False
        )
        psd_time.append(psd)
        times.append(start / sfreq)

    psd = np.array(psd_time)  # shape: (time, n_channels, n_freqs)
    psd = np.transpose(psd, [1,2,0]) # chans x time x freq
    times = np.array(times)

    return psd, times, freqs
