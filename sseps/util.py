##
import pandas as pd
import numpy as np
import h5py
import json
import os
import pickle
import gzip
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.signal import butter, iirnotch, tf2zpk, zpk2sos, sosfiltfilt
from multitaper import multitaper_spectrogram as mts
from multitaper import nanpow2db

##
def norm(x):
    return (x-np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def filter_eeg(data, fs, lo=50, hi=0.5, notch=60):

    b, a = butter(2, [hi, lo], fs=fs, btype='bandpass', analog=False)
    b0, a0 = iirnotch(notch, 20.0, fs=fs)
    
    b = np.convolve(b, b0)
    a = np.convolve(a, a0)

    Z, P, K = tf2zpk(b, a)
    sos = zpk2sos(Z, P, K)

    out = sosfiltfilt(sos, data, axis=0)
    return out

def sliding_window(array, win_size, causal=True, axis=0):
    if causal:
        pad = [[0, 0] for i in range(len(array.shape))]
        pad[axis] = [win_size-1, 0]
        array = array.astype(float)
        array = np.pad(array, pad, mode='constant', constant_values=np.nan)
    return swv(array, win_size)

##
