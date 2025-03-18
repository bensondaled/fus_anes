import numpy as np
from scipy.signal import resample, butter, lfilter, iirnotch, tf2zpk, zpk2sos, sosfilt, sosfilt_zi
from scipy import convolve
from numpy.lib.stride_tricks import as_strided as ast

import fus_anes.config as config

def sliding_window(a, ws, ss=1, pad=True, pos='center', pad_kw=dict(mode='constant', constant_values=np.nan)):
    """Generate sliding window version of an array

    * note that padding and positioning are not implemented for step size (ss) other than 1

    Parameters
    ----------
    a (np.ndarray): input array
    ws (int): window size
    ss (int): step size
    pad (bool): maintain size of supplied array by padding resulting array
    pos (str): center / right / left, applies only if pad==True
    pad_kw (dict): kwargs for np.pad

    Returns
    -------
    Array in which iteration along the 0th dimension provides requested data windows

    """
    if pad:
        npad = ws-1
        if pos=='right':
            pad_size = (npad, 0)
        elif pos=='left':
            pad_size = (0, npad)
        elif pos=='center':
            np2 = npad/2.
            pad_size = (int(np.ceil(np2)), int(np.floor(np2))) # is there a more principled way to choose which end takes the bigger pad in the even window size scenario?
        pad_size = [pad_size] + [(0,0) for i in range(len(a.shape)-1)]
        a = np.pad(a, pad_size, **pad_kw)

    l = a.shape[0]
    n_slices = ((l - ws) // ss) + 1
    newshape = (n_slices,ws) + a.shape[1:] 
    newstrides = (a.strides[0]*ss,) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)

    return strided
            
class LiveFilter():
    def __init__(self):
        
        self.fs = config.fs
        self.lo = config.eeg_lopass
        self.hi = config.eeg_hipass
        self.notch = config.eeg_notch

        self.zi = None
        self.refresh_filter()

    def refresh_filter(self):
        # chose order 2 via buttord(55, 80, 3, 8, False, fs=500)
        b, a = butter(2, [self.hi, self.lo], fs=self.fs, btype='bandpass', analog=False)
        b0, a0 = iirnotch(self.notch, 20.0, fs=self.fs)
        
        b = convolve(b, b0)
        a = convolve(a, a0)

        Z, P, K = tf2zpk(b, a)
        self.sos = zpk2sos(Z, P, K)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)
            self.zi = np.repeat(self.zi[:, :, np.newaxis], config.n_channels, axis=2)

    def __call__(self, x):
        out, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return out

def nanpow2db(y):
    if isinstance(y, (int, float)):
        return np.nan if y==0 else 10*np.log10(y)
    else:
        y = np.asarray(y).astype(float)
        y[y == 0] = np.nan
        return 10*np.log10(y)

