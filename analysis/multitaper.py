import numpy as np
from scipy.signal.windows import dpss

def nanpow2db(y):
    if isinstance(y, (int, float)):
        return np.nan if y==0 else 10*np.log10(y)
    else:
        y = np.asarray(y).astype(float)
        y[y == 0] = np.nan
        return 10*np.log10(y)

def multitaper_spectrogram(data, fs,
                           window_size = 5, # sec
                           window_step = 2.5, # sec
                           df = 1, #Hz
                           ):
    '''see PMID 27927806, pg 72'''

    frequency_range = [0, fs/2]
    time_bandwidth = 0.5 * window_size * df
    num_tapers = int(np.floor(2 * time_bandwidth) - 1) # optimal
    winsize_samples = int(np.rint(window_size * fs))
    winstep_samples = np.rint(window_step * fs)
    window_start = np.arange(0, len(data) - winsize_samples + 1, winstep_samples)
    stimes = (window_start + round(winsize_samples / 2)) / fs

    window_idxs = (np.atleast_2d(window_start).T + np.arange(0, winsize_samples, 1)).astype(int)
    data_segments = data[window_idxs]

    sfreqs = np.fft.rfftfreq(winsize_samples, d=1/fs)
    dpss_tapers = dpss(winsize_samples, time_bandwidth, num_tapers)

    def calc_mts_segment(data_segment, dpss_tapers, num_tapers):

        if np.all(data_segment == 0): return np.zeros_like(sfreqs)
        if np.any(np.isnan(data_segment)): return np.zeros_like(sfreqs) * np.nan

        tapered_data = data_segment.reshape([-1, 1]) * dpss_tapers.T
        fft_data = np.fft.rfft(tapered_data, axis=0)
        spower = np.abs(fft_data)**2
        spec = np.mean(spower, axis=1)
        return spec

    mts_params = (dpss_tapers, num_tapers)
    mt_spectrogram = np.apply_along_axis(calc_mts_segment, 1, data_segments, *mts_params)
    mt_spectrogram = mt_spectrogram.T # freq x windows

    # Preserve total energy while collapsing into 1d
    mult = np.ones([len(sfreqs), 1])
    mult[((sfreqs!=0) & (sfreqs!=fs/2)), :] = 2
    mt_spectrogram = mt_spectrogram * mult / fs # why the /fs?

    # mt_spect is freq x time if input was 1d
    # and chans x freq x time if input was 2d (where input should be time x channels)

    return mt_spectrogram, stimes, sfreqs


