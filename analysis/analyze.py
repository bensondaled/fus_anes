##
import numpy as np
import pandas as pd
import mne
import os
from util import mts, filter_eeg, detect_switch
from fus_anes.constants import MONTAGE

from threshs import switch_thresh, ssep_thresh

## Params
session_path = '/Users/bdd/data/fus_anes/2025-07-22_13-56-10_subject-b001.h5'
name = os.path.splitext(os.path.split(session_path)[-1])[0]
is_TUS = True

## Load
with pd.HDFStore(session_path, 'r') as h:
    eeg = h.eeg
    bl_eyes = h.bl_eyes
    markers = h.markers
    squeeze = h.squeeze
    oddball = h.oddball
    chirp = h.chirp

eeg_time = eeg.index.values
nominal_fs = 500.0
empiric_fs = len(eeg_time) / (eeg_time[-1] - eeg_time[0])
assert np.abs(empiric_fs-nominal_fs) < 0.1
fs = nominal_fs
eeg_time = np.linspace(eeg_time[0], eeg_time[-1], len(eeg_time))
def t2i(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t2i(x) for x in t])
    else:
        return np.argmin(np.abs(t - eeg_time))

channel_names = MONTAGE
eeg = eeg.values[:, :len(channel_names)]
eeg *= 1e-6
eeg_raw = eeg

## Format into MNE

ch_types = [{'ssep':'stim', 'gripswitch':'stim', 'ecg':'ecg'}.get(c, 'eeg') for c in channel_names]
info = mne.create_info(ch_names=channel_names,
                       sfreq=fs,
                       ch_types=ch_types)
eeg = mne.io.RawArray(eeg_raw.T, info)
eeg.set_montage(mne.channels.make_standard_montage('standard_1020'))

if is_TUS:
    eeg = eeg.drop_channels(['Oz', 'Fz']) # for TUS sessions

# filter
eeg.notch_filter(freqs=60.0, fir_design='firwin')
eeg.filter(l_freq=1., h_freq=50., fir_design='firwin')

# reference
#eeg.set_eeg_reference('average')
eeg.set_eeg_reference(['M1','M2'])



## ---- Analyses

## Squeeze
switch = filter_eeg(eeg_raw[:,[channel_names.index('gripswitch')]],
                    fs=fs,
                    lo=20,
                    hi=0.1,
                    notch=60)[:,0]

fig,ax = pl.subplots(figsize=(8,1.5), gridspec_kw=dict(bottom=0.2))
time = eeg_time - eeg_time[0]
ax.plot(time, switch, color='k')

sq_onset = squeeze[squeeze.event.str.endswith('mp3')].onset_ts.values
for t in sq_onset:
    ax.axvline(t-eeg_time[0], color='grey')
ax.axvline(t-eeg_time[0], color='grey', label='Squeeze command')

flip = detect_switch(np.abs(switch), switch_thresh[name])
for idx in flip:
    ax.axvline(time[idx], color='pink')
ax.axvline(time[idx], color='pink', label='Detected squeeze')

ax.legend()

# analyze response time
max_lag = 1.0 # secs
squeeze_times = np.array([eeg_time[idx] for idx in flip])
rts = []
for cmd_idx, cmd_t in enumerate(sq_onset):
    candidates = squeeze_times[squeeze_times > cmd_t]
    candidates = candidates[candidates - cmd_t <= max_lag]
    if cmd_idx < len(sq_onset)-1: # there was no next command in that time
        candidates = candidates[candidates <= sq_onset[cmd_idx+1]]
    rt = candidates[0] - cmd_t if len(candidates) else np.nan
    rts.append(float(rt) * 1000)
rts = np.array(rts)
pct_resp = 100.0 * np.mean(~np.isnan(rts))

fig, ax = pl.subplots()
ax.hist(rts[~np.isnan(rts)], bins=15, color='grey')
ax.set_title(f'% responses: {pct_resp:0.0f}%')
ax.set_xlabel('RT (ms)')


## Chirp
eeg.set_eeg_reference(['M1','M2'])

chirp_onset = t2i(chirp[chirp.event=='c'].onset_ts.values)
events = np.array([[int(idx), 0, 1] for idx in chirp_onset])
event_id = dict(chirp=1)

epochs = mne.Epochs(eeg,
                    events,
                    event_id=event_id,
                    tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0),
                    detrend=1,
                    preload=True)
epochs = epochs.pick('eeg')  # or just use all: comment this out

frequencies = np.linspace(25, 55, 30)
n_cycles = frequencies / 2.0  # higher freqs need more cycles

power, itc = epochs.compute_tfr(method='morlet',
                                freqs=frequencies,
                                n_cycles=n_cycles,
                                use_fft=True,
                                return_itc=True,
                                average=True,
                                decim=2,
                                )

power.plot(picks='FCz',
           baseline=(-0.1, 0),
           mode='logratio',
           title='Evoked power')

itc.plot(picks='FCz',
         title='Inter-trial Coherence')

# Average ITC during the 0–0.5s chirp period
itc_data = itc.copy().crop(tmin=0.0, tmax=0.5).data  # shape: (n_channels, n_freqs, n_times)
mean_itc = itc_data[0].mean(axis=1)  # average over time, channel 0

fig, ax = pl.subplots()
ax.plot(frequencies, mean_itc, marker='o')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Mean ITC (0–0.5 s)')
ax.set_title('Cortical tracking of chirp')


## Oddball
eeg.set_eeg_reference(['M1','M2'])
#eeg.set_eeg_reference('average')

events_standard = t2i(oddball[oddball.event=='s'].onset_ts.values)
events_deviant = t2i(oddball[oddball.event=='d'].onset_ts.values)

events = np.concatenate([
    np.column_stack((events_standard, np.zeros_like(events_standard), np.ones_like(events_standard))),
    np.column_stack((events_deviant, np.zeros_like(events_deviant), np.full_like(events_deviant, 2)))
]).astype(int)
event_id = {'standard': 1, 'deviant': 2}

epochs = mne.Epochs(eeg,
                    events,
                    event_id=event_id,
                    tmin=-0.2,
                    tmax=0.8,
                    baseline=(-0.2, 0),
                    detrend=1,
                    preload=True)
epochs = epochs.pick('eeg')

mean_standard = epochs['standard'].average()
mean_deviant = epochs['deviant'].average()
err_standard = epochs['standard'].standard_error()
err_deviant = epochs['deviant'].standard_error()

fig, ax = pl.subplots(figsize=(8, 5))
times = mean_standard.times * 1000

ch_name = 'F4'
ch_idx = mean_standard.ch_names.index(ch_name)
std = mean_standard.data[ch_idx] * 1e6
std_err = err_standard.data[ch_idx] * 1e6
dev = mean_deviant.data[ch_idx] * 1e6
dev_err = err_deviant.data[ch_idx] * 1e6
dif = dev - std
ax.plot(times, std, label='Standard', color='k')
ax.fill_between(times, std-std_err, std+std_err, alpha=0.5, lw=0, color='k')
ax.plot(times, dev, label='Deviant', color='red')
ax.fill_between(times, dev-dev_err, dev+dev_err, alpha=0.5, lw=0, color='red')
ax.plot(times, dif, label='Diff', ls='--', color='grey')

ax.axvline(0, color='grey', linestyle='--')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title(f'Auditory Oddball ERPs at {ch_name}')
ax.legend()
ax.grid(True)

# topo map
# Compute peak-to-peak amplitude per channel in a post-stimulus window (e.g., 20-60 ms)
tmin_pp, tmax_pp = 0.200, 0.300  # in seconds
evoked_s_crop = mean_standard.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
evoked_d_crop = mean_deviant.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
ptp_s_amplitudes = np.ptp(evoked_s_crop.data, axis=1) * 1e6  # Convert to µV
ptp_d_amplitudes = np.ptp(evoked_d_crop.data, axis=1) * 1e6  # Convert to µV
dif = ptp_d_amplitudes - ptp_s_amplitudes

# Plot the topomap of these amplitudes
fig_topo, ax_topo = pl.subplots(1, 1)
mne.viz.plot_topomap(ptp_d_amplitudes, mean_deviant.info, axes=ax_topo,
                     show=True, cmap='Reds', contours=0)



## SSEP
eeg.set_eeg_reference(['M1', 'M2'])

ssep = filter_eeg(eeg_raw[:,[channel_names.index('ssep')]],
                    fs=fs,
                    lo=fs/2-0.1,
                    hi=60,
                    notch=60)[:,0]

fig,ax = pl.subplots(figsize=(8,1.5), gridspec_kw=dict(bottom=0.2))
time = eeg_time - eeg_time[0]
ax.plot(time, ssep, color='k')

onset_idx = detect_switch(ssep, ssep_thresh[name], min_duration=1)
for idx in onset_idx:
    ax.axvline(time[idx], color='pink')
ax.axvline(time[idx], color='pink', label='Detected SSEP pulse')

events = np.column_stack((onset_idx, np.zeros_like(onset_idx), np.ones_like(onset_idx))).astype(int)

epochs = mne.Epochs(eeg,
                    events,
                    event_id=1,
                    tmin=-0.020,
                    tmax=0.100,
                    baseline=(-0.010, 0),
                    preload=True)
evoked = epochs.average()

fig, ax = pl.subplots(figsize=(8, 5))
times_ms = evoked.times * 1000

cols = ['slateblue', 'darkorange', 'grey']
cnames = ['C3', 'C4', 'Pz']
for ch_name, col in zip(cnames, cols):
    ch_idx = evoked.ch_names.index(ch_name)
    ax.plot(times_ms, evoked.data[ch_idx] * 1e6, label=ch_name, color=col, lw=2)

ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title('Median nerve SSEP')
ax.legend()
ax.grid(True)

tmin_pp, tmax_pp = 0.010, 0.050  # in seconds
evoked_crop = evoked.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
ptp_amplitudes = np.ptp(evoked_crop.data, axis=1) * 1e6

# Plot the topomap of these amplitudes
fig_topo, ax_topo = pl.subplots(1, 1)
mne.viz.plot_topomap(ptp_amplitudes, evoked.info, axes=ax_topo,
                     show=True, cmap='Reds', contours=0)
ax_topo.set_title(f'Peak-to-peak SSEP (µV) {int(tmin_pp*1000)}–{int(tmax_pp*1000)} ms')
##
