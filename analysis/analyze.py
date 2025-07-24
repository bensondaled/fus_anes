##
import numpy as np
import pandas as pd
import mne
import os
from util import mts, filter_eeg, detect_switch
from fus_anes.constants import MONTAGE

from threshs import switch_thresh, ssep_thresh

## Params
session_path = '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5'
name = os.path.splitext(os.path.split(session_path)[-1])[0]
is_TUS = False

## Load
with pd.HDFStore(session_path, 'r') as h:
    eeg = h.eeg
    bl_eyes = h.bl_eyes
    markers = h.markers
    squeeze = h.squeeze
    oddball = h.oddball
    chirp = h.chirp
    tci_cmd = h.tci_cmd
    print(h.keys())

true_eeg_time = eeg.index.values
nominal_fs = 500.0
empiric_fs = len(true_eeg_time) / (true_eeg_time[-1] - true_eeg_time[0])
assert np.abs(empiric_fs-nominal_fs) < 0.1
fs = nominal_fs
def t2i(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t2i(x) for x in t])
    else:
        return np.argmin(np.abs(t - true_eeg_time))
#eeg_time = np.linspace(eeg_time[0], eeg_time[-1], len(eeg_time))
slope, intercept = np.polyfit(np.arange(len(true_eeg_time)), true_eeg_time, deg=1)
eeg_time = slope * np.arange(len(true_eeg_time)) + intercept

channel_names = MONTAGE
eeg = eeg.values[:, :len(channel_names)]
eeg *= 1e-6
eeg_raw = eeg

## Label propofol levels
goto = tci_cmd[tci_cmd.kind == 'goto']
target = goto.ce_target
def t_to_phase_idx(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t_to_phase_idx(x) for x in t])
    else:
        edges = target.index.values
        return np.searchsorted(edges, t)

phase_starts = np.append(eeg_time[0], target.index.values)
phase_levels = np.append(0, target.values)

def t_to_phase_level(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t_to_phase_level(x) for x in t])
    else:
        idx = t_to_phase_idx(t)
        return phase_levels[idx]

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

## Spectrogram
frontal = np.isin(channel_names, ['F3', 'Fz', 'FCz', 'F4'])
posterior = np.isin(channel_names, ['P7', 'P8', 'Oz', 'P3', 'P4'])
e_f = eeg._data[frontal]
e_p = eeg._data[posterior]
sp_frontal, sp_t, sp_f = mts(e_f.T, fs=fs, window_size=10.0, window_step=10.0)
sp_posterior, sp_t, sp_f  = mts(e_p.T, fs=fs, window_size=10.0, window_step=10.0)
f_keep = sp_f < 40
sp_f = sp_f[f_keep]

fig, axs = pl.subplots(2, 1, sharex=True)
for ax, sp in zip(axs, [sp_frontal, sp_posterior]):
    sp = np.median(sp, axis=0)
    sp = sp[f_keep]

    vmin, vmax = np.percentile(sp, [1, 95])
    ax.pcolormesh(sp_t, sp_f, sp,
                 vmin=vmin, vmax=vmax,
                 cmap=pl.cm.rainbow)

ax = axs[0]
for ps, plev in zip(phase_starts, phase_levels):
    ax.axvline(ps-eeg_time[0], color='white', ls=':')
    ax.text(ps-eeg_time[0], 42, f'to {plev:0.1f}', clip_on=False,
            ha='center', va='center')


## Chirp
eeg.set_eeg_reference('average')

chirp_onset_t = chirp[chirp.event=='c'].onset_ts.values
level_id = t_to_phase_idx(chirp_onset_t)
chirp_onset = t2i(chirp_onset_t)
events = np.array([[int(idx), 0, lid] for idx, lid in zip(chirp_onset, level_id)])
event_id = {f'{phase_levels[x]:0.1f}':x for x in np.unique(level_id)}

epochs = mne.Epochs(eeg,
                    events,
                    event_id=event_id,
                    tmin=-0.2, tmax=1.5,
                    baseline=(-0.2, 0),
                    detrend=1,
                    preload=True)
epochs = epochs.pick('eeg')  # or just use all: comment this out

frequencies = np.linspace(25, 55, 30)
n_cycles = frequencies / 2.0  # higher freqs need more cycles

n_levels = len(event_id)
fig, axs = pl.subplots(1, n_levels, figsize=(15,4),
                       sharex=True, sharey=True,
                       gridspec_kw=dict(left=0.05, right=0.98))

#eids = sorted(list(event_id.keys()))
eids = list(event_id.keys())
summary = []
for eid, ax in zip(eids, axs):
    epochs_cond = epochs[eid]
    power, itc = epochs_cond.compute_tfr(method='morlet',
                                        freqs=frequencies,
                                        n_cycles=n_cycles,
                                        use_fft=True,
                                        return_itc=True,
                                        average=True,
                                        decim=2,
                                        )

    #power.plot(picks='Fz',
    #           baseline=(-0.1, 0),
    #           mode='logratio',
    #           title='Evoked power')

    itc.plot(picks='Fz',
             axes=ax,
             vlim=(0.0, 0.8),
             cnorm=None,
             colorbar=False)
    ax.set_title(f'{eid}', fontsize=9)
    if ax is not axs[0]:
        ax.set_ylabel('')

    itc_data = itc.copy().crop(tmin=0.0, tmax=0.5).data  # shape: (n_channels, n_freqs, n_times)
    mean_itc = itc_data[0].mean(axis=1)  # average over time, channel 0
    summary.append([float(eid), mean_itc])

fig, ax = pl.subplots()
summ = np.array([(xval, np.mean(yval)) for xval, yval in summary])
ax.scatter(*summ.T, color='grey', s=150, marker='o')
ax.set_xlabel('Propofol level')
ax.set_ylabel('Mean chirp responses (ITC)')

## Oddball
#eeg.set_eeg_reference(['M1','M2'])
eeg.set_eeg_reference('average')

ob_events_t = oddball[oddball.event.isin(['s','d'])].onset_ts.values
s_d = oddball[oddball.event.isin(['s','d'])].event.values
level_id = t_to_phase_idx(ob_events_t)
ob_onset = t2i(ob_events_t)

n_levels = len(np.unique(level_id))
fig, axs = pl.subplots(3, 3, figsize=(15,4),
                       sharex=True, sharey=True,
                       gridspec_kw=dict(left=0.05, right=0.98))
axs = axs.ravel()

summary = []
for lev,ax in zip(np.unique(level_id), axs):
    events_s = ob_onset[(level_id == lev) & (s_d == 's')]
    events_d = ob_onset[(level_id == lev) & (s_d == 'd')]

    events = np.concatenate([
        np.column_stack((events_s, np.zeros_like(events_s), np.ones_like(events_s))),
        np.column_stack((events_d, np.zeros_like(events_d), np.full_like(events_d, 2)))
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

    times = mean_standard.times * 1000

    ch_name = 'Fz'
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
    ax.set_title(f'{ch_name} at {phase_levels[lev]:0.1f}', fontsize=8)
    ax.grid(True)

    # Compute peak-to-peak amplitude per channel in a post-stimulus window (e.g., 20-60 ms)
    tmin_pp, tmax_pp = 0.100, 0.200  # in seconds
    evoked_s_crop = mean_standard.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    evoked_d_crop = mean_deviant.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    #ptp_s_amplitudes = np.ptp(evoked_s_crop.data, axis=1) * 1e6  # Convert to µV
    #ptp_d_amplitudes = np.ptp(evoked_d_crop.data, axis=1) * 1e6  # Convert to µV
    ch_idx = eeg.ch_names.index('Fz')
    s_trace = evoked_s_crop.data[ch_idx] * 1e6
    d_trace = evoked_d_crop.data[ch_idx] * 1e6
    dif = d_trace - s_trace
    dif = np.min(dif)

    # Plot the topomap of these amplitudes
    #fig_topo, ax_topo = pl.subplots(1, 1)
    #mne.viz.plot_topomap(ptp_d_amplitudes, mean_deviant.info, axes=ax_topo,
    #                     show=True, cmap='Reds', contours=0)

    summary.append([phase_levels[lev], dif])
ax.legend()

fig, ax = pl.subplots()
summ = np.array([(xval, np.min(yval)) for xval, yval in summary])
ax.scatter(*summ.T, color='grey', s=150, marker='o')
ax.set_xlabel('Propofol level')
ax.set_ylabel('Mean oddball mmn')


## SSEP
#eeg.set_eeg_reference(['M1', 'M2'])
eeg.set_eeg_reference('average')

ssep = filter_eeg(eeg_raw[:,[channel_names.index('ssep')]],
                    fs=fs,
                    lo=fs/2-0.1,
                    hi=60,
                    notch=60)[:,0]

fig,ax = pl.subplots(figsize=(8,1.5), gridspec_kw=dict(bottom=0.2))
time = eeg_time - eeg_time[0]
ax.plot(time, ssep, color='k')

onset_idx = detect_switch(ssep, ssep_thresh[name], min_duration=1)
onset_t = np.array([eeg_time[i] for i in onset_idx])
for idx in onset_idx:
    ax.axvline(time[idx], color='pink')
ax.axvline(time[idx], color='pink', label='Detected SSEP pulse')

level_id = t_to_phase_idx(onset_t)

n_levels = len(np.unique(level_id))
fig, axs = pl.subplots(1, n_levels, figsize=(15,4),
                       sharex=True, sharey=True,
                       gridspec_kw=dict(left=0.05, right=0.98))

summary = []

for lev,ax in zip(np.unique(level_id), axs):
    onset = onset_idx[level_id == lev]
    events = np.column_stack((onset, np.zeros_like(onset), np.ones_like(onset))).astype(int)

    epochs = mne.Epochs(eeg,
                        events,
                        event_id=1,
                        tmin=-0.020,
                        tmax=0.100,
                        baseline=(-0.005, 0.010),
                        preload=True)
    evoked = epochs.average()
    times_ms = evoked.times * 1000

    cols = ['slateblue', 'darkorange', 'grey']
    cnames = ['C3', 'C4', 'Pz']
    for ch_name, col in zip(cnames, cols):
        ch_idx = evoked.ch_names.index(ch_name)
        ax.plot(times_ms, evoked.data[ch_idx] * 1e6, label=ch_name, color=col, lw=2)

    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'SSEP at {phase_levels[lev]:0.1f}', fontsize=8)
    ax.grid(True)

    evoked_zero = evoked.copy().crop(tmin=0, tmax=0)
    tmin_pp, tmax_pp = 0.005, 0.100  # in seconds
    evoked_crop = evoked.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    ch_idx = evoked.ch_names.index('C3')
    trace = evoked_crop.data[ch_idx] * 1e6
    zero = evoked_zero.data[ch_idx].squeeze() * 1e6
    amplitude = np.abs(np.min(trace) - zero)
    latency = np.argmin(trace)

    summary.append([phase_levels[lev], amplitude, latency])

    # Plot the topomap of these amplitudes
    #fig_topo, ax_topo = pl.subplots(1, 1)
    #mne.viz.plot_topomap(ptp_amplitudes, evoked.info, axes=ax_topo,
    #                     show=True, cmap='Reds', contours=0)
    #ax_topo.set_title(f'Peak-to-peak SSEP (µV) {int(tmin_pp*1000)}–{int(tmax_pp*1000)} ms')

ax.legend()

summary = np.array(summary)
fig, axs = pl.subplots(1, 2)
axs[0].scatter(summary[:,0], summary[:,1], color='grey', s=150, marker='o') # amplitude
axs[1].scatter(summary[:,0], summary[:,2], color='grey', s=150, marker='o') # latency
axs[0].set_xlabel('Propofol level')
axs[1].set_xlabel('Propofol level')
axs[0].set_ylabel('Amplitude')
axs[1].set_ylabel('Latency')

##
