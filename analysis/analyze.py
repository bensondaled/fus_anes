##
import numpy as np
import pandas as pd
import mne
import os
import json
from util import mts, filter_eeg, detect_switch, nanpow2db
from fus_anes.constants import MONTAGE

from threshs import switch_thresh, ssep_thresh

## Params
#session_path = '/Users/bdd/data/fus_anes/2025-07-25_08-38-29_subject-b003.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5'
session_path = '/Users/bdd/data/fus_anes/2025-07-29_08-07-02_subject-b004.h5'
name = os.path.splitext(os.path.split(session_path)[-1])[0]

## Load
with pd.HDFStore(session_path, 'r') as h:
    eeg = h.eeg
    bl_eyes = h.bl_eyes
    markers = h.markers
    squeeze = h.squeeze
    oddball = h.oddball
    chirp = h.chirp
    if 'tci_cmd' in h:
        tci_cmd = h.tci_cmd
    else:
        tci_cmd = None
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

def ch_name_to_idx(name):
    if isinstance(name, (str,)):
        return channel_names.index(name) 
    elif isinstance(name, (list, np.ndarray)):
        return [ch_name_to_idx(n) for n in name]

## Label propofol levels
if tci_cmd is not None:
    goto = tci_cmd[tci_cmd.kind == 'goto']
    _p_target = goto.ce_target
else:
    # for ultrasound sessions, mock label as prop 0 vs 1 for pre-post respectively
    us = markers[markers.text.str.contains('ultrasound')].t.iloc[0]
    _p_target = pd.Series([1], index=[us])

def t_to_phase_idx(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t_to_phase_idx(x) for x in t])
    else:
        edges = _p_target.index.values
        return np.searchsorted(edges, t)

phase_starts = np.append(eeg_time[0], _p_target.index.values)
phase_levels = np.append(0, _p_target.values)

prop_rising = np.arange(len(phase_levels)) <= np.argmax(phase_levels)
prop_falling = np.arange(len(phase_levels)) > np.argmax(phase_levels)
prop_direction = prop_rising - prop_falling.astype(int)

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
eeg = mne.io.RawArray(eeg_raw.T.copy(), info)
eeg.set_montage(mne.channels.make_standard_montage('standard_1020'))

# filter
#eeg.notch_filter(freqs=61.0, fir_design='firwin')
notch_freqs = np.arange(60, 241, 60)
notch_freqs = np.concatenate([notch_freqs, notch_freqs-1, notch_freqs+1])
eeg = eeg.notch_filter(freqs=notch_freqs, method='spectrum_fit', picks='eeg')
eeg = eeg.filter(l_freq=0.1, h_freq=58, fir_design='firwin', picks='eeg')

# reference
eeg.set_eeg_reference('average')
#eeg.set_eeg_reference(['M1','M2'])

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

press_idx = detect_switch(np.abs(switch), switch_thresh[name])
squeeze_times = np.array([eeg_time[i] for i in press_idx])
for idx in press_idx:
    ax.axvline(time[idx], color='pink')
ax.axvline(time[idx], color='pink', label='Detected squeeze')

ax.legend()

level_id = t_to_phase_idx(sq_onset)
fig, axs = pl.subplots(1, len(np.unique(level_id)))
summary = []
for lev, ax in zip(np.unique(level_id), axs):
    max_lag = 1.0 # secs
    rts = []
    sq_lev = sq_onset[level_id == lev]
    for cmd_idx, cmd_t in enumerate(sq_lev):
        candidates = squeeze_times[squeeze_times > cmd_t]
        candidates = candidates[candidates - cmd_t <= max_lag]
        if cmd_idx < len(sq_lev)-1: # there was no next command in that time
            candidates = candidates[candidates <= sq_lev[cmd_idx+1]]
        rt = candidates[0] - cmd_t if len(candidates) else np.nan
        rts.append(float(rt) * 1000)
    rts = np.array(rts)
    pct_resp = 100.0 * np.mean(~np.isnan(rts))

    ax.hist(rts[~np.isnan(rts)], bins=15, color='grey')
    ax.set_title(f'% resp: {pct_resp:0.0f}%, lev {phase_levels[lev]:0.2f}', fontsize=8)
    ax.set_xlabel('RT (ms)')

    summary.append([phase_levels[lev], np.nanmean(rts), pct_resp])

summary = np.array(summary)
fig, axs = pl.subplots(1, 2)
axs[0].scatter(summary[:,0], summary[:,1], c=prop_direction, cmap=pl.cm.Spectral, s=150, marker='o') # rt
axs[1].scatter(summary[:,0], summary[:,2], c=prop_direction, cmap=pl.cm.Spectral, s=150, marker='o') # % resp
axs[0].set_xlabel('Propofol level')
axs[1].set_xlabel('Propofol level')
axs[0].set_ylabel('RT')
axs[1].set_ylabel('% response')

## Spectrogram
eeg.set_eeg_reference('average')
frontal = np.isin(channel_names, ['F3', 'Fz', 'FCz', 'F4'])
posterior = np.isin(channel_names, ['P7', 'P8', 'Oz', 'P3', 'P4'])
e_f = eeg._data[frontal]
e_p = eeg._data[posterior]
sp_frontal, sp_t, sp_f = mts(e_f.T, fs=fs, window_size=10.0, window_step=10.0)
sp_posterior, sp_t, sp_f  = mts(e_p.T, fs=fs, window_size=10.0, window_step=10.0)
def spect_t2i(t):
    return np.argmin(np.abs(t - sp_t))
f_keep = sp_f < 40
sp_f = sp_f[f_keep]
sp_frontal = sp_frontal[:, f_keep, :]
sp_posterior = sp_posterior[:, f_keep, :]

fig, axs = pl.subplots(2, 1, sharex=True)
for ax, sp in zip(axs, [sp_frontal, sp_posterior]):
    sp = np.median(sp, axis=0)

    vmin, vmax = np.percentile(sp, [1, 90])
    ax.pcolormesh(sp_t, sp_f, sp,
                 vmin=vmin, vmax=vmax,
                 cmap=pl.cm.rainbow)

ax = axs[0]
for ps, plev in zip(phase_starts, phase_levels):
    ax.axvline(ps-eeg_time[0], color='white', ls=':')
    ax.text(ps-eeg_time[0], 42, f'to {plev:0.1f}', clip_on=False,
            ha='center', va='center')

summary = []
alpha = (sp_f>=8) & (sp_f<15)
delta = (sp_f>=0.5) & (sp_f<4)
pss = np.append(phase_starts, 1e15)
for t0, t1, lev in zip(pss[:-1], pss[1:], phase_levels):
    i0 = spect_t2i(t0-eeg_time[0])
    i1 = spect_t2i(t1-eeg_time[0])
    chunk = sp_frontal[:, :, i0:i1]
    chunk *= 1e6
    #chunk = nanpow2db(chunk)

    chunk_a = chunk[:, alpha, :]
    chunk_d = chunk[:, delta, :]
    mean_a = np.nanmedian(chunk_a)
    mean_d = np.nanmedian(chunk_d)
    summary.append([lev, mean_a, mean_d])
summary = np.array(summary)

fig, axs = pl.subplots(1, 2)
ax = axs[0]
ax.scatter(summary[:,0], summary[:,1], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
ax.set_xlabel('Propofol level')
ax.set_ylabel('Alpha power')

ax = axs[1]
ax.scatter(summary[:,0], summary[:,2], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
ax.set_xlabel('Propofol level')
ax.set_ylabel('Delta power')

## Chirp
#chirp_ch_name = ['Fz'] #['Fz', 'Cz', 'FCz',]  # always use a list even if just one; ?F4
chirp_ch_name = ['F3', 'F4']
chirp_ch_idx = ch_name_to_idx(chirp_ch_name)
#eeg.set_eeg_reference('average')
eeg.set_eeg_reference(['M1', 'M2'])

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

frequencies = np.linspace(25, 55, 35)
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

    #power.plot(picks=chirp_ch_name,
    #          baseline=(-0.1, 0),
    #           mode='logratio',
    #           title='Evoked power',
    #           axes=ax)
    #itc.plot(axes=ax,
    #         baseline=(-0.100, 0),
    #         mode='mean',
    #         vlim=(0.0, 0.3), # 0.8
    #         cnorm=None,
    #         colorbar=False)

    to_show = itc

    mean_to_show = np.mean(np.abs(to_show.data[chirp_ch_idx, :, :]), axis=0) 

    ax.imshow(mean_to_show,
              aspect='auto',
              origin='lower',
              vmin=0.0,
              vmax=0.3,
              extent=[itc.times[0], itc.times[-1], itc.freqs[0], itc.freqs[-1]],
              cmap='rainbow')
    
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Freq (Hz)')
    ax.set_title(f'{eid}', fontsize=9)
    if ax is not axs[0]:
        ax.set_ylabel('')

    use_data = to_show.copy().crop(tmin=0.0,
                                   tmax=0.5,
                                   fmin=30,  # note this could be 25?
                                   fmax=55).data  # shape: (n_channels, n_freqs, n_times)
    mean = use_data[chirp_ch_idx].mean(axis=0).mean(axis=1)  # avg chans thens time
    summary.append([float(eid), mean])
    
    '''
    mne.viz.plot_topomap(use_data.mean(axis=(1,2)), itc.info, axes=ax,
                         show=True, cmap='Reds', contours=0)
    '''

fig, ax = pl.subplots()
summ = np.array([(xval, np.mean(yval)) for xval, yval in summary])
ax.scatter(*summ.T, s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
ax.set_xlabel('Propofol level')
ax.set_ylabel('Mean chirp responses (ITC)')

## Oddball
ch_name = ['Fz',] #['FCz', 'Fz', 'Cz'] # always list
ch_idx = ch_name_to_idx(ch_name)
eeg_ob = eeg.copy().filter(l_freq=0.1, h_freq=40, fir_design='firwin')
eeg_ob.set_eeg_reference('average')
#eeg_ob.set_eeg_reference(['M1', 'M2'])

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

    epochs = mne.Epochs(eeg_ob,
                        events,
                        event_id=event_id,
                        tmin=-0.100,
                        tmax=0.500,
                        baseline=(-0.100, 0),
                        detrend=1,
                        preload=True)
    epochs = epochs.pick('eeg')

    mean_standard = epochs['standard'].average()
    mean_deviant = epochs['deviant'].average()
    err_standard = epochs['standard'].standard_error()
    err_deviant = epochs['deviant'].standard_error()

    times = mean_standard.times * 1000

    std = mean_standard.data[ch_idx].mean(axis=0) * 1e6
    std_err = err_standard.data[ch_idx].mean(axis=0) * 1e6
    dev = mean_deviant.data[ch_idx].mean(axis=0) * 1e6
    dev_err = err_deviant.data[ch_idx].mean(axis=0) * 1e6
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
    tmin_pp, tmax_pp = 0.100, 0.300  # in seconds
    evoked_s_crop = mean_standard.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    evoked_d_crop = mean_deviant.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    s_trace = evoked_s_crop.data[ch_idx].mean(axis=0) * 1e6
    d_trace = evoked_d_crop.data[ch_idx].mean(axis=0) * 1e6
    dif_trace = d_trace - s_trace
    dif = np.min(dif_trace)
    latency = 1000 * ((np.argmin(dif_trace)) / fs + tmin_pp)

    # Plot the topomap of these amplitudes
    #fig_topo, ax_topo = pl.subplots(1, 1)
    #mne.viz.plot_topomap(ptp_d_amplitudes, mean_deviant.info, axes=ax_topo,
    #                     show=True, cmap='Reds', contours=0)

    summary.append([phase_levels[lev], dif, latency])
ax.legend()

summ = np.array(summary)
fig, axs = pl.subplots(1, 2)
ax = axs[0]
ax.scatter(summ[:,0], summ[:,1], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
ax.set_xlabel('Propofol level')
ax.set_ylabel('Mean oddball mmn ampl')
ax = axs[1]
ax.scatter(summ[:,0], summ[:,2], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
ax.set_xlabel('Propofol level')
ax.set_ylabel('Mean oddball mmn latency')


## SSEP
eeg_ssep = eeg.copy().filter(l_freq=0.5, h_freq=50, fir_design='firwin', picks='eeg')
#eeg_ssep.set_eeg_reference('average')
eeg_ssep.set_eeg_reference(['Fz', 'FCz', 'Cz'])

# ssep-channel-specific filters applied only to it
eeg_ssep.notch_filter(freqs=[60, 120, 180], method='spectrum_fit', picks=['ssep'])
eeg_ssep.filter(l_freq=65., h_freq=fs/2-0.1, fir_design='firwin', picks=['ssep'])
ssep_channel = channel_names.index('ssep')
ssep = eeg_ssep._data[ssep_channel]

#ssep = filter_eeg(eeg_raw[:,[channel_names.index('ssep')]],
#                    fs=fs,
#                    lo=fs/2-0.1,
#                    hi=60,
#                    notch=60)[:,0]

fig,ax = pl.subplots(figsize=(8,1.5), gridspec_kw=dict(bottom=0.2))
time = eeg_time - eeg_time[0]
ax.plot(time, ssep, color='k')

onset_idx = detect_switch(np.abs(ssep), ssep_thresh[name], min_duration=1)
onset_t = np.array([eeg_time[i] for i in onset_idx])

# filter to include only those inside manually marked bounds of ssep testing made during session
bounds = markers[markers.text.str.startswith('ssep')]
bs = bounds.text.str.strip().str.replace('ssep ','').values
assert np.all(bs[0::2] == 'start')
assert np.all(bs[1::2] == 'stop')
bounds = np.array(list(zip(bounds.t.values[0::2], bounds.t.values[1::2])))
keep = np.array([np.any([((t>=b0) & (t<=b1)) for b0,b1 in bounds]) for t in onset_t])
onset_t = onset_t[keep]
onset_idx = onset_idx[keep]

inferred_ssep_rate = 1/np.median(np.diff(onset_t))
print(f'Inferred rate of {inferred_ssep_rate:0.1f} Hz - if far from 7, inspect.')

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

    epochs = mne.Epochs(eeg_ssep,
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
    tmin_pp, tmax_pp = 0.00, 0.060  # in seconds
    evoked_crop = evoked.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    trace = evoked_crop.data * 1e6
    zero = evoked_zero.data.squeeze() * 1e6
    #amplitude = np.abs(np.min(trace) - zero)
    amplitude = np.ptp(np.abs(trace), axis=1)
    #latency = np.argmax(np.abs(trace), axis=1)
    latency = np.argmin(trace, axis=1)
    ch_idx = evoked.ch_names.index('C3')

    summary.append([phase_levels[lev], amplitude[ch_idx], latency[ch_idx]])

    # Plot the topomap of these amplitudes
    #mne.viz.plot_topomap(amplitude, evoked.info, axes=ax,
    #                     show=True, cmap='Reds', contours=0)
    #ax_topo.set_title(f'Peak-to-peak SSEP (µV) {int(tmin_pp*1000)}–{int(tmax_pp*1000)} ms')

ax.legend()

summary = np.array(summary)
fig, axs = pl.subplots(1, 2)
axs[0].scatter(summary[:,0], summary[:,1], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral) # amplitude
axs[1].scatter(summary[:,0], summary[:,2], s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral) # latency
axs[0].set_xlabel('Propofol level')
axs[1].set_xlabel('Propofol level')
axs[0].set_ylabel('Amplitude')
axs[1].set_ylabel('Latency')

## Vigilance
data_file = '/Users/bdd/data/fus_anes/2025-07-24_vigilance_b003.txt'
with open(data_file, 'r') as f:
    vdata = f.readlines()
vdata = [json.loads(l) for l in vdata]

dt = [dt for dt,task,dat in vdata]
dat = [dat for dt,task,dat in vdata]
task = [task for dt,task,dat in vdata]
vdata = pd.DataFrame(dat)
vdata['ts'] = dt
vdata['task'] = task

def drop_leading_repeats(df, column):
    first_value = df[column].iloc[0]
    change_index = (df[column] != first_value).idxmax()
    if df[column].nunique() == 1:
        return df.iloc[0:0]  # empty DataFrame
    return df.iloc[change_index:].reset_index(drop=True)
vdata = drop_leading_repeats(vdata, 'task') # drop leading pvt
vdata = drop_leading_repeats(vdata, 'task') # drop leading dsst

pvt = vdata[vdata.task == 'pvt']
dsst = vdata[vdata.task == 'dsst']

# pvt
rt = pvt[~(pvt.note == 'early')].rt.values
rt *= 1000
fig, ax = pl.subplots()
ax.hist(rt, color='grey', bins=25)
ax.set_xlabel('RT (ms)')
# TODO analyze early presses

# dsst
dsst = dsst[~dsst.target.isna()]
target = dsst.target.str.replace('.jpeg','').values
choice = dsst.key.str.replace('num_','').values
print(f'{np.mean(choice == target):0.2f} correct')
print(f'{len(choice)} completed')

# TODO: separate by pre/post (in future will be labeled with start token) 

## Artifact sandbox
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

ica = ICA(n_components=15, random_state=97, max_iter='auto')
ica.fit(eeg)

# Find EOG (eye blink) components
eog_chans = ['F3','Fz','F4']
eog_epochs = create_eog_epochs(eeg, ch_name=eog_chans)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=eog_chans)
ica.exclude.extend(eog_inds)

# Find ECG (heartbeat) components
ecg_epochs = create_ecg_epochs(eeg)
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
ica.exclude.extend(ecg_inds)

eeg_clean = ica.apply(eeg.copy())
eeg.plot(title='Before cleaning', duration=30)
eeg_clean.plot(title='After cleaning', duration=30)


##
