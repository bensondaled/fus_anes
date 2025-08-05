##
import numpy as np
import pandas as pd
import mne
import os
import json
import matplotlib.pyplot as pl
pl.ion()
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_icalabel import label_components

from util import mts, filter_eeg, detect_switch, nanpow2db, fit_sigmoid
from fus_anes.constants import MONTAGE as channel_names
from threshs import switch_thresh, ssep_thresh

## Params
#session_path = '/Users/bdd/data/fus_anes/2025-07-25_08-38-29_subject-b003.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-07-30_merge_subject-b004.h5'
session_path = '/Users/bdd/data/fus_anes/2025-08-04_08-48-05_subject-b001.h5'

src_dir = os.path.split(session_path)[0]
name = os.path.splitext(os.path.split(session_path)[-1])[0]
clean_eeg_path = os.path.join(src_dir, f'{name}.fif.gz')
ica_path = os.path.join(src_dir, f'{name}_ica.fif.gz')

already_clean = os.path.exists(clean_eeg_path)

# Display params
prop_direction_cols = {d:c for c,d in zip(pl.cm.Spectral([0.2,0.8]), [1,-1])}
prop_direction_markers = {1:'>', -1:'<'}

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
slope, intercept = np.polyfit(np.arange(len(true_eeg_time)), true_eeg_time, deg=1)
eeg_time = slope * np.arange(len(true_eeg_time)) + intercept

eeg = eeg.values[:, :len(channel_names)]
eeg *= 1e-6
eeg_raw = eeg # Volts

def ch_name_to_idx(name):
    if isinstance(name, (str,)):
        return channel_names.index(name) 
    elif isinstance(name, (list, np.ndarray)):
        return [ch_name_to_idx(n) for n in name]

## Format into MNE
ch_types = [{'ssep':'stim', 'gripswitch':'stim', 'ecg':'ecg'}.get(c, 'eeg') for c in channel_names]
info = mne.create_info(ch_names=channel_names,
                       sfreq=fs,
                       ch_types=ch_types)
eeg = mne.io.RawArray(eeg_raw.T.copy(), info)
eeg.set_montage(mne.channels.make_standard_montage('standard_1020'))

# in case nan's are present (should only be for merged sessions due to a software close in session
nan_samples = np.where(np.isnan(eeg._data).any(axis=0))[0]
if len(nan_samples):
    start_nan = nan_samples[0] / fs
    end_nan = nan_samples[-1] / fs
    eeg.annotations.append(onset=start_nan, duration=end_nan-start_nan, description='bad_nan')


## Clean EEG
if not already_clean:
    eeg_clean = eeg.copy()
    eeg_clean._data = np.nan_to_num(eeg_clean._data)
    eeg_clean.set_eeg_reference('average', projection=True) # project True so it doesn't lose original data
    eeg_clean.plot(block=True, duration=30, use_opengl=True,
                   highpass=1.0, lowpass=40)

    eeg_clean.save(clean_eeg_path,) #overwrite=True)
elif already_clean:
    eeg_clean = mne.io.read_raw_fif(clean_eeg_path, preload=True)
eeg = eeg_clean

# ICA 
if os.path.exists(ica_path):
    ica = mne.preprocessing.read_ica(ica_path)
else:
    eeg_for_ica = eeg_clean.copy()
    eeg_for_ica = eeg_for_ica.notch_filter(freqs=[60,120,180,200,240], method='fir', picks=None, notch_widths=2.0)
    eeg_for_ica = eeg_for_ica.filter(l_freq=1.0, h_freq=100.0)
    eeg_for_ica.set_eeg_reference('average', projection=False)
    ica = mne.preprocessing.ICA(n_components=14, method='infomax', fit_params=dict(extended=True))
    ica.fit(eeg_for_ica, picks='eeg')
    labels = label_components(eeg_for_ica, ica, method='iclabel')
    print(labels['labels'])
    ica.exclude = [i for i in range(len(labels['labels'])) if labels['labels'][i] != 'brain']
    ica.save(ica_path)

## Preprocessing

# filter
eeg = eeg.notch_filter(freqs=[60, 120, 180, 200, 240], method='fir', notch_widths=2.0, picks=None)
eeg = eeg.filter(l_freq=0.1, h_freq=58, fir_design='firwin', picks='eeg')

# reference
eeg.set_eeg_reference('average', projection=False)  # default unless analyses undo it
eeg = ica.apply(eeg.copy())

# extract peripheral signals
eeg_ssep = eeg.copy()
eeg_ssep = eeg_ssep.notch_filter(freqs=[60,120,180], method='spectrum_fit', picks=['ssep']) # 2nd notch
eeg_ssep.filter(l_freq=65., h_freq=fs/2-0.1, fir_design='firwin', picks=['ssep'])
ssep_channel = channel_names.index('ssep')
ssep = eeg_ssep._data[ssep_channel]
eeg_ssep.set_eeg_reference(['Fz'])#, 'FCz', 'Cz'])

eeg_switch = eeg.copy()
eeg_switch.filter(l_freq=0.1, h_freq=20, fir_design='firwin', picks=['gripswitch'])
switch_channel = channel_names.index('gripswitch')
switch = eeg_switch._data[switch_channel]

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


## ---- Analyses

## Squeeze

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
    pct_resp = 100.0 * np.nanmean(~np.isnan(rts))

    ax.hist(rts[~np.isnan(rts)], bins=15, color='grey')
    ax.set_title(f'% resp: {pct_resp:0.0f}%, lev {phase_levels[lev]:0.2f}', fontsize=8)
    ax.set_xlabel('RT (ms)')

    summary.append([phase_levels[lev], np.nanmean(rts), pct_resp])

summary = np.array(summary)

lvals, rts, pcts = summary.T

fig, axs = pl.subplots(1, 2, gridspec_kw=dict(wspace=0.5))

for pd in [1,-1]:
    m = prop_direction_markers[pd]
    c = prop_direction_cols[pd]

    use = prop_direction == pd
    lv = lvals[use]

    axs[0].scatter(lv, rts[use], color=c, s=150, marker=m) # rt
    axs[1].plot(lv, pcts[use],  color=c, markersize=10, marker=m) # % resp

axs[0].set_xlabel('Propofol level')
axs[1].set_xlabel('Propofol level')
axs[0].set_ylabel('RT')
axs[1].set_ylabel('% response')

pl.savefig(f'/Users/bdd/Desktop/squeeze_{name}.pdf')

## Spectrogram
frontal = np.isin(channel_names, ['F3', 'Fz', 'FCz', 'F4'])
posterior = np.isin(channel_names, ['P7', 'P8', 'Oz', 'P3', 'P4'])
eeg_spect = eeg.copy()
e_f = eeg_spect._data[frontal] * 1e6 # to uV
e_p = eeg_spect._data[posterior] * 1e6 # to uV
decim = 4
sp_frontal, sp_t, sp_f = mts(e_f.T[::decim], fs=fs/decim, window_size=40.0, window_step=20.0)
sp_posterior, sp_t, sp_f  = mts(e_p.T[::decim], fs=fs/decim, window_size=40.0, window_step=20.0)
def spect_t2i(t):
    return np.argmin(np.abs(t - sp_t))
f_keep = sp_f < 40
sp_f = sp_f[f_keep]
sp_frontal = sp_frontal[:, f_keep, :]
sp_posterior = sp_posterior[:, f_keep, :]

fig, axs = pl.subplots(2, 1, sharex=True)
for ax, sp in zip(axs, [sp_frontal, sp_posterior]):
    sp = np.nanmedian(sp, axis=0)

    vmin, vmax = np.nanpercentile(sp, [1, 95])
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

    chunk = nanpow2db(chunk)

    chunk_a = chunk[:, alpha, :]
    chunk_d = chunk[:, delta, :]
    mean_a = np.nanmedian(chunk_a)
    mean_d = np.nanmedian(chunk_d)
    summary.append([lev, mean_a, mean_d])
summary = np.array(summary)

lvals, apows, dpows = summary.T

fig, axs = pl.subplots(1, 2, gridspec_kw=dict(wspace=0.5), figsize=(9,5))

for pd in [1,-1]:
    m = prop_direction_markers[pd]
    c = prop_direction_cols[pd]

    use = prop_direction == pd
    lv = lvals[use]

    ax = axs[0]
    ap = apows[use]
    ax.scatter(lv, ap, s=150, marker=m, color=c)
    sx, sy, s50 = fit_sigmoid(lv, ap, return_ec50=True, b0=0.1)
    ax.plot(sx, sy, color=c, label=f'ec50: {s50:0.1f}')

    ax = axs[1]
    dp = dpows[use]
    ax.scatter(lv, dp, s=150, marker=m, color=c)
    sx, sy, s50 = fit_sigmoid(lv, dp, return_ec50=True, b0=0.5)
    ax.plot(sx, sy, color=c, label=f'ec50: {s50:0.1f}')

ax = axs[0]
ax.set_xlabel('Propofol level')
ax.set_ylabel(r'$\alpha$ power (dB)')
ax.legend()

ax = axs[1]
ax.set_xlabel('Propofol level')
ax.set_ylabel(r'$\Delta$ power (dB)')
ax.legend()

pl.savefig(f'/Users/bdd/Desktop/power_{name}.pdf')

## Chirp
#eeg_chirp = eeg.copy() # worked v well w/ b004, ie avg ref
eeg_chirp = eeg.copy().set_eeg_reference(['M1','M2'], projection=False)
#chirp_ch_name = ['Cz'] #['Fz', 'Cz', 'FCz',]  # always use a list even if just one; ?F4
chirp_ch_name = ['F3', 'F4'] # worked v well w/ b004
chirp_ch_idx = ch_name_to_idx(chirp_ch_name)

chirp_onset_t = chirp[chirp.event=='c'].onset_ts.values
level_id = t_to_phase_idx(chirp_onset_t)
chirp_onset = t2i(chirp_onset_t)
events = np.array([[int(idx), 0, lid] for idx, lid in zip(chirp_onset, level_id)])
event_id = {f'{phase_levels[x]:0.1f}':x for x in np.unique(level_id)}

epochs = mne.Epochs(eeg_chirp,
                    events,
                    event_id=event_id,
                    tmin=-0.2, tmax=0.600,
                    baseline=(-0.2, 0),
                    detrend=1,
                    reject_by_annotation=True,
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
                                         #decim=2,
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
              vmax=0.4,
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
SHUF = False # False / 'sd' / 'rand'
ch_name = ['Fz']# ['FCz', 'Fz', 'Cz'] # always list
ch_idx = ch_name_to_idx(ch_name)
eeg_ob = eeg.copy()
eeg_ob = eeg_ob.filter(l_freq=1, h_freq=20, fir_design='firwin', picks='eeg') # note this is on top of main loading filter as of now
eeg_ob.set_eeg_reference('average', projection=False)

ob_events_t = oddball[oddball.event.isin(['s','d'])].onset_ts.values
s_d = oddball[oddball.event.isin(['s','d'])].event.values.copy()
level_id = t_to_phase_idx(ob_events_t)
ob_onset = t2i(ob_events_t)

if SHUF == 'sd':
    np.random.shuffle(s_d)
elif SHUF == 'rand':
    ob_onset = np.random.randint(np.min(ob_onset), np.max(ob_onset), size=ob_onset.shape)

n_levels = len(np.unique(level_id))
fig, axs = pl.subplots(3, 3, figsize=(15,4),
                       sharex='row', sharey='row',
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
                        baseline=(-0.050, 0),
                        detrend=1,
                        reject_by_annotation=True,
                        preload=True)
    epochs = epochs.pick('eeg')

    mean_standard = epochs['standard'].average()
    mean_deviant = epochs['deviant'].average()
    err_standard = epochs['standard'].standard_error()
    err_deviant = epochs['deviant'].standard_error()

    # SANDBOX: using some MNE tools to dig deeper
    #evoked_diff = mne.combine_evoked([mean_deviant, mean_standard], weights=[1, -1])
    evoked_diff = mne.combine_evoked([mean_deviant], weights=[1,]) # TODO TEMP
    fig = evoked_diff.plot_joint(picks=['Cz','FCz','Fz','M1','M2','Oz','C3','C4'],
                                 times=[0.0, 0.100, 0.250],
                                 title=f'level {phase_levels[lev]:0.1f}')
    fig.savefig(f'/Users/bdd/Desktop/level_idx{lev}_ce{phase_levels[lev]:0.1f}.png')
    pl.close(fig)
    # --

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
    tmin_pp, tmax_pp = 0.00, 0.300  # in seconds
    evoked_s_crop = mean_standard.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    evoked_d_crop = mean_deviant.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    full_dif = 1e6 * (evoked_d_crop.data - evoked_s_crop.data)
    dif_trace = full_dif[ch_idx].mean(axis=0)
    dif = np.min(dif_trace)
    latency = 1000 * ((np.argmin(dif_trace)) / fs + tmin_pp)

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
#ch_idx = eeg.ch_names.index('C3')
#ch_idx = eeg.ch_names.index('P3')
ch_idx = eeg.ch_names.index('P7')

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
print(f'Inferred rate of {inferred_ssep_rate:0.2f} Hz - if far from 7, inspect.')

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
                        tmax=0.080,
                        baseline=(-0.010, 0.00),
                        reject_by_annotation=True,
                        preload=True)
    evoked = epochs.average()
    times_ms = evoked.times * 1000

    cols = ['slateblue', 'darkorange', 'grey']
    cnames = ['C3', 'P3', 'P7']
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
    amplitude = np.ptp(np.abs(trace), axis=1)
    latency = 1000 * (np.argmin(trace, axis=1) / fs + tmin_pp)

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
data_file = '/Users/bdd/data/fus_anes/2025-07-30_vigilance_b004.txt'
with open(data_file, 'r') as f:
    vdata = f.readlines()
vdata = [json.loads(l) for l in vdata]

dt = [dt for dt,task,dat in vdata]
dat = [dat for dt,task,dat in vdata]
task = [task for dt,task,dat in vdata]
vdata = pd.DataFrame(dat)
vdata['ts'] = dt
vdata['task'] = task

start_idx = np.argwhere(vdata.task == 'start').squeeze()
start_idx = np.append(start_idx, 1e11)
for r, (si0, si1) in enumerate(zip(start_idx[:-1], start_idx[1:])):
    rng = np.arange(len(vdata))
    in_grp = (rng > si0) & (rng < si1)
    vdata.loc[in_grp, 'rep'] = r

vdata = vdata[vdata.task != 'start']
n_reps = len(np.unique(vdata.rep.values))

pvt = vdata[vdata.task == 'pvt']
dsst = vdata[vdata.task == 'dsst']
dsst = dsst[~dsst.target.isna()]

# pvt
# TODO analyze early presses
fig, ax = pl.subplots(1, sharex=True)
for rep in range(n_reps):
    use = (pvt.rep == rep) & (pvt.note != 'early')
    rt_ = pvt[use].rt.values
    ax.hist(1000*rt_, histtype='step', label=rep, density=True)
ax.set_xlabel('RT (ms)')
ax.legend()

# dsst
fig, axs = pl.subplots(1, 3)
for rep in range(n_reps):
    use = (dsst.rep == rep)
    dsst_ = dsst[use]
    target = dsst_.target.str.replace('.jpeg','').values
    choice = dsst_.key.str.replace('num_','').values

    n_completed = len(choice)
    rt = np.mean(dsst_.rt.values)
    pct_corr = np.mean(choice == target)
    print(f'Rep {rep}')
    print(f'\t{pct_corr:0.2f} correct')
    print(f'\t{n_completed} completed')

    ax = axs[0]
    ax.bar([rep], [n_completed], color='grey')
    
    ax = axs[1]
    ax.bar([rep], [100*pct_corr], color='grey')
    
    ax = axs[2]
    ax.bar([rep], [1000*rt], color='grey')

axs[0].set_title('N completed')
axs[1].set_title('% correct')
axs[1].set_ylim([0,100])
axs[2].set_title('RT (ms)')
# TODO: separate by pre/post (in future will be labeled with start token) 

## EEG artifact cleaning sandbox

#ica = ICA(n_components=13)
#ica.fit(eeg)
#ica.plot_components()
#ica.plot_sources(eeg.pick_types(eeg=True))
#
#ica.exclude = []
#ica.apply(eeg_ica)
#
## Find EOG (eye blink) components
#eog_chans = ['F3','Fz','F4']
#eog_epochs = create_eog_epochs(eeg_ica, ch_name=eog_chans)
#eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=eog_chans)
#ica.exclude.extend(eog_inds)
#
## Find ECG (heartbeat) components
#ecg_epochs = create_ecg_epochs(eeg_ica)
#ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
#ica.exclude.extend(ecg_inds)
#
#eeg_clean = ica.apply(eeg_ica.copy())
#eeg_ica.plot(title='Before cleaning', duration=30)
#eeg_clean.plot(title='After cleaning', duration=30)



##
