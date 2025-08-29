##
import numpy as np
import pandas as pd
import mne
import os
import json
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory as blend
pl.ion()
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_icalabel import label_components

from util import mts, filter_eeg, detect_switch, nanpow2db, fit_sigmoid, mts_mne
from fus_anes.constants import MONTAGE as channel_names
from fus_anes.tci import TCI_Propofol as TCI
from threshs import switch_thresh, ssep_thresh

## Params
#session_path = '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-08-04_08-48-05_subject-b001.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-08-05_11-52-41_subject-b001.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-07-30_merge_subject-b004.h5'
session_path = '/Users/bdd/data/fus_anes/2025-08-12_09-11-34_subject-b004.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-07-24_08-38-41_subject-b003.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-07-25_08-38-29_subject-b003.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-08-29_08-54-34_subject-b003.h5'

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
    sconfig = h.config
    eeg = h.eeg
    bl_eyes = h.bl_eyes
    markers = h.markers
    squeeze = h.squeeze
    oddball = h.oddball
    chirp = h.chirp
    if 'tci_cmd' in h:
        tci_cmd = h.tci_cmd
        pump = h.pump
    else:
        tci_cmd = None
        pump = None
    print(h.keys())

sconfig = json.loads(sconfig.iloc[0])

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
    ica = mne.preprocessing.ICA(n_components=13, method='infomax', fit_params=dict(extended=True))
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
if name == '2025-08-29_08-54-34_subject-b003': # ssep reset issue
    ssep_markers = markers[markers.text.str.contains('ssep')]
    drop = ssep_markers.iloc[[7,8,9]]
    assert 'reset' in drop.iloc[-1].text
    markers.drop(index=drop.index.values, inplace=True)

eeg_switch = eeg.copy()
eeg_switch.filter(l_freq=0.1, h_freq=20, fir_design='firwin', picks=['gripswitch'])
switch_channel = channel_names.index('gripswitch')
switch = eeg_switch._data[switch_channel]

## Compute effect site concentrations
if name == '2025-07-25_08-38-29_subject-b003':
    pump = pump.iloc[2:]

t = TCI(age=sconfig['age'],
        sex=sconfig['sex'],
        weight=sconfig['weight'],
        height=sconfig['height'])
t.infuse(0)
starttime = eeg_time[0]
endtime = eeg_time[-1]
sec = starttime
pidx = 0
lev = []
while sec < endtime:
    while pidx<len(pump) and sec >= pump.index.values[pidx]:
        rate = pump.iloc[pidx].rate
        rate = 10000 * rate / sconfig['weight']
        t.infuse(rate)
        t.wait(1)
        pidx += 1
        sec += 1
        lev.append(t.level)
    t.wait(1)
    sec += 1
    lev.append(t.level)
for _ in range(60*5):
    t.wait(1)
    lev.append(t.level)

ce_vals = np.array(lev)
ce_time = np.arange(len(lev)) + starttime

def ce_t2i(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([ce_t2i(x) for x in t])
    else:
        return np.argmin(np.abs(t - ce_time))

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

## Summary with spectrogram
if tci_cmd is None:
    summary_start_time = eeg_time[0]
else:
    summary_start_time = tci_cmd.index.values[tci_cmd.ce_target==0.8][0] - 18*60
summary_end_time = summary_start_time + 8800 #tci_cmd.index.values[tci_cmd.ce_target==0.4][0] + 18*60
total_secs = summary_end_time-summary_start_time
total_mins = total_secs / 60
s_win_size = 20.0 # secs
n_topo = 16

eeg_spect = eeg.copy().pick('eeg')
#eeg_spect.set_eeg_reference(['Cz'])
eeg_spect = eeg_spect.drop_channels(eeg.info['bads'])
eeg_spect = eeg_spect.resample(100)

# nan bad segments
sfreq = eeg_spect.info['sfreq']
mask = np.ones(eeg_spect.n_times, dtype=np.float32)
for ann in eeg_spect.annotations:
    desc = ann['description']
    onset = ann['onset']
    duration = ann['duration']
    if desc.upper().startswith("BAD"):
        start = int(onset * sfreq)
        stop = int((onset + duration) * sfreq)
        mask[start:stop] = np.nan
mask = mask[:, None]

# compute spect
in_data = eeg_spect._data.T * 1e6 * mask
spect, sp_t, sp_f = mts(in_data,
                        window_size=s_win_size,
                        window_step=s_win_size,
                        fs=eeg_spect.info['sfreq'])
keep = (sp_f <= 30) & (sp_f > 0.5) # max frequency to show/analyze
spect = spect[:,keep,:]
sp_f = sp_f[keep]
sp = np.nanmedian(spect, axis=0)

sp = nanpow2db(sp)
#spect2, sp_t2, sp_f2 = mts_mne(eeg_spect, window_size=s_win_size) # works but slower and worse
#sp2 = np.nanmedian(spect2, axis=0)
def spect_t2i(t):
    return np.argmin(np.abs(sp_t - t))

# prep the chirp data
eeg_chirp = eeg.copy().set_eeg_reference(['M1','M2'], projection=False)
chirp_ch_name = ['F3', 'F4'] 
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
eids = list(event_id.keys())
chirps = []
for eid in eids:
    epochs_cond = epochs[eid]
    power, itc = epochs_cond.compute_tfr(method='morlet',
                                        freqs=frequencies,
                                        n_cycles=n_cycles,
                                        use_fft=True,
                                        return_itc=True,
                                        average=True,
                                         #decim=2,
                                        )

    mean_to_show = np.mean(np.abs(itc.data[chirp_ch_idx, :, :]), axis=0) 
    chirps.append([float(eid), mean_to_show])


# prep the AEP (oddball) data
ob_frontal = ['Cz','FCz','Fz','C3','C4']
ob_posterior = ['Oz','M1','M2','P3','P4']
ch_frontal = ch_name_to_idx(ob_frontal)
ch_posterior = ch_name_to_idx(ob_posterior)
eeg_ob = eeg.copy()
eeg_ob = eeg_ob.filter(l_freq=1, h_freq=20, fir_design='firwin', picks='eeg') # note this is on top of main loading filter as of now
eeg_ob.set_eeg_reference('average', projection=False)

ob_events_t = oddball[oddball.event.isin(['s','d'])].onset_ts.values
s_d = oddball[oddball.event.isin(['s','d'])].event.values.copy()
level_id = t_to_phase_idx(ob_events_t)
ob_onset = t2i(ob_events_t)

n_levels = len(np.unique(level_id))
oddballs = []
for lev in np.unique(level_id):
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
                        tmax=0.250,
                        baseline=(-0.050, 0),
                        detrend=1,
                        reject_by_annotation=True,
                        preload=True)
    epochs = epochs.pick('eeg')

    mean_standard = epochs['standard'].average()
    mean_deviant = epochs['deviant'].average()

    evoked = mne.combine_evoked([mean_standard], weights=[1,])
    sig_frontal = evoked.data[ch_frontal].mean(axis=0) * 1e6
    sig_posterior = evoked.data[ch_posterior].mean(axis=0) * 1e6
    oddballs.append([phase_levels[lev], sig_frontal, sig_posterior])

# prepare ssep
onset_idx = detect_switch(np.abs(ssep), ssep_thresh[name], min_duration=1)
onset_t = np.array([eeg_time[i] for i in onset_idx])

# filter to include only those inside manually marked bounds of ssep testing made during session
bounds = markers[markers.text.str.startswith('ssep')]
bs = bounds.text.str.strip().str.replace('ssep ','').values
bs = np.array([b for b in bs if 'reset' not in b]) # 2025-08-29_08-54-34_subject-b003 had a reset
assert np.all(bs[0::2] == 'start')
assert np.all(bs[1::2] == 'stop')
bounds = np.array(list(zip(bounds.t.values[0::2], bounds.t.values[1::2])))
keep = np.array([np.any([((t>=b0) & (t<=b1)) for b0,b1 in bounds]) for t in onset_t])
onset_t = onset_t[keep]
onset_idx = onset_idx[keep]

inferred_ssep_rate = 1/np.median(np.diff(onset_t))
print(f'Inferred rate of {inferred_ssep_rate:0.2f} Hz - if far from 7, inspect.')

level_id = t_to_phase_idx(onset_t)
n_levels = len(np.unique(level_id))
ssep_traces = []
for lev in np.unique(level_id):
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

    tmin_pp, tmax_pp = 0.00, 0.060  # in seconds
    evoked_crop = evoked.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
    ch_idxs = list(map(evoked.ch_names.index, ['C3','P3','P7',]))
    trace = evoked.data[ch_idxs] * 1e6
    ssep_traces.append([phase_levels[lev], trace])


## display the summary

gs = GridSpec(6, n_topo+1, left=0.1, right=0.9, top=0.98, bottom=0.15,
              width_ratios=[1]*n_topo + [0.1],
              height_ratios=[2,2,3,2,3,5],
              hspace=0.45)
fig = pl.figure(figsize=(11,9))

# show propofol ce
if tci_cmd is not None:
    ax = fig.add_subplot(gs[0,:-1])
    ax.plot((ce_time-summary_start_time)/60, ce_vals, color='k', lw=3)
    #ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Propofol\nlevel')
    ax.set_yticks(np.arange(0, 3.5, 1.0))
    ax.set_xlim([0, total_mins])
    ax.grid(True)

# show spect
ax = fig.add_subplot(gs[4,:-1])
vmin, vmax = np.nanpercentile(sp, [5, 95])
pcm = ax.pcolormesh(sp_t/60, sp_f, sp,
             vmin=vmin, vmax=vmax,
             cmap=pl.cm.rainbow)
cax = fig.add_subplot(gs[4,-1])
cbar = pl.colorbar(pcm, cax=cax, shrink=0.5, label='Power (dB)')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlim([0, total_mins])

# show topo
alpha = (sp_f>=8) & (sp_f<15)
delta = (sp_f>=0.8) & (sp_f<4)

t_idxs = np.array_split(np.arange(len(sp_t)), n_topo)
for i_topo in range(n_topo):
    idx = t_idxs[i_topo]
    start_idx = idx[0]
    end_idx = idx[-1]

    alpha_power = np.nanmean(spect[:, alpha, start_idx:end_idx], axis=(1, 2))

    ax = fig.add_subplot(gs[2, i_topo])
    mne.viz.plot_topomap(alpha_power, eeg_spect.info, axes=ax, show=False)

    if i_topo == 0:
        ax.set_ylabel('Alpha power')

# show alpha and delta numerically
chunk_dt = 90.0 # secs
time_chunks = np.arange(0, total_secs+1, chunk_dt)
is_frontal = np.isin(eeg_spect.ch_names, ['F3', 'Fz', 'FCz', 'F4'])
fspect = spect[is_frontal, ...]
response_traj = []
for t0 in time_chunks:
    t1 = t0 + chunk_dt
    i0 = spect_t2i(t0)
    i1 = spect_t2i(t1)
    alpha_power = np.nanmean(spect[:, alpha, i0:i1+1], axis=(0, 1, 2))
    f_alpha_power = np.nanmean(fspect[:, alpha, i0:i1+1], axis=(0, 1, 2))
    delta_power = np.nanmean(spect[:, delta, i0:i1+1], axis=(0, 1, 2))
    response_traj.append([t0+chunk_dt//2, alpha_power, f_alpha_power, delta_power])
response_traj = np.array(response_traj)
ax = fig.add_subplot(gs[3, :-1])
t, a, fa, d = response_traj.T
rat = nanpow2db(fa/a)
a = nanpow2db(a)
d = nanpow2db(d)
ax.plot(t/60, a, color='indianred', lw=1.5)
fax = ax.twinx()
fax.plot(t/60, rat, color='grey', lw=1.5)
fax.set_yticks([])
tax = ax.twinx()
tax.plot(t/60, d, color='indigo', lw=1.5)
ax.set_xlim([0, total_mins])
ax.set_ylabel('Power (dB)')
#ax.set_xlabel('Time (minutes)')
ax.tick_params(axis='y', labelcolor='indianred')
tax.tick_params(axis='y', labelcolor='indigo')
ax.spines['right'].set_visible(True)
ax.text(0.9, 0.98, 'Alpha', fontsize=10,
        color='indianred',
        ha='left',
        va='center',
        transform=ax.transAxes)
ax.text(0.9, 0.78, 'Delta', fontsize=10,
        color='indigo',
        ha='left',
        va='center',
        transform=ax.transAxes)

# show squeeze
chunk_dt = 30
time_chunks = np.arange(0, total_secs+1, chunk_dt)
response_traj = []
max_lag = 1.0 # secs
sq_onset = squeeze[squeeze.event.str.endswith('mp3')].onset_ts.values - summary_start_time
press_idx = detect_switch(np.abs(switch), switch_thresh[name])
squeeze_times = np.array([eeg_time[i] for i in press_idx]) - summary_start_time
for t0 in time_chunks:
    t1 = t0 + chunk_dt
    sq_lev = sq_onset[(sq_onset>=t0) & (sq_onset<t1)]
    rts = []
    for cmd_idx, cmd_t in enumerate(sq_lev):
        candidates = squeeze_times[squeeze_times > cmd_t]
        candidates = candidates[candidates - cmd_t <= max_lag]
        if cmd_idx < len(sq_lev)-1: # there was no next command in that time
            candidates = candidates[candidates <= sq_lev[cmd_idx+1]]
        rt = candidates[0] - cmd_t if len(candidates) else np.nan
        rts.append(float(rt) * 1000)
    rts = np.array(rts)
    pct_resp = 100.0 * np.nanmean(~np.isnan(rts))
    response_traj.append([t0+chunk_dt//2, pct_resp])
response_traj = np.array(response_traj)
ax = fig.add_subplot(gs[1, :-1])
rshow = response_traj.T[1]
rfilled = pd.Series(rshow).ffill()
ax.plot(response_traj.T[0]/60, rfilled, color='purple', lw=3, ls=':')
ax.scatter(response_traj.T[0]/60, rshow, color='darkviolet', lw=3, marker='o')
ax.set_xlim([0, total_mins])
ax.set_ylabel('% command\nfollowing')
ax.set_xlabel('Time (minutes)')
ax.set_yticks([0, 50, 100])
ax.grid(axis='y')

# show chirp and oddball and SSEP

# grab the correct y position on figure, but will make x's manually
row_pos_ax = fig.add_subplot(gs[5, 0])
_,y,_,h = row_pos_ax.get_position().bounds
h *= 0.5
y += 0.08
row_pos_ax.remove()

ob_ax = None
ssep_ax = None
for (c_plev, chirp_i), (o_plev, ob_i_fr, ob_i_ps), (s_plev, ssep_trace) in zip(chirps, oddballs, ssep_traces):
    assert c_plev == o_plev == s_plev
    t0 = phase_starts[phase_levels.tolist().index(c_plev)]
    t0 = (t0 - summary_start_time) / 60 + 5 # left side of panel 5 mins into new prop level
    
    x,_ = fig.axes[0].transData.transform([t0, -1])
    x,_ = fig.transFigure.inverted().transform([x, -1])
    w = 0.08

    # chirp
    ax = fig.add_axes([x, y, w, h*0.8])

    ax.imshow(chirp_i,
              aspect='auto',
              origin='lower',
              vmin=0.0,
              vmax=0.3,
              cmap=pl.cm.viridis)
    ax.axis('off')

    # oddball
    ax = fig.add_axes([x, y-h*1.0, w, h*0.8])
    #ax.plot(ob_i_fr - ob_i_ps, color='teal')
    ax.plot(ob_i_fr, color='teal')
    #ax.plot(ob_i_ps, color='crimson')
    if ob_ax: ax.sharey(ob_ax)
    ob_ax = ax
    ax.set_yticklabels([])
    ax.set_xlim([0, 180])
    ax.set_xticklabels([])

    # ssep
    ax = fig.add_axes([x, y-h*2.0, w, h*0.8])
    ax.plot(ssep_trace.T, color='crimson')
    if ssep_ax: ax.sharey(ssep_ax)
    ssep_ax = ax
    ax.set_yticklabels([])
    ax.set_xticklabels([])

# save summary fig
fig.savefig(f'/Users/bdd/Desktop/summary_{name}.jpg', dpi=350)






#-- spect summary analyses
'''
is_frontal = np.isin(eeg_spect.ch_names, ['F3', 'Fz', 'FCz', 'F4'])
is_frontal[:] = True # TEMP TODO, maybe keep? helps SNR

summary = []

# use entire phase
#pss = np.append(phase_starts, 1e15)
#for t0, t1, lev in zip(pss[:-1], pss[1:], phase_levels):

# use just the final 2 mins
pss = np.append(phase_starts[1:], phase_starts[-1]+20*60) - 60*2
pse = pss + 60*2
for t0, t1, lev in zip(pss, pse, phase_levels):

    i0 = spect_t2i(t0-eeg_time[0])
    i1 = spect_t2i(t1-eeg_time[0])
    chunk = spect[is_frontal, :, i0:i1]

    chunk = np.nanmean(chunk, axis=0)

    chunk_a = chunk[alpha, :]
    chunk_d = chunk[delta, :]

    mean_a = np.nanmedian(chunk_a)
    mean_d = np.nanmedian(chunk_d)
    
    try:
        cent_a = np.nanmedian(sp_f[alpha][np.nanargmax(chunk_a, axis=0)])
    except ValueError:
        cent_a = np.nan

    summary.append([lev, mean_a, mean_d, cent_a])
summary = np.array(summary)

lvals, apows, dpows, acents = summary.T

fig, axs = pl.subplots(1, 3, gridspec_kw=dict(wspace=0.5), figsize=(9,5))

for prd in [1,-1]:
    m = prop_direction_markers[prd]
    c = prop_direction_cols[prd]

    use = prop_direction == prd
    lv = lvals[use]

    ax = axs[0]
    ap = apows[use]
    ax.scatter(lv, ap, s=150, marker=m, color=c)
    sx, sy, s50 = fit_sigmoid(lv, ap, return_ec50=True, b0=0.1)
    ax.plot(sx, sy, color=c, label=f'ec50: {s50:0.1f}')
    
    ax = axs[1]
    ac = acents[use]
    ax.scatter(lv, ac, s=150, marker=m, color=c)

    ax = axs[2]
    dp = dpows[use]
    ax.scatter(lv, dp, s=150, marker=m, color=c)
    sx, sy, s50 = fit_sigmoid(lv, dp, return_ec50=True, b0=0.5)
    ax.plot(sx, sy, color=c, label=f'ec50: {s50:0.1f}')

ax = axs[0]
ax.set_xlabel('Propofol level')
ax.set_ylabel(r'$\alpha$ power (dB)')
ax.set_ylim([-6,11])
ax.legend()

ax = axs[1]
ax.set_xlabel('Propofol level')
ax.set_ylabel(r'$\alpha$ center')
ax.set_ylim([8,14])

ax = axs[2]
ax.set_xlabel('Propofol level')
ax.set_ylabel(r'$\Delta$ power (dB)')
ax.legend()

pl.savefig(f'/Users/bdd/Desktop/power_{name}.pdf')
np.save(f'/Users/bdd/Desktop/power_summ_{name}.npy', summary)
'''


## anteriorization sandbox analysis
is_ant = np.isin(eeg_spect.ch_names, ['F3', 'Fz', 'FCz', 'F4'])
is_post = np.isin(eeg_spect.ch_names, ['P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'])
spect_ant = np.nanmean(spect[is_ant], axis=0)
spect_post = np.nanmean(spect[is_post], axis=0)

res = []
for t, sa, sp in zip(sp_t, spect_ant.T, spect_post.T):
    t = t + summary_start_time
    a = np.nanmean(sa[alpha])
    p = np.nanmean(sp[alpha])
    ce = ce_vals[ce_t2i(t)]
    res.append([ce,a,p,])
res = np.array(res)
res = np.array([np.nanmean(r, axis=0) for r in np.array_split(res, 12, axis=0)])
c,a,p = res.T
keep = np.diff(c, prepend=-1) > 0

c = c[keep]
a = a[keep]
p = p[keep]

fig, axs = pl.subplots(1, 3, figsize=(12,4))
ax = axs[0]
ax.scatter(c, a, c=np.arange(len(c)), s=200)
ax.set_title('anterior')
ax = axs[1]
ax.scatter(c ,p, c=np.arange(len(c)), s=200)
ax.set_title('posterior')
ax = axs[2]
ax.scatter(c, a/p, c=np.arange(len(c)), s=200)
ax.set_title('a/p ratio')
ax.set_ylim([0,4])
ax.set_xlim([-0.3,3.6])
ax.set_xticks(np.arange(0, 3.3, 0.5))
ax.grid(True)
pl.savefig(f'/Users/bdd/Desktop/alpha_{name}.pdf')

## ------------ Individual analyses


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
np.save(f'/Users/bdd/Desktop/sqz_summ_{name}.npy', summary)

lvals, rts, pcts = summary.T

fig, axs = pl.subplots(1, 2, gridspec_kw=dict(wspace=0.5))

for prd in [1,-1]:
    m = prop_direction_markers[prd]
    c = prop_direction_cols[prd]

    use = prop_direction == prd
    lv = lvals[use]

    axs[0].scatter(lv, rts[use], color=c, s=150, marker=m) # rt
    axs[1].plot(lv, pcts[use],  color=c, markersize=10, marker=m) # % resp

axs[0].set_xlabel('Propofol level')
axs[1].set_xlabel('Propofol level')
axs[0].set_ylabel('RT')
axs[1].set_ylabel('% response')

pl.savefig(f'/Users/bdd/Desktop/squeeze_{name}.pdf')

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
ob_frontal = ['Cz','FCz','Fz','C3','C4']
ob_posterior = ['Oz','M1','M2','P3','P4']
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
fig, axs = pl.subplots(1, n_levels, sharex=True, sharey=True)
agg = []
for lev, ax in zip(np.unique(level_id), axs):
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
                        tmax=0.250,
                        baseline=(-0.050, 0),
                        detrend=1,
                        reject_by_annotation=True,
                        preload=True)
    epochs = epochs.pick('eeg')

    mean_standard = epochs['standard'].average()
    mean_deviant = epochs['deviant'].average()

    #evoked_diff = mne.combine_evoked([mean_deviant, mean_standard], weights=[1, -1])
    #evoked_diff = mne.combine_evoked([mean_deviant], weights=[1,])
    evoked_diff = mne.combine_evoked([mean_standard], weights=[1,])

    fig = evoked_diff.plot_joint(picks=ob_frontal + ob_posterior,
                                 times=[0.0, 0.100, 0.250],
                                 title=f'level {phase_levels[lev]:0.1f}')
    fig.savefig(f'/Users/bdd/Desktop/level_idx{lev}_ce{phase_levels[lev]:0.1f}.png')
    pl.close(fig)

    # posterior minus frontal
    ch_frontal = ch_name_to_idx(ob_frontal)
    ch_posterior = ch_name_to_idx(ob_posterior)
    sig_frontal = evoked_diff.data[ch_frontal].mean(axis=0) * 1e6
    sig_posterior = evoked_diff.data[ch_posterior].mean(axis=0) * 1e6
    posterofrontal = sig_posterior - sig_frontal
    agg.append([posterofrontal[np.argmax(np.abs(posterofrontal))], phase_levels[lev]])
    t = (np.arange(len(sig_frontal)) / fs) + epochs.tmin
    ax.plot(t, posterofrontal, color='k')
    ax.plot(t, sig_frontal, color='grey', lw=0.5)
    ax.plot(t, sig_posterior, color='dimgrey', lw=0.5)
    ax.axvline(0, color='grey', ls=':')
    ax.set_ylabel('microvolts')
    ax.set_xlabel('Secs from beep')
    ax.set_title(f'{lev}')

a, l = zip(*agg)
a = np.array(a)
l = np.array(l)
fig, ax = pl.subplots(figsize=(7,3.5))
#ax.scatter(l, a, s=150, marker='o', c=prop_direction, cmap=pl.cm.Spectral)
l = l[prop_direction==1]
a = a[prop_direction==1]
a = a / max(a)
ax.scatter(l, a, s=150, marker='o', color='slateblue')
ax.tick_params(labelsize=25)
#ax.set_xlabel('Propofol level')
#ax.set_ylabel('Auditory-evoked response magnitude (normalized)')


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
#data_file = '/Users/bdd/data/fus_anes/2025-07-25_vigilance_b003.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-07-30_vigilance_b004.txt'
data_file = '/Users/bdd/data/fus_anes/2025-08-12_vigilance_b004.txt'
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
    lab = f'{rep} - {1000*np.median(rt_):0.0f}'
    ax.hist(1000*rt_, histtype='step', label=lab, density=True)
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

## Sandbox 1 - squeeze detailed view
lev = ce_vals

fig, ax = pl.subplots(figsize=(15,9))
ax.plot(ce_time/60, lev[::5], color='k', lw=2)

sq_cmd = sq_onset - starttime
sq_did = squeeze_times - starttime
for c in sq_cmd:
    ax.axvline(c/60, color='grey', ls=':', alpha=0.5)
ax.axvline(c/60, color='k', ls=':', alpha=0.5, label='Squeeze command')
for d in sq_did:
    ax.axvline(d/60, color='red', ls=':', alpha=0.5)
ax.axvline(d/60, color='red', ls=':', alpha=0.5, label='Squeezed')

pumpds = pump.copy()
pumpds['tstamp'] = pumpds.index.values
pumpds = pumpds.groupby(np.arange(len(pumpds)) // 8).median()
pumprate = pumpds.rate.values
pumpidx = pumpds.tstamp.values
mkm = 10000 * pumprate / sconfig['weight']
for t, m in zip(pumpidx, mkm):
    ax.text((t-starttime)/60, 1.02, f'{m:0.0f}',
            fontsize=6,
            rotation=90,
            #ha='left',
            #va='bottom',
            transform=blend(ax.transData, ax.transAxes))

ax.legend()
ax.set_xlim([0, 160])
ax.set_xlabel('Minutes')
ax.set_ylabel('Propofol level')
fig.savefig(f'/Users/bdd/Desktop/{name}_squeeze_display.pdf')
pl.close(fig)

## Sandbox 2 - comparing power across sessions
seshs = [
        #'/Users/bdd/Desktop/power_summ_2025-07-23_12-05-45_subject-b001.npy',
        #'/Users/bdd/Desktop/power_summ_2025-08-05_11-52-41_subject-b001.npy',
        '/Users/bdd/Desktop/power_summ_2025-07-30_merge_subject-b004.npy',
        '/Users/bdd/Desktop/power_summ_2025-08-12_09-11-34_subject-b004.npy',
        '/Users/bdd/Desktop/power_summ_2025-07-25_08-38-29_subject-b003.npy',
        ]
labs = [
        'A',
        'B',
        'X']
cols = [
        'dimgrey',
        'darkorange',
        'darkorange',]
lss = [
       '-',
       '-',
       ':']

fig, axs = pl.subplots(1, 2, figsize=(8,5))

for hyster,ax in zip([1, -1], axs):
    for sesh, col, ls, lab in zip(seshs, cols, lss, labs):
        dat = np.load(sesh)
        prop = dat.T[0]
        powr = dat.T[1]
        powr = 10*np.log10(powr / powr[0])
        if hyster == 1:
            prop = prop[:5]
            powr = powr[:5]
        elif hyster == -1:
            prop = prop[5:]
            powr = powr[5:]
        ax.plot(prop, powr, color=col, marker='o', lw=1, ls=ls, label=lab)
        
        #sx, sy, s50 = fit_sigmoid(prop, powr, return_ec50=True, b0=0.1)
        #m, b = np.polyfit(prop, powr, 1)
        #sx = np.linspace(prop.min(), prop.max(), 100)
        #sy = m * sx + b
        #ax.plot(sx, sy, label=f'ec50: {s50:0.1f}',
        #        color=col, ls=ls)

axs[-1].invert_xaxis()
axs[0].legend()

axs[0].set_title('induction')
axs[1].set_title('emergence')
axs[0].set_ylabel('frontal alpha power change from baseline (dB)')
axs[0].set_xlabel('propofol concentration')
axs[1].set_xlabel('propofol concentration')

fig.savefig('/Users/bdd/Desktop/pwr.pdf')
pl.close(fig)

## Sandbox 3 - comparing squeeze across sessions
seshs = [
        #'/Users/bdd/Desktop/sqz_summ_2025-07-23_12-05-45_subject-b001.npy',
        #'/Users/bdd/Desktop/sqz_summ_2025-08-05_11-52-41_subject-b001.npy',
        '/Users/bdd/Desktop/sqz_summ_2025-07-30_merge_subject-b004.npy',
        '/Users/bdd/Desktop/sqz_summ_2025-08-12_09-11-34_subject-b004.npy',
        '/Users/bdd/Desktop/sqz_summ_2025-07-25_08-38-29_subject-b003.npy',
        ]
labs = [
        'A',
        'B',
        'X']
cols = [
        'dimgrey',
        'darkorange',
        'darkorange',]
lss = [
       '-',
       '-',
       ':']

fig, axs = pl.subplots(1, 2, figsize=(8,5))

for hyster,ax in zip([1, -1], axs):
    for sesh, col, ls, lab in zip(seshs, cols, lss, labs):
        dat = np.load(sesh)
        prop = dat.T[0]
        pct = dat.T[2]
        if hyster == 1:
            prop = prop[:5]
            pct = pct[:5]
        elif hyster == -1:
            prop = prop[5:]
            pct = pct[5:]
        ax.plot(prop, pct, color=col, marker='o', lw=1, ls=ls, label=lab)

axs[-1].invert_xaxis()
axs[0].legend()

axs[0].set_title('induction')
axs[1].set_title('emergence')
axs[0].set_ylabel('% response')
axs[0].set_xlabel('propofol concentration')
axs[1].set_xlabel('propofol concentration')

fig.savefig('/Users/bdd/Desktop/sq.pdf')
pl.close(fig)
##
