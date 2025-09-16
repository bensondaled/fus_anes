##
import numpy as np
import pandas as pd
import h5py
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
from timings import us_startstop

## Params
session_path = '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-08-04_08-48-05_subject-b001.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-08-05_11-52-41_subject-b001.h5'

#session_path = '/Users/bdd/data/fus_anes/2025-07-29_08-07-02_subject-b004.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-07-30_merge_subject-b004.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-08-11_07-54-24_subject-b004.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-08-12_09-11-34_subject-b004.h5'

#session_path = '/Users/bdd/data/fus_anes/2025-07-24_08-38-41_subject-b003.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-07-25_08-38-29_subject-b003.h5'
#session_path = '/Users/bdd/data/fus_anes/2025-08-28_08-50-10_subject-b003.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-08-29_08-54-34_subject-b003.h5'

#session_path = '/Users/bdd/data/fus_anes/2025-09-04_08-06-39_subject-b008.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-09-05_08-10-33_subject-b008.h5'

#session_path = '/Users/bdd/data/fus_anes/2025-09-11_07-42-12_subject-b006.h5' # u/s
#session_path = '/Users/bdd/data/fus_anes/2025-09-12_merge_subject-b006.h5'

# intermediate data paths
processed_path = '/Users/bdd/data/fus_anes/intermediate/processed.h5'

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
        is_us_session = False
        _tci = h.tci
    else:
        tci_cmd = None
        pump = None
        is_us_session = True
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
if not is_us_session:
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
if not is_us_session:
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

## Label propofol levels
if not is_us_session:
    goto = tci_cmd[tci_cmd.kind == 'goto']
    _p_target = goto.ce_target
else:
    # for ultrasound sessions, mock label as prop 0 vs 1 for pre-post respectively
    _p_target = pd.Series([1, 2], index=us_startstop[name])

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

## Compute effect site concentrations
if not is_us_session:
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

elif is_us_session:
    assert len(phase_starts) == 3 # in u/s sessions, this is pre-us, during, after
    ce_time = np.concatenate([ np.linspace(phase_starts[0], phase_starts[1], 100),
                               np.linspace(phase_starts[1], phase_starts[2], 100),
                               np.linspace(phase_starts[2], eeg_time[-1], 100),
                            ])
    ce_vals = np.array([phase_levels[0]]*100 + [phase_levels[1]]*100 + [phase_levels[2]]*100)

def ce_t2i(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([ce_t2i(x) for x in t])
    else:
        return np.argmin(np.abs(t - ce_time))


# ---- Analyses

## Summary with spectrogram
if is_us_session:
    summary_start_time = eeg_time[0]
else:
    first_target = tci_cmd.ce_target.values
    first_target = first_target[first_target>0][0]
    summary_start_time = tci_cmd.index.values[tci_cmd.ce_target==first_target][0] - 18*60
summary_end_time = summary_start_time + 9800 #tci_cmd.index.values[tci_cmd.ce_target==0.4][0] + 18*60
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
sp_t += eeg_time[0]
keep = (sp_f <= 30) & (sp_f > 0.5) # max frequency to show/analyze
spect = spect[:,keep,:]
sp_f = sp_f[keep]
sp = np.nanmedian(spect, axis=0)
alpha = (sp_f>=8) & (sp_f<15)
delta = (sp_f>=0.8) & (sp_f<4)

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
expanded_chirps = []
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
    expanded_chirps.append([float(eid), itc.data])


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

# anteriorization analysis
ce_for_spect = np.array([ce_vals[ce_t2i(t)] for t in sp_t])
ds0 = f'{name}_ce'
ds1 = f'{name}_spect'
with h5py.File(processed_path, 'a') as h:
    if ds0 in h:
        del h[ds0]
    if ds1 in h:
        del h[ds1]
    h.create_dataset(ds0, data=ce_for_spect, compression='lzf')
    ds = h.create_dataset(ds1, data=spect, compression='lzf')
    ds.attrs['channels'] = eeg_spect.ch_names
    ds.attrs['freq'] = sp_f
    ds.attrs['time'] = sp_t

## display the summary

gs = GridSpec(6, n_topo+1, left=0.1, right=0.9, top=0.98, bottom=0.15,
              width_ratios=[1]*n_topo + [0.1],
              height_ratios=[2,2,3,2,3,5],
              hspace=0.45)
fig = pl.figure(figsize=(11,9))

# show propofol ce
ax = fig.add_subplot(gs[0,:-1])
ax.plot((ce_time-summary_start_time)/60, ce_vals, color='k', lw=3)
#ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Propofol\nlevel')
ax.set_yticks(np.arange(0, 3.5, 1.0))
ax.set_xlim([0, total_mins])
ax.grid(True)
ax_prop = ax

# show spect
ax = fig.add_subplot(gs[4,:-1])
vmin, vmax = np.nanpercentile(sp, [5, 95])
pcm = ax.pcolormesh((sp_t-summary_start_time)/60, sp_f, sp,
             vmin=vmin, vmax=vmax,
             cmap=pl.cm.rainbow)
cax = fig.add_subplot(gs[4,-1])
cbar = pl.colorbar(pcm, cax=cax, shrink=0.5, label='Power (dB)')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlim([0, total_mins])

# show topo
t_secs = np.array_split(np.arange(total_secs)+summary_start_time, n_topo)
for i_topo in range(n_topo):
    sec = t_secs[i_topo]
    start_sec = sec[0]
    end_sec = sec[-1]

    start_idx = spect_t2i(start_sec)
    end_idx = spect_t2i(end_sec)

    alpha_power = np.nanmean(spect[:, alpha, start_idx:end_idx], axis=(1, 2))
    alpha_power = nanpow2db(alpha_power) # NOTE

    ax = fig.add_subplot(gs[2, i_topo])
    mne.viz.plot_topomap(alpha_power, eeg_spect.info, axes=ax, show=False,
                         vlim=(-2, 11.0), cmap=pl.cm.Reds)

    if i_topo == 0:
        ax.set_ylabel('Alpha power')

# show alpha and delta numerically
chunk_dt = 90.0 # secs
time_chunks = np.arange(0, total_secs+1, chunk_dt) + summary_start_time
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
t -= summary_start_time
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
chunk_dt = 60
time_chunks = np.arange(0, total_secs+1, chunk_dt)
response_traj = []
max_lag = 1.0 # secs
sq_onset = squeeze[squeeze.event.str.endswith('mp3')].onset_ts.values - summary_start_time
press_idx = detect_switch(np.abs(switch), switch_thresh[name])
squeeze_times = np.array([eeg_time[i] for i in press_idx]) - summary_start_time
#fig_s, ax_s = pl.subplots()
#ax_s.vlines(sq_onset, 0, 1, color='k')
#ax_s.vlines(squeeze_times, 0, 1, color='r')
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
#ax.vlines(sq_onset/60, 0, 100, color='red', lw=0.25)
#ax.vlines(squeeze_times/60, 0, 100, color='green', lw=0.25)
ax.sharex(ax_prop)

# save squeeze data for other analyses
res = []
for sqo in sq_onset+summary_start_time:
    c = ce_vals[ce_t2i(sqo)]
    resp = np.any([ st>sqo and st-sqo<1.5 for st in squeeze_times+summary_start_time])
    res.append([c, resp, sqo])
res = np.array(res)
sq_start = squeeze[squeeze.event.str.strip() == 'play'].index.values
ds0 = f'{name}_squeeze'
ds1 = f'{name}_squeeze_starts'
with h5py.File(processed_path, 'a') as h:
    if ds0 in h:
        del h[ds0]
    if ds1 in h:
        del h[ds1]
    h.create_dataset(ds0, data=res, compression='lzf')
    h.create_dataset(ds1, data=sq_start, compression='lzf')

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
              vmax=0.4,
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


## --- sandbox
'''
fig, axs = pl.subplots(1, len(expanded_chirps), figsize=(15,5))
for (l, c), ax in zip(expanded_chirps, axs):
    mag = np.mean(c, axis=(1,2))
    mne.viz.plot_topomap(mag, eeg_chirp.info, axes=ax, show=False)
'''
##
