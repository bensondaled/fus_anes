##
import numpy as np
import pandas as pd
import h5py
import mne
import os, sys
import json
import matplotlib.pyplot as pl
pl.ion()
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_connectivity import spectral_connectivity_epochs
from mne_icalabel import label_components

from util import mts, filter_eeg, detect_switch, nanpow2db, fit_sigmoid, mts_mne, make_sq_probability
from fus_anes.constants import MONTAGE as channel_names
from fus_anes.tci import TCI_Propofol as TCI

## Params
sessions = [
    '/Users/bdd/data/fus_anes/2025-07-23_12-05-45_subject-b001.h5', # 0,
    '/Users/bdd/data/fus_anes/2025-08-05_11-52-41_subject-b001.h5', # 1,

    '/Users/bdd/data/fus_anes/2025-07-30_merge_subject-b004.h5',    # 2,
    '/Users/bdd/data/fus_anes/2025-08-12_09-11-34_subject-b004.h5', # 3,

    '/Users/bdd/data/fus_anes/2025-07-25_08-38-29_subject-b003.h5', # 4,
    '/Users/bdd/data/fus_anes/2025-08-29_08-54-34_subject-b003.h5', # 5,

    '/Users/bdd/data/fus_anes/2025-09-05_08-10-33_subject-b008.h5', # 6,
    '/Users/bdd/data/fus_anes/2025-09-19_07-52-47_subject-b008.h5', # 7,
    '/Users/bdd/data/fus_anes/2025-10-24_07-54-48_subject-b008.h5', # 8,

    '/Users/bdd/data/fus_anes/2025-09-12_merge_subject-b006.h5',    # 9,
    '/Users/bdd/data/fus_anes/2025-10-03_07-38-36_subject-b006.h5', # 10,

    '/Users/bdd/data/fus_anes/2025-09-17_07-57-44_subject-b002.h5', # 11
    '/Users/bdd/data/fus_anes/2025-09-23_07-51-59_subject-b002.h5', # 12
    
    '/Users/bdd/data/fus_anes/2025-10-08_07-45-31_subject-b007.h5', # 13
    '/Users/bdd/data/fus_anes/2025-10-22_07-51-53_subject-b007.h5', # 14
    
    '/Users/bdd/data/fus_anes/2025-10-16_08-04-53_subject-b010.h5', # 15 # messy session
    '/Users/bdd/data/fus_anes/2025-11-05_merge_subject-b010.h5', # 16
    
    '/Users/bdd/data/fus_anes/2025-10-29_07-49-12_subject-b013.h5', # 17
    '/Users/bdd/data/fus_anes/2025-11-12_07-45-42_subject-b013.h5', # 18

    ]

try:
    selection = int(sys.argv[1]) # argument-based
except:
    selection = 18 # manual within-script selection

session_path = sessions[selection]

# intermediate data paths
src_dir = os.path.split(session_path)[0]
name = os.path.splitext(os.path.split(session_path)[-1])[0]
clean_eeg_path = os.path.join(src_dir, f'{name}.fif.gz')
ica_path = os.path.join(src_dir, f'{name}_ica.fif.gz')

already_clean = os.path.exists(clean_eeg_path)

## Load
with pd.HDFStore(session_path, 'r') as h:
    sconfig = json.loads(h.config.iloc[0])
    eeg = h.eeg
    bl_eyes = h.bl_eyes
    markers = h.markers
    squeeze = h.squeeze
    oddball = h.oddball
    chirp = h.chirp
    tci_cmd = h.tci_cmd
    pump = h.pump
    is_us_session = False
    _tci = h.tci

if name == '2025-09-16_07-48-02_subject-b002':
    eeg = eeg.iloc[:-25000] # time reset issue and unimportant segment at end to cut

# absolute time for high-res stuff
precise_eeg_time = eeg.hardware_ts.values + eeg.hardware_offset.values
def precise_t2i(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t2i(x) for x in t])
    else:
        return np.argmin(np.abs(t - precise_eeg_time))

# uniform time inference
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

## Load and process
eeg = mne.io.read_raw_fif(clean_eeg_path, preload=True)
ica = mne.preprocessing.read_ica(ica_path)
eeg = eeg.notch_filter(freqs=[60, 120, 180, 200, 240], method='fir', notch_widths=2.0, picks=None)
eeg = eeg.filter(l_freq=0.1, h_freq=58, fir_design='firwin', picks='eeg')
eeg.set_eeg_reference('average', projection=False)  # default unless analyses undo it
eeg = ica.apply(eeg.copy())

## Label propofol levels
goto = tci_cmd[tci_cmd.kind == 'goto']
_p_target = goto.ce_target
_p_target = pd.DataFrame(_p_target).drop_duplicates(keep='first').ce_target

def t_to_phase_idx(t):
    if isinstance(t, (list, np.ndarray)):
        return np.array([t_to_phase_idx(x) for x in t])
    else:
        edges = _p_target.index.values
        return np.searchsorted(edges, t)

bl_done = markers[markers.text.str.strip() == 'baseline_eyes complete'].iloc[0].t
phase0_start = markers[markers.text.str.strip() == 'steady start']
if len(phase0_start):
    phase0_start = phase0_start.iloc[0].t
else:
    phase0_start = bl_done + 60.0
assert phase0_start > bl_done and phase0_start - bl_done < 20*60
phase_starts = np.append(bl_done, _p_target.index.values)
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

# prep the AEP (oddball) data
eeg_ob = eeg.copy()
eeg_ob = eeg_ob.filter(l_freq=1, h_freq=20, fir_design='firwin', picks='eeg') # note this is on top of main loading filter as of now
eeg_ob.set_eeg_reference('average', projection=False)

ob_events_t = oddball[oddball.event.isin(['s','d'])].onset_ts.values
s_d = oddball[oddball.event.isin(['s','d'])].event.values.copy()
ob_onset_samples = precise_t2i(ob_events_t)

event_ids = {'standard': 1, 'deviant': 2}
events = np.zeros((len(ob_onset_samples), 3), dtype=int)
events[:, 0] = ob_onset_samples  # sample indices in MNE Raw object
events[:, 2] = [event_ids['standard'] if s == 's' else event_ids['deviant'] for s in s_d]

tmin, tmax = -0.2, 0.8  # -200ms to 800ms around stimulus
baseline = (None, 0)  # baseline correction using pre-stimulus period

epochs = mne.Epochs(eeg_ob, events, event_id=event_ids, 
                    tmin=tmin, tmax=tmax, 
                    baseline=baseline, 
                    preload=True,
                    reject=dict(eeg=150e-6),  # reject epochs with >150µV
                    reject_by_annotation=True)

##
outpath = '/Users/bdd/Desktop'

# Get propofol levels for each epoch
epoch_times = epochs.events[:, 0] / fs + eeg_time[0]
prop_levels = t_to_phase_level(epoch_times)

# Get unique propofol levels in temporal order
_, unique_indices = np.unique(prop_levels, return_index=True)
unique_levels = prop_levels[np.sort(unique_indices)]
print(f"Propofol levels: {unique_levels}")

# Plot responses by propofol level with shuffled controls
n_levels = len(unique_levels)
fig, axes = pl.subplots(n_levels, 2, figsize=(16, 3*n_levels), sharex=True)

if n_levels == 1:
    axes = axes.reshape(1, -1)

for i, level in enumerate(unique_levels):
    # Real epochs at this level
    mask = prop_levels == level
    epochs_level = epochs[mask]
    erp_real = epochs_level.average()
    
    # Shuffled control - random times
    n_epochs = len(epochs_level)
    random_times = np.random.uniform(eeg_time[0] + 1, eeg_time[-1] - 1, n_epochs)
    random_samples = ((random_times - eeg_time[0]) * fs).astype(int)
    
    events_shuffled = np.zeros((n_epochs, 3), dtype=int)
    events_shuffled[:, 0] = random_samples
    events_shuffled[:, 2] = 1
    
    epochs_shuffled = mne.Epochs(eeg_ob, events_shuffled, event_id={'random': 1},
                                tmin=-0.2, tmax=0.8, baseline=(None, 0),
                                preload=True, reject=dict(eeg=150e-6),
                                event_repeated='drop',
                                verbose=False)
    erp_shuffled = epochs_shuffled.average()
    
    # Plot real
    ax = axes[i, 0]
    times = erp_real.times * 1000
    colors = pl.cm.rainbow(np.linspace(0, 1, len(erp_real.ch_names)))
    
    for ch_idx in range(len(erp_real.ch_names)):
        ax.plot(times, erp_real.data[ch_idx] * 1e6, linewidth=0.8, 
               alpha=0.7, color=colors[ch_idx])
    
    gfp = np.std(erp_real.data, axis=0) * 1e6
    ax.plot(times, gfp, linewidth=2.5, color='black', label='GFP')
    ax.axvline(0, color='r', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Real - {level:.2f} µg/mL', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Plot shuffled
    ax = axes[i, 1]
    for ch_idx in range(len(erp_shuffled.ch_names)):
        ax.plot(times, erp_shuffled.data[ch_idx] * 1e6, linewidth=0.8,
               alpha=0.7, color=colors[ch_idx])
    
    gfp_shuf = np.std(erp_shuffled.data, axis=0) * 1e6
    ax.plot(times, gfp_shuf, linewidth=2.5, color='black', label='GFP')
    ax.axvline(0, color='r', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Shuffled - {level:.2f} µg/mL', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

axes[-1, 0].set_xlabel('Time (ms)')
axes[-1, 1].set_xlabel('Time (ms)')
fig.suptitle(f'{name} - AEP by Propofol Level', fontsize=14)
pl.tight_layout()
pl.savefig(os.path.join(outpath, f'{name}_aep_simple.png'), dpi=150, bbox_inches='tight')
pl.close(fig)
print(f"Saved: {name}_aep_simple.png")

##
# Chirp Analysis - Auditory steady-state response (ASSR)

# Prep chirp data - use broader frequency range for ASSR
eeg_chirp = eeg.copy()
eeg_chirp = eeg_chirp.filter(l_freq=1, h_freq=100, fir_design='firwin', picks='eeg')
eeg_chirp.set_eeg_reference('average', projection=False)

# Get chirp events (c = chirp, w = white noise control)
chirp_events = chirp[chirp.event.isin(['c', 'w'])]
chirp_times = chirp_events.onset_ts.values
chirp_type = chirp_events.event.values

# Convert to sample indices
chirp_samples = precise_t2i(chirp_times)

# Create events array
event_ids_chirp = {'chirp': 1, 'white_noise': 2}
events_chirp = np.zeros((len(chirp_samples), 3), dtype=int)
events_chirp[:, 0] = chirp_samples
events_chirp[:, 2] = [event_ids_chirp['chirp'] if c == 'c' else event_ids_chirp['white_noise'] for c in chirp_type]

# Create epochs - chirp is typically 500-1000ms, use longer window
tmin_chirp, tmax_chirp = -0.5, 2.0
epochs_chirp = mne.Epochs(eeg_chirp, events_chirp, event_id=event_ids_chirp,
                         tmin=tmin_chirp, tmax=tmax_chirp,
                         baseline=(None, 0),
                         preload=True,
                         reject=dict(eeg=150e-6),
                         reject_by_annotation=True)

print(f"Chirp epochs: {len(epochs_chirp['chirp'])}, White noise: {len(epochs_chirp['white_noise'])}")

# Get propofol levels for chirp epochs
epoch_times_chirp = epochs_chirp.events[:, 0] / fs + eeg_time[0]
prop_levels_chirp = t_to_phase_level(epoch_times_chirp)

# Get unique levels
_, unique_indices_chirp = np.unique(prop_levels_chirp, return_index=True)
unique_levels_chirp = prop_levels_chirp[np.sort(unique_indices_chirp)]

# Plot chirp responses by propofol level
n_levels_chirp = len(unique_levels_chirp)
fig, axes = pl.subplots(n_levels_chirp, 2, figsize=(16, 3*n_levels_chirp), sharex=True)

if n_levels_chirp == 1:
    axes = axes.reshape(1, -1)

for i, level in enumerate(unique_levels_chirp):
    mask = prop_levels_chirp == level
    epochs_level = epochs_chirp[mask]
    
    # Separate chirp and white noise
    chirp_mask = epochs_level.events[:, 2] == event_ids_chirp['chirp']
    noise_mask = epochs_level.events[:, 2] == event_ids_chirp['white_noise']
    
    if chirp_mask.sum() > 0:
        erp_chirp = epochs_level[chirp_mask].average()
    else:
        continue
    
    if noise_mask.sum() > 0:
        erp_noise = epochs_level[noise_mask].average()
    else:
        # Create random control if no white noise
        n_epochs = chirp_mask.sum()
        random_times = np.random.uniform(eeg_time[0] + 1, eeg_time[-1] - 1, n_epochs)
        random_samples = ((random_times - eeg_time[0]) * fs).astype(int)
        events_shuffled = np.zeros((n_epochs, 3), dtype=int)
        events_shuffled[:, 0] = random_samples
        events_shuffled[:, 2] = 1
        epochs_shuffled = mne.Epochs(eeg_chirp, events_shuffled, event_id={'random': 1},
                                    tmin=tmin_chirp, tmax=tmax_chirp, baseline=(None, 0),
                                    preload=True, reject=dict(eeg=150e-6),
                                    event_repeated='drop', verbose=False)
        erp_noise = epochs_shuffled.average()
    
    # Plot chirp
    ax = axes[i, 0]
    times = erp_chirp.times * 1000
    colors = pl.cm.rainbow(np.linspace(0, 1, len(erp_chirp.ch_names)))
    
    for ch_idx in range(len(erp_chirp.ch_names)):
        ax.plot(times, erp_chirp.data[ch_idx] * 1e6, linewidth=0.8,
               alpha=0.7, color=colors[ch_idx])
    
    gfp = np.std(erp_chirp.data, axis=0) * 1e6
    ax.plot(times, gfp, linewidth=2.5, color='black', label='GFP')
    ax.axvline(0, color='r', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Chirp - {level:.2f} µg/mL', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Plot white noise control
    ax = axes[i, 1]
    for ch_idx in range(len(erp_noise.ch_names)):
        ax.plot(times, erp_noise.data[ch_idx] * 1e6, linewidth=0.8,
               alpha=0.7, color=colors[ch_idx])
    
    gfp_noise = np.std(erp_noise.data, axis=0) * 1e6
    ax.plot(times, gfp_noise, linewidth=2.5, color='black', label='GFP')
    ax.axvline(0, color='r', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'White Noise Control - {level:.2f} µg/mL', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

axes[-1, 0].set_xlabel('Time (ms)')
axes[-1, 1].set_xlabel('Time (ms)')
fig.suptitle(f'{name} - Chirp ASSR by Propofol Level', fontsize=14)
pl.tight_layout()
pl.savefig(os.path.join(outpath, f'{name}_chirp_simple.png'), dpi=150, bbox_inches='tight')
pl.close(fig)
print(f"Saved: {name}_chirp_simple.png")

# Time-frequency analysis for chirp (shows frequency following)
from mne.time_frequency import tfr_morlet

for i, level in enumerate(unique_levels_chirp):
    mask = prop_levels_chirp == level
    epochs_level = epochs_chirp[mask]
    chirp_mask = epochs_level.events[:, 2] == event_ids_chirp['chirp']
    
    if chirp_mask.sum() < 5:
        continue
    
    epochs_chirp_level = epochs_level[chirp_mask]
    
    # Compute time-frequency for ASSR (focus on 20-100 Hz range)
    freqs = np.arange(20, 100, 2)
    n_cycles = freqs / 2.
    
    power = tfr_morlet(epochs_chirp_level, freqs=freqs, n_cycles=n_cycles,
                      return_itc=False, average=True, n_jobs=1, verbose=False)
    
    # Plot for one representative channel
    ch = 'Cz' if 'Cz' in power.ch_names else power.ch_names[0]
    
    fig, ax = pl.subplots(1, 1, figsize=(10, 6))
    power.plot([ch], baseline=(None, 0), mode='logratio',
              axes=ax, show=False, colorbar=True,
              title=f'Chirp ASSR - {level:.2f} µg/mL ({ch})')
    pl.savefig(os.path.join(outpath, f'{name}_chirp_tf_{level:.1f}.png'),
               dpi=150, bbox_inches='tight')
    pl.close(fig)

print("Chirp analysis complete")

##
