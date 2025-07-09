##
import pandas as pd
from scipy.signal import butter, iirnotch, tf2zpk, zpk2sos, sosfiltfilt
from matplotlib.transforms import blended_transform_factory as blend

def filter_eeg(data, fs, lo=50, hi=0.5, notch=60):

    b, a = butter(2, [hi, lo], fs=fs, btype='bandpass', analog=False)
    b0, a0 = iirnotch(notch, 20.0, fs=fs)
    
    b = np.convolve(b, b0)
    a = np.convolve(a, a0)

    Z, P, K = tf2zpk(b, a)
    sos = zpk2sos(Z, P, K)

    out = sosfiltfilt(sos, data, axis=0)
    return out


## -------------- oddball/chirp
mode = 'oddball'

if mode == 'oddball':
    with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-04_17-38-41_subject-test_subject.h5', 'r') as h:
        eeg = h.eeg
        ob = h.oddball

if mode == 'chirp':
    with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-04_18-07-45_subject-test_subject.h5', 'r') as h:
        eeg = h.eeg
        chirp = h.chirp

eeg_time = eeg.index.values
eeg = eeg.iloc[:, :16].values

downsamp = 4
eeg = eeg[::downsamp]
eeg_time = eeg_time[::downsamp]
fs = 500 / downsamp

eeg = filter_eeg(eeg, fs=fs)

#eeg[:,:12] -= eeg[:,12][:,None] # reref
#eeg[:,:12] -= eeg[:,:12].mean(axis=1, keepdims=True) # avg reref

def t2i(t):
    return np.argmin(np.abs(eeg_time-t))


##
if mode == 'oddball':
    ob = ob[ob.event.isin(['s','d'])]
    ob_s = ob[ob.event=='s'].onset_ts.values
    ob_d = ob[ob.event=='d'].onset_ts.values
    b0 = ob_s
    b1 = ob_d
elif mode == 'chirp':
    chirp = chirp[chirp.event.isin(['c','w'])]
    chirp_c = chirp[chirp.event=='c'].onset_ts.values
    chirp_w = chirp[chirp.event=='w'].onset_ts.values
    b0 = chirp_c
    b1 = chirp_w

##
fig, ax = pl.subplots()
cols = ['blue', 'red', 'grey']
res = []
b_r = np.random.choice(np.linspace(np.min(b0), np.max(b0), 10000), size=len(b1))
for grp, col in zip([b0, b1, b_r], cols):
    sigs = []
    for evt in grp:
        i = t2i(evt)
        sig = eeg[i-int(fs//5) : i+int(fs), :]
        sigs.append(sig)

    #choose_chan = [0,1,2,3,4,5,6,7,8]
    choose_chan = [4,]
    choose = np.array(sigs)[:,:, choose_chan]
    mean = choose.mean(axis=-1) # just avg over channels for now

    sd = np.std(mean, axis=0)/np.sqrt(len(mean))
    mean = np.mean(mean, axis=0)
    
    time = (np.arange(len(mean)) - (fs//5)) / fs
    ax.plot(time, mean, color=col)
    ax.fill_between(time, mean-sd, mean+sd, lw=0, color=col, alpha=0.4)
    res.append(mean)

ax.axvline(0, color='grey', ls=':')
ax.set_xlim([-.200, .800])

fig, ax = pl.subplots()
ax.plot(time, res[1] - res[0], color='red', lw=2)
ax.plot(time, res[2] - res[0], color='grey', lw=2)
ax.axvline(0, color='grey', ls=':')
ax.set_xlim([-.200, .800])

## -------------- ssep
with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-04_17-38-41_subject-test_subject.h5', 'r') as h:
    eeg = h.eeg
    markers = h.markers
eeg_time = eeg.index.values
eeg = eeg.iloc[:, :16].values

downsamp = 1
eeg = eeg[::downsamp]
eeg_time = eeg_time[::downsamp]
fs = 500 / downsamp

eeg = filter_eeg(eeg, fs=fs, lo=100, hi=5, notch=60)
#eeg[:,:12] -= eeg[:,:12].mean(axis=1, keepdims=True) # avg reref
eeg[:,:12] -= eeg[:,4][..., None] # cz ref
#eeg[:,:12] -= eeg[:,[1,4,7]].mean(axis=-1)[..., None] # cz/fz/pz ref
#eeg[:,:12] -= eeg[:,5][..., None] # c4 ref

ssep = eeg[:,14]
ssep = filter_eeg(ssep, fs=fs, lo=fs/2-0.1, hi=25, notch=60)
ecg = eeg[:,13]

eeg_time = np.linspace(eeg_time[0], eeg_time[-1], len(eeg_time))

'''
fig, ax = pl.subplots()
ax.plot(eeg_time, ssep)
ax.plot(eeg_time, ecg-200, color='red')
for t,m in markers.iterrows():
    ax.axvline(t, color='grey', ls=':')
    ax.text(t, 1.01, m.text,
            fontsize=8,
            transform=blend(ax.transData, ax.transAxes))
'''


def t2i(t):
    return np.argmin(np.abs(eeg_time-t))

# first round
t0 = 4870321
t1 = 4870658
height = 200

# highest intensity round
#t0 = 4870678
#t1 = 4870818
#height = 2000

# lowest intensity (final) round
#t0 = 4870825
#t1 = 4870995
#height = 200

i0 = t2i(t0)
i1 = t2i(t1)
eeg_time = eeg_time[i0:i1]
ssep = ssep[i0:i1]

onset = (ssep[1:]>height) & (ssep[:-1]<=height)

onset_idx = np.arange(len(ssep)-1)[onset] +1
onset_idx = onset_idx[onset_idx > 100] # so there's pad before it
onset_idx -= int(0.008 * fs)# this is a manual pullback because i have to detect peaks but the pulse started before those peaks, visually verify its right
hz = len(onset_idx) / (eeg_time[-1] - eeg_time[0])
print(f'{len(onset_idx)} in {eeg_time[-1]-eeg_time[0]:0.1f}secs suggests ~{hz:0.1f}hz')
print('if thats very wrong, youre detecting pulses incorrectly')

fig, ax = pl.subplots()
ax.plot(eeg_time, ssep, color='k')
ax.vlines(eeg_time[onset_idx], -500, height, color='grey')

##
shuf_idx = np.random.choice(np.arange(onset_idx.min(), onset_idx.max()), size=onset_idx.shape)

pad0 = 0.080
pad1= 0.200

sigs = []
for idx in onset_idx:
    sig = eeg[idx-int(fs*pad0) : idx+int(fs*pad1), :]
    sigs.append(sig)
sigs = np.array(sigs)

sigs_shuf = []
for idx in shuf_idx:
    sig = eeg[idx-int(fs*pad0) : idx+int(fs*pad1), :]
    sigs_shuf.append(sig)
sigs_shif = np.array(sigs_shuf)

fxn = np.mean
mean = fxn(sigs,axis=0)
sd = np.std(sigs, axis=0) / np.sqrt(len(sigs))
mean_shuf = fxn(sigs_shuf,axis=0)
sd_shuf = np.std(sigs_shuf, axis=0) / np.sqrt(len(sigs_shuf))

from fus_anes.constants import MONTAGE # NOTE THIS WILL BE WRONG SINCE CHANGED

fig, axs = pl.subplots(3,4, sharex=True, sharey=True)
axs = axs.ravel()
time = (np.arange(len(mean)) - (fs*pad0)) / fs
for chan in range(0, 12):
    ax = axs[chan]
    ax.plot(time, mean[:, chan], color='blue')
    ax.fill_between(time, mean[:, chan]-sd[:,chan], mean[:,chan]+sd[:,chan], color='blue', alpha=0.15, lw=0)

    ax.plot(time, mean_shuf[:, chan], color='grey')
    ax.fill_between(time, mean_shuf[:, chan]-sd_shuf[:,chan], mean_shuf[:,chan]+sd_shuf[:,chan], color='k', alpha=0.1, lw=0)
    ax.set_title(MONTAGE[chan], fontsize=10)
    ax.axvline(0, color='grey', ls=':')





## ----------squeeze
from fus_anes.constants import MONTAGE
with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-07_18-48-25_subject-p005.h5', 'r') as h:
    eeg = h.eeg
    sq = h.squeeze
eeg_time = eeg.index.values
eeg = eeg.iloc[:, :18].values
t = eeg_time - eeg_time[0]
fs = 500
eeg = filter_eeg(eeg, fs=fs, lo=20, hi=0.1, notch=60)
switch = eeg[:,17]
fig,ax = pl.subplots(figsize=(8,1.5), gridspec_kw=dict(bottom=0.2))
ax.plot(t,switch)
sq = sq[sq.event.str.endswith('mp3')]
for t in sq.onset_ts.values:
    ax.axvline(t-eeg_time[0]+1.4, color='grey') # 1.4 just for clip duration
ax.set_xlim([37, 63])
ax.set_ylim([-6077136.377254958, 1815696.37])


## fancy chirp
import mne
import pandas as pd

with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-07_18-48-25_subject-p005.h5', 'r') as h:
    eeg = h.eeg
    chirp = h.chirp
eeg_time = eeg.index.values
data = eeg.iloc[:,:14].values.T * 1e-6
sfreq = 500.0  # Hz
from fus_anes.constants import MONTAGE
channel_names = MONTAGE[:14]
def t2i(t):
    return np.argmin(np.abs(eeg_time - t))
onsets = chirp[chirp.event=='c'].onset_ts.values
onsets = [t2i(t) for t in onsets]

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# Optional: set montage for scalp plotting
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

raw.set_eeg_reference('average')

raw.notch_filter(freqs=60., fir_design='firwin')
raw.filter(l_freq=1., h_freq=80., fir_design='firwin')

events = np.array([[int(idx), 0, 1] for idx in onsets])
event_id = dict(chirp=1)

epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.2, tmax=0.8,  # 200 ms baseline + 800 ms post-stim
                    baseline=(-0.2, 0),
                    detrend=1, preload=True)

#epochs.pick_channels(['Cz', 'Fz', 'Pz'])  # or just use all: comment this out
#epochs.pick_channels(['P7', 'Oz', 'P8'])

frequencies = np.linspace(25, 55, 30)
n_cycles = frequencies / 2.0  # higher freqs need more cycles

power, itc = mne.time_frequency.tfr_morlet(
    epochs, freqs=frequencies, n_cycles=n_cycles,
    use_fft=True, return_itc=True, decim=2, n_jobs=1)

power.plot(picks='Fz', baseline=(-0.1, 0), mode='logratio',
           title='Evoked power')

# Inter-trial coherence (phase-locking)
itc.plot(picks='Fz', title='ITC')
#itc.plot(picks='C4', title='ITC')

# ----------------------
# 7. Optional: Plot frequency tracking curve
# ----------------------

# Average ITC during the 0–0.5s chirp period
itc_data = itc.copy().crop(tmin=0.0, tmax=0.5).data  # shape: (n_channels, n_freqs, n_times)
mean_itc = itc_data[0].mean(axis=1)  # average over time, channel 0

pl.figure()
pl.plot(frequencies, mean_itc, marker='o')
pl.xlabel('Frequency (Hz)')
pl.ylabel('Mean ITC (0–0.5 s)')
pl.title('Cortical tracking of chirp')
pl.grid(True)



## oddball ERPs with MNE
import numpy as np
import mne

# --- 1. Load your data and events ---
with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-07_18-48-25_subject-p005.h5', 'r') as h:
    eeg = h.eeg
    ob = h.oddball

eeg_time = eeg.index.values
data = eeg.iloc[:,:14].values.T * 1e-6
sfreq = 500.0
#from fus_anes.constants import MONTAGE
channel_names = MONTAGE[:14]
def t2i(t):
    return np.argmin(np.abs(eeg_time - t))
montage = mne.channels.make_standard_montage('standard_1020')

onsets = ob[ob.event=='s'].onset_ts.values
events_standard = [t2i(t) for t in onsets]
onsets = ob[ob.event=='d'].onset_ts.values
events_deviant = [t2i(t) for t in onsets]

# Combine into one events array: [sample, 0, event_id]
events = np.concatenate([
    np.column_stack((events_standard, np.zeros_like(events_standard), np.ones_like(events_standard))),
    np.column_stack((events_deviant, np.zeros_like(events_deviant), np.full_like(events_deviant, 2)))
])
events = events.astype(int)
event_id = {'standard': 1, 'deviant': 2}

# --- 2. Create MNE Raw object ---

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.set_montage(montage)
#raw.set_eeg_reference('average')
raw.set_eeg_reference(['M1', 'M2'])


# Filter data (optional but recommended)
raw.notch_filter(freqs=60., fir_design='firwin')
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')

# --- 3. Epoch data around events ---

epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
                    detrend=1, preload=True)

# --- 4. Average ERPs for standards and deviants ---

evoked_standard = epochs['standard'].average()
evoked_deviant = epochs['deviant'].average()

# --- 5. Plot ERPs at Cz (classic auditory oddball site) ---

fig, ax = pl.subplots(figsize=(8, 5))
times = evoked_standard.times * 1000  # convert to ms

ax.plot(times, evoked_standard.data[evoked_standard.ch_names.index('Cz')]*1e6, label='Standard')
ax.plot(times, evoked_deviant.data[evoked_deviant.ch_names.index('Cz')]*1e6, label='Deviant')
ax.plot(times, (evoked_deviant.data - evoked_standard.data)[evoked_standard.ch_names.index('Cz')]*1e6,
        label='Deviant - Standard', linestyle='--', color='k')

ax.axvline(0, color='grey', linestyle='--')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title('Auditory Oddball ERPs at Cz')
ax.legend()
ax.grid(True)

# topo map
# Compute peak-to-peak amplitude per channel in a post-stimulus window (e.g., 20-60 ms)
tmin_pp, tmax_pp = 0.100, 0.300  # in seconds
evoked_s_crop = evoked_standard.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
evoked_d_crop = evoked_deviant.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
ptp_s_amplitudes = evoked_s_crop.data.ptp(axis=1) * 1e6  # Convert to µV
ptp_d_amplitudes = evoked_d_crop.data.ptp(axis=1) * 1e6  # Convert to µV
dif = ptp_d_amplitudes - ptp_s_amplitudes

# Plot the topomap of these amplitudes
fig_topo, ax_topo = pl.subplots(1, 1)
mne.viz.plot_topomap(ptp_d_amplitudes, evoked_deviant.info, axes=ax_topo,
                     show=True, cmap='Reds', contours=0)


## MNE ssep
import numpy as np
import mne
from fus_anes.constants import MONTAGE

# 1. Load your data and events
with pd.HDFStore('/Users/bdd/data/fus_anes/2025-07-07_18-48-25_subject-p005.h5', 'r') as h:
    eeg = h.eeg
    markers = h.markers
eeg_time = eeg.index.values
eeg_time = np.linspace(eeg_time[0], eeg_time[-1], len(eeg_time))
def t2i(t):
    return np.argmin(np.abs(eeg_time-t))
eeg = eeg.iloc[:, :16].values
data = eeg[:,:14].T * 1e-6

fs = 500.0
channel_names = MONTAGE[:14]
montage = mne.channels.make_standard_montage('standard_1020')

# Event onsets in samples (stimulus triggers)
ssep = eeg[:,14]
ssep = filter_eeg(ssep, fs=fs, lo=fs/2-0.1, hi=60, notch=60)
# first round
t0 = markers.iloc[4].t
t1 = markers.iloc[5].t
#t0 = markers.iloc[13].t
#t1 = markers.iloc[14].t
#t0 = markers.iloc[16].t
#t1 = markers.iloc[17].t

height = 50
i0 = t2i(t0)
i1 = t2i(t1)
eeg_time = eeg_time[i0:i1]
data = data[:,i0:i1]
ssep = ssep[i0:i1]
onset = (ssep[1:]>height) & (ssep[:-1]<=height)
onset_idx = np.arange(len(ssep)-1)[onset] +1
onset_idx = onset_idx[onset_idx > 100] # so there's pad before it
onset_idx -= int(0.002 * fs)# this is a manual pullback because i have to detect peaks but the pulse started before those peaks, visually verify its right

hz = len(onset_idx) / (eeg_time[-1] - eeg_time[0])
print(f'{len(onset_idx)} in {eeg_time[-1]-eeg_time[0]:0.1f}secs suggests ~{hz:0.1f}hz')
print('if thats very wrong, youre detecting pulses incorrectly')
fig, ax = pl.subplots()
ax.plot(eeg_time, ssep, color='k')
ax.vlines(eeg_time[onset_idx], -500, height, color='grey')

events_samples = onset_idx

# Build MNE events array: [sample, 0, event_id]
events = np.column_stack((events_samples, np.zeros_like(events_samples), np.ones_like(events_samples))).astype(int)

# 2. Create Raw object
info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.set_montage(montage)
raw.set_eeg_reference(['M1', 'M2'])


# 3. Filter data to typical SSEP band (e.g., 1-100 Hz)
raw.notch_filter(freqs=60., fir_design='firwin')
raw.filter(l_freq=1., h_freq=100., fir_design='firwin')

# 4. Re-reference - e.g., linked mastoids or average reference
raw.set_eeg_reference('average')  # or
#raw.set_eeg_reference(['M1', 'M2'])  # if mastoids available

# 5. Epoch data around stim pulse (e.g., -50 ms to +200 ms)
epochs = mne.Epochs(raw, events, event_id=1,
                    tmin=-0.05, tmax=0.2,
                    baseline=(-0.05, 0),
                    preload=True)

# 6. Average epochs to get SSEP evoked response
evoked = epochs.average()

# 7. Plot evoked waveforms for C3, C4, and P7 (ipsilateral and contralateral)
fig, ax = pl.subplots(figsize=(8, 5))
times_ms = evoked.times * 1000

#for ch_name in ['C3', 'C4', 'P7', 'P8', 'F3', 'F4', 'Oz']:
cols = ['maroon', 'lightgrey']
for ch_name, col in zip(['C4', 'Oz'], cols):
    ch_idx = evoked.ch_names.index(ch_name)
    ax.plot(times_ms, evoked.data[ch_idx] * 1e6, label=ch_name, color=col, lw=2)

ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title('Median Nerve SSEP')
ax.legend()
ax.grid(True)

# topo map
# Compute peak-to-peak amplitude per channel in a post-stimulus window (e.g., 20-60 ms)
tmin_pp, tmax_pp = 0.020, 0.050  # in seconds
#tmin_pp, tmax_pp = 0.050, 0.075  # in seconds
evoked_crop = evoked.copy().crop(tmin=tmin_pp, tmax=tmax_pp)
ptp_amplitudes = evoked_crop.data.ptp(axis=1) * 1e6  # Convert to µV

# Plot the topomap of these amplitudes
fig_topo, ax_topo = pl.subplots(1, 1)
mne.viz.plot_topomap(ptp_amplitudes, evoked.info, axes=ax_topo,
                     show=True, cmap='Reds', contours=0)
ax_topo.set_title(f'Peak-to-peak SSEP (µV) {int(tmin_pp*1000)}–{int(tmax_pp*1000)} ms')

##
