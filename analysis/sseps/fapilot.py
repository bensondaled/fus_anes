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
mode = 'chirp'

if mode == 'oddball':
    with pd.HDFStore('/Users/bdd/Desktop/2025-07-04_17-38-41_subject-test_subject.h5', 'r') as h:
        eeg = h.eeg
        ob = h.oddball

if mode == 'chirp':
    with pd.HDFStore('/Users/bdd/Desktop/2025-07-04_18-07-45_subject-test_subject.h5', 'r') as h:
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
eeg[:,:12] -= eeg[:,:12].mean(axis=1, keepdims=True) # avg reref

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

    choose_chan = [0,1,2,3,4,5,6,7,8]
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
with pd.HDFStore('/Users/bdd/Desktop/2025-07-04_17-38-41_subject-test_subject.h5', 'r') as h:
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

from fus_anes.constants import MONTAGE

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
with pd.HDFStore('/Users/bdd/Desktop/2025-07-04_18-15-38_subject-test_subject.h5', 'r') as h:
    eeg = h.eeg
    sq = h.squeeze
eeg_time = eeg.index.values
eeg = eeg.iloc[:, :16].values
fs = 500
eeg = filter_eeg(eeg, fs=fs)
switch = eeg[:,15]
fig,ax = pl.subplots()
ax.plot(eeg_time,switch)
sq = sq[sq.event.str.endswith('mp3')]
for t in sq.index.values:
    ax.axvline(t, color='k')
##
