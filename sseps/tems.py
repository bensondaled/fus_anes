##
'''
tems device:
100Hz for 250ms
2 Hz

my head:
nasion to inion 36cm
ear to ear 33cm
1=black=cz
2=white=left C3
3=red=right C4
brown-ground-forehead
4=blue=back of neck
'''

from util import filter_eeg, mts, sliding_window
import mne 

path = 'fusanes/bd3.vhdr'
raw = mne.io.read_raw_brainvision(path)
ch_names = raw.ch_names
assert ch_names[:4] == ['bd2','bd3','bd4','bdaux']
data = raw.get_data()
data = data[:4,:].T
time = np.arange(data.shape[0])/500

keep = time<160
data = data[keep]
time = time[keep]

pl.plot(time, data)
##

eeg = filter_eeg(data[:,:-1], fs=500)
eeg = eeg - np.nanmedian(eeg, axis=1)[:,None]
stim = data[:,-1]

spects,stime,sfreq = mts(eeg, fs=500, window_size=5.0, window_step=2.0)
vmin,vmax = np.percentile(spects, [5,99])

fig,axs = pl.subplots(4,1,
                      sharex=True,
                      sharey=False)
for ax, spect in zip(axs[1:], spects):
    ax.pcolormesh(stime, sfreq, spect, cmap=pl.cm.rainbow,
                  vmin=vmin, vmax=vmax)
    ax.set_ylim([0,40])

ax = axs[0]
ax.plot(time, stim, color='k')
##

is_stim_time = time>65.9
stim_sig = stim[is_stim_time]
stim_time = time[is_stim_time]
roll = sliding_window(stim_sig, 20)
roll = np.nanmax(roll, axis=1)
roll -= np.nanmin(roll)
roll = np.diff(roll)
switch = roll > 0.2
on, = np.where(switch)
on_time = stim_time[on]

fig, ax = pl.subplots()
ax.plot(stim_time, stim_sig)
ax.vlines(on_time, color='grey', ymin=-5.08, ymax=-4.4)

## k good
eeg_stim = eeg[is_stim_time]
slices = [np.arange(ot-100, ot+151) for ot in on]
stims = np.take(stim_sig, slices)
eegs = np.take(eeg_stim, slices, axis=0)

fig, ax = pl.subplots()
mean_stim = np.mean(stims, axis=0)
#ax.plot(mean_stim, color='grey')
#ax = ax.twinx()
mean_eeg = np.mean(eegs, axis=0).T
sd_eeg = (np.std(eegs, axis=0) / np.sqrt(eegs.shape[0])).T
cols = ['maroon', 'steelblue', 'forestgreen']
for me, se, col in zip(mean_eeg, sd_eeg, cols):
    ax.plot(me, color=col)
    ax.fill_between(np.arange(len(me)), me-se, me+se, color=col, alpha=0.2, lw=0)
ax.axvline(100, color='k')

##
