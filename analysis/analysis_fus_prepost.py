##
import pandas as pd
import numpy as np
import mne
import json, os
from fus_anes.constants import MONTAGE as channel_names
from util import mts, filter_eeg, nanpow2db, mts_mne

session_paths = [
    '/Users/bdd/data/fus_anes/2025-09-16_07-48-02_subject-b002.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-09-22_07-58-44_subject-b002.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-07-24_08-38-41_subject-b003.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-08-28_08-50-10_subject-b003.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-07-29_08-07-02_subject-b004.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-08-11_07-54-24_subject-b004.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-09-11_07-42-12_subject-b006.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-10-02_07-40-09_subject-b006.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-10-07_07-51-10_subject-b007.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-10-21_07-51-24_subject-b007.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-09-04_08-06-39_subject-b008.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-09-18_07-47-23_subject-b008.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-10-15_07-48-20_subject-b010.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-11-04_07-47-36_subject-b010.h5', # u/s

    '/Users/bdd/data/fus_anes/2025-10-28_07-43-57_subject-b013.h5', # u/s
    '/Users/bdd/data/fus_anes/2025-11-11_07-41-58_subject-b013.h5', # u/s
    ]

with open('/Users/bdd/data/fus_anes/grps.json', 'r') as f:
    grps = json.load(f)


##
agg = []

for session_path in session_paths:
    name = os.path.splitext(os.path.split(session_path)[-1])[0]

    with pd.HDFStore(session_path, 'r') as h:
        sconfig = h.config
        eeg = h.eeg
        bl_eyes = h.bl_eyes
        markers = h.markers

    sconfig = json.loads(sconfig.iloc[0])

    if name == '2025-09-16_07-48-02_subject-b002':
        eeg = eeg.iloc[:-25000] # time reset issue and unimportant segment at end to cut
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

    # Format into MNE
    ch_types = [{'ssep':'stim', 'gripswitch':'stim', 'ecg':'ecg'}.get(c, 'eeg') for c in channel_names]
    info = mne.create_info(ch_names=channel_names,
                           sfreq=fs,
                           ch_types=ch_types)
    eeg = mne.io.RawArray(eeg_raw.T.copy(), info)
    eeg.set_montage(mne.channels.make_standard_montage('standard_1020'))

    def ch_name_to_idx(name):
        all_channel_names = eeg.info['ch_names']
        bad_channel_names = eeg.info['bads']
        good_channel_names = [ch for ch in all_channel_names if ch not in bad_channel_names]
        if isinstance(name, (str,)):
            if name in good_channel_names:
                return good_channel_names.index(name) 
            else:
                return None
        elif isinstance(name, (list, np.ndarray)):
            return [ch_name_to_idx(n) for n in name if ch_name_to_idx(n) is not None]


    # Preprocessing
    eeg = eeg.notch_filter(freqs=[60, 120, 180, 200, 240], method='fir', notch_widths=2.0, picks=None)
    eeg = eeg.filter(l_freq=0.1, h_freq=58, fir_design='firwin', picks='eeg')
    eeg = eeg.set_eeg_reference('average', projection=False)  # default unless analyses undo it

    # spect
    s_win_size = 20.0 # secs
    eeg_spect = eeg.copy().pick('eeg')
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
    delta = (sp_f>=0.8) & (sp_f<4)
    theta = (sp_f>=4) & (sp_f<8)
    alpha = (sp_f>=8) & (sp_f<18)
    beta = (sp_f>=18) & (sp_f<30)
    bands = dict(delta=delta, theta=theta, alpha=alpha, beta=beta)

    def spect_t2i(t):
        if isinstance(t, (list, np.ndarray)):
            return np.array([spect_t2i(x) for x in t])
        else:
            return np.argmin(np.abs(t - sp_t))

    # plot it
    mtxt = markers.text.str.strip()
    t0s = markers[mtxt=='squeeze start'].t.values
    t0a,t0b = t0s[0],t0s[-1]
    t1s = markers[mtxt=='chirp complete'].t.values
    t1a,t1b = t1s[0],t1s[-1]

    i0a, i1a = spect_t2i([t0a, t1a])
    i0b, i1b = spect_t2i([t0b, t1b])

    name_frontal = ['F3','Fz','F4']
    name_posterior = ['Oz','P3','P4']
    ch_frontal = ch_name_to_idx(name_frontal)
    ch_posterior = ch_name_to_idx(name_posterior)

    agg.append([name, ch_frontal, ch_posterior, spect, bands, i0a, i1a, i0b, i1b])


##
#fig, axs = pl.subplots(2, len(bands), sharey=True, sharex=True)
#cols = ['blue', 'red']
results = []

for sesh in agg:
    name, ch_frontal, ch_posterior, spect, bands, i0a, i1a, i0b, i1b = sesh
    subj = name.split('-')[-1]
    for fpidx, (fplab, fp) in enumerate(zip(['frontal','posterior'], [ch_frontal, ch_posterior])):
        for bidx, (bandlab, band) in enumerate(bands.items()):
            x = np.nanmean(spect[fp], axis=0)
            x = np.nanmean(x[band], axis=0)
            pre_tus = np.nanmean(x[i0a:i1a])
            post_tus = np.nanmean(x[i0b:i1b])

            #ax = axs[fpidx, bidx]
            #ax.plot([0,1], [pre_tus,post_tus], label=name[-10:], color=cols[grps[name]])
            #ax.set_title(f'{fplab}-{bandlab}')
            results.append(dict(subj=subj, val=pre_tus, pp='pre', band=bandlab, fp=fplab, cond=grps[name]))
            results.append(dict(subj=subj, val=post_tus, pp='post', band=bandlab, fp=fplab, cond=grps[name]))
results = pd.DataFrame(results)

##
# stats
import statsmodels.formula.api as smf

df = results[(results.fp=='frontal') & (results.band=='delta')]
#df = results[(results.fp=='frontal') & (results.band=='delta') & (results.pp=='post')]

df['cond'] = df['cond'].astype('category')
df['time'] = df['pp'].astype('category')
model = smf.mixedlm("val ~ cond * time",
                                        df,
                                        groups=df['subj'])
mod_fit = model.fit()
print(mod_fit.summary())

##
