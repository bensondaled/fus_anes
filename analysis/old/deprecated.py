
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
