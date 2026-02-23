##
import pandas as pd
import numpy as np
import json
import os
import h5py
from datetime import datetime, timedelta

with open('/Users/bdd/data/fus_anes/grps.json', 'r') as f:
    grps = json.loads(f.read())
def parse(s):
    subj = s[-4:]
    f = s[5:10]
    return f'{subj}_{f}'
def parsedt(d, add=0, keep_word_subj=False):
    subj = d.split('_')[-1]
    d = d[:d.index('_')]
    date_obj = datetime.strptime(d, "%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=add)
    new_date_str = next_day_obj.strftime("%Y-%m-%d")
    if keep_word_subj:
        return f'{new_date_str}_subject-{subj}'
    else:
        return f'{new_date_str}_{subj}'
grps = {parsedt(g):v for g,v in grps.items()}
grps2 = {parse(s):g for s,g in grps.items()}
grp_dt = {g:parsedt(g, add=1) for g,v in grps.items()} # us session: anesthesia session

##
csv = '/Users/bdd/data/fus_anes/sleep/All_mean_durations_of_sleep_stages.csv'
sleep_data = pd.read_csv(csv)
sleep_data['grp'] = sleep_data.iloc[:,0].map(grps2)
sleep_data['subj'] = sleep_data.iloc[:,0].str.slice(0,4)
sleep_data = sleep_data.rename(columns={sleep_data.columns[0]: 'session_id'})
wide = sleep_data.pivot(index='subj', columns='grp')
wide.columns = [f"{var}_grp{grp}" for var, grp in wide.columns]

# TEMP
#sadict = {0:'sham',1:'active'}
#wide.columns = [f"{var}_{sadict[grp]}" for var, grp in wide.columns]

var = 'N2'
v0 = wide[f'{var}_grp0']
v1 = wide[f'{var}_grp1']

v0[np.isnan(v0)] = 0 # nan means none happened
v1[np.isnan(v1)] = 0

n2_changes = (v1 - v0).to_dict()

##
order = [
        ['2025-08-29_08-54-34_subject-b003',
        '2025-07-25_08-38-29_subject-b003',],

        ['2025-07-30_merge_subject-b004',
        '2025-08-12_09-11-34_subject-b004',],
        
        [
        '2025-09-05_08-10-33_subject-b008',
        '2025-09-19_07-52-47_subject-b008',
        ],
        
        [
        '2025-10-03_07-38-36_subject-b006',
        '2025-09-12_merge_subject-b006',],
        
        [
        '2025-09-17_07-57-44_subject-b002',
        '2025-09-23_07-51-59_subject-b002',],
        
        [
        '2025-10-22_07-51-53_subject-b007',
        '2025-10-08_07-45-31_subject-b007',
        ],
        
        [
        '2025-10-16_08-04-53_subject-b010',
        '2025-11-05_merge_subject-b010',
        ],
        
        [
        '2025-10-29_07-49-12_subject-b013',
        '2025-11-12_07-45-42_subject-b013',
        ],
        ]
ant = {}
processed_path = '/Users/bdd/data/fus_anes/intermediate/processed.h5'
with h5py.File(processed_path, 'r') as h:
    for name in np.ravel(order):
        ce = np.array(h[f'{name}_ce']).copy()
        cprop = np.array(h[f'{name}_cprop']).copy()
        phase_info = np.array(h[f'{name}_phases']).copy()
        spect_ds = h[f'{name}_spect']
        spect = np.array(spect_ds).copy()
        channels = spect_ds.attrs['channels']
        sp_f = spect_ds.attrs['freq']
        ant[name] = [ce, cprop, spect, channels, sp_f, phase_info]

changes = {}
peak_ants = {}
for idx, names in enumerate(order):
    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        ce, cprop, spect, channels, sp_f, phase_info = ant[name]

        ph_idx, ph_time, ph_lab = phase_info

        keep = np.arange(len(ce)) >= phase_info[0][0] # the true beginning of level-0
        keep = keep & (np.arange(len(ce)) <= ph_idx[np.argmax(ph_lab)+1])
        ce = ce[keep]
        cprop = cprop[keep]
        spect = spect[...,keep]

        _pq = ce
        is_alpha = (sp_f>=8) & (sp_f<=17)
        is_ant = np.isin(channels, ['F3', 'Fz', 'FCz', 'F4'])
        is_post = np.isin(channels, ['P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'])

        spect_ant = np.nanmean(spect[is_ant], axis=0)
        spect_post = np.nanmean(spect[is_post], axis=0)
        ant_alpha = np.nanmean(spect_ant[is_alpha], axis=0)
        post_alpha = np.nanmean(spect_post[is_alpha], axis=0)

        ap_ratio = ant_alpha / post_alpha

        #hi,lo = np.nanpercentile(ap_ratio, [95,5])
        #change = hi/lo
        #changes[name] = change
        peak_ants[name] = np.nanmax(ap_ratio) - np.nanmin(ap_ratio) # with or wirthout the subtract min

def reparse(n):
    date = n.split('_')[0]
    subj = n.split('-')[-1]
    return f'{date}_{subj}'
peak_ants = {reparse(c):v for c,v in peak_ants.items()}
pants = pd.DataFrame()
pants['session_id'] = list(peak_ants.keys())
pants['session_id_yest'] = [parsedt(i, add=-1, keep_word_subj=True) for i in pants.session_id.values]
pants['pant'] = list(peak_ants.values())
pants['grp'] = pants.session_id_yest.map(grps)
pants['subj'] = pants.session_id.str.slice(-4,None)

wide = pants.pivot(index='subj', columns='grp')
wide.columns = [f"{var}_grp{grp}" for var, grp in wide.columns]
var = 'pant'
v0 = wide[f'{var}_grp0']
v1 = wide[f'{var}_grp1']
pant_changes = (v1 - v0).to_dict()
##

assert list(n2_changes.keys()) == list(pant_changes.keys())
agg = pd.DataFrame()
agg['subj'] = list(n2_changes.keys())
agg['n2_change'] = list(n2_changes.values())
agg['pant_change'] = list(pant_changes.values())

# -- end place for basic comparison
##
csv = '/Users/bdd/data/fus_anes/sleep/prepost_All_PSDs_raw.csv'
psd_data = pd.read_csv(csv, index_col=0)
psd_data['grp'] = psd_data['Session'].map(grps2)
psd_data['subj'] = psd_data['Session'].str.slice(0,4)
data_cols = psd_data.columns[:-5]
freqs = np.array(data_cols).astype(float)

psd_res = {}
for subj in psd_data.subj.unique():
    psd_res[subj] = {}
    for cond in [0,1]:
        dat_pre = psd_data[(psd_data.subj==subj) & (psd_data.grp==cond) & (psd_data['Pre/Post']=='pre')]
        dat_post = psd_data[(psd_data.subj==subj) & (psd_data.grp==cond) & (psd_data['Pre/Post']=='post')]
        assert len(dat_pre) == 1
        assert len(dat_post) == 1
        pre = dat_pre[data_cols].values[0]
        post = dat_post[data_cols].values[0]
        dif = post - pre
        psd_res[subj][cond] = dif

## peak ants vs psds
fig, axs = pl.subplots(2, 8, sharex='row', sharey='row')
for idx, subj in enumerate(psd_res.keys()):
    sham_psd = psd_res[subj][0]
    active_psd = psd_res[subj][1]
    sham_pant = pants[(pants.subj==subj) & (pants.grp==0)].pant.values[0]
    active_pant = pants[(pants.subj==subj) & (pants.grp==1)].pant.values[0]

    ax = axs[0, idx]
    ax.plot([0,1], [sham_pant, active_pant])
    
    ax = axs[1, idx]
    ax.plot(freqs, sham_psd, color='grey')
    ax.plot(freqs, active_psd, color='steelblue')

##
