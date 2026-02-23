##
import pandas as pd
import numpy as np
import json
import h5py
import os
from datetime import datetime, timedelta

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
##
with open('/Users/bdd/data/fus_anes/grps.json', 'r') as f:
    grps = json.loads(f.read())
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
grps_us_to_condition = {parsedt(g):v for g,v in grps.items()}
grps_us_to_anes = {g:parsedt(g, add=1) for g,v in grps.items()} # us session: anesthesia session
grps_anes_to_us = {sesh:parsedt(sesh, add=-1) for sublist in order for sesh in sublist}

##
ant = {}
processed_path = '/Users/bdd/data/fus_anes/intermediate/processed.h5'
with h5py.File(processed_path, 'r') as h:
    for name in np.ravel(order):
        ce = np.array(h[f'{name}_ce']).copy()
        cprop = np.array(h[f'{name}_cprop']).copy()
        phase_info = np.array(h[f'{name}_phases']).copy()
        spect_ds = h[f'{name}_spect']
        spect_time = spect_ds.attrs['time']
        spect = np.array(spect_ds).copy()
        channels = spect_ds.attrs['channels']
        sp_f = spect_ds.attrs['freq']
        ant[name] = [ce, cprop, spect, channels, sp_f, phase_info, spect_time]

res = {}
for idx, names in enumerate(order):
    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        ce, cprop, spect, channels, sp_f, phase_info, spect_time = ant[name]

        ph_idx, ph_time, ph_lab = phase_info

        keep = np.arange(len(ce)) >= phase_info[0][0] # the true beginning of level-0
        #keep = keep & (np.arange(len(ce)) <= ph_idx[np.argmax(ph_lab)+1])
        ce = ce[keep]
        cprop = cprop[keep]
        spect = spect[...,keep]
        spect_time = spect_time[keep]
        
        spect_time -= spect_time[0]

        _pq = ce
        is_alpha = (sp_f>=8) & (sp_f<=17)
        is_ant = np.isin(channels, ['F3', 'Fz', 'FCz', 'F4'])
        is_post = np.isin(channels, ['P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'])

        spect_ant = np.nanmean(spect[is_ant], axis=0)
        spect_post = np.nanmean(spect[is_post], axis=0)
        ant_alpha = np.nanmean(spect_ant[is_alpha], axis=0)
        post_alpha = np.nanmean(spect_post[is_alpha], axis=0)
        
        key = {0:'sham', 1:'active'}[grps_us_to_condition[grps_anes_to_us[name]]]
        key = f'{grps_anes_to_us[name].replace('_subject','')}-{key}'
        res[key] = dict(anterior_alpha=ant_alpha, posterior_alpha=post_alpha, effect_site_concentration=ce, seconds=spect_time)
        print(ant_alpha.shape, post_alpha.shape, ce.shape, spect_time.shape)

with pd.ExcelWriter("/Users/bdd/Desktop/for_boris.xlsx", engine="xlsxwriter") as writer:
    for sheet_name, columns in res.items():
        df = pd.DataFrame(columns)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
##
