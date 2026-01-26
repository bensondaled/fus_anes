##
import pandas as pd
import numpy as np
import json
import os
from scipy.stats import wilcoxon, ttest_rel
stattest = wilcoxon

with open('/Users/bdd/data/fus_anes/grps.json', 'r') as f:
    grps = json.loads(f.read())
def parse(s):
    subj = s[-4:]
    f = s[5:10]
    return f'{subj}_{f}'
grps = {parse(s):g for s,g in grps.items()}

## generic comparison
csvs = ['/Users/bdd/data/fus_anes/sleep/All_mean_durations_of_sleep_stages.csv',
        '/Users/bdd/data/fus_anes/sleep/All_number_of_episodes_by_sleep_stages.csv',
        '/Users/bdd/data/fus_anes/sleep/Summary_bandpower_by_state.csv',
        ]

for csv in csvs:
    fname = os.path.splitext(os.path.split(csv)[-1])[0]
    sleep_data = pd.read_csv(csv)
    sleep_data['grp'] = sleep_data.iloc[:,0].map(grps)
    sleep_data['subj'] = sleep_data.iloc[:,0].str.slice(0,4)

    var = [c for c in sleep_data.columns[1:] if c not in ['grp','subj']]

    fig, axs = pl.subplots(1, len(var), figsize=(15,4), 
                           gridspec_kw=dict(wspace=0.8))
    for column, ax in zip(var, axs):
        dat = sleep_data.groupby(['grp','subj'])[column].mean()
        assert np.all(dat[0].index == dat[1].index)

        a = dat[0].values
        b = dat[1].values

        isnan = np.isnan(a) | np.isnan(b)
        pval = stattest(a[~isnan],b[~isnan]).pvalue

        ax.plot([0,1], [a, b])
        ax.set_title(f'{column}')#, p={pval:0.2f}')
        ax.set_title(f'{column} p={pval:0.2f}')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Sham','Active'])

    fig.savefig(f'/Users/bdd/Desktop/{fname}.png')
    pl.close(fig)

## sleep types specifically
sleep_data = pd.read_csv('/Users/bdd/data/fus_anes/sleep/All_manual_sleep_stats.csv')
sleep_data['grp'] = sleep_data.iloc[:,0].map(grps)
sleep_data['subj'] = sleep_data.iloc[:,0].str.slice(0,4)

slp_types = ['S1','S2','S3']
metrics = ['%TST','min']

fig, axs = pl.subplots(len(metrics),len(slp_types),
                       sharey='row',
                       sharex=True,
                       squeeze=False,
                       figsize=(7,6),
                       gridspec_kw=dict(hspace=0.6, wspace=0.8))

for slti,slt in enumerate(slp_types):
    for mi,metric in enumerate(metrics):

        ax = axs[mi, slti]
        column = f'{slt}_{metric}'

        dat = sleep_data.groupby(['grp','subj'])[column].mean()
        assert np.all(dat[0].index == dat[1].index)

        a = dat[0].values
        b = dat[1].values
        pval = stattest(a,b).pvalue

        ax.plot([0,1], [a, b])
        ax.set_title(f'{column}')#, p={pval:0.2f}')
        ax.set_title(f'{column} p={pval:0.2f}')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Sham','Active'])

for ax in axs.ravel():
    ax.tick_params(length=5, width=0.5)
    for spine in ax.spines:
        ax.spines[spine].set_linewidth(0.5)
##
