##
import pandas as pd
import numpy as np
import json

with open('/Users/bdd/data/fus_anes/grps.json', 'r') as f:
    grps = json.loads(f.read())
def parse(s):
    subj = s[-4:]
    f = s[5:10]
    return f'{subj}_{f}'
grps = {parse(s):g for s,g in grps.items()}

sleep_data = pd.read_csv('/Users/bdd/data/fus_anes/All_manual_sleep_stats.csv')
sleep_data['grp'] = sleep_data.iloc[:,0].map(grps)
sleep_data['subj'] = sleep_data.iloc[:,0].str.slice(0,4)

##
from scipy.stats import wilcoxon, ttest_rel
stattest = ttest_rel

slp_types = ['S1','S2','S3']
metrics = ['%TST','min']

fig, axs = pl.subplots(len(metrics),len(slp_types),
                       sharey='row',
                       sharex=True,
                       squeeze=False,
                       gridspec_kw=dict(hspace=0.7, wspace=0.7))

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
        ax.set_xticklabels(['Sham','CM'])

for ax in axs.ravel():
    ax.tick_params(length=5, width=0.5)
    for spine in ax.spines:
        ax.spines[spine].set_linewidth(0.5)
##
