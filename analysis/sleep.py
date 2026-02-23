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
        '/Users/bdd/data/fus_anes/sleep/Summary_bandpower_by_state2.csv',
        '/Users/bdd/data/fus_anes/sleep/All_Ferrarelli_spindle_stats.csv',
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
        ids = dat[0].index

        isnan = np.isnan(a) | np.isnan(b)
        pval = stattest(a[~isnan],b[~isnan]).pvalue
        
        for ai, bi, idi in zip(a,b,ids):
            ax.plot([0,1], [ai, bi], label=idi)
        ax.set_title(f'{column}')#, p={pval:0.2f}')
        ax.set_title(f'{column} p={pval:0.2f}')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Sham','Active'])
        #ax.legend()

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

## full traces stuff
#csv = '/Users/bdd/data/fus_anes/sleep/All_PSDs_by_state_normalized.csv'
csv = '/Users/bdd/data/fus_anes/sleep/All_PSDs_by_state_raw.csv'
#csv = '/Users/bdd/data/fus_anes/sleep/prepost_All_PSDs_raw.csv'
#csv = '/Users/bdd/data/fus_anes/sleep/prepost_All_PSDs_normalized.csv'

sleep_data = pd.read_csv(csv, index_col=0)
sleep_data['cond'] = sleep_data['Session'].map(grps)
sleep_data['subj'] = sleep_data['Session'].str.slice(0,4)
data_cols = sleep_data.columns[:-5]

if 'Wake/Sleep Category' not in sleep_data.columns:
    sleep_data['Wake/Sleep Category'] = sleep_data['Pre/Post']

sleep_data[data_cols] = 10 * np.log10(sleep_data[data_cols])
grouped = sleep_data.groupby(['Wake/Sleep Category', "cond"])
mean_df = grouped[data_cols].mean()
sem_df  = grouped[data_cols].sem()

x = np.array(data_cols).astype(float)
groups = sleep_data["Wake/Sleep Category"].unique()
fig, axs = pl.subplots(2, len(groups), sharex=True, sharey='row', squeeze=False)
for idx,g in enumerate(groups):
    hold = []
    for cond, condstr, color in zip([0, 1], ['sham', 'active'], ["tab:blue", "tab:orange"]):
        mean_curve = mean_df.loc[(g, cond)].values
        sem_curve  = sem_df.loc[(g, cond)].values
        
        ax = axs[0, idx]
        ax.plot(x, mean_curve, label=f"{condstr}", color=color, lw=0.5)
        ax.fill_between(
            x,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=color,
            alpha=0.3,
            lw=0,
        )
        ax.set_title(g)
        ax.legend()

        hold.append([cond,mean_curve])
    ax = axs[1, idx]
    ax.plot(x, hold[1][1] - hold[0][1], color='k')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('active-sham power')
axs[0,0].set_ylabel('Power (dB)')
axs[1,1].set_xlabel('Frequency (Hz)')
   
## per subj pre post diff
csv = '/Users/bdd/data/fus_anes/sleep/prepost_All_PSDs_raw.csv'
#csv = '/Users/bdd/data/fus_anes/sleep/prepost_All_PSDs_normalized.csv'

sleep_data = pd.read_csv(csv, index_col=0)
sleep_data['cond'] = sleep_data['Session'].map(grps)
sleep_data['subj'] = sleep_data['Session'].str.slice(0,4)
data_cols = sleep_data.columns[:-5]
sleep_data[data_cols] = 10 * np.log10(sleep_data[data_cols])
df = sleep_data

df_pre  = df[df["Pre/Post"] == "pre"]
df_post = df[df["Pre/Post"] == "post"]
idx_cols = ["subj", "cond"]
pre_data  = df_pre.set_index(idx_cols)[data_cols]
post_data = df_post.set_index(idx_cols)[data_cols]

common_idx = pre_data.index.intersection(post_data.index)
pre_data  = pre_data.loc[common_idx]
post_data = post_data.loc[common_idx]

diff = post_data - pre_data

mean_diff_by_cond = diff.groupby("cond").mean()
sem_diff_by_cond  = diff.groupby("cond").sem()

x = np.array(data_cols).astype(float)
fig, ax = pl.subplots()
for cond in [0, 1]:
    y = mean_diff_by_cond.loc[cond].values
    yerr = sem_diff_by_cond.loc[cond].values
    ax.plot(x, y, label={0:'sham',1:'active'}[cond])
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.3)

ax.legend()
ax.axhline(0, linestyle="--", linewidth=1, color='k')
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("Post − Pre power")
ax.set_title('within-subject post-pre difference,\nmean ± sem across subjects')

# 
q = np.array(data_cols).astype(float)
dc = data_cols[(q>=7) & (q<=10)]
indiv_diff = diff[dc].mean(axis=1)
s_wide = indiv_diff.unstack(level='cond')  
fig, ax = pl.subplots()
for subj, row in s_wide.iterrows():
    ax.plot([0,1], [row[0], row[1]], marker='o')
ax.set_xticks([0,1])
ax.set_xticklabels(['Sham','Active'])
ax.set_ylabel('Post-Pre 7-10Hz power')

## transition matrices
root = '/Users/bdd/data/fus_anes/sleep/transition_matrices'
dats = {}
subjs = []
for f in os.listdir(root):
    subj,date,*_ = f.split('_')
    dats[f'{subj}_{date}'] = pd.read_csv(os.path.join(root, f), index_col=0)
    subjs.append(subj)

diffs = []
for subj in np.unique(subjs):
    shams = [d for k,d in dats.items() if k.split('_')[0]==subj and grps[k]==0]
    actives = [d for k,d in dats.items() if k.split('_')[0]==subj and grps[k]==1]

    assert len(shams) == 1
    assert len(actives) == 1
    sham = shams[0]
    active = actives[0]

    dif = active.values - sham.values
    diffs.append(dif)

fig,axs = pl.subplots(2,4)
axs = axs.ravel()
for ax, dif in zip(axs, diffs):
    ax.imshow(dif)

fig, ax = pl.subplots(figsize=(11,9))
mean = np.mean(diffs,axis=0)
mean = pd.DataFrame(mean, index=sham.index, columns=sham.columns)
sns.heatmap(mean, cmap="viridis", ax=ax)
ax.set_title('Active-Sham transition matrix, mean over subjects')

##
