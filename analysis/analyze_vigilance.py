##
import json, os
import pandas as pd
import numpy as np

meta = {}

##

#data_file = '/Users/bdd/data/fus_anes/2025-07-25_vigilance_b003.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-07-30_vigilance_b004.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-08-12_vigilance_b004.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-09-05_vigilance_b008.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-09-19_vigilance_b008.txt'
#data_file = '/Users/bdd/data/fus_anes/2025-09-17_vigilance_b002.txt'
data_file = '/Users/bdd/data/fus_anes/2025-09-23_vigilance_b002.txt'

meta[data_file] = {}

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
meta[data_file]['pvt'] = []
fig, ax = pl.subplots(1, sharex=True)
for rep in range(n_reps):
    use = (pvt.rep == rep) & (pvt.note != 'early')
    rt_ = pvt[use].rt.values
    lab = f'{rep} - {1000*np.median(rt_):0.0f}'
    ax.hist(1000*rt_, histtype='step', label=lab, density=True)
    meta[data_file]['pvt'].append(rt_)
ax.set_xlabel('RT (ms)')
ax.legend()

# dsst
meta[data_file]['dsst'] = []
fig, axs = pl.subplots(1, 3,)
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

    meta[data_file]['dsst'].append(n_completed)

axs[0].set_title('N completed')
axs[1].set_title('% correct')
axs[1].set_ylim([0,100])
axs[2].set_title('RT (ms)')

## -- meta
fig, axs = pl.subplots(2,6,
                       sharex=True,
                       sharey='row')
for idx,(name, rdict) in enumerate(meta.items()):
    pvt = rdict['pvt']
    dsst = np.array(rdict['dsst']).astype(float)
    
    sem = lambda x: np.std(x)/np.sqrt(len(x))
    pvt_means = np.array([p.mean() for p in pvt])
    pvt_err = [sem(p) for p in pvt]

    #pvt_means /= pvt_means[0]

    ax = axs[0, idx]
    ax.errorbar(np.arange(3), pvt_means, )#yerr=pvt_err)

    ax.set_title(os.path.split(name)[-1], fontsize=8)

    ax = axs[1, idx]
    #dsst /= dsst[0]
    ax.plot(dsst)

##
