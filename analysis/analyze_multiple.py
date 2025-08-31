## 
import h5py
import numpy as np
import pandas as pd
import os
from util import fit_sigmoid2

anteriorization_path = '/Users/bdd/data/fus_anes/intermediate/anteriorization.h5'

##
ant = {}
with h5py.File(anteriorization_path, 'r') as h:
    for name in h:
        ant[name] = np.array(h[name])


## big full one
order = [
        '2025-07-23_12-05-45_subject-b001',

        '2025-08-04_08-48-05_subject-b001', # us
        '2025-08-05_11-52-41_subject-b001',

        '2025-07-24_08-38-41_subject-b003', # us
        '2025-07-25_08-38-29_subject-b003',

        '2025-08-28_08-50-10_subject-b003', # us
        '2025-08-29_08-54-34_subject-b003',

        '2025-07-29_08-07-02_subject-b004', # us
        '2025-07-30_merge_subject-b004',

        '2025-08-11_07-54-24_subject-b004', # us
        '2025-08-12_09-11-34_subject-b004',
        
        ]

fig, axs = pl.subplots(2, len(ant), figsize=(15,9),
                       sharey='row', sharex=True,
                       gridspec_kw=dict(wspace=1.0))
for idx, name in enumerate(order):
    res = ant[name]
    res = np.array([np.nanmean(r, axis=0) for r in np.array_split(res, 50, axis=0)])

    ax_col = axs[:, idx]

    c,a,p,delt = res.T
    up = np.diff(c, prepend=-1) >= 0
    down = ~up
    
    for ki,keep in enumerate([up, down]):
        if np.all(keep == False):
            continue
        
        mkr = ['>','<'][ki]

        c_ = c[keep]
        a_ = a[keep]
        p_ = p[keep]
        ratio = a_ / p_

        col = np.arange(len(c_))
        kw = dict(c=col, s=40, marker=mkr, cmap=pl.cm.turbo)
        
        '''
        ax = ax_col[3*ki + 0]
        ax.scatter(c_, a_, **kw)
        ax.set_ylabel('anterior')

        ax = ax_col[3*ki + 1]
        ax.scatter(c_ ,p_, **kw)
        ax.set_ylabel('posterior')
        '''

        ax = ax_col[ki]
        ax.scatter(c_, ratio, **kw)
        ax.set_ylabel('a/p ratio')

        xv, yv, (A,y0,_,_) = fit_sigmoid2(c_, ratio, return_params=True)
        ax.plot(xv, yv, color='grey', lw=0.5)
        ax.text(0.2, 0.8, f'A={A:0.1f}\ny0={y0:0.1f}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8,
                color='grey')


        #ax.set_ylim([0,4])
        #ax.set_xlim([-0.3,3.6])
        #ax.set_xticks(np.arange(0, 3.3, 0.5))
        #ax.grid(True)

    ax_col[0].set_title(name[-4:])

## small aesthetic one
order = [

        ['2025-07-23_12-05-45_subject-b001',
        '2025-08-05_11-52-41_subject-b001',],

        ['2025-08-29_08-54-34_subject-b003',
        '2025-07-25_08-38-29_subject-b003',],

        ['2025-07-30_merge_subject-b004',
        '2025-08-12_09-11-34_subject-b004',],
        
        ]

fig, axs = pl.subplots(1, len(order), figsize=(13,4),
                       sharey='row', sharex=True,
                       gridspec_kw=dict(wspace=1.0, bottom=0.25))
for idx, names in enumerate(order):

    ax = axs[idx]

    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        res = ant[name].copy()
        
        # include only rising/falling prop levels
        c = res.T[0]
        drop_starts = -(np.where(c[::-1] > 3.19)[0][0])
        res = res[:drop_starts] # rising
        #res = res[drop_starts:] # falling
        
        # bin them
        '''
        res = pd.DataFrame(res)
        bins = [-0.01, #0 level
                0.4, #0.8 level
                1.2, #1.6 level
                2.0, #2.4 level
                2.8, #3.2 level
                4.0, #3.2 level
                ]
        res['bin'] = pd.cut(res.iloc[:,0], bins=bins)
        res = res.groupby('bin', as_index=False).mean()
        res = res.values[:,1:].astype(float)

        # or bin them agnostically
        #res = np.array([np.nanmean(r, axis=0) for r in np.array_split(res, 5, axis=0)])
        '''

        c,a,p,delt = res.T

        keep = ~(np.isnan(a))
        
        c_ = c[keep]
        a_ = a[keep]
        p_ = p[keep]
        delt = delt[keep]
        show = a_ / p_
        #show = 10*np.log10(a_ / p_)

        kw = dict(s=15, marker='o', color=col, alpha=0.5, lw=0)
        
        ax.scatter(c_, show, **kw)
        ax.set_ylabel(r'Ant:post $\alpha$ ratio')

        xv, yv, (A,y0,ec50,B) = fit_sigmoid2(c_, show, return_params=True)

        rise_pt = ec50 + np.log(0.05/0.95)/B
        #ax.axvline(rise_pt, color=col, lw=2, ls=':')

        ax.plot(xv, yv, color=col, lw=1.5,
                label=f'?{cond}\nA={A:0.1f}\ny0={y0:0.1f}\nec50={ec50:0.1f}\nrise={rise_pt:0.1f}',)


        ax.set_xticks(np.arange(0, 3.2, 0.5))
        ax.grid(True, lw=0.25)
    
    ax.set_title(name[-4:])
    ax.tick_params(rotation=90)
    ax.legend(fontsize=7)
    ax.set_xlabel('Propofol conc.', labelpad=20)
##
