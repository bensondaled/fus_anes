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

order = [
        '2025-07-23_12-05-45_subject-b001',
        '2025-08-05_11-52-41_subject-b001',
        '2025-08-29_08-54-34_subject-b003',
        '2025-07-25_08-38-29_subject-b003',
        '2025-07-30_merge_subject-b004',
        '2025-08-12_09-11-34_subject-b004',
        ]

fig, axs = pl.subplots(6, len(ant), figsize=(15,9),
                       sharey='row', sharex=True,
                       gridspec_kw=dict(wspace=1.0))
for idx, name in enumerate(order):
    res = ant[name]

    ax_col = axs[:, idx]

    c,a,p = res.T
    up = np.diff(c, prepend=-1) >= 0
    down = ~up
    
    for ki,keep in enumerate([up, down]):
        mkr = ['>','<'][ki]

        c_ = c[keep]
        a_ = a[keep]
        p_ = p[keep]
        ratio = a_ / p_

        col = np.arange(len(c_))
        kw = dict(c=col, s=40, marker=mkr, cmap=pl.cm.turbo)

        ax = ax_col[3*ki + 0]
        ax.scatter(c_, a_, **kw)
        ax.set_ylabel('anterior')

        ax = ax_col[3*ki + 1]
        ax.scatter(c_ ,p_, **kw)
        ax.set_ylabel('posterior')

        ax = ax_col[3*ki + 2]
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

##
