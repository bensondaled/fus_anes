## 
import h5py
import numpy as np
import pandas as pd
import os
from util import fit_sigmoid2
from matplotlib.gridspec import GridSpec

anteriorization_path = '/Users/bdd/data/fus_anes/intermediate/anteriorization.h5'

order = [

        ['2025-07-23_12-05-45_subject-b001',
        '2025-08-05_11-52-41_subject-b001',],

        ['2025-08-29_08-54-34_subject-b003',
        '2025-07-25_08-38-29_subject-b003',],

        ['2025-07-30_merge_subject-b004',
        '2025-08-12_09-11-34_subject-b004',],
        
        [
        #'2025-09-04_08-06-39_subject-b008', # u/s
        '2025-09-05_08-10-33_subject-b008',
        '2025-09-05_08-10-33_subject-b008',
         ],
        
        ]

##
ant = {}
with h5py.File(anteriorization_path, 'r') as h:
    print(list(h.keys()))
    for name in np.ravel(order):
        ce = np.array(h[f'{name}_ce']).copy()
        spect_ds = h[f'{name}_spect']
        spect = np.array(spect_ds).copy()
        channels = spect_ds.attrs['channels']
        sp_f = spect_ds.attrs['freq']
        ant[name] = [ce, spect, channels, sp_f]


## figure

gs = GridSpec(1, len(order)*3,
              right=0.98,
              left=0.1,
              top=0.9,
              bottom=0.25,
              wspace=0.05)
fig = pl.figure(figsize=(15,3.5))
all_axs = []
for idx, names in enumerate(order):

    ax_list = [
                fig.add_subplot(gs[0, idx*3]),
                fig.add_subplot(gs[0, idx*3+1]),
            ]
    all_axs.append(ax_list)

    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        ce, spect, channels, sp_f = ant[name]

        is_alpha = (sp_f>=8) & (sp_f<15)
        is_delta = (sp_f>=0.8) & (sp_f<4)

        is_ant = np.isin(channels, ['F3', 'Fz', 'FCz', 'F4'])
        is_post = np.isin(channels, ['P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'])
        spect_ant = np.nanmean(spect[is_ant], axis=0)
        spect_post = np.nanmean(spect[is_post], axis=0)
        spect_full = np.nanmean(spect, axis=0)

        ant_alpha = np.nanmean(spect_ant[is_alpha], axis=0)
        post_alpha = np.nanmean(spect_post[is_alpha], axis=0)
        total_alpha = np.nanmean(spect_full[is_alpha], axis=0)
        total_delta = np.nanmean(spect_full[is_delta], axis=0)
        total_power = np.nanmean(spect_full, axis=0)

        _to_plot = np.array([ce,
                            ant_alpha,
                            post_alpha,
                            total_alpha,
                            total_delta,
                            total_power,
                           ]).T

        # include only rising/falling prop levels
        if ce.max() > 3.1: # original protocol
            bins_category = 0
        else: # new protocol 2025-09-05
            bins_category = 1
        drop_starts = -(np.where(ce[::-1] > 2.99)[0][0])
        for direction, ax in zip([1, -1], ax_list):
            if direction == 1:
                to_plot = _to_plot[:drop_starts] # rising
            elif direction == -1:
                to_plot = _to_plot[drop_starts:] # falling
            
            # bin them
            do_bin = False
            if do_bin:
                to_plot = pd.DataFrame(to_plot)

                if direction == 1:
                    if bins_category == 0:
                        bins = [-0.01, #0 level
                                0.4, #0.8 level
                                1.2, #1.6 level
                                2.0, #2.4 level
                                2.8, #3.2 level
                                4.0, #3.2 level
                                ]
                    elif bins_category == 1:
                        bins = [-0.01, #0 level
                                0.25, #0.5 level
                                0.75, #1.0 level
                                1.25, #1.5 level
                                1.75, #2.0 level
                                2.25, #2.5 level
                                2.75, #3.0 level
                                3.25, #3.0 level
                                ]
                elif direction == -1:
                    if bins_category == 0:
                        bins = [-0.01,
                                0.8, #0.4 level
                                1.6, #1.2 level
                                2.4, #2.0 level
                                3.2, #2.8 level
                                ]
                    elif bins_category == 1:
                        bins = [-0.01,
                                0.9, #0.6 level
                                1.5, #1.2 level
                                2.1, #1.8 level
                                2.7, #2.4 level
                                ]
                to_plot['bin'] = pd.cut(to_plot.iloc[:,0], bins=bins)
                to_plot = to_plot.groupby('bin', as_index=False).mean()
                to_plot = to_plot.values[:,1:].astype(float)

                # or bin them agnostically
                #to_plot = np.array([np.nanmean(r, axis=0) for r in np.array_split(to_plot, 5, axis=0)])

            c,a,p,ta,td,tp = to_plot.T

            keep = ~(np.isnan(a))
            
            c_ = c[keep]
            a_ = a[keep]
            p_ = p[keep]
            ta_ = ta[keep]
            td_ = td[keep]
            tp_ = tp[keep]

            show = a_ / p_
            #show = a_
            #show = p_
            #show = a_ / ta_
            #show = ta_
            #show = tp_
            #show = td_
            #show = 10*np.log10(a_ / p_)

            kw = dict(s=15, marker='o', color=col, alpha=0.5, lw=0)
            
            ax.scatter(c_, show, **kw)
            
            do_fit = True
            if do_fit:
                xv, yv, (A,y0,ec50,B) = fit_sigmoid2(c_, show, return_params=True)
                ax.plot(xv, yv, color=col, lw=1.,
                    label=f'?{cond}\nA={A:0.1f}\ny0={y0:0.1f}\nx0={ec50:0.1f}',)


            ax.sharey(all_axs[0][0])

            ax.tick_params(rotation=90)
            ax.grid(True, lw=0.25)
            #ax.legend(fontsize=7)
            
            if ax is all_axs[0][0]:
                ax.set_ylabel(r'Ant:post $\alpha$ ratio')
                #ax.set_ylabel(r'anything')
            else:
                pl.setp(ax.get_yticklabels(), visible=False)

            if direction == 1:
                ax.set_title(name[-4:])
                ax.set_xlabel('Propofol conc.', labelpad=20)
                ax.set_xlim([0, 3.3])
                ax.set_xticks(np.arange(0, 3.2, 0.5))
            elif direction == -1:
                ax.set_xlim([3.3, 0])
                ax.set_xticks(np.arange(0, 3.2, 0.5)[::-1])
                ax.tick_params(axis='y', length=0)
                ax.spines['left'].set_visible(False)
##
