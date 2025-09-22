##
import h5py
import numpy as np
import pandas as pd
import os
from util import fit_sigmoid2
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory as blend

processed_path = '/Users/bdd/data/fus_anes/intermediate/processed.h5'
prop_quantity = 'cce' # ce / cprop / cce

order = [

        ['2025-07-23_12-05-45_subject-b001',
        '2025-08-05_11-52-41_subject-b001',],

        ['2025-08-29_08-54-34_subject-b003',
        '2025-07-25_08-38-29_subject-b003',],

        ['2025-07-30_merge_subject-b004',
        '2025-08-12_09-11-34_subject-b004',],
        
        [
        '2025-09-05_08-10-33_subject-b008',
        '2025-09-19_07-52-47_subject-b008',],
        
        [
        '2025-09-12_merge_subject-b006',
        '2025-09-12_merge_subject-b006',],
        
        [
        '2025-09-17_07-57-44_subject-b002',
        '2025-09-17_07-57-44_subject-b002',],
        
        ]

##
ant = {}
sq = {}
with h5py.File(processed_path, 'r') as h:
    for name in np.ravel(order):
        ce = np.array(h[f'{name}_ce']).copy()
        cprop = np.array(h[f'{name}_cprop']).copy()
        spect_ds = h[f'{name}_spect']
        spect = np.array(spect_ds).copy()
        channels = spect_ds.attrs['channels']
        sp_f = spect_ds.attrs['freq']
        cce = np.cumsum(ce)
        ant[name] = [ce, cce, cprop, spect, channels, sp_f]
        
        sq_dat = np.array(h[f'{name}_squeeze'])
        ss = np.array(h[f'{name}_squeeze_starts'])
        sq[name] = sq_dat, ss
        
## squeeze and LOR stats
lors = {}
#fig, axs = pl.subplots(1, len(order))
for idx, names in enumerate(order):
    #ax = axs[idx]
    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        sqdat, starts = sq[name]
        ce, yn, ts, cprop = sqdat.T
        cce = np.cumsum(ce)

        sqp_time, sq_prob = make_sq_probability(yn, ts)

        # -- experimental
        df = pd.DataFrame(sqdat, columns=['ce','yn','ts','cprop'])

        # -- end experimental
        
        starts = np.append(starts, starts[-1]+20*60)
        
        asc = np.arange(len(ce)) <= np.argmax(ce)
        ce = ce[asc]
        cce = cce[asc]
        yn = yn[asc]
        ts = ts[asc]
        cprop = cprop[asc]

        res = []

        for ss, se in zip(starts[:-1], starts[1:]):

            use = (ts>=ss) & (ts<se)
            if np.all(use == False): continue
            _yn = yn[use]
            _ce = ce[use]
            _cce = cce[use]
            _ts = ts[use]
            _cprop = cprop[use]
    
            dat = pd.DataFrame()
            dat['ce'] = _ce
            dat['cce'] = _cce
            dat['resp'] = _yn
            dat['ts'] = _ts
            dat['cprop'] = _cprop
            if prop_quantity == 'ce':
                dat['bin'] = pd.cut(dat['ce'], bins=2)
            elif prop_quantity == 'cce':
                dat['bin'] = pd.cut(dat['cce'], bins=10)
            elif prop_quantity == 'cprop':
                dat['bin'] = pd.cut(dat['cprop'], bins=10)
            #assert np.all(dat.groupby('bin', observed=True).count().values[:,0] > 4) # had to have at least 4 squeeze commands in a given level to get a percentage response
            mean = dat.groupby('bin', as_index=False).mean()

            mean['label'] = mean.bin.apply(lambda x: (x.left+x.right)/2)
            res.append(mean)
        
        res = pd.concat(res)
        #ax.plot(res['label'].values,
        #        res['resp'].values,
        #        color=col)

        #lor = np.where(res.resp.values < 0.5)[0][0] # first time low resp rate recorded
        lor = np.where(res.resp.values >= 0.9)[0][-1] # last time high rate recorded; only works when using ascending-only times, as we are
        if prop_quantity == 'ce':
            lor = res.ce.values[lor]
        elif prop_quantity == 'cce':
            lor = res.cce.values[lor]
        elif prop_quantity == 'cprop':
            lor = res.cprop.values[lor]
        lors[name] = lor

## anteriorization figure
rise_only = True
do_bin = True
show_lor = True

gs = GridSpec(1, len(order)*3,
              right=0.99,
              left=0.05,
              top=0.85,
              bottom=0.25,
              wspace=0.05)
fig = pl.figure(figsize=(15,3.5))
all_axs = []
for idx, names in enumerate(order):
    
    if rise_only:
        ax_list = [
                    fig.add_subplot(gs[0, idx*3:idx*3+2]),
                ]
    else:
        ax_list = [
                    fig.add_subplot(gs[0, idx*3]),
                    fig.add_subplot(gs[0, idx*3+1]),
                ]
    all_axs.append(ax_list)

    for name,cond,col in zip(names, ['sham','active'], ['cadetblue', 'coral']): 
        ce, cce, cprop, spect, channels, sp_f = ant[name]

        if prop_quantity == 'ce':
            _pq = ce
        elif prop_quantity == 'cce':
            _pq = cce
        elif prop_quantity == 'cprop':
            _pq = cprop

        is_alpha = (sp_f>=8) & (sp_f<=17)
        is_theta = (sp_f>=4) & (sp_f<8)
        is_delta = (sp_f>=0.8) & (sp_f<4)
        is_beta = sp_f>20

        is_ant = np.isin(channels, ['F3', 'Fz', 'FCz', 'F4'])
        is_post = np.isin(channels, ['P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'])

        spect_ant = np.nanmean(spect[is_ant], axis=0)
        spect_post = np.nanmean(spect[is_post], axis=0)
        spect_full = np.nanmean(spect, axis=0)

        ant_alpha = np.nanmean(spect_ant[is_alpha], axis=0)
        post_alpha = np.nanmean(spect_post[is_alpha], axis=0)
        total_alpha = np.nanmean(spect_full[is_alpha], axis=0)
        total_power = np.nanmean(spect_full, axis=0)

        ap_ratio = ant_alpha / post_alpha

        _to_plot = np.array([
                            _pq,
                            ap_ratio,
                           ]).T

        # include only rising/falling prop levels
        if prop_quantity == 'ce':
            if ce.max() > 3.1 and (name!='2025-09-12_merge_subject-b006'): # original protocol
                bins_category = 0
            else: # new protocol 2025-09-05
                bins_category = 1
        elif prop_quantity == 'cprop':
            bins_category = 2
        elif prop_quantity == 'cce':
            bins_category = 2
        drop_starts = -(np.where(ce[::-1] > 2.99)[0][0])

        if rise_only:
            dirs = [1]
        else:
            dirs = [1,-1]

        for direction, ax in zip(dirs, ax_list):
            if direction == 1:
                to_plot = _to_plot[:drop_starts] # rising
            elif direction == -1:
                to_plot = _to_plot[drop_starts:] # falling
            
            # bin them
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
                    elif bins_category == 2:
                        bins = 10
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
                    elif bins_category == 2:
                        bins = 10
                to_plot['bin'] = pd.cut(to_plot.iloc[:,0], bins=bins)
                to_plot_mean = to_plot.groupby('bin', as_index=False).mean()
                to_plot_mean = to_plot_mean.values[:,1:].astype(float)
                to_plot_err = to_plot.groupby('bin', as_index=False).std()
                to_plot_err = to_plot_err.values[:,1:].astype(float)

                # or bin them agnostically
                #to_plot = np.array([np.nanmean(r, axis=0) for r in np.array_split(to_plot, 5, axis=0)])
            
            if do_bin:
                c,apr = to_plot_mean.T
                e_c,e_apr = to_plot_err.T
            else:
                c,apr = to_plot.T

            keep = ~(np.isnan(apr))
            
            c_ = c[keep]
            apr_ = apr[keep]
            if do_bin:
                e_apr_ = e_apr[keep]
            else:
                e_apr_ = None

            # TEMP TODO
            apr_ = 10*np.log10(apr_)
            apr_ -= apr_[-1]
    
            if not do_bin:
                kw = dict(s=15, marker='o', color=col, alpha=0.5, lw=0)
                ax.scatter(c_, apr_, **kw)
            elif do_bin:
                ax.errorbar(c_,
                            apr_,
                            yerr=e_apr_,
                            marker='o',
                            color=col,
                            capsize=5,
                            markersize=6,
                            elinewidth=1,
                            alpha=0.9,
                            lw=0)
            
            do_fit = False
            if do_fit:
                xv, yv, (A,y0,ec50,B) = fit_sigmoid2(c_, apr_, return_params=True)
                ax.plot(xv, yv, color=col, lw=.5,
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
                ax.set_title(name[-4:], pad=25)
                ax.set_xlabel('Propofol conc.', labelpad=20)
                #ax.set_xlim([-0.3, 3.4])
                #ax.set_xticks(np.arange(0, 3.2, 0.5))
                ax.tick_params(axis='x', labelsize=8)
            elif direction == -1:
                #ax.set_xlim([3.4, -0.3])
                #ax.set_xticks(np.arange(0, 3.2, 0.5)[::-1])
                ax.tick_params(axis='y', length=0)
                ax.tick_params(axis='x', labelsize=8)
                ax.spines['left'].set_visible(False)

            if show_lor and direction==1:
                ax.axvline(lors[name], color=col, lw=2, alpha=0.95, ls=':')
                ax.text(lors[name], 1.01, 'LOR',
                        rotation=90,
                        ha='center',
                        va='bottom',
                        alpha=0.5,
                        fontsize=9,
                        color=col,
                        transform=blend(ax.transData, ax.transAxes))

##
