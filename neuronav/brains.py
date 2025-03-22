##

# flirt -in input_scan.nii.gz -ref /Users/bdd/fsl/data/standard/MNI152_T1_2mm.nii.gz  -out input_scan_registered.nii.gz -omat transform.mat -cost mutualinfo

# https://www.sciencedirect.com/science/article/pii/S105381191730407X

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nilearn
from nilearn import datasets, plotting

'''

RsC:
    https://www.researchgate.net/figure/Sagittal-coronal-and-horizontal-view-MNI-coordinates-and-3D-model-of-the_fig3_306392818
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5321500/
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6866613/

LC
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10406215/
'''

##

regions = dict( # L, R
   rsc=[(-5, -50, 12), (5, -50, 12)], 
   md=[(-2, -12, 4), (2, -12, 4)],
   lc=[(-6, -37, -25), (6, -37, -25)],
   s1=[(-48, -21, 50), (48, -21, 50)],
)

for region, coords in regions.items():
    print(region)

    mean = np.array(coords).mean(axis=0)
    mean[0] = coords[0][0] # in mediolateral axis, go fully to one side

    template_img = datasets.load_mni152_template()
    display = plotting.plot_stat_map(template_img,
                                     display_mode='ortho',
                                     cut_coords=mean,
                                     cmap='grey',
                                     colorbar=False,
                                     radiological=True,)

    display.add_markers(coords,
                        marker_color='orange',
                        marker_size=75,
                        marker='o',
                        facecolor='none',
                        alpha=0.5)


    fig = pl.gcf()
    fig.savefig(f'/Users/bdd/Desktop/{region}.pdf')
    pl.close(fig)

##

mni152_template = datasets.load_mni152_template()
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm')
atlas_filename = atlas.filename

plotting.plot_roi(
    atlas_filename,
    display_mode='ortho',
    draw_cross=True,
    cut_coords=(0, -30, 60),
    cmap='jet',
    annotate=False,
    alpha=0.5,
    view_type='contours',
    bg_img=mni152_template,
)



##
'''
# Load the MRI scan
#mri_path = 'input_scan_registered.nii.gz'
#mri_img = nib.load(mri_path)
#mri_img_arr = mri_img.get_fdata()

# Load the Harvard-Oxford atlas amygala
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
atlas_img = atlas.maps
atlas_img_arr = atlas_img.get_fdata()
amygdala_labels = [17, 18]
amygdala_mask = np.isin(atlas_img.get_fdata(), amygdala_labels)

alpha = amygdala_mask.copy().astype(float) * 0.5
mask = np.zeros((*amygdala_mask.shape, 4))  
mask[..., 0] = 1 # red full
mask[..., -1] = alpha

fig, axs = pl.subplots(1, 3)
for ax, axis in zip(axs, [0,1,2]):
    ax.imshow(atlas_img_arr.sum(axis=axis), cmap='gray')
    #ax.imshow(mask.sum(axis=axis), cmap='autumn', alpha=0.5)
    ax.axis('off')
'''

##
