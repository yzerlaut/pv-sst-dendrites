# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules
import sys, os
import numpy as np
import matplotlib.pylab as plt

sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
sys.path.append('../')
import plot_tools as pt

folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'SST-WT-NR1-GluN3-2023')

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

DATASET = scan_folder_for_NWBfiles(folder,
                                   verbose=False)

# %%
data = Data(DATASET['files'][2])
data.build_rawFluo()
data.build_neuropil()

data.build_dFoF(roi_to_neuropil_fluo_inclusion_factor=1.2,
                neuropil_correction_factor=0.8,
                sliding_window=180,
                method_for_F0='sliding_percentile',
                percentile=10,
                with_correctedFluo_and_F0=True,
                verbose=False)

# %%
frac = len(data.valid_roiIndices)/data.iscell.sum()
X = [frac, 1-frac]
pt.pie(X,COLORS=['tab:green', 'tab:grey'],
       ext_labels=['%s\n%.1f%%'%(label, 100.*x) for label, x in zip(['kept', 'discarded'], X)])

# %%
Nrois = 10
fig, AX = plt.subplots(Nrois, 3, figsize=(9,0.6*Nrois))

for i, roi in enumerate(np.random.choice(np.arange(data.vNrois), Nrois, replace=False)):
    
    AX[i][0].plot(data.t_rawFluo, data.rawFluo[roi,:], color='tab:green', label='ROI')
    AX[i][0].plot(data.t_rawFluo, data.neuropil[roi,:], color='tab:red', label='neuropil')
    
    AX[i][1].plot(data.t_rawFluo, data.correctedFluo[roi,:], color='tab:green' ,label='$F$')
    AX[i][1].plot(data.t_rawFluo, data.correctedFluo0[roi,:], color='tab:orange', lw=2, label='$F_0$')

    AX[i][2].plot(data.t_rawFluo, data.dFoF[roi,:], color='tab:green')

for ax in pt.flatten(AX):
    pt.set_plot(ax, xlim=[0, data.t_rawFluo[-1]], xticks=[])
    
pt.draw_bar_scales(AX[0][0], Xbar=60, Xbar_label='1min', Ybar=1e-21)
AX[0][0].legend(frameon=False, loc=(0.8,1), fontsize=6)
AX[0][1].legend(frameon=False, loc=(0.8,1), fontsize=6)

AX[0][0].set_title('raw Fluorescence')
AX[0][1].set_title('corrected Fluo.')
AX[0][2].set_title('$\Delta$F/F');

# %%
