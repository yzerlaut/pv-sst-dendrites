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

# %% [markdown]
# # Simulation of Clustered-Input on Dendritic Segments 

# %%
from PV_template import *

from clustered_input_stim import * 

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

# %%
# load cell
ID = '864691135396580129_296758' # Basket Cell example
cell = PVcell(ID=ID, debug=False)
cell.check_that_all_dendritic_branches_are_well_covered(verbose=False)

# %% [markdown]
# ## Distal Clusters

# %%
for seed in [2, 4]:

    props ={'iDistance':2, # 2 -> means "distal" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)

# %% [markdown]
# ## Proximal Clusters

# %%
for seed in [2, 4]:

    props ={'iDistance':0, # 0 -> means "proximal" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)

# %% [markdown]
# ## Intermediate Clusters

# %%
for seed in [2, 4]:

    props ={'iDistance':1, # 1 -> means "intermediate" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)


# %%
