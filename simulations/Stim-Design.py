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
import sys, os

import numpy as np

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils import plot_tools as pt

# %% [markdown]
# ## Load morphology

# %%
mFile = 'morphologies/Pvalb-IRES-Cre_Ai14-305768.04.02.01_656999337_m.swc'
morpho = nrn.Morphology.from_swc_file(mFile) 
FULL = nrn.morpho_analysis.compute_segments(morpho)
SEGMENTS = nrn.morpho_analysis.compute_segments(morpho, without_axon=True)

# %%
fig, [ax0, ax] = pt.plt.subplots(1, 2, figsize=(4,1.3))
visFull, vis = nrnvyz(FULL), nrnvyz(SEGMENTS)
visFull.plot_segments(ax=ax0, color='tab:grey')
ax0.annotate('dendrite', (0,0), xycoords='axes fraction', fontsize=6, color='tab:blue')
vis.plot_segments(ax=ax0, color='tab:blue')
ax.hist(1e6*SEGMENTS['distance_to_soma'], density=True)
ax.set_xlabel('path dist. to soma ($\mu$m)')
ax.set_ylabel('density')
ax.set_yticks([])

# %% [markdown]
# ## Distribute synapses

# %%
x = np.linspace(SEGMENTS['distance_to_soma'].min(), SEGMENTS['distance_to_soma'].max(), 25)
uniform = 0.5 +0*x
uniform /= np.sum(uniform) #np.trapz(uniform, x=1e6*x)

biased = 1.-(x-x.min())/(x.max()-x.min())
biased /= np.sum(biased) # np.trapz(biased, x=1e6*x)

# %%
Nsynapses = 40

np.random.seed(20)
LOCS = {}

digitized_dist = np.digitize(SEGMENTS['distance_to_soma'], bins=x, right=True)

for case, proba in zip(['uniform', 'biased'], [uniform, biased]):
    
    LOCS[case] = []
    iDist = np.random.choice(np.arange(len(x)), Nsynapses, p=proba)
    
    for i in iDist:
        # pick one segment loc for the synapse with the distance condition:
        LOCS[case].append(np.random.choice(np.arange(len(SEGMENTS['x']))[digitized_dist==i], 1)[0])


# %%
fig, AX = pt.plt.subplots(2, 2, figsize=(7,5))
pt.plt.subplots_adjust(wspace=.2, hspace=.6)

for c, y, case in zip(range(2), [uniform, biased], ['uniform', 'biased']):
    
    ax=pt.inset(AX[0][c], [0.3, 0.2, 0.4, 0.6])
    ax.set_ylabel('synaptic density')
    AX[0][c].axis('off')
    ax.plot(1e6*x, y, '-', color='tab:grey', lw=2)
    ax.set_yticks([0,0.1])
    ax.set_title(case)
    ax.set_xlabel('path dist. to soma ($\mu$m)');
    
    vis.plot_segments(ax=AX[1][c], color='tab:grey')
    vis.add_dots(AX[1][c], LOCS[case], 1)

    inset = pt.inset(AX[1][c], [0.9, 0., 0.4, 0.3])
    inset.hist(SEGMENTS['distance_to_soma'][LOCS[case]], color='tab:red')
    inset.set_xlabel('dist.');inset.set_xticks([]);inset.set_yticks([])
    inset.set_xlim([x.min(), x.max()])

# %%
# np.trapz?

# %%
fig.sup
