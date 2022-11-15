# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4
import pcg_skel # version 0.3.0 
from caveclient import CAVEclient # version 4.16.2

# %%
datastack_name = 'minnie65_public_v343'
client = CAVEclient(datastack_name)
client.materialize.version = 343

# %%
# from the Allen database
cells = {'Basket': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'BC' in fn],
         'Martinotti': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'MC' in fn]}


# %% [markdown]
# # Analyze presynaptic cells

# %%
def load_cell(nrn_h5_file):
    """
    we translate everything in terms of skeleton indices ! (mesh properties)
    """
    nrn = meshwork.load_meshwork(nrn_h5_file)
    nrn.root_id = int(nrn_h5_file.split('-')[-1].replace('.h5', ''))

    nrn.post_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno.post_syn.df['post_pt_mesh_ind']]
    nrn.pre_pt_root_id = nrn.anno.post_syn.df['pre_pt_root_id']
    
    return nrn


# %%
def compute_single_cell(nrn_h5_file, 
                        bins = np.linspace(0, 400, 30),
                        with_fig=False):
    """
    we loop over cover_paths to compute the segment density along the paths
    """
    
    nrn = load_cell(nrn_h5_file)
    
    if with_fig:
        fig, AX = plt.subplots(1, 2, figsize=(12,4))

    PATHS_TO_SOMA = []
    for i, path in enumerate(nrn.dend_cover_paths):
        
        PATHS_TO_SOMA += [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

    return .5*(bins[:-1]+bins[1:]), np.histogram(PATHS_TO_SOMA, bins=bins, density=True)[0]


# %%
nrn = load_cell(cells['Martinotti'][0])

# %%
nrn.skeleton.cover_paths

# %%
bins, hist = compute_single_cell(cells['Martinotti'][0])
plt.plot(bins, hist)


# %%
Nsample, colors = 7, ['r', 'b']
fig, AX = plt.subplots(2, Nsample, figsize=(1.7*Nsample,3))
xlim, ylim = [np.inf, -np.inf], [np.inf, -np.inf] 
for i, cType in enumerate(['Basket', 'Martinotti']):
    for j, c in enumerate(np.random.choice(len(cells[cType]), Nsample, replace=False)):
        nrn = load_cell(cells[cType][c])
        AX[i,j].scatter(nrn.pre_X, nrn.pre_Z, color=colors[i], s=1)
        AX[i,j].scatter([nrn.root_X], [nrn.root_Z], color='y', s=40)
        AX[i,j].annotate(' %s #%s' % (cType, c+1), (0,0.98),
                         xycoords='axes fraction', va='top', color=colors[i], fontsize=8)
        xlim = [min([xlim[0], AX[i,j].get_xlim()[0]]), max([xlim[1], AX[i,j].get_xlim()[1]])]
        ylim = [min([ylim[0], AX[i,j].get_ylim()[0]]), max([ylim[1], AX[i,j].get_ylim()[1]])]
        
for Ax in AX:
    for ax in Ax:
        ax.axis('equal')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
AX[1,0].set_xlabel('                             <-- medial  |  lateral-->   ($\mu$m)')
AX[1,0].set_ylabel('              <-- posterior  |  anterior-->  ($\mu$m)')
fig.suptitle('soma locations of synaptic afferents with respect to target soma location')
plt.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/pre-location-examples.png', dpi=300)

# %%
Nsample = 2
bins = np.linspace(10, 400, 50)
fig, AX = plt.subplots(1, 3, figsize=(9,3))
FINAL = {}
for i, cType in enumerate(['Basket', 'Martinotti']):
    FINAL[cType] = []
    AX[i].set_title(cType, color=colors[i])
    for j, c in enumerate(range(len(cells[cType]))):
        nrn = load_cell(cells[cType][c])
        distance = np.sqrt((nrn.pre_X-nrn.root_X)**2+(nrn.pre_Z-nrn.root_Z)**2)
        hist, be = np.histogram(distance, bins=bins, density=True)
        AX[i].plot(0.5*(bins[1:]+bins[:-1]), hist, color=colors[i], lw=0.1)
        FINAL[cType].append(hist)
    AX[i].annotate('n=%i' % len(cells[cType]), (0.95, 0.95), fontsize=9,
                   xycoords='axes fraction', va='top', ha='right', color=colors[i])
    AX[2].plot(0.5*(bins[1:]+bins[:-1]), np.mean(FINAL[cType], axis=0), color=colors[i], lw=2)
for ax in AX:
    ax.set_ylabel('density (norm.)')
    ax.set_xlabel('2D distance (um) \nfrom post-syn. soma')
plt.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/pre-location.png', dpi=300)

# %%
bins = np.linspace(10, 400, 50)
FINAL = {}
for i, cType in enumerate(['Basket', 'Martinotti']):
    FINAL[cType+'-frac'], FINAL[cType+'-sum'] = [], []
    for j, c in enumerate(range(len(cells[cType]))):
        nrn = load_cell(cells[cType][c])
        FINAL[cType+'-sum'].append(len(nrn.synaptic_sign))
        FINAL[cType+'-frac'].append(len(nrn.pre_X)/len(nrn.pre_loc))

# %%
fig, AX = plt.subplots(1, 2, figsize=(8,2))

for i, cType in enumerate(['Basket', 'Martinotti']):
    mSum, sSum = np.mean(FINAL[cType+'-sum']), np.std(FINAL[cType+'-sum'])
    AX[i].set_title(cType+'\nn=%i+/-%i syn.' % (mSum, sSum), color=colors[i], fontsize=9)
    
    mSyn, sSyn = 100.*np.mean(FINAL[cType+'-frac']), 100.*np.std(FINAL[cType+'-frac'])

    AX[i].pie([mSyn, 100-mSyn], colors=['purple', 'lightgray'],
              labels=['located:\n%.1f+/-%.1f%%' % (mSyn, sSyn), ''])
    AX[i].annotate('unlocated', (0,-0.2), color='w', ha='center', va='top')
fig.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/Loc-sampling.png', dpi=300)
