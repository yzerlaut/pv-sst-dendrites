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


# %%
def load_cell(nrn_h5_file):
    """
    we translate everything in terms of skeleton indices ! (mesh properties)
    """
    nrn = meshwork.load_meshwork(nrn_h5_file)
    
    nrn.pre_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno.pre_syn.df['pre_pt_mesh_ind']]
    
    nrn.post_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno.post_syn.df['post_pt_mesh_ind']]
    
    axon_inds, Q = meshwork.algorithms.split_axon_by_annotation(nrn, 
                                                                pre_anno='pre_syn',
                                                                post_anno='post_syn')
    
    if Q>0.5:
        nrn.axon_inds = nrn.skeleton.mesh_to_skel_map[axon_inds]
        nrn.is_axon = np.array([(m in nrn.axon_inds) for m in nrn.skeleton_indices], dtype=bool)
        # dendritic cover paths
        nrn.dend_cover_paths = []
        for cover_path in nrn.skeleton.cover_paths:
            path = [c for c in cover_path if c not in nrn.axon_inds]
            if len(path)>0:
                nrn.dend_cover_paths.append(path)
    else:
        print('axon splitting not trusted...')

    return nrn


# %% [markdown]
# # path to soma density

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
bins, hist = compute_single_cell(cells['Martinotti'][0])
#plt.plot(bins, hist)


# %%
bins = np.linspace(0, 400, 30)

Martinotti_Density = []
for cell in cells['Martinotti']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Martinotti_Density.append(density)
    except BaseException as be:
        print(be)

Basket_Density = []
for cell in cells['Basket']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Basket_Density.append(density)
    except BaseException as be:
        print(be)

# %%
fig, AX = plt.subplots(1, 3, figsize=(11,3))

for ax, density, c, title in zip(AX, [Basket_Density, Martinotti_Density],
                                 ['red', 'blue'], ['Basket', 'Martinotti']):
    
    for d in density:
        ax.plot(bins[1:], d, lw=0.1)
        
    x, y, sy = bins[1:], np.nanmean(density, axis=0), np.nanstd(density, axis=0)
    ax.plot(x, y, color=c)
    ax.fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    # --
    ax.annotate('n=%i ' % len(density), (1,0.95), color='grey', xycoords='axes fraction', ha='right', va='top')
    ax.set_title('%s cells' % title, color=c)
    ax.set_xlabel('path dist. to soma ($\mu$m)')
    ax.set_ylabel('path density (norm.)')
    # --
    AX[2].plot(x, y, color=c, label=title)
    #AX[2].fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    
AX[2].set_xlabel('path dist. to soma ($\mu$m)')
AX[2].set_ylabel('path density (norm.)')
AX[2].legend()
fig.tight_layout()
#fig.savefig('/home/yann.zerlaut/Desktop/figs/path-density.png', dpi=300)

# %% [markdown]
# # Branching

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

    COUNTS = np.zeros(len(bins))
    for p, path in enumerate(nrn.dend_cover_paths):
        
        path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]
        hist = np.digitize(path_to_soma, bins=bins)
        for i in np.unique(hist):
            if i>0 and i<len(bins):
                COUNTS[i]+=1

    return bins, COUNTS


# %%
bins, hist = compute_single_cell(cells['Basket'][0])
#plt.plot(bins, hist)

# %%
bins = np.linspace(0, 400, 30)

Martinotti_Density = []
for cell in cells['Martinotti']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Martinotti_Density.append(density)
    except BaseException as be:
        print(be)

Basket_Density = []
for cell in cells['Basket']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Basket_Density.append(density)
    except BaseException as be:
        print(be)

# %%
fig, AX = plt.subplots(1, 3, figsize=(11,3))

for ax, density, c, title in zip(AX, [Basket_Density, Martinotti_Density],
                                 ['red', 'blue'], ['Basket', 'Martinotti']):
    
    for d in density:
        ax.plot(bins, d, lw=0.1)
        
    x, y, sy = bins, np.nanmean(density, axis=0), np.nanstd(density, axis=0)
    ax.plot(x, y, color=c)
    ax.fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    # --
    ax.annotate('n=%i ' % len(density), (1,0.95), color='grey', xycoords='axes fraction', ha='right', va='top')
    ax.set_title('%s cells' % title, color=c)
    ax.set_xlabel('path dist. to soma ($\mu$m)')
    ax.set_ylabel('branch number (cover-paths)')
    # --
    AX[2].plot(x, y, color=c, label=title)
    #AX[2].fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    
AX[2].set_xlabel('path dist. to soma ($\mu$m)')
AX[2].set_ylabel('branch number (cover-paths)')
AX[2].legend()
fig.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/branching.png', dpi=300)


# %% [markdown]
# # Net path length

# %%
def compute_single_cell(nrn_h5_file, 
                        bins = np.linspace(0, 400, 50),
                        with_fig=False):
    """
    we loop over cover paths
        we bin pieces of paths
            we look for contiguous pieces -> we increment the net path length in that bien
    """
    
    nrn = load_cell(nrn_h5_file)
    
    if with_fig:
        fig, AX = plt.subplots(1, 2, figsize=(12,4))

    COUNTS = np.zeros((len(bins), len(nrn.dend_cover_paths)))
    for i, path in enumerate(nrn.dend_cover_paths):
        
        path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]
        hist = np.digitize(path_to_soma, bins=bins)
        for j in np.unique(hist):
            if j>0 and j<len(bins):
                indices = np.flatnonzero(hist==j)
                path_within_bin = np.array(path)[indices]
                if len(path_within_bin)>1:
                    #print(path_within_bin)
                    jumps = np.flatnonzero(np.abs(np.diff(path_within_bin))>2)+1
                    #print(jumps)
                    for istart, istop in zip(np.concatenate([[0], jumps]),
                                             np.concatenate([jumps, [-1]])):
                        #print(istart, istop, path_within_bin[istart:istop])
                        COUNTS[j, i] += nrn.path_length(path_within_bin[istart:istop])/1_000

    return bins, COUNTS.sum(axis=1)

#compute_single_cell(cells['Basket'][0])


# %%
bins, hist = compute_single_cell(cells['Basket'][0], bins=np.linspace(40, 300, 10))
plt.bar(bins, hist, width=bins[1]-bins[0])

# %%
bins = np.linspace(20, 400, 20)

Martinotti_Density = []
for cell in cells['Martinotti']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Martinotti_Density.append(density)
    except BaseException as be:
        print(be)

Basket_Density = []
for cell in cells['Basket']:
    try:
        _, density = compute_single_cell(cell,
                                         bins = bins)
        Basket_Density.append(density)
    except BaseException as be:
        print(be)

# %%
fig, AX = plt.subplots(1, 3, figsize=(11,3))

for ax, density, c, title in zip(AX, [Basket_Density, Martinotti_Density],
                                 ['red', 'blue'], ['Basket', 'Martinotti']):
    
    for d in density:
        ax.plot(bins, d, lw=0.1)
        
    x, y, sy = bins, np.nanmean(density, axis=0), np.nanstd(density, axis=0)
    ax.plot(x, y, color=c)
    ax.fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    # --
    ax.annotate('n=%i ' % len(density), (1,0.95), color='grey', xycoords='axes fraction', ha='right', va='top')
    ax.set_title('%s cells' % title, color=c)
    ax.set_xlabel('path dist. to soma ($\mu$m)')
    ax.set_ylabel('net path length ($\mu$m)')
    # --
    AX[2].plot(x, y, color=c, label=title)
    #AX[2].fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    
AX[2].set_xlabel('path dist. to soma ($\mu$m)')
AX[2].set_ylabel('net path length ($\mu$m)')
AX[2].legend()
fig.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/path-length.png', dpi=300)

# %%
# nrn.path_length?

# %%
