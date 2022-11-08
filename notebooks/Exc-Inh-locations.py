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
client.materialize.get_tables() # to explore the available data

# %%
nucleus_table = client.materialize.query_table('nucleus_detection_v0')

# %%
cell_table = client.materialize.query_table('allen_v1_column_types_slanted')

# %% [markdown]
# ## Available Cell Types 

# %%
np.unique(full_table['cell_type'])

# %% [markdown]
# ### We split cells into excitatory and inhibitory types

# %%
excitatory = ['23P', '4P', '5P-IT', '5P-NP', '5P-PT', '6P-CT', '6P-IT', '6P-U', 'Unsure E']
inhibitory = ['BC', 'BPC', 'MC', 'NGC', 'Unsure I']

# %%
MCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'MC'})
BCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'BC'})

# %%
# from the Allen database
cells = {'Basket': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'BC' in fn],
         'Martinotti': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'MC' in fn]}


# %% [markdown]
# # Analyze presynaptic cells

# %%
def analyze_afferent_neurons(nrn, cell_table, nucleus_table):
    """
    analyzes the properties of the afferent neurons:
    - whether excitatory or inhibitory (or unclassified) 
            through an array of -1, +1, 0 values
    - the spatial location of the nucleus in the sample
    """
    nrn.synaptic_sign = np.zeros(len(nrn.pre_pt_root_id), dtype=int)
    nrn.pre_loc = []
    
    for i, pre_pt in enumerate(nrn.pre_pt_root_id):

        # EXC vs INH 
        i_pt = np.flatnonzero(cell_table['pt_root_id']==pre_pt)
        if len(i_pt)>0:
            if cell_table['cell_type'].values[i_pt[0]] in inhibitory:
                nrn.synaptic_sign[i] = -1
            elif cell_table['cell_type'].values[i_pt[0]] in excitatory:
                nrn.synaptic_sign[i] = +1
                
        # PRE NUCLEUS POSITION
        i_pt = np.flatnonzero(nucleus_table['pt_root_id']==pre_pt)
        if len(i_pt)>0:
            nrn.pre_loc.append(nucleus_table['pt_position'].values[i_pt[0]])
        else:
            nrn.pre_loc.append(None)
            
    nrn.root_position = nucleus_table['pt_position'].values[\
                np.flatnonzero(nucleus_table['pt_root_id']==nrn.root_id)][0]
            
    nrn.pre_X = 4e-3*np.array([(pre_loc[0]-nrn.root_position[0]) for pre_loc in nrn.pre_loc if pre_loc is not None])
    nrn.pre_Y = 4e-3*np.array([(pre_loc[1]-nrn.root_position[1]) for pre_loc in nrn.pre_loc if pre_loc is not None])
    nrn.pre_Z = 40e-3*np.array([(pre_loc[2]-nrn.root_position[2]) for pre_loc in nrn.pre_loc if pre_loc is not None])

def load_cell(nrn_h5_file):
    """
    we translate everything in terms of skeleton indices ! (mesh properties)
    """
    nrn = meshwork.load_meshwork(nrn_h5_file)
    nrn.root_id = int(nrn_h5_file.split('-')[-1].replace('.h5', ''))

    nrn.post_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno.post_syn.df['post_pt_mesh_ind']]
    nrn.pre_pt_root_id = nrn.anno.post_syn.df['pre_pt_root_id']
    
    analyze_afferent_neurons(nrn, cell_table, nucleus_table)
    
    return nrn


# %%

def compute_single_cell(nrn_h5_file, 
                        bins = np.linspace(0, 400, 30),
                        with_fig=False):
    """
    we loop over cover_paths to compute the exc/inh ratio 
    """
    
    nrn = load_cell(nrn_h5_file)
    
    if with_fig:
        fig, AX = plt.subplots(1, 2, figsize=(12,4))

    RATIO, Ntot_syn = [], 0

    for i, path in enumerate(nrn.dend_cover_paths):
        
        path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

        count_along_path = np.zeros(len(path))
        for i, p in enumerate(path):
            count_along_path[i] = np.sum(nrn.post_syn_sites==p)

        binned_dist = np.digitize(path_to_soma, bins=bins)
        
        density_hist = np.ones(len(bins))*np.nan # nan by default
        for b in range(len(bins)):
            if np.sum(binned_dist==b)>0:
                
                # we sum all synapses that fall into this bin and we divide by the bin length
                density_hist[b] = np.sum(count_along_path[binned_dist==b])/(bins[1]-bins[0])

        DENSITY_HIST.append(density_hist)
        Ntot_syn += count_along_path.sum()
    
    print('synapses counted: %i/%i' % (Ntot_syn, len(nrn.anno.post_syn.df['post_pt_mesh_ind'])))

    if with_fig:
        AX[1].plot(bins, np.nanmean(DENSITY_HIST, axis=0)) # only non-infinite values contributing

        AX[0].set_title('looping over dendritic paths')
        AX[0].axis('equal')
        
        AX[1].set_xlabel('path dist. to soma ($\mu$m)'); 
        AX[1].set_ylabel('linear density (syn./$\mu$m)')
    else:
        fig = None
        
    return bins, np.nanmean(DENSITY_HIST, axis=0), fig



# %%
nrn = load_cell('../data/MC-%s.h5' % MCs.pt_root_id[0])


# %%
Nsample, colors = 3, ['r', 'b']
fig, AX = plt.subplots(2, Nsample, figsize=(10,4))
for i, cType in enumerate(['Basket', 'Martinotti']):
    for j, c in enumerate(np.random.choice(len(cells[cType]), Nsample)):
        nrn = load_cell(cells[cType][c])
        AX[i,j].scatter(nrn.pre_X, nrn.pre_Z, color=colors[i], s=1)
        AX[i,j].annotate(' %s #%s' % (cType, c), (0,0.95), xycoords='axes fraction', va='top', color=colors[i])
for Ax in AX:
    for ax in Ax:
        ax.axis('equal')
        ax.set_ylim([-300, 300])
        ax.set_xlim([-400, 600])
AX[1,0].set_xlabel('<-- medial  |  lateral-->')
AX[1,0].set_ylabel('                 <-- posterior  |  anterior-->')
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
        distance = np.sqrt(nrn.pre_X**2+nrn.pre_Z**2)
        hist, be = np.histogram(distance, bins=bins, density=True)
        AX[i].plot(0.5*(bins[1:]+bins[:-1]), hist, color=colors[i], lw=0.5)
        FINAL[cType].append(hist)
    AX[2].plot(0.5*(bins[1:]+bins[:-1]), np.mean(FINAL[cType], axis=0), color=colors[i], lw=2)
for ax in AX:
    ax.set_ylabel('density (norm.)')
    ax.set_xlabel('distance (um) \nfrom target soma')
plt.tight_layout()
fig.savefig('/home/yann.zerlaut/Desktop/figs/pre-location.png', dpi=300)

# %%
hist, be = np.histogram(distance, bins=bins, density=True)

# %%
len([x for x in nrn.pre_loc if x is not None])

# %%
for pre_pt in nrn.pre_pt_root_id:
    i_pt = np.flatnonzero(cell_table['pt_root_id']==pre_pt)
    if len(i_pt)>0:
        print(cell_table['pt_position'].values[i_pt[0]])

# %%
# client.materialize.query_table?

# %%
np.sum(full_table['pt_root_id']==nrn.pre_pt_root_id[2])

# %%
i_pt = np.flatnonzero(full_table['pt_root_id']==nrn.pre_pt_root_id[2])
print(i_pt)
full_table['cell_type'][i_pt]

# %%
pre_pt = nrn.pre_pt_root_id[3]
#print(pre_pt)
pre_pt = client.materialize.query_table('allen_v1_column_types_slanted',
                                        filter_equal_dict={'pt_root_id':pre_pt})
pre_pt

# %%
len(nrn.post_syn_sites)

# %%
client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'BC'})

# %%
