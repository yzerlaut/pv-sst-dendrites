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

# %% [markdown]
# # Synaptic Targetting
#
# Which location on the dendrite does the different input targets ?

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
np.unique(cell_table['cell_type'])

# %% [markdown]
# ### We split cells into excitatory and inhibitory types

# %%
excitatory = ['23P', '4P', '5P-IT', '5P-NP', '5P-PT', '6P-CT', '6P-IT', '6P-U', 'Unsure E']
inhibitory = ['BC', 'BPC', 'MC', 'NGC', 'Unsure I']

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
            
    nrn.root_X,nrn.root_Y,nrn.root_Z=4e-3*nrn.root_position[0],4e-3*nrn.root_position[1],40e-3*nrn.root_position[2]
    nrn.pre_X = 4e-3*np.array([pre_loc[0] for pre_loc in nrn.pre_loc if pre_loc is not None])
    nrn.pre_Y = 4e-3*np.array([pre_loc[1] for pre_loc in nrn.pre_loc if pre_loc is not None])
    nrn.pre_Z = 40e-3*np.array([pre_loc[2] for pre_loc in nrn.pre_loc if pre_loc is not None])

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
                        bins = np.linspace(1, 400, 30),
                        with_fig=False):
    """
    we loop over cover_paths to compute the exc/inh ratio 
    """
    
    nrn = load_cell(nrn_h5_file)
    
    if with_fig:
        fig, ax = plt.subplots(1, figsize=(6,4))

    path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in nrn.post_syn_sites]
    hist, be = np.histogram(path_to_soma, bins=bins, density=True)
    
    if with_fig:
        ax.plot(.5*(bins[1:]+bins[:-1]), hist)
        ax.set_xlabel('path dist. to soma ($\mu$m)'); 
        ax.set_ylabel('norm. synaptic density')
    else:
        fig = None
        
    return bins, hist, fig



# %%
#nrn = load_cell(cells['Martinotti'][0])
compute_single_cell(cells['Martinotti'][0], with_fig=True)


# %%
bins = np.linspace(-1, 350, 20)

Martinotti_Density = []
for cell in cells['Martinotti']:
    try:
        _, density, _ = compute_single_cell(cell,
                                            bins = bins)
        Martinotti_Density.append(density)
    except BaseException as be:
        print(be)

Basket_Density = []
for cell in cells['Basket']:
    try:
        _, density, _ = compute_single_cell(cell,
                                            bins = bins)
        Basket_Density.append(density)
    except BaseException as be:
        print(be)

# %%
from scipy.stats import ttest_ind

fig, AX = plt.subplots(1, 3, figsize=(11,3))
AX[2].set_xlabel('path dist. to soma ($\mu$m)')
AX[2].set_ylabel('norm. synaptic  density')
plt.tight_layout()

fig2, ax2 = plt.subplots(1, figsize=(3.5,2))

means = {}
for ax, density, c, title in zip(AX, [Basket_Density, Martinotti_Density],
                                 ['red', 'blue'], ['Basket', 'Martinotti']):
    means[title] = []
    x, y, sy = .5*(bins[1:]+bins[:-1]), np.nanmean(density, axis=0), np.nanstd(density, axis=0)
    
    for d in density:
        ax.plot(x, d, lw=0.1)
        means[title].append(np.mean(x*d/d.mean()))
        
    ax.plot(x, y, color=c)
    ax.fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    # --
    ax.annotate('n=%i ' % len(density), (1,0.95), color='grey', xycoords='axes fraction', ha='right', va='top')
    ax.set_title('%s cells' % title, color=c)
    ax.set_xlabel('path dist. to soma ($\mu$m)')
    ax.set_ylabel('norm. synaptic  density')
    # --
    AX[2].plot(x, y, color=c, label=title)
    #AX[2].fill_between(x, y-sy, y+sy, color=c, alpha=0.2, lw=0)
    
    ax2.plot(len(means.keys())-1+np.random.randn(len(means[title]))*0.2,
             means[title], '.', color=c, ms=1)
    ax2.bar([len(means.keys())-1], [np.mean(means[title])], 
            yerr=[np.std(means[title])], 
            color=c, label=title, alpha=0.7)
    
    
AX[2].legend()
ax2.legend(loc=(1.1,0.2))
ax2.set_ylabel('mean path dist.\n to soma ($\mu$m)')
ax2.set_title('ttest_ind: p=%.1e' % ttest_ind(means['Basket'], means ['Martinotti']).pvalue, fontsize=7)
plt.tight_layout()

fig2.savefig('/home/yann.zerlaut/Desktop/figs/syn-targeting-summary.png', dpi=300)
fig.savefig('/home/yann.zerlaut/Desktop/figs/syn-targeting.png', dpi=300)

# %%
