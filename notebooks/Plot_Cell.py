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
# general packages
import matplotlib.pyplot as plt
import numpy as np
import os

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4

# %% [markdown]
# https://github.com/AllenInstitute/swdb_2022/blob/main/DynamicBrain/EM_reference_Materials/EM_Meshwork_Creation.ipynb

# %%
# our handpicked cells:
cells = {'Basket': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'Basket' in fn],
         'Martinotti': [os.path.join('..', 'data', fn) for fn in os.listdir('../data') if 'Martinotti' in fn]}

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


# %%
# example
nrn = load_cell(cells['Basket'][1])


# %%
def build_fig(key,
              cmap=plt.cm.magma,
              with_pre = False,
              with_post = False,
              Ncol = 5):
    
    Nrow = int(len(cells[key]['nrn'])/Ncol+0.99)

    fig, AX = plt.subplots(Nrow, 1, 
                           figsize=(12, 2*Nrow), frameon=False)
    
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.suptitle(key+ ' cells', fontsize=10)

    for i in range(len(cells[key]['nrn'])):

        plot_cell(cells[key]['nrn'][i], AX[int(i/Ncol)],
                  shift=i*5e2,
                  with_pre = with_pre,
                  with_post = with_post,
                  color=cmap(0.25+i/len(cells[key]['nrn'])/2.))
        
    for ax in AX:
        ax.invert_yaxis()
        ax.axis('equal')
        ax.plot(ax.get_xlim()[0]*np.ones(2), ax.get_ylim()[1]+200*np.arange(2), 'k-', lw=1)
        ax.annotate('200$\mu$m ', (ax.get_xlim()[0], ax.get_ylim()[1]), rotation=90, ha='right', va='top')
        ax.axis('off')
        
    return fig, AX

def plot_cell(nrn, 
              ax=None, 
              shift=0,
              proj_axis1=0, proj_axis2=1, 
              dend_color='orange',
              no_axon=False,
              no_soma=False,
              with_pre = False,
              with_post = False,
              with_segID=True,
              color='k',
              synapse_size=6,
              synapse_color='blue',
              synapse_alpha=0.5,
              clean=True, 
              lw=0.2):
    
    if ax is None:
        ax = plt.gca()

    if with_pre:
        ax.scatter(shift+\
                   nrn.anno.pre_syn.points[:,proj_axis1]/1e3,
                   nrn.anno.pre_syn.points[:,proj_axis2]/1e3,
                   color='tomato', s=synapse_size,
                   alpha=synapse_alpha, label='pre syn.')

    if with_post:
        ax.scatter(shift+\
                   nrn.anno.post_syn.points[:,proj_axis1]/1e3,
                   nrn.anno.post_syn.points[:,proj_axis2]/1e3,
                   color=synapse_color, s=synapse_size,
                   alpha=synapse_alpha, label='post syn.')
    
    # plotting using the "cover_paths"
    if not no_axon:
        for cover_path in nrn.skeleton.cover_paths:
            # dendrites
            path_verts = nrn.skeleton.vertices[cover_path,:]
            ax.plot(shift+\
                    path_verts[:,proj_axis1]/1e3, 
                    path_verts[:,proj_axis2]/1e3,
                    color=color, lw=lw)
        
    if hasattr(nrn, 'dend_cover_paths'):
        if not no_axon:
            ax.annotate('\n dendrites ', (1,1), xycoords='axes fraction', color='orange', va='top', ha='right')
        for cover_path in nrn.dend_cover_paths:
            # dendrites
            path_verts = nrn.skeleton.vertices[cover_path,:]
            ax.plot(shift+\
                    path_verts[:,proj_axis1]/1e3, 
                    path_verts[:,proj_axis2]/1e3,
                    color=dend_color, lw=2*lw)

    if with_segID:
        ax.annotate('\n%s' % nrn.seg_id,
                    (shift+nrn.anno.post_syn.points[:,proj_axis1].min()/1e3,
                     nrn.anno.post_syn.points[:,proj_axis2].min()/1e3), 
                    xycoords='data', fontsize=5)


    # plot soma
    if not no_soma:
        ax.scatter(shift+\
                       nrn.skeleton.vertices[:, proj_axis1][nrn.skeleton.root]/1e3,
                   nrn.skeleton.vertices[:, proj_axis2][nrn.skeleton.root]/1e3,
                   s=40,color='violet', label='put. soma')
        
def set_view_with_scale(ax, x0, y0, width, bar,
                        frac=0.1):
    ax.set_xlim([x0, x0+width])
    ax.set_ylim([y0, y0+width])
    dx = frac*width
    ax.plot([x0+dx, x0+dx], [y0+dx, y0+dx+bar], 'k', lw=1)
    ax.plot([x0+dx, x0+dx+bar], [y0+dx, y0+dx], 'k', lw=1)
    ax.annotate(' %ium\n'%bar, (x0+dx, y0+dx))
    ax.axis('off')
   


# %%
fig, AX = plt.subplots(1, 2, figsize=(8,4))
plt.subplots_adjust(left=0, bottom=0, top=1., right=1, wspace=0)

nrn = load_cell('../data/BC-864691135100167712.h5')

AX[0].annotate('Basket cell', (1,1), ha='center', va='top', xycoords='axes fraction')

# large view
x0, y0, width = 500, 350, 400
plot_cell(nrn, ax=AX[0],
          with_post=True, no_axon=True, no_soma=True, with_segID=False, 
          dend_color='k', synapse_color='r', synapse_alpha=1.,
          synapse_size=0.02)
set_view_with_scale(AX[0], x0, y0, width, 50)

# zoom view
x0, y0, width = 610, 450, 200
plot_cell(nrn, ax=AX[1],
          with_post=True, no_axon=True, no_soma=True, with_segID=False, 
          dend_color='k', synapse_color='r', synapse_alpha=1.,
          synapse_size=0.02)
set_view_with_scale(AX[1], x0, y0, width, 10)

fig.savefig('/home/yann.zerlaut/Desktop/Basket.png', dpi=300)

# %%
fig, AX = plt.subplots(1, 2, figsize=(8,4))
plt.subplots_adjust(left=0, bottom=0, top=1., right=1, wspace=0)

nrn = load_cell('../data/MC-864691135467660940.h5')

AX[0].annotate('Martinotti cell', (1,1), ha='center', va='top', xycoords='axes fraction')

# large view
x0, y0, width = 550, 580, 400
plot_cell(nrn, ax=AX[0],
          with_post=True, no_axon=True, no_soma=True, with_segID=False, 
          dend_color='k', synapse_color='r', synapse_alpha=1.,
          synapse_size=0.02)
set_view_with_scale(AX[0], x0, y0, width, 50)

# zoom view
x0, y0, width = 630, 650, 200
plot_cell(nrn, ax=AX[1],
          with_post=True, no_axon=True, no_soma=True, with_segID=False, 
          dend_color='k', synapse_color='r', synapse_alpha=1.,
          synapse_size=0.02)
set_view_with_scale(AX[1], x0, y0, width, 10)

fig.savefig('/home/yann.zerlaut/Desktop/Martinotti.png', dpi=300)
