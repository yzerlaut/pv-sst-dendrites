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
# REDUCED
cells = {
         'Basket':{'segID':[864691135644647151,
                            #
                            864691136965924814,
                            864691135875962451,
                            864691135307240262,
                            864691136601891793,
                            864691135341149893,
                            864691135212725632,
                            864691135939796646,
                            864691135269913253,
                            864691135396580129,
                            864691135771743563,
                            864691135446872916,
                            #
                            864691135403927534,
                            864691135815629903,
                            864691135737012612,
                            864691135873741966,
                            864691135644647151,
                            864691135807467037,
                            864691135528023492,
                            864691135212725632,
                            864691137197014081,
                            864691135307240262]},
    
         'Martinotti':{'segID':[864691135654096066,
                                864691136483096108,
                                864691136866902638,
                                864691135390890482,
                                864691136296781083,
                                864691135697569813,
                                864691135575230238,
                                864691136056318680,
                                864691136296781083,
                                #
                                864691136871043694,
                                864691136436386718,
                                864691135394616306,
                                864691135584090435]}
        }

# %%
# general packages
import matplotlib.pyplot as plt
import numpy as np

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4
import pcg_skel # version 0.3.0 
from caveclient import CAVEclient # version 4.16.2

datastack_name = 'minnie65_public_v343'
client = CAVEclient(datastack_name)
client.materialize.version = 343

def compute_skeleton_with_synapses(neuron_id,
                                   refine='all', # switch to None for fast computation
                                   voxel_resolution = np.array([4,4,40]),
                                   soma_radius = 10*1000):

    # --------------------------------------------------------
    # get soma position of neuron ID in the "nucleus" database
    # --------------------------------------------------------
    soma = client.materialize.query_table('nucleus_detection_v0',
                                          filter_equal_dict={'pt_root_id':neuron_id})
    soma_pt= soma.loc[0, 'pt_position']*voxel_resolution

    
    # --------------------------------------------------------
    #           compute skeleton with the `pcg_skel` package
    # --------------------------------------------------------
    sk, mesh, (l2dict_mesh, l2dict_mesh_r) = pcg_skel.pcg_skeleton(neuron_id,
                                                                   client=client,
                                                                   refine=refine, # 'all'
                                                                   root_point=soma_pt,
                                                                   root_point_resolution=[1,1,1],
                                                                   collapse_soma=True,
                                                                   collapse_radius=soma_radius,
                                                                   save_to_cache=True,
                                                                   return_mesh=True,
                                                                   return_l2dict_mesh=True,
                                                                   n_parallel=8)
    
    # --------------------------------------------------------
    #           build Meshwork object from mesh and skeleton
    # --------------------------------------------------------
    nrn = meshwork.Meshwork(mesh, 
                            seg_id=neuron_id, 
                            skeleton=sk)

    # -----------------------------------------------------------------------
    #  add synapses from the "synapses_pni_2" database on the reconstruction
    # -----------------------------------------------------------------------
    pcg_skel.features.add_synapses(nrn,
                                   "synapses_pni_2",
                                   l2dict_mesh,
                                   client,
                                   root_id=neuron_id,
                                   pre=True,
                                   post=True,
                                   remove_self_synapse=True)
    
    return nrn


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
              with_pre = False,
              with_post = False,
              with_segID=True,
              color='k',
              clean=False, lw=0.1):
    
    if ax is None:
        ax = plt.gca()

    if with_pre:
        ax.scatter(shift+\
                   nrn.anno.pre_syn.points[:,proj_axis1]/1e3,
                   nrn.anno.pre_syn.points[:,proj_axis2]/1e3,
                   color='tomato', s=8, alpha=.5, label='pre syn.')

    if with_post:
        ax.scatter(shift+\
                   nrn.anno.post_syn.points[:,proj_axis1]/1e3,
                   nrn.anno.post_syn.points[:,proj_axis2]/1e3,
                   color='turquoise', s=8, alpha=.5, label='post syn.')
    
    if clean:
        for cover_path in nrn.skeleton.cover_paths:
            path_verts = nrn.skeleton.vertices[cover_path,:]
            ax.plot(shift+\
                    path_verts[:,proj_axis1]/1e3, 
                    path_verts[:,proj_axis2]/1e3,
                    color=color, lw=lw)

    else:
        ax.plot(shift+\
                   nrn.skeleton.vertices[:, proj_axis1]/1e3,
                nrn.skeleton.vertices[:, proj_axis2]/1e3, 
                '_', lw=0, ms=0.03, color=color)
        
    if with_segID:
        ax.annotate('\n%s' % nrn.seg_id,
                    (shift+nrn.anno.post_syn.points[:,proj_axis1].min()/1e3,
                     nrn.anno.post_syn.points[:,proj_axis2].min()/1e3), 
                    xycoords='data', fontsize=5)


    # plot soma
    ax.scatter(shift+\
                   nrn.skeleton.vertices[:, proj_axis1][nrn.skeleton.root]/1e3,
               nrn.skeleton.vertices[:, proj_axis2][nrn.skeleton.root]/1e3,
               s=40,color='violet', label='put. soma')


# %%
neuron_id = cells['Basket']['segID'][0]
nrn = compute_skeleton_with_synapses(neuron_id, refine=None)

# %%
nrn.save_meshwork('bc-example.h5')

# %%
nrn = meshwork.load_meshwork('bc-example.h5')

# %%
#has_inds = np.full(nrn.skeleton.n_vertices, 0)
#W = meshwork.utils.window_matrix(nrn.skeleton, 2000)

#print(has_inds.sum())
nrn._convert_to_meshindex(nrn.anno.post_syn.df['post_pt_mesh_ind'])

# %%

distances = nrn.skeleton_property_to_mesh(nrn.distance_to_root(nrn.skeleton.mesh_index))/1_000

width = 2000

rho = nrn.linear_density(nrn.anno.post_syn.df['post_pt_mesh_ind'],
                         width=width, normalize=False, exclude_root=True)

plt.plot(distances, rho/width, '.')

# %%
# nrn.linear_density?

# %%
plot_cell(nrn, clean=True, lw=0.2, with_post=True)

# %%
for cType in ['Basket', 'Martinotti']:
    
    print('\n --- \n %s ' % cType)
    
    cells[cType]['nrn'] = []
    
    for neuron_ID in cells[cType]['segID']:
        
        try:
            cells[cType]['nrn'].append(compute_skeleton_with_synapses(neuron_ID))
            print('  [ok] --> skeleton built for cell:', neuron_id)

        except BaseException as be:
            print(be)
            print('  [X] --> FAILED for cell:', cells[cType]['segID'][n])

# %%
bins = np.linspace(30, 250, 50) # um

fig, AX = plt.subplots(3, 1, figsize=(6,3))
for ax, key, color in zip(AX, ['Basket', 'Martinotti'], [plt.cm.tab10(0), plt.cm.tab10(1)]):
    Density = []
    for nrn in cells[key]['nrn']:
        try:
            d_syn_path_um = nrn.distance_to_root(nrn.anno.post_syn.df['post_pt_mesh_ind'] ) / 1_000
            hist, be = np.histogram(d_syn_path_um, bins=bins)
            Density.append(hist/(bins[1]-bins[0])) # linear density
            ax.plot(0.5*(be[1:]+be[:-1]), Density[-1], color=color)
        except BaseException as be:
            pass
    norm = np.mean(Density, axis=0).max()
    mean, std = np.mean(Density, axis=0)/norm, np.std(Density, axis=0)/norm
    AX[2].plot(0.5*(be[1:]+be[:-1]), mean, color=color)
    AX[2].fill_between(0.5*(be[1:]+be[:-1]), mean-std, mean+std, color=color, alpha=0.3, lw=0)
    
    ax.set_ylabel('lin. density\n (count/$\mu$m)')
    ax.annotate('%s \n(n=%i) ' % (key, len(cells[key]['nrn'])),
                (1,0.9), ha='right', va='top', color=color, xycoords='axes fraction')
_ = plt.xlabel('path length from soma ($\mu$m)')
_ = plt.ylabel('density\n peak norm.')

# %%
fig, AX = build_fig('Basket')
fig.suptitle('Basket cells', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/basket.png', dpi=300)

# %%
fig, AX = build_fig('Basket', with_post=True)
fig.suptitle('Basket cells: "post" sites', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/basket-post.png', dpi=300)

# %%
fig, AX = build_fig('Basket', with_pre=True)
fig.suptitle('Basket cells: "pre" sites', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/basket-pre.png', dpi=300)

# %%
fig, AX = build_fig('Martinotti')
fig.suptitle('Martinotti cells', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/martinotti.png', dpi=300)

# %%
fig, AX = build_fig('Martinotti', with_pre=True)
fig.suptitle('Martinotti cells', fontsize=10)
fig.suptitle('Martinotti cells: "pre" sites', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/martinotti-pre.png', dpi=300)

# %%
fig, AX = build_fig('Martinotti', with_post=True)
fig.suptitle('Martinotti cells: "post" sites', fontsize=10)
fig.savefig('/home/yann.zerlaut/Desktop/figs/martinotti-post.png', dpi=300)

# %%
