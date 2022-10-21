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

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4

nrn = meshwork.load_meshwork('../data/Basket-864691135341149893.h5')


# %%
# ls ../data

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
              with_pre = False,
              with_post = False,
              with_segID=True,
              color='k',
              clean=True, lw=0.2):
    
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
    
    # plotting using the "cover_paths"
    for cover_path in nrn.skeleton.cover_paths:
        path_verts = nrn.skeleton.vertices[cover_path,:]
        ax.plot(shift+\
                path_verts[:,proj_axis1]/1e3, 
                path_verts[:,proj_axis2]/1e3,
                color=color, lw=lw)

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
nrn.skeleton.path_length(nrn.skeleton.cover_paths[0])

# %%
fig, AX = plt.subplots(2, 3, figsize=(15,7))

path = nrn.skeleton.cover_paths[0] # 0, 10

bins = np.linspace(0, 800, 30)

for ax, title in zip(AX[0][:3], ['cell', 'single path', 'w. synapses']):
    plot_cell(nrn, ax=ax)
    ax.set_title(title)

for ax in AX[0][1:3]:
    ax.plot(nrn.skeleton.vertices[path,0]/1e3, 
            nrn.skeleton.vertices[path,1]/1e3, color='r', lw=2)
    ax.plot(nrn.vertices[path,0]/1e3+nrn.skeleton.vertices[nrn.root,0]/1e3, 
            nrn.vertices[path,1]/1e3+nrn.skeleton.vertices[nrn.root,1]/1e3, color='r', lw=2)

synapses_pt_in_path = np.array([d for d in nrn.anno.post_syn.df['post_pt_mesh_ind'] if (d in path)])

# need to translate synapses into 
#post_pt_skel_ind = [nrn.skeleton.mesh_to_skel_map[pt] for pt in nrn.anno.post_syn.df['post_pt_mesh_ind']]
#synapses_pt_in_path = np.array([d for d in post_pt_skel_ind if (d in path)])


print(len(synapses_pt_in_path))

if len(synapses_pt_in_path)>0:
    AX[0][2].plot(nrn.skeleton.vertices[synapses_pt_in_path,0]/1e3+3*np.random.randn(len(synapses_pt_in_path)), 
                  nrn.skeleton.vertices[synapses_pt_in_path,1]/1e3+3*np.random.randn(len(synapses_pt_in_path)),
                  'b.', ms=4)

path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

AX[1][0].plot(path, path_to_soma)
AX[1][0].set_xlabel('path-point index'); AX[1][0].set_ylabel('path dist. to soma (um)')

count_along_path = np.zeros(len(path))
for i, p in enumerate(path):
    count_along_path[i] = np.sum(nrn.anno.post_syn.df['post_pt_mesh_ind']==p)
    
AX[1][1].plot(path, count_along_path, 'b.')
AX[1][1].set_xlabel('path-point index'); AX[1][1].set_ylabel('synaptic count')

binned_dist = np.digitize(path_to_soma, bins=bins)
density_hist = np.zeros(len(bins))
for b in range(len(bins)):
    # we sum all synapses that fall into this bin and we divide by the bin length
    density_hist[b] = np.sum(count_along_path[binned_dist==b])/(bins[1]-bins[0])

AX[1][2].plot(bins, density_hist, 'b-')
AX[1][2].set_xlabel('path dist. to soma ($\mu$m)'); AX[1][2].set_ylabel('linear density (syn./$\mu$m)')
  

# %%
fig, AX = plt.subplots(1, 2, figsize=(12,4))

bins = np.linspace(0, 800, 30)

DENSITY_HIST = []
for path in nrn.skeleton.cover_paths:

    # we plot all paths with a different color
    AX[0].plot(nrn.skeleton.vertices[path,0]/1e3, 
            nrn.skeleton.vertices[path,1]/1e3,)

    path_to_soma = [nrn.skeleton.distance_to_root[p]/1_000 for p in path]

    count_along_path = np.zeros(len(path))
    for i, p in enumerate(path):
        count_along_path[i] = np.sum(nrn.anno.post_syn.df['post_pt_mesh_ind']==p)

    binned_dist = np.digitize(path_to_soma, bins=bins)
    density_hist = np.ones(len(bins))*np.nan # nan by default
    for b in range(len(bins)):
        if np.sum(binned_dist==b)>0:
            # we sum all synapses that fall into this bin and we divide by the bin length
            density_hist[b] = np.sum(count_along_path[binned_dist==b])/(bins[1]-bins[0])

    # we 
    DENSITY_HIST.append(density_hist)
    
AX[1].plot(bins, np.nanmean(DENSITY_HIST, axis=0)) # only non-infinite values contributing

AX[0].set_title('looping over paths')

AX[1].set_xlabel('path dist. to soma ($\mu$m)'); 
AX[1].set_ylabel('linear density (syn./$\mu$m)')

# %%
# np.nanmean?

# %%
fig, AX = plt.subplots(1, 2, figsize=(8,4))
AX[1].plot(bins, np.mean(DENSITY_HIST, axis=0))
AX[1].fill_between(bins, 
           np.mean(DENSITY_HIST, axis=0)-np.std(DENSITY_HIST, axis=0),
           np.mean(DENSITY_HIST, axis=0)+np.std(DENSITY_HIST, axis=0), alpha=0.4)

# %%
print(np.max(nrn.anno.post_syn.df['post_pt_mesh_ind']))
print(np.max(path))
print(np.max(nrn.mesh_indices))
print(len(nrn.skeleton.vertices), np.max(nrn.skeleton.cover_paths[0]))

# %%
mesh_to_skel_indices = nrn._convert_to_skelindex(nrn.mesh_indices)
len(nrn.mesh_indices), len(np.unique(nrn.mesh_indices)), np.max(nrn.skeleton_indices)

# %%
nrn.skeleton.vertices[nrn.root,0]

# %%
len(nrn._convert_to_skelindex(nrn.mesh_indices))

# %%
nrn.skeleton.path_length

# %%
#has_inds = np.full(nrn.skeleton.n_vertices, 0)
#W = meshwork.utils.window_matrix(nrn.skeleton, 2000)
path = nrn.skeleton.cover_paths[0]
#print(has_inds.sum())
#nrn._convert_to_meshindex()

# %%
distances = nrn.skeleton_property_to_mesh(nrn.distance_to_root(nrn.skeleton.mesh_index))/1_000

width = 2500

rho = nrn.linear_density(nrn.anno.post_syn.df['post_pt_mesh_ind'],
                         width=width, 
                         normalize=True, 
                         exclude_root=True)

bins = np.linspace(5, 500)
dist_indices = np.digitize(distances, bins=bins)
mean_density, std_density = np.zeros(len(bins)), np.zeros(len(bins))

for b in range(len(bins)):
    if np.sum(dist_indices==b)>0:
        mean_density[b] = np.mean(rho[dist_indices==b])
        std_density[b] = np.std(rho[dist_indices==b])

plt.plot(0.5*(bins[:-1]+bins[1:]), 1e3*mean_density[1:])
plt.fill_between(0.5*(bins[:-1]+bins[1:]), 
                 1e3*mean_density[1:]-1e3*std_density[1:],
                 1e3*mean_density[1:]+1e3*std_density[1:], alpha=.3)

# %%
# nrn.linear_density?

# %%
plt.hist(rho[np.isfinite(rho)])

# %%
import pcg_skel # version 0.3.0 
# pcg_skel.pcg_skeleton?

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
