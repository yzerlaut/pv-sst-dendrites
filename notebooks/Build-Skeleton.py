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
from meshparty import meshwork, trimesh_io
import pcg_skel
import matplotlib.pyplot as plt
from caveclient import CAVEclient

#datastack_name = 'minnie65_public_v117'
datastack_name = 'minnie65_public_v343' 
client = CAVEclient(datastack_name)
client.materialize.version = 343

mesh_folder = 'meshes'
mm = trimesh_io.MeshMeta(cv_path ="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m343",
                         disk_cache_path=mesh_folder,
                         cache_size=0)

# %%
cells = {
         'Basket':{'segID':[864691135269913253,
                            864691135994717610,
                            864691135644647151,
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
for neuron_id in cells['Basket']['segID']:
    try:
        post_mesh = mm.mesh(seg_id = neuron_id, lod=2)
        print('  [ok] --> mesh found for cell:', neuron_id)
    except BaseException as be:
        print('  [X] --> NOT found for cell:', neuron_id)

# %%
for neuron_id in cells['Martinotti']['segID']:
    try:
        post_mesh = mm.mesh(seg_id = neuron_id, lod=2)
        print('  [ok] --> mesh found for cell:', neuron_id)
    except BaseException as be:
        print('  [X] --> NOT found for cell:', neuron_id)

# %%
from meshparty import skeleton, skeletonize, trimesh_io, meshwork
neuron_id = cells['Martinotti']['segID'][1]
mesh = mm.mesh(seg_id = neuron_id, lod=1)
sk = skeletonize.skeletonize_mesh(mesh,
                                  collapse_soma=False,
                                  compute_radius=False,
                                  cc_vertex_thresh=100,
                                  remove_zero_length_edges=False,
                                  meta={
                                        "root_id": root_id,
                                        "skeleton_type": "pcg_skel",
                                        "meta": {"space": "l2cache", "datastack": client.datastack_name}
                                        })

# %%
new_v, new_e, _, new_skel_map = skeletonize.calculate_skeleton_paths_on_mesh(mesh,
                                                                            return_map=True)
sk = skeletonize.Skeleton(new_v, new_e,
                          mesh_to_skel_map=new_skel_map)

# %%
plt.title('plotting skeleton')
plt.scatter(sk.vertices[:,0], sk.vertices[:,1], s=1)
plt.axis('equal')

# %%
nrn = meshwork.Meshwork(mesh, seg_id=neuron_id, skeleton=sk)

# %%
lvl2_eg = client.chunkedgraph.level2_chunk_graph(neuron_id)
cv = client.info.segmentation_cloudvolume(progress=False)
_, l2dict_mesh, _, _ = pcg_skel.build_spatial_graph(lvl2_eg, cv)

# %%
pcg_skel.features.add_synapses(nrn,
                               "synapses_pni_2",
                               l2dict_mesh,
                               client,
                               root_id=neuron_id,
                               pre=True,
                               post=False,
                               remove_self_synapse=True)
                               #timestamp=timestamp,
                               #live_query=live_query)

# %%
# here's a simple way to plot vertices of the skeleton
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# %matplotlib notebook 

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.scatter3D(nrn.skeleton.vertices[:,0], nrn.skeleton.vertices[:,1], nrn.skeleton.vertices[:,2], s=1)

# %%

soma_df = client.materialize.query_table('nucleus_neuron_svm', filter_equal_dict={'pt_root_id': neuron_id})
nucleus_id = soma_df.loc[0].id


# %%
pre_syn_df = nrn.anno.pre_syn.df
# Direct approach:
d_syn_path_um = nrn.distance_to_root( pre_syn_df['pre_pt_mesh_ind'] ) / 1_000

# %%
# Get euclidean distance for each synapse
voxel_resolution = np.array([4,4,40])

soma_loc_nm = soma_df.loc[0]['pt_position'] * voxel_resolution
syn_loc_nm = np.vstack(pre_syn_df['ctr_pt_position'].values) * voxel_resolution

# Simpler version using the filter query object:
# syn_loc_nm = axon_filter.points

d_syn_euc_um = np.linalg.norm(syn_loc_nm - soma_loc_nm, axis=1) / 1_000 # Convert to microns

# %%
# Visualize the two distances as a histogram
# For bins, use 50 micron bins up to a distance of 1000 microns.


bins = np.arange(0, 1000, 50)

fig, ax = plt.subplots(figsize=(4,3), dpi=150)
ax.hist(d_syn_euc_um, bins=bins, label='Euclidean')
ax.hist(d_syn_path_um, bins=bins, label='Path')
ax.legend()


# %%
new_v, new_e, new_skel_map = skel_verts, skel_edges, skel_map
sk = skeletonize.Skeleton(new_v, new_e,
                          mesh_to_skel_map=new_skel_map)

# %%
skel_map_full_mesh = np.full(mesh.node_mask.shape, -1, dtype=np.int64)

# %%
import numpy as np

# %%
skel_map.shape

# %%
# mm.mesh?

# %%
from meshparty import skeleton, skeletonize, trimesh_io, meshwork
neuron_id = cells['Martinotti']['segID'][1]
mesh = mm.mesh(seg_id = neuron_id, 
               lod=2)

# %%
len(mesh.vertices), len(mesh.edges)

# %%
