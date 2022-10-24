# general packages
import os
import matplotlib.pyplot as plt
import numpy as np

# packages from Allen Institute:
from meshparty import meshwork # version 1.16.4
import pcg_skel # version 0.3.0 
from caveclient import CAVEclient # version 4.16.2

def compute_meshwork_with_synapses(neuron_id,
                                   client,
                                   refine=None, # switch to None for fast computation
                                   voxel_resolution = np.array([4,4,40]),
                                   soma_radius = 20*1000):

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

if __name__=='__main__':

    import sys
    import pandas as pd

    if '.csv' in sys.argv[-1]:

        # run client
        datastack_name = 'minnie65_public_v343'
        client = CAVEclient(datastack_name)
        client.materialize.version = 343


        # load cell database
        df = pd.read_csv(sys.argv[-1])

        for cell_type in np.unique(df['cell-type']):

            print('\n   ---- %s ---- \n ' % cell_type)

            for root_id in df['root-ID'][df['cell-type']==cell_type]:
              
                neuron_id = int(root_id.replace('"', '')) 

                filename = 'data/%s-%s.h5' % (cell_type, neuron_id) 

                print('\n- fetching and saving %s [...]' % filename)

                if not os.path.isfile(filename):
                    try:
                        nrn  = compute_meshwork_with_synapses(neuron_id,
                                                              client,
                                                              refine='all') # switch to None for testing

                        nrn.save_meshwork(filename)
                        print('        ----> succeded [V]')

                    except BaseException as be:
                        print('        ----> failed [X]')



    else:

        print('need to provide a csv file')


