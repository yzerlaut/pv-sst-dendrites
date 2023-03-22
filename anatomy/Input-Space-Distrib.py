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
# client.materialize.get_tables() # to explore the available data

# %%
MCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'MC'})
BCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'BC'})


# %%
def load_cell(nrn_h5_file):
    """
    we translate everything in terms of skeleton indices ! (mesh properties)
    """
    nrn = meshwork.load_meshwork(nrn_h5_file)
    
    nrn.post_syn_sites = nrn.skeleton.mesh_to_skel_map[nrn.anno.post_syn.df['post_pt_mesh_ind']]
    
    return nrn


# %%
nrn = load_cell('../data/MC-%s.h5' % MCs.pt_root_id[0])


# %%
nrn.anno.post_syn.df['pre_pt_root_id']

# %%
