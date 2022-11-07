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
client.materialize.get_tables()

# %%
'MC' in np.unique(client.materialize.query_table('allen_v1_column_types_slanted').cell_type)

# %%
MCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'MC'})
BCs = client.materialize.query_table('allen_v1_column_types_slanted',
                                     filter_equal_dict={'cell_type':'BC'})

# %%
for mc_id in MCs.pt_root_id:
    print(mc_id)
    

# %%
MCs.pt_root_id

# %%
