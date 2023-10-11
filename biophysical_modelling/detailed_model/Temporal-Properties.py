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
import numpy as np

from cell_template import BRANCH_COLORS
from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


# %%
results = np.load('single_sim.npy', allow_pickle=True).item()
t = np.arange(len(results['Vm_dend']))*results['dt']
plt.plot(t, results['Vm_dend'], lw=0.5)
plt.plot(t, results['Vm_soma'])
plt.show()


# %%
# sim = Parallel(\
        # filename='../../data/detailed_model/Basket_bgStim_sim.zip')

# sim.load()
# sim.fetch_quantity_on_grid('Vm', dtype=object) 
# sim.fetch_quantity_on_grid('Vm_dend', dtype=object) 
# sim.fetch_quantity_on_grid('output_rate', dtype=float) 
# dt = sim.fetch_quantity_on_grid('dt', dtype=float) 
# dt = np.unique(sim.dt)[0]
