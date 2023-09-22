# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Current Step Responses in the Morphologically-detailed models

# %%
exec(open('PV_template.py').read())

import sys, os, json, pandas
import numpy as np

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils.params import load as load_params

sys.path.append('..')

import plot_tools as pt
import matplotlib.pylab as plt

sys.path.append('../src')
import morphology

# load parameters
Params = load_params('PV-parameters.json')

# %% [markdown]
# ## Load the detailed morphology

"""
# %%
ID = '864691135396580129_296758' # Basket Cell example
#ID = '864691135571546917_264824' # Martinotti Cell example

cell = morphology.load(ID)
print(len(cell.SEGMENTS['x']))

cell = PVcell(ID=ID, debug=True)

ic = h.IClamp(cell.soma[0](0.5))
ic.amp = 0. 
ic.dur =  1e9 * ms
ic.delay = 0 * ms

dt, tstop = 0.025, 500

t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
Vm = h.Vector()

Vm.record(cell.soma[0](0.5)._ref_v)

h.finitialize()

for i in range(int(50/dt)):
    h.fadvance()

for i in range(1, 11):

    ic.amp = i/10.
    for i in range(int(100/dt)):
        h.fadvance()
    ic.amp = 0
    for i in range(int(100/dt)):
        h.fadvance()

import matplotlib.pylab as plt
plt.figure(figsize=(9,3))
plt.plot(np.arange(len(Vm))*dt, np.array(Vm))
plt.show()

"""

