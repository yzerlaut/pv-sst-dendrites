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
# # Simulation of Morphologically-detailed models

# %%
import sys, os, json, pandas
import numpy as np


sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils.params import load as load_params

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt
from src.morphology import load as load_morphology

# %% [markdown]
# ## Basket cell

# %%
#cell = load_morphology('864691135396580129_296758')
cell = load_morphology('864691137053906294_301107')
Params = load_params('PV-parameters.json')

# %%
fig, ax = plt.subplots(1, figsize=(10,10))
vis = nrnvyz(cell.SEGMENTS)
vis.plot_segments(cond=(cell.SEGMENTS['comp_type']!='axon'),
                  bar_scale_args={'Ybar':100, 'Xbar':1e-9,
                                  'Ybar_label':'100$\mu$m ', 'fontsize':6}, ax=ax)
# plot number of synapses in each segment !
vis.add_dots(ax, range(len(cell.SEGMENTS['x'])), cell.SEGMENTS['Nsyn']/4)

# %% [markdown]
# ### Input Impedance Characterization

# %%
net = nrn.Network()

EL = Params['EL']*nrn.mV
#gL = 2*nrn.psiemens/nrn.umeter**2
eqs='''
Im = gL * (EL - v) : amp/meter**2
gL : siemens/meter**2
I : amp (point current)
'''
neuron = nrn.SpatialNeuron(morphology=cell.morpho, model=eqs,
                           Cm=Params['cm'] * nrn.uF / nrn.cm ** 2,
                           Ri=Params['Ri'] * nrn.ohm * nrn.cm)
neuron.v = Params['EL']*nrn.mV

for comp in ['axon', 'dend', 'soma']:
    neuron.gL[cell.SEGMENTS['comp_type']==comp] = Params['gL_%s' % comp]*nrn.psiemens/nrn.um**2

net.add(neuron)

# recording
#M = nrn.StateMonitor(neuron, ('v'), record=[0]) # soma + dend loc
#net.add(M)

# relax
net.run(30.*nrn.ms)

results = {'distance_to_soma':[],
           'input_res' : []}

step = 50. # pA
for comp, nsyn in enumerate(cell.SEGMENTS['Nsyn']):
    # current step
    neuron.I[comp] = step*nrn.pA
    net.run(50.*nrn.ms)
    for n in range(nsyn):
        # we just duplicate the value according to the number of synapses in this compartment
        results['distance_to_soma'].append(cell.SEGMENTS['distance_to_soma'][comp])
        results['input_res'].append(1e3*(neuron.v[0]/nrn.mV-Params['EL'])/step) # MOhm
    # relaxation
    neuron.I[comp] = 0*nrn.pA
    net.run(50.*nrn.ms)
    
for key in results.keys():
    results[key] = np.array(results[key])
    
#t, v = M.t/nrn.second, M.v[0]/nrn.mV
#net.remove(M)
net.remove(neuron)
net, M, neuron = None, None, None


# %%
bins = np.linspace(0, 200, 100)

binned = np.digitize(1e6*np.array(results['distance_to_soma']), bins=bins)

fig, ax = pt.plt.subplots(1, figsize=(2,1))

for b in np.unique(binned):
    cond = (binned==b)
    ax.errorbar([bins[b]], [np.mean(np.array(results['input_res'])[cond])],
                yerr=[np.std(np.array(results['input_res'])[cond])], fmt='ko-', ms=2, lw=1)
pt.set_plot(ax, xlim=[0,200])

# %%
binned==1

# %%
binned

# %%
pt.plt.hist(1e6*np.array(results['distance_to_soma']))

# %%
fig, ax = pt.plt.subplots(1, figsize=(5,1.4))
ax.plot(t, v)
ax.set_ylabel('$V_m$ (mV)')
ax.set_xlabel('time (ms)')

# %%
COMP_LIST, COMP_NAMES, NSEG_INDICES = nrn.morpho_analysis.get_compartment_list(cell.morpho)

# %%
getattr(neuron.morphology, COMP_NAMES[1])

# %%
NSEG_INDICES

# %%
len(cell.SEGMENTS['Nsyn'])

# %%
len(np.concatenate(NSEG_INDICES))

# %%
cell.SEGMENTS.keys()

# %%
