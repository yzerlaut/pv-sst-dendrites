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

sys.path.append('../src')
import morphology

# %% [markdown]
# ## Basket cell

# %%
#cell = load_morphology('864691135396580129_296758')
cell = morphology.load('864691137053906294_301107')
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

for comp in np.unique(cell.synapses_morpho_index):
    
    # current step
    neuron.I[comp] = step*nrn.pA
    net.run(50.*nrn.ms)
    
    for n in range(cell.SEGMENTS['Nsyn'][comp]):
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
bins = np.linspace(0, 200, 20)

binned = np.digitize(1e6*np.array(results['distance_to_soma']), bins=bins)

fig, ax = pt.plt.subplots(1, figsize=(2,1))

for b in np.unique(binned)[:-1]:
    cond = (binned==b)
    ax.errorbar([bins[b]], [np.mean(np.array(results['input_res'])[cond])],
                yerr=[np.std(np.array(results['input_res'])[cond])], fmt='ko-', ms=2, lw=1)
pt.set_plot(ax, xlim=[0,210], xlabel='dist to soma ($\mu$m)', ylabel='M$\Omega$')

# %% [markdown]
# ## Responses to current steps

# %%
net = nrn.Network()

nrn.defaultclock.dt = 0.025*nrn.ms

# Starting from an empty equation string:
Equation_String= '''
Im = + 0*amp/meter**2 : amp/meter**2
vc = clip(v/mV, -90, 50) : 1 # UNITLESS CLIPPED VOLTAGE, useful for mechanisms
I_inj : amp (point current)
'''

# intrinsic currents
Na_inactivation_speed_factor = 3.0
CURRENTS = [nrn.PassiveCurrent(name='Pas', params={'El':-68}),
            nrn.SodiumChannelCurrent(name='Na', params={'tha':-35,
                                                    'E_Na':55.,
                                                    'Ra':Na_inactivation_speed_factor*0.182,
                                                    'Rb':Na_inactivation_speed_factor*0.124}),
            nrn.DelayedRectifierPotassiumChannelCurrent(name='K')]


for current in CURRENTS:
    Equation_String = current.insert(Equation_String)

eqs = nrn.Equations(Equation_String)

neuron = nrn.SpatialNeuron(morphology=cell.morpho, model=eqs,
                           Cm=Params['cm'] * nrn.uF / nrn.cm ** 2,
                           Ri=Params['Ri'] * nrn.ohm * nrn.cm)
net.add(neuron)

# initial conditions:
neuron.v = Params['EL']*nrn.mV

for current in CURRENTS:
    current.init_sim(neuron)

## -- PASSIVE PROPS -- ##
neuron.gbar_Pas = 1.3e-4*nrn.siemens/nrn.cm**2

## -- SPIKE PROPS (Na & Kv) -- ##
# soma
neuron.gbar_Na = 1.3e-1*nrn.siemens/nrn.cm**2
neuron.gbar_K =  3.6e-2*nrn.siemens/nrn.cm**2
# dendrites
neuron.dend.gbar_Na = 0*nrn.siemens/nrn.cm**2 #40*1e-12*siemens/um**2
neuron.dend.gbar_K = 0*nrn.siemens/nrn.cm**2 #30*1e-12*siemens/um**2
# neuron.dend.distal.gbar_Na = 40*1e-12*siemens/um**2
# neuron.dend.distal.gbar_K = 30*1e-12*siemens/um**2

soma_loc, dend_loc = 0, 2
mon = nrn.StateMonitor(neuron, ['v', 'I_inj'], record=[soma_loc, dend_loc])
net.add(mon)

net.run(100*nrn.ms)
neuron.main.I_inj = 200*nrn.pA
net.run(200*nrn.ms)
neuron.main.I_inj = 0*nrn.pA
net.run(100*nrn.ms)
neuron.dend.I_inj = 200*nrn.pA
net.run(200*nrn.ms)
neuron.dend.I_inj = 0*nrn.pA
net.run(200*nrn.ms)

# %%
import matplotlib.pylab as plt
fig, AX = plt.subplots(3,1, figsize=(12,4))

AX[0].plot(mon.t / nrn.ms, mon[soma_loc].v/nrn.mV, color='blue', label='soma')
AX[0].plot(mon.t / nrn.ms, mon[dend_loc].v/nrn.mV, color='red', label='dend')
AX[0].set_ylabel('Vm (mV)')
AX[0].legend()

net.remove(neuron)
net.remove(mon)

net, M, neuron = None, None, None

# %%

# %%
