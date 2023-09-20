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

# %%
ID = '864691135396580129_296758' # Basket Cell example
#ID = '864691135571546917_264824' # Martinotti Cell example

cell = morphology.load(ID)

# %%
net = nrn.Network()

nrn.defaultclock.dt = 0.025*nrn.ms

Equation_String= """
Im = + 0*amp/meter**2 : amp/meter**2
vc = clip(v/mV, -90, 50) : 1 # UNITLESS CLIPPED VOLTAGE, needed for mechanisms
I_inj : amp (point current)
"""

# intrinsic currents
Na_inactivation_speed_factor = 5.
CURRENTS = [nrn.PassiveCurrent(name='Pas', params={'El':Params['EL']}),
            nrn.SodiumChannelCurrent(name='Na', params={'tha':-35,
                                                        'E_Na':55.,
                                                        'Ra':Na_inactivation_speed_factor*0.182,
                                                        'Rb':Na_inactivation_speed_factor*0.124}),
            nrn.DelayedRectifierPotassiumChannelCurrent(name='K'),
            nrn.ATypePotassiumCurrentProximal(name='KAprox'),
            nrn.ATypePotassiumCurrentDistal(name='KAdist')]
CURRENTS = [nrn.PassiveCurrent(name='Pas', params={'El':Params['EL']}),
            nrn.SodiumChannelCurrent(name='Na', params={'tha':-35,
                                                        'E_Na':55.,
                                                        'Ra':Na_inactivation_speed_factor*0.182,
                                                        'Rb':Na_inactivation_speed_factor*0.124}),
            nrn.PotassiumChannelCurrent(name='K')]

for current in CURRENTS:
    Equation_String = current.insert(Equation_String)

eqs = nrn.Equations(Equation_String)

neuron = nrn.SpatialNeuron(morphology=cell.morpho,
                           model=eqs,
                           Cm=Params['cm'] * nrn.uF / nrn.cm ** 2,
                           Ri=Params['Ri'] * nrn.ohm * nrn.cm)
net.add(neuron)

# initial conditions:
neuron.v = Params['EL']*nrn.mV

for current in CURRENTS:
    current.init_sim(neuron)

CONDS = {\
    'soma': (cell.SEGMENTS['comp_type']=='soma'),
    'prox': (cell.SEGMENTS['comp_type']=='dend') & (cell.SEGMENTS['distance_to_soma']<100e-6),
    'dist': (cell.SEGMENTS['comp_type']=='dend') & (cell.SEGMENTS['distance_to_soma']>=100e-6),
    'axon': (cell.SEGMENTS['comp_type']=='axon')}

for c in CONDS:

    ## --       PASSIVE CURRENT       -- ##
    neuron.gbar_Pas[CONDS[c]] = Params['gPas_%s'%c]*nrn.siemens/nrn.cm**2
    ## --       SODIUM CURRENT        -- ##
    neuron.gbar_Na[CONDS[c]] = 0*Params['gNa_%s'%c]*nrn.siemens/nrn.cm**2
    ## --     POTASSIUM CURRENT       -- ##
    neuron.gbar_K[CONDS[c]] = 0*Params['gK_%s'%c]*nrn.siemens/nrn.cm**2
    """
    ## -- PROX A-TYPE K CURRENT       -- ##
    neuron.gbar_KAprox[CONDS[c]] = Params['gKAprox_%s'%c]*nrn.siemens/nrn.cm**2
    ## --  DIST A-TYPE K CURRENT      -- ##
    neuron.gbar_KAdist[CONDS[c]] = Params['gKAdist_%s'%c]*nrn.siemens/nrn.cm**2
    """

soma_loc, dend_loc = 0, 2
mon = nrn.StateMonitor(neuron, ['v'], record=[soma_loc])
net.add(mon)

net.run(1*nrn.ms)
# net.run(50*nrn.ms)
# neuron.I_inj[0] = 200*nrn.pA
# net.run(200*nrn.ms)

# %%
# # # ## Run the various variants of the model to reproduce Figure 12
import matplotlib.pylab as plt
fig, ax = plt.subplots(3,1, figsize=(8,2))

ax.plot(mon.t / nrn.ms, mon[soma_loc].v/nrn.mV, color='blue', label='soma')
ax.set_ylabel('Vm (mV)')
ax.legend()

net.remove(neuron)
net.remove(mon)

net, M, neuron = None, None, None

plt.show()

# %%
# # # ##
