# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Active Mechanism in Morphologically-detailed models

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

Params = load_params('PV-parameters.json')

# %% [markdown]
# ## On a reduced morphology

net = nrn.Network()

nrn.defaultclock.dt = 0.025*nrn.ms

# Starting from an empty equation string:
Equation_String= '''
Im = + 0*amp/meter**2 : amp/meter**2
vc = clip(v/mV, -90, 50) : 1 # UNITLESS CLIPPED VOLTAGE, useful for mechanisms
I_inj : amp (point current)
'''

# calcium dynamics following: HighVoltageActivationCalciumCurrent + LowThresholdCalciumCurrent
Equation_String = nrn.CalciumConcentrationDynamics(contributing_currents='IHVACa',
                                               name='CaDynamics').insert(Equation_String)

# intrinsic currents
Na_inactivation_speed_factor = 10 
CURRENTS = [nrn.PassiveCurrent(name='Pas', params={'El':Params['EL']}),
            nrn.SodiumChannelCurrent(name='Na', params={'tha':-35,
                                                        'E_Na':55.,
                                                        'Ra':Na_inactivation_speed_factor*0.182,
                                                        
                                                        'Rb':Na_inactivation_speed_factor*0.124}),
            nrn.DelayedRectifierPotassiumChannelCurrent(name='K'),
            nrn.SlowlyInactivatingPotassiumCurrent(name='Kslowin'),
            nrn.HighVoltageActivationCalciumCurrent(name='HVACa'),
            nrn.LowThresholdCalciumCurrent(name='T'),
            nrn.MuscarinicPotassiumCurrent(name='Musc'),
            nrn.CalciumDependentPotassiumCurrent(name='KCa')]


for current in CURRENTS:
    Equation_String = current.insert(Equation_String)

eqs = nrn.Equations(Equation_String)

class Cell:
    def __init__(self):
        self.morpho = nrn.Cylinder(x=[0, 30]*nrn.um, diameter=20*nrn.um)
        self.morpho.dend = nrn.Cylinder(x=[0, 20]*nrn.um, diameter=10*nrn.um)
        self.morpho.dend.distal = nrn.Cylinder(x=[0, 500]*nrn.um, diameter=3*nrn.um)
cell = Cell()

neuron = nrn.SpatialNeuron(morphology=cell.morpho, model=eqs,
                           Cm=Params['cm'] * nrn.uF / nrn.cm ** 2,
                           Ri=Params['Ri'] * nrn.ohm * nrn.cm)
net.add(neuron)

# initial conditions:
neuron.v = Params['EL']*nrn.mV
neuron.InternalCalcium = 100*nrn.nM

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

neuron.gbar_Kslowin = 7.3e-4*nrn.siemens/nrn.cm**2

## -- HIGH-VOLTAGE-ACTIVATION CALCIUM CURRENT -- ##
neuron.gbar_HVACa = 0.00e-00*nrn.siemens/nrn.cm**2

## -- CALCIUM-DEPENDENT POTASSIUM CURRENT -- ##
neuron.gbar_KCa =0.00e-00*nrn.siemens/nrn.cm**2

## -- T-CURRENT (Calcium) -- ##
neuron.gbar_T =0.00e-00*nrn.siemens/nrn.cm**2

## -- M-CURRENT (Potassium) -- ##
neuron.gbar_Musc = 0.00e-00*nrn.siemens/nrn.cm**2

# ## -- H-CURRENT (non-specific) -- ##
# neuron.gbar_H = 0*1e-12*siemens/um**2 # set to zero !!

soma_loc, dend_loc = 0, 2
mon = nrn.StateMonitor(neuron, ['v', 'I_inj', 'InternalCalcium'], record=[soma_loc, dend_loc])
net.add(mon)

net.run(100*nrn.ms)
neuron.main.I_inj = 200*nrn.pA
net.run(200*nrn.ms)
neuron.main.I_inj = 0*nrn.pA
net.run(100*nrn.ms)
neuron.dend.I_inj = 300*nrn.pA
net.run(200*nrn.ms)
neuron.dend.I_inj = 0*nrn.pA
net.run(200*nrn.ms)

# %%
# # # ## Run the various variants of the model to reproduce Figure 12
import matplotlib.pylab as plt
fig, AX = plt.subplots(3,1, figsize=(12,4))

AX[0].plot(mon.t / nrn.ms, mon[soma_loc].v/nrn.mV, color='blue', label='soma')
AX[0].plot(mon.t / nrn.ms, mon[dend_loc].v/nrn.mV, color='red', label='dend')
AX[0].set_ylabel('Vm (mV)')
AX[0].legend()

AX[1].plot(mon.t / nrn.ms, mon[soma_loc].InternalCalcium/nrn.nM, color='blue', label='soma')
AX[1].plot(mon.t / nrn.ms, mon[dend_loc].InternalCalcium/nrn.nM, color='red', label='dend')
AX[1].set_ylabel('[Ca$^{2+}$] (nM)')

net.remove(neuron)
net.remove(mon)

net, M, neuron = None, None, None

plt.show()
# %%
