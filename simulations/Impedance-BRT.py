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
import sys, os

import numpy as np

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils import plot_tools as pt
import matplotlib.pylab as plt

# %%
Model = {
    #################################################
    # ---------- MORPHOLOGY PARAMS  --------------- #
    #################################################
    'Nbranch':4, # 
    'branch_length':100, # [um]
    'radius_soma':10, # [um]
    'diameter_root_dendrite':2.0, # [um]
    'nseg_per_branch': 10,
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 1, # [pS/um2] = 10*[S/m2] # FITTED --- Farinella et al. 0.5pS/um2 = 0.5*1e-12*1e12 S/m2, NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 1., # [uF/cm2] NEURON default
    "Ri": 50., # [Ohm*cm]
    "EL": -75, # [mV]
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'dt':0.025,# [ms]
    'seed':1, #
    'interstim':250, # [ms]
    'interspike':5, #[ms]
}


# %%
BRT = nrn.morphologies.BallandRallsTree.build_morpho(Nbranch=Model['Nbranch'],
                                                     branch_length=Model['branch_length'],
                                                     soma_radius=Model['radius_soma'],
                                                     root_diameter=Model['diameter_root_dendrite'],
                                                     Nperbranch=Model['nseg_per_branch'])

SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

vis = nrnvyz(SEGMENTS)
BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['Nbranch']+1)
fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, BRANCH_LOCS, 2)
ax.set_title('n=%i segments' % len(BRANCH_LOCS), fontsize=6)


# %%

def run_imped_charact(Model, 
                      pulse={'amp':1,
                             'duration':200}):
    """
    current in pA, durations in ms
    """
    
    # simulation params
    nrn.defaultclock.dt = Model['dt']*nrn.ms
    
    # passive
    gL = Model['gL']*1e-4*nrn.siemens/nrn.cm**2
    EL = Model['EL']*nrn.mV                
    # equation
    eqs='''
    Im = gL * (EL - v) : amp/meter**2
    I : amp (point current)
    '''
    
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(Nbranch=Model['Nbranch'],
                                                         branch_length=Model['branch_length'],
                                                         soma_radius=Model['radius_soma'],
                                                         root_diameter=Model['diameter_root_dendrite'],
                                                         Nperbranch=Model['nseg_per_branch'])
    
    SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)
    BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['Nbranch']+1)
    
    neuron = nrn.SpatialNeuron(morphology=BRT,
                               #model=Equation_String.format(**Model),
                               model=eqs,
                               method='euler',
                               Cm=Model['cm'] * nrn.uF / nrn.cm ** 2,
                               Ri=Model['Ri'] * nrn.ohm * nrn.cm)
    
    
    output = {'loc':[],
              'input_resistance':[],
              'transfer_resistance_to_soma':[]}
    
    for b in BRANCH_LOCS:
        
        neuron.v = EL # init to rest

        # recording and running
        Ms = nrn.StateMonitor(neuron, ('v'), record=[0, b]) # soma

        # run a pre-period for relaxation
        nrn.run(100*nrn.ms)
        # turn on the current pulse
        neuron.I[b] = pulse['amp']*nrn.pA
        # ru nuntil relaxation (long pulse)
        nrn.run(pulse['duration']*nrn.ms)
        # turn off the current pulse
        neuron.I[b] = 0*nrn.pA
        
        # measure all quantities
        output['input_resistance'].append(\
                                1e6*(neuron.v[b]-EL)/nrn.volt/pulse['amp']) # 1e6*V/pA = MOhm
        output['transfer_resistance_to_soma'].append(\
                                1e6*(neuron.v[0]-EL)/nrn.volt/pulse['amp']) # 1e6*V/pA = MOhm
        output['loc'].append(b/len(BRANCH_LOCS)*Model['Nbranch']*Model['branch_length'])
    
    t, neuron, BRT = None, None, None
    return output
    
results = run_imped_charact(Model)

# %%
fig, AX = plt.subplots(1, 2, figsize=(5,1.5))
plt.subplots_adjust(wspace=0.6)
AX[0].set_title('input resistance')
AX[0].plot(results['loc'], results['input_resistance'])
AX[1].set_title('transfer resistance')
AX[1].plot(results['loc'], results['transfer_resistance_to_soma'])
for ax in AX:
    ax.set_xlabel('dist. to soma ($\mu$m)')
    ax.set_ylabel('resistance (M$\Omega$)')

# %%
full_length = 400 # [um]

BRANCHES, RESULTS = range(2, 6), []
for i, Nbranch in enumerate(BRANCHES):
    cModel = Model.copy()
    cModel['Nbranch'] = Nbranch
    cModel['branch_length'] = full_length/Nbranch
    RESULTS.append(run_imped_charact(cModel))

# %%
full_length = 400 # [um]

fig, AX = plt.subplots(1, 2, figsize=(5,1.5))
plt.subplots_adjust(wspace=0.6, right=0.8)
AX[0].set_title('input resistance')
AX[1].set_title('transfer resistance')
for ax in AX:
    ax.set_xlabel('path dist. to soma ($\mu$m)')
    ax.set_ylabel('resistance (M$\Omega$)')

for i, results in enumerate(RESULTS):
    color = plt.cm.viridis(i/(len(BRANCHES)-1))
    AX[0].plot(results['loc'], results['input_resistance'], color=color)
    AX[1].plot(results['loc'], results['transfer_resistance_to_soma'], color=color)

# %%
