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

# %% [markdown]
# # Firing Response to Background Synaptic Stimulation

# %%
from cell_template import *

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

# %% [markdown]
# ## Perform numerical simulation
#
# run:
# ```
# python bgStim_to_firingFreq.py --cellType Basket # for the Basket Cell
# python bgStim_to_firingFreq.py --cellType Martinotti # for the Martinotti Cell
# ```

# %% [markdown]
# ## Analyze and Plot Simulations

# %%
import numpy as np

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

from parallel import Parallel
sim = Parallel(\
        filename='../../data/detailed_model/Basket_bgStim_sim.zip')

sim.load()
sim.fetch_quantity_on_grid('Vm', dtype=object) 
sim.fetch_quantity_on_grid('Vm_dend', dtype=object) 
sim.fetch_quantity_on_grid('output_rate', dtype=float) 
dt = sim.fetch_quantity_on_grid('dt', dtype=float) 
dt = np.unique(sim.dt)[0]

# %%
import numpy as np
import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

fig, AX = pt.figure(axes=(len(np.unique(sim.iBranch)), len(np.unique(sim.bgStimFreq))),
                    figsize=2*np.array((1.,0.7)), reshape_axes=False)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

trial = 2
dt = 0.025

for iB, iBranch in enumerate(np.unique(sim.iBranch)):
    c= plt.cm.tab10(iB)
    AX[0][iB].set_title('   branch #%i' % (1+iBranch), fontsize=7, color=c)
    for iF, bgStimFreq in enumerate(np.unique(sim.bgStimFreq)):
        if iB==0:
            pt.annotate(AX[iF][0], '$\\nu$=%.1eHz ' % bgStimFreq, (0, 0), ha='right', fontsize=7)
        for from_uniform, color, lw in zip([False, True], [c, 'tab:grey'], [1, 0.5]):
            Vm = sim.get('Vm', dict(from_uniform=from_uniform,
                                    iBranch=iBranch,
                                    bgStimFreq=bgStimFreq))[0]
            AX[iF][iB].plot(np.arange(len(Vm))*dt, Vm, color=color, lw=lw)
            AX[iF][iB].axis('off')
            
pt.set_common_ylims(AX)
pt.set_common_ylims(AX)
pt.draw_bar_scales(AX[0][0], loc='top-right',
                   Xbar=50, Xbar_label='50ms',
                   Ybar=20, Ybar_label='20mV')

fig.savefig('/tmp/1.svg')

# %%
fig, ax = pt.figure(figsize=(3,3))

iBranch = 0
bgStimFreq = 5e-3

for from_uniform, color, label in zip([False, True], ['tab:red', 'tab:blue'], ['real', 'uniform']):
    Vm = sim.get('Vm', dict(from_uniform=from_uniform, iBranch=iBranch, bgStimFreq=bgStimFreq))[0]
    ax.plot(np.arange(len(Vm))*dt, Vm, color=color, lw=1)
    Vm = sim.get('Vm_dend', dict(from_uniform=from_uniform, iBranch=iBranch, bgStimFreq=bgStimFreq))[0]
    ax.plot(np.arange(len(Vm))*dt, Vm, color=color, lw=1, label=label)
ax.legend(frameon=False, loc=(0,1))
ax.axis('off')

pt.draw_bar_scales(ax, loc='top-right',
                   Xbar=50, Xbar_label='50ms',
                   Ybar=20, Ybar_label='20mV')
#fig.savefig('/tmp/1.svg')

# %%
fig, AX = pt.figure(axes=(len(np.unique(sim.iBranch)), 1),
                    figsize=(0.9,1.0), reshape_axes=False)
plt.subplots_adjust(wspace=0.9)

for iB, iBranch in enumerate(np.unique(sim.iBranch)):
    c= plt.cm.tab10(iB)
    AX[iB][0].set_title('   branch #%i' % (1+iBranch), fontsize=7, color=c)
    for from_uniform, color, lw in zip([False, True], [c, 'tab:grey'], [1, 0.5]):
        params = {'iBranch':iBranch, 'from_uniform':from_uniform}
        mX = [sim.get('output_rate', dict(**params, bgStimFreq=f)).mean() for f in np.unique(sim.bgStimFreq)]
        sX = [sim.get('output_rate', dict(**params, bgStimFreq=f)).std() for f in np.unique(sim.bgStimFreq)]
        pt.plot(np.unique(sim.bgStimFreq), mX, sy=sX, ax=AX[iB][0], color=color, ms=1)
    pt.set_plot(AX[iB][0], xlabel='freq. (Hz)', ylabel='rate (Hz)', xscale='log')
    
# pt.set_common_ylims(AX)
# pt.set_common_ylims(AX)

# %%
props ={'synShuffleSeed':2,
        'distance': 100,
        'nCluster': 10,
        'subsampling_fraction':10./100.}

fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
for iBranch in range(6):
    c= plt.cm.tab10(iBranch)
    AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
    find_clustered_input(cell, iBranch, **props,
            with_plot=True, ax=AX[0][iBranch], syn_color=c)
    find_clustered_input(cell, iBranch, synShuffled=True, **props,
            with_plot=True, syn_color=c, ax=AX[1][iBranch])
    pt.annotate(AX[0][iBranch], 'real', (-0.1,0.1), bold=True, color=c)
    pt.annotate(AX[1][iBranch], 'shuffled', (-0.1,0.1), bold=True, color=c)

# %%
from clustered_input_stim import * 
ID = '864691135396580129_296758' # Basket Cell example
cell = PVcell(ID=ID, debug=False)
index = 0
find_clustered_input(cell, 0, with_plot=True)
find_clustered_input(cell, 0,
        synShuffled=True, with_plot=True)


# %%
import numpy as np
results = np.load('single_sim.npy', allow_pickle=True).item()
t0=200
ISI=200
delay=5
dt=0.025
Vm_soma = results['Vm_soma']
Vm_dend = results['Vm_dend']
real, linear = results['real_dend'], results['linear_dend']

# %%

# real, linear = build_linear_pred(Vm_dend, dt, t0, ISI, delay, len(results['synapses']))
fig, AX = pt.figure(axes=(len(real), 1))
for real, linear, ax in zip(real, linear, AX):
    ax.plot(np.arange(len(real))*dt, real, 'k-', lw=0.5)
    ax.plot(np.arange(len(real))*dt, linear, 'r:')
    ax.set_xlim([0,50])
pt.set_common_ylims(AX)

# %%
import matplotlib.pylab as plt

def build_linear_pred(Vm, dt, t0, ISI, delay):
    t = np.arange(len(Vm))*dt
    # extract single EPSPs
    sEPSPS = []
    for i in range(nCluster):
        tstart = t0+i*ISI
        cond = (t>tstart) & (t<(tstart+ISI))
        sEPSPS.append(Vm[cond]-Vm[cond][0])
    # compute real responses
    tstart = t0+nCluster*ISI
    cond = (t>tstart) & (t<(tstart+ISI))
    real = Vm[cond]
    # then linear pred
    linear = np.ones(np.sum(cond))*real[0]
    t = np.arange(len(real))*dt
    for i, epsp in enumerate(sEPSPS):
        cond = (t>i*delay)
        linear[cond] += epsp[:np.sum(cond)]

    return real, linear

real, linear = build_linear_pred(Vm_dend, dt, t0, ISI, delay)
t = np.arange(len(real))*dt

plt.plot(t, real)
plt.plot(t, linear)
    
    

# %%
sEPSPS = []
for i in range(nCluster):
    tstart = t0+i*ISI
    cond = (t>tstart) & (t<(tstart+ISI))
    sEPSPS.append(Vm_soma[cond])

import matplotlib.pylab as plt
t = np.arange(len(results['Vm_soma']))*results['dt']
tstart = t0+(1+nCluster)*ISI
tstart = t0+nCluster*ISI
cond = (t>(tstart-0*ISI/2.)) & (t<(tstart+ISI))
plt.plot(t[cond], results['Vm_soma'][cond])


# %%

