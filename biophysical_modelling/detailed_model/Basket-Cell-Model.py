)# ---
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
# # Simulation of Morphologically-detailed models of Basket Cell

# %%
from PV_template import *

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt

# %% [markdown]
# ## FI-curve

# %%
"""
ID = '864691135396580129_296758' # Basket Cell example
cell = PVcell(ID=ID, debug=False)

ic = h.IClamp(cell.soma[0](0.5))
ic.amp = 0. 
ic.dur =  1e9 * ms
ic.delay = 0 * ms

dt, tstop = 0.025, 500

t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
Vm = h.Vector()

Vm.record(cell.soma[0](0.5)._ref_v)

apc = h.APCount(cell.soma[0](0.5))

h.finitialize(cell.El)

for i in range(int(50/dt)):
    h.fadvance()

duration = 200 # ms
AMPS, RATES = np.array([-0.1]+list(0.1*np.arange(1, 10))), []
for a, amp in enumerate(AMPS):

    ic.amp = amp
    apc.n = 0
    for i in range(int(duration/dt)):
        h.fadvance()
    if a==0:
        # calculate input res.
        Rin = (cell.soma[0](0.5).v-cell.El)/amp # Mohm
    RATES.append(apc.n*1e3/duration) # rates in Hz
    ic.amp = 0
    for i in range(int(duration/dt)):
        h.fadvance()
"""

# %%
"""
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(np.arange(len(Vm))*dt, np.array(Vm), color='tab:grey')
ax.axis('off')
pt.draw_bar_scales(ax, loc='top-right',
                   Xbar=100, Xbar_label='100ms',
                   Ybar=10, Ybar_label='10mV')
pt.annotate(ax, '%imV ' % cell.El, (0, cell.El), xycoords='data', va='center', ha='right')
pt.annotate(ax, '      R$_{in}$=%.1fM$\Omega$ ' % Rin, (0, 0), va='top')

inset = pt.inset(ax, [0, 0.6, 0.2, 0.4])
inset.plot(AMPS, RATES, 'ko-', lw=0.5)
pt.set_plot(inset, xlabel='amp. (nA)', ylabel='firing rate (Hz)')

fig.savefig('../figures/BC-FI-curve.svg')
"""

# %% [markdown]
# ## Resistance Profile

# %%
"""
ID = '864691135396580129_296758' # Basket Cell example
cell = PVcell(ID=ID, debug=False)
cell.check_that_all_dendritic_branches_are_well_covered(show=False)

amp, duration, dt = -25e-3, 300, 0.1

DISTANCE, RIN, RT = [], [], []
for iB, branch in enumerate(cell.branches['branches']):

    Distance = []
    Rin, Rt = [], [] # input and transfer resistance
    for b in branch:

        x = cell.SEGMENTS['NEURON_segment'][b]/cell.SEGMENTS['NEURON_section'][b].nseg
        Distance.append(h.distance(cell.SEGMENTS['NEURON_section'][b](x),
                                   cell.soma[0](0.5)))

        ic = h.IClamp(cell.SEGMENTS['NEURON_section'][b](x))
        ic.amp, ic.dur = 0. , 1e3

        h.finitialize(cell.El)
        ic.amp = amp
        for i in range(int(duration/dt)):
            h.fadvance()

        Rin.append((cell.SEGMENTS['NEURON_section'][b](x).v-cell.El)/amp) # Mohm
        Rt.append((cell.soma[0](0.5).v-cell.El)/amp) # Mohm
    RIN.append(Rin)
    RT.append(Rt)
    DISTANCE.append(Distance)

np.save('../data/BC-Input-Resistance.npy',
        {'distance':DISTANCE, 'Rin':RIN, 'Rt':RT})
"""

# %%
R = np.load('../data/BC-Input-Resistance.npy', allow_pickle=True).item()

bins = np.linspace(0, 180, 20)
fig, AX = plt.subplots(1, 2, figsize=(5,2))
plt.subplots_adjust(wspace=0.8, bottom=0.3, right=0.8)
for i, dist, rin, rt in zip(range(len(R['Rt'])), R['distance'], R['Rin'], R['Rt']):
    AX[0].plot(dist, rin, 'o', ms=0.5, color=plt.cm.tab10(i))
    AX[1].plot(dist, rt, 'o', ms=0.5, color=plt.cm.tab10(i))
    pt.annotate(AX[1], i*'\n'+'branch #%i' % (i+1), (1,1), va='top', color=plt.cm.tab10(i))
pt.set_plot(AX[0], xlabel='dist. to soma ($\mu$m)',
            ylabel='Input Res. (M$\Omega$', yscale='log')
pt.set_plot(AX[1], xlabel='dist. to soma ($\mu$m)',
            ylabel='Transfer Res. (M$\Omega$)\n to soma ')
fig.savefig('../figures/BC-Resistance-Profile.svg')
plt.show()

# %%
import numpy as np
import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

fig, AX = pt.figure(axes=(len(np.unique(sim.iBranch)), len(np.unique(sim.bgStimFreq))),
                    figsize=(1.5,0.4))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for iB, iBranch in enumerate(np.unique(sim.iBranch)):
    c= plt.cm.tab10(iB)
    AX[0][iB].set_title('   branch #%i' % (1+iBranch), fontsize=7, color=c)
    for iF, freq in enumerate(np.unique(sim.bgStimFreq)):
        if iB==0:
            pt.annotate(AX[iF][0], '$\\nu$=%.1eHz ' % freq, (0, 0), ha='right', fontsize=7)
        c= plt.cm.tab10(iB)
        for iS, color, lw in zip([0, 1], [c, 'tab:grey'], [1, 0.5]):
            Vm, dt = sim.Vm[iB, 2, iF, iS] , 0.025
            AX[iF][iB].plot(np.arange(len(Vm))*dt, Vm, color=color, lw=lw)
            AX[iF][iB].axis('off')
pt.set_common_ylims(AX)
pt.set_common_ylims(AX)
pt.draw_bar_scales(AX[0][0], loc='top-right',
                   Xbar=50, Xbar_label='50ms',
                   Ybar=20, Ybar_label='20mV')


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
sim.fetch_quantity_on_grid('output_rate', dtype=float) 
RESULTS = {}
print(sim.keys)
print(sim.output_rate)
print(sim.VALUES)

# %%

# RESULTS = np.load(\
        # '../../data/detailed_model/spikingFreq_vs_stim_PV_relationship.npy',
        # allow_pickle=True).item()

fig, AX = pt.figure(axes=(len(np.unique(sim.iBranch)), 1),
                    figsize=(0.9,1.0))
plt.subplots_adjust(wspace=0.9)

for iB, iBranch in enumerate(np.unique(sim.iBranch)):
    c= plt.cm.tab10(iB)
    AX[iB].set_title('   branch #%i' % (1+iBranch), fontsize=7, color=c)
    for synShuffled, color, lw in zip([0, 1], [c, 'tab:grey'], [1, 0.5]):
        print(sim.output_rate[iB,:,:,synShuffled])
        # print(sim.output_rate[iB,:,:,synShuffled].mean(axis=0))
        pt.plot(np.unique(sim.bgStimFreq), 
                    sim.output_rate[iB,:,:,synShuffled].mean(axis=0),
                    sy=sim.output_rate[iB,:,:,synShuffled].std(axis=0),
                    ax=AX[iB], color=color, ms=1)
    pt.set_plot(AX[iB], xlabel='freq. (Hz)', ylabel='rate (Hz)', xscale='log')
# pt.set_common_ylims(AX)
# pt.set_common_ylims(AX)

# %%


