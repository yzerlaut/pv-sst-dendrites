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

# %% [markdown]
# ## Load morphology

# %%
mFile = 'morphologies/Pvalb-IRES-Cre_Ai14-305768.04.02.01_656999337_m.swc'
morpho = nrn.Morphology.from_swc_file(mFile) 
FULL = nrn.morpho_analysis.compute_segments(morpho)
SEGMENTS = nrn.morpho_analysis.compute_segments(morpho, without_axon=True)

# %%
fig, [ax0, ax] = pt.plt.subplots(1, 2, figsize=(4,1.3))
visFull, vis = nrnvyz(FULL), nrnvyz(SEGMENTS)
visFull.plot_segments(ax=ax0, color='tab:grey')
ax0.annotate('dendrite', (0,0), xycoords='axes fraction', fontsize=6, color='tab:blue')
vis.plot_segments(ax=ax0, color='tab:blue')
ax.hist(1e6*SEGMENTS['distance_to_soma'], density=True)
ax.set_xlabel('path dist. to soma ($\mu$m)')
ax.set_ylabel('density')
ax.set_yticks([]);

# %% [markdown]
# ## Distribute synapses

# %%
x = np.linspace(SEGMENTS['distance_to_soma'].min(), SEGMENTS['distance_to_soma'].max(), 25)
uniform = 0.5 +0*x
uniform /= np.sum(uniform) #np.trapz(uniform, x=1e6*x)

biased = 1.-(x-x.min())/(x.max()-x.min())
biased /= np.sum(biased) # np.trapz(biased, x=1e6*x)

# %%
Nsynapses = 100

np.random.seed(20)
LOCS = {}

digitized_dist = np.digitize(SEGMENTS['distance_to_soma'], bins=x, right=True)

for case, proba in zip(['uniform', 'biased'], [uniform, biased]):
    
    LOCS[case] = []
    iDist = np.random.choice(np.arange(len(x)), Nsynapses, p=proba)
    
    for i in iDist:
        # pick one segment loc for the synapse with the distance condition:
        LOCS[case].append(np.random.choice(np.arange(len(SEGMENTS['x']))[digitized_dist==i], 1)[0])


# %%
fig, AX = pt.plt.subplots(2, 2, figsize=(7,5))
pt.plt.subplots_adjust(wspace=.2, hspace=.6)

for c, y, case in zip(range(2), [uniform, biased], ['uniform', 'biased']):
    
    ax=pt.inset(AX[0][c], [0.3, 0.2, 0.4, 0.6])
    ax.set_ylabel('synaptic density')
    AX[0][c].axis('off')
    ax.plot(1e6*x, y, '-', color='tab:grey', lw=2)
    ax.set_yticks([0,0.1])
    ax.set_title(case)
    ax.set_xlabel('path dist. to soma ($\mu$m)');
    
    vis.plot_segments(ax=AX[1][c], color='tab:grey')
    vis.add_dots(AX[1][c], LOCS[case], 1)

    inset = pt.inset(AX[1][c], [0.9, 0., 0.4, 0.3])
    inset.hist(SEGMENTS['distance_to_soma'][LOCS[case]], color='tab:red')
    inset.set_xlabel('dist.', fontsize=7);inset.set_xticks([]);inset.set_yticks([])
    inset.set_xlim([x.min(), x.max()])
    inset.set_title('%i synapses' % Nsynapses, fontsize=6)

# %% [markdown]
# ## Synaptic integration with "uniform" and "biased" distributions

# %%
gL = 5e-5*nrn.siemens/nrn.cm**2
EL = -70*nrn.mV                
Es = 0*nrn.mV                  

eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gs * (Es - v) : amp (point current)
gs : siemens
'''

# %%
results = {}

Nstim = 10
results['events'] = np.arange(Nstim)*200
results['Nsyns'] = 1+np.arange(Nstim)*10

for case in ['uniform', 'biased']:

    results[case] = {'Vm':[]}
    
    for repeat in range(2):
        
        np.random.seed(repeat)

        spike_IDs, spike_times, synapses = np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)

        neuron = nrn.SpatialNeuron(morphology=morpho, 
                           model=eqs,
                           Cm=1 * nrn.uF / nrn.cm ** 2,    
                           Ri=100 * nrn.ohm * nrn.cm)
        neuron.v = EL

        taus = 5.*nrn.ms
        w = 0.2*nrn.nS

        for e, ns in zip(results['events'], results['Nsyns']):

            s = np.random.choice(np.arange(Nsynapses), ns, replace=False)

            spike_times = np.concatenate([spike_times, e+np.arange(len(s))*0.01])
            spike_IDs = np.concatenate([spike_IDs, np.array(s, dtype=int)])

        stimulation = nrn.SpikeGeneratorGroup(Nsynapses,
                                              np.array(spike_IDs, dtype=int),
                                              spike_times*nrn.ms)

        ES = nrn.Synapses(stimulation, neuron,
                           model='''dg/dt = -g/taus : siemens
                                    gs_post = g : siemens (summed)''',
                           on_pre='g += w',
                           method='exponential_euler')

        for ipre, iseg_post in enumerate(LOCS[case]): # connect spike IDs to a given location
            ES.connect(i=ipre, j=iseg_post)

        # recording and running
        M = nrn.StateMonitor(neuron, ('v'), record=[0, 1000])
        nrn.run((400+np.max(spike_times))*nrn.ms)
        results[case]['Vm'].append(np.array(M.v[0]/nrn.mV))
        
results['t'] = np.array(M.t/nrn.ms)

# %%
for case in ['uniform', 'biased']:

    results[case]['depol'] = []
    results[case]['depol-sd'] = []

    for event in results['events']:

        t_cond = (results['t']>event) & (results['t']<=event+100)

        imax = np.argmax(np.mean(results[case]['Vm'], axis=0)[t_cond])
        results[case]['depol'].append(np.mean(results[case]['Vm'], axis=0)[t_cond][imax])
        results[case]['depol-sd'].append(np.std(results[case]['Vm'], axis=0)[t_cond][imax])

# %%
fig, AX = pt.plt.subplots(2, figsize=(7,2.7))
pt.plt.subplots_adjust(right=.7, hspace=0.1)

axS = pt.inset(AX[1], (1.15,0.5,0.35,1.2))
axS.set_ylabel('peak depol. (mV)')
axS.set_xlabel(' $N_{synapses}$ ')
for ax, case, color in zip(AX, ['uniform', 'biased'], ['tab:blue', 'tab:green']):
    ax.plot(results['t'], np.mean(results[case]['Vm'],axis=0), color=color)
    ax.set_ylabel('$V_m$ (mV)')
    ax.set_xlabel('time (ms)')
    #pt.draw_bar_scales(ax, Ybar=1, Ybar_label='10mV', Xbar=500, Xbar_label='500ms');ax.axis('off');
    pt.draw_bar_scales(ax, Xbar=200, Xbar_label='200ms', Ybar=1e-12)
    #ax.ax.xaxis('off');
    ax.axes.get_xaxis().set_visible(False)
    axS.plot(results['Nsyns'], results[case]['depol'], color=color, label=case, lw=2)
axS.legend(loc=(0,1), frameon=False)
