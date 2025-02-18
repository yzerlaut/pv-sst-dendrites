# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Response to Flashed-Gratings (2s duration)
#
# we feed the neuron with spikes drawn from an inhomogeneous Poisson process whose time-varying rate is set by a waveform ressembling activity evoked by full-field-gratings

# %%
import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from scipy import stats
from parallel import Parallel

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


# %% [markdown]
# # Drafting the Input pattern

# %%
def sigmoid(x, width=0.1):
    return (1+erf(x/width))/2.

P = dict(t1=0.2, t2=0.45, t3=0.7, t4=2.1,
         w1=0.08, w2=0.35, w3=0.3, w4=0.2)

def inputRate(x,
              t1=0, t2=0, t3=0, t4=0,
              w1=0, w2=0, w3=0, w4=0,
              Amp=0):
    y = sigmoid(x-t1, w1)*sigmoid(-(x-t2), w2)+\
            Amp*(sigmoid(x-t3, w3)*sigmoid(-(x-t4), w4))
    return y/y.max()
    
tstop, dt = 4e3, 0.1
t = np.arange(int(tstop/dt))*dt
fig, ax = pt.figure(figsize=(1.,1))
pt.plot(1e-3*t, inputRate(1e-3*t-0.5, Amp=0.3, **P), ax=ax, no_set=True)
ax.fill_between([0.5,2.5], [0,0], [1,1], color='gray', alpha=0.2, lw=0)
pt.set_plot(ax, yticks=[0,1],  xlabel='time (s)', ylabel='input rate\n(norm.)')

# %%
# now implemented in module
from grating_stim import input_signal

tstop, dt = 4e3, 0.1
t = np.arange(int(tstop/dt))*dt
fig, ax = pt.figure(figsize=(1.,1))
pt.plot(1e-3*t, input_signal(1e-3*t-0.5), ax=ax, no_set=True)
ax.fill_between([0.5,2.5], [0,0], [1,1], color='gray', alpha=0.2, lw=0)
pt.set_plot(ax, yticks=[0,1],  xlabel='time (s)', ylabel='input rate\n(norm.)')

# %% [markdown]
# # Test Simulation
# ```
# python grating_stim.py --test -c Martinotti\
#         --with_NMDA\
#         --with_presynaptic_spikes\
#         --stimFreq 2 --stepAmpFactor 4\
#         --synapse_subsampling 1 --Inh_fraction 0.2\
#         --iBranch 1 --spikeSeed 3
# ```

# %%
results = np.load('single_sim.npy', allow_pickle=True).item()

t = np.arange(len(results['Vm_soma']))*results['dt']
fig, AX = pt.figure(axes_extents=[[(1,4)],[(1,2)],[(1,1)]],
                    figsize=(2,0.4), left=0, bottom=0., hspace=0.)
#AX[0].plot(t, results['Vm_dend'], 'k:', lw=0.5, label=' distal\ndendrite')
AX[2].fill_between(t[1:], 0*t[1:], results['Stim'], color='tab:grey', label='soma')
AX[0].plot(t, results['Vm_soma'], 'tab:brown', label='soma')
AX[0].plot(t, -60+0*t, 'k:')
pt.annotate(AX[0], '-60mV ', (0,-60), xycoords='data', ha='right', va='center')
pt.draw_bar_scales(AX[0], Xbar=100, Xbar_label='100ms', Ybar=10, Ybar_label='10mV')
for i, events in enumerate(results['presynaptic_exc_events']):
    AX[1].scatter(events, i*np.ones(len(events)), facecolor='g', edgecolor=None, alpha=.35, s=3)
for i, events in enumerate(results['presynaptic_inh_events']):
    AX[1].scatter(events, len(results['presynaptic_exc_events'])+i*np.ones(len(events)),
                  facecolor='r', edgecolor=None, alpha=.35, s=3)
    
pt.annotate(AX[1], 'Inh.', (0,1), ha='right', va='top', color='r')
pt.annotate(AX[1], 'Exc.', (0,0), ha='right', va='bottom', color='g')

print('\n - number of excitatory events: %i' %\
              np.sum([len(E) for E in results['presynaptic_exc_events']]))
print(' - output rate: %.1f Hz\n' % (1e3*len(results['spikes'].flatten())/results['tstop']))
pt.set_common_xlims(AX, lims=[t[0], t[-1]])
for ax in AX:
    ax.axis('off')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Summary Effect

# %%
rate_smoothing = 100. # ms

def load_sim(results, cellType):

    rates = []
    for iBranch in range(6):
        sim = Parallel(\
                filename='../data/detailed_model/full-grating6/GratingSim_%s_branch%i.zip' % (cellType, iBranch))
        sim.load()
        sim.fetch_quantity_on_grid('spikes', dtype=list)
        seeds = np.unique(sim.spikeSeed)
        dt = sim.fetch_quantity_on_grid('dt', return_last=True)
        tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
        spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
        for i, spikes in enumerate(sim.spikes):
            spikes_matrix[i,(spikes/dt).astype('int')] = True
        rates.append(1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                   int(rate_smoothing/dt)))
        input = sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)

    results['stimFreq_%s' % cellType] = sim.fetch_quantity_on_grid('stimFreq', return_last=True)
    results['rate_%s' % cellType] = rates
    results['input_%s' % cellType] = input
    results['t'] = np.arange(len(rates[0]))*dt
    

results = {}
for cellType in ['MartinottiFull', 'MartinottinoNMDA', 'BasketFull']:
    load_sim(results, cellType)

# %%
fig1, ax1 = pt.figure(figsize=(.9,1.0))
fig2, ax2 = pt.figure(figsize=(.9,.9))

pt.annotate(ax2, '2s', (0.2, 0), va='top', color='k', fontsize=7)

for c, cellType, color in zip(range(3),\
    ['MartinottinoNMDA', 'MartinottiFull', 'BasketFull'],
    ['tab:purple', 'tab:orange', 'tab:red']):

    rates = np.array(results['rate_%s' % cellType])
    pt.plot(1e-3*results['t'], np.mean(rates, axis=0), sy=np.std(rates, axis=0), ax=ax1, color=color, lw=1, no_set=True)

    norm_factor = 1./(np.mean(rates, axis=0).max()-np.mean(rates, axis=0).min())
    #pt.plot(1e-3*results['t'], norm_factor*(np.mean(rates, axis=0)-np.mean(rates, axis=0).min()),
    rates = (rates.T/rates.max(axis=1)).T
    pt.plot(1e-3*results['t'][::1000], np.mean(rates, axis=0)[::1000],
            sy = stats.sem(rates, axis=0)[::1000],
            ax=ax2, color=color, lw=1, no_set=True)
        
    if c==0:
        pt.draw_bar_scales(ax2, Xbar=1e-12, Ybar=0.14, loc='top-right')
        input = results['input_%s' % cellType]
        inset = pt.inset(ax2, [0,1,1,0.4])
        inset.fill_between(1e-3*results['t'][1:][::1000], 0*results['t'][1:][::1000], input[::1000]/np.max(input), color='lightgrey')
        pt.draw_bar_scales(inset, Ybar=0.25/1.15, Xbar=1e-12, loc='bottom-right')
        inset.axis('off')
        ax2.plot([0,2], [0,0], 'k-', lw=1)
    pt.annotate(ax2, '%.1fHz' % (0.14/norm_factor)+c*'\n', (1, 0),
                color=color, fontsize=7)
    pt.annotate(inset, '%.1fHz' % (results['stimFreq_%s' % cellType]/1.15)+c*'\n', (1, 0),
                color=color, fontsize=7)
pt.set_plot(ax1, xlabel='time (s)', ylabel='rate (Hz)')
pt.set_plot(ax2, [])

#fig.savefig('../figures/Temp-Properties-Pred/PV-vs-SST.svg')

# %%
fig2.savefig('/Users/yann/Desktop/fig.svg')

# %% [markdown]
# # Example Simulations

# %%
rate_smoothing = 10. # ms
zoom = [0,4000]

PATH = '../data/detailed_model/grating-demo/GratingSim_demo_%s.zip'

"""
def load_sim(cellType, RESULTS):
    
    sim = Parallel(filename=PATH % cellType)
    sim.load()

    sim.fetch_quantity_on_grid('spikes', dtype=list)

    seeds = np.unique(sim.spikeSeed)
    
    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
    RESULTS['%s_rate' % cellType] = []
    for i, spikes in enumerate(sim.spikes):
        spikes_matrix[i,(spikes/dt).astype('int')] = True
    RESULTS['rate_%s' % cellType] = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                           int(rate_smoothing/dt))
    RESULTS['stimFreq_%s' % cellType] = sim.fetch_quantity_on_grid('stimFreq', return_last=True)
    print(cellType, ' --->', sim.fetch_quantity_on_grid('stimFreq', return_last=True))
    RESULTS['t'] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt    

    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
    mean_input_rate = np.mean(sim.Stim[0])*RESULTS['stimFreq_%s' % cellType]
    syn_exc_rates = [np.mean([1e3*len(E)/tstop for E in sim.presynaptic_exc_events[i]]) for i in range(len(seeds))]
    #print('           exc syn. rate: %.1f +/- %.1f Hz' % (np.mean(syn_exc_rates), np.std(syn_exc_rates)))
    #print('              --> average release proba (of single events): %.2f ' % (np.mean(syn_exc_rates)/mean_input_rate))

"""

def load_example_index(cellType, RESULTS):
    
    sim = Parallel(filename=PATH % cellType)
    sim.load()

    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
    RESULTS['Input_%s' % cellType] = sim.Stim[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
    RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
    RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
        

def plot_sim(cellType, RESULTS, iBranch, color='k', example_index=None, figsize=(1.8,.5), Ybar=10):

    fig, AX = pt.figure(axes_extents=[[(1,1)],[(1,1)],[(1,4)],[(1,2)]],
                        figsize=figsize, left=0, bottom=0., hspace=0.)

    # input
    AX[0].fill_between(RESULTS['t'][:-1][::20], 0*RESULTS['t'][:-1][::20], RESULTS['Input_%s' % cellType][::20],
                       color='tab:grey', lw=1)

    # Vm
    AX[2].plot(RESULTS['t'][::10], RESULTS['Vm_%s' % cellType][::10], color=color, lw=0.5)
    AX[2].plot(RESULTS['t'][::100], -60+0*RESULTS['t'][::100], 'k:', lw=0.3)

    # rate
    if RESULTS['rate_%s' % cellType] is not None:
        AX[3].fill_between(RESULTS['t'][::20], 0*RESULTS['t'][::20], 
                           np.array(RESULTS['rate_%s' % cellType])[iBranch, ::20],
                           color=color, lw=0)
        
    # events
    if 'pre_inh_%s' % cellType in RESULTS:
        subsampling = 4 if 'Basket' in cellType else 1
        # 1/4 for display
        for i, events in enumerate(RESULTS['pre_exc_%s' % cellType]):
            if len(events)>0:
                e = np.random.choice(events, np.min([int(len(events)/subsampling+1),1]), replace=False)
                AX[1].plot(e, i*np.ones(len(e)), 'o', fillstyle='full', color='g', ms=.3)
        for i, events in enumerate(RESULTS['pre_inh_%s' % cellType]):
            if len(events)>0:
                e = np.random.choice(events, np.min([int(len(events)/subsampling+1),1]), replace=False)
                AX[1].plot(e, len(RESULTS['pre_exc_%s' % cellType])+i*np.ones(len(e)), 'o', 
                           fillstyle='full', color='r', ms=.3)

    pt.set_common_xlims(AX, lims=zoom)
    
    pt.draw_bar_scales(AX[0], Xbar=200, Xbar_label='200ms', Ybar=RESULTS['stimFreq_%s' % cellType],
                       Ybar_label='%.0fHz' % (RESULTS['stimFreq_%s' % cellType]))
    pt.annotate(AX[2], '-60mV ', (zoom[0],-60), xycoords='data', ha='right', va='center')
    pt.draw_bar_scales(AX[2], Xbar=1e-12, Ybar=20,Ybar_label='20mV')
    for ax in AX:
        ax.axis('off')
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=Ybar,Ybar_label='%iHz' % Ybar)
    return fig, AX

#for cellType, color, index in zip(['Martinotti', 'Basket'],
for cellType, color, index, iBranch in zip(
            ['MartinottiFull', 'BasketFull', 'MartinottinoNMDA'],
            ['tab:orange', 'tab:red', 'tab:purple'],
            [1, 0, 2], [1, 0, 1]):
    results['%s_example_index' % cellType] = index # change here !
    load_example_index(cellType, results) 
            
    fig, _ = plot_sim(cellType, results, iBranch, color=color, Ybar=2 if 'noNMDA' in cellType else 10)
    fig.savefig('../figures/Temp-Properties-Pred/GratingSim_example_%s.svg' % cellType)

# %%
