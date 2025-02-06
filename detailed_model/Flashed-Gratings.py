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
# # Response to Step inputs
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

P = dict(t1=0.2, t2=0.45, t3=0.75, t4=2.1,
         w1=0.08, w2=0.3, w3=0.2, w4=0.2,
         Amp=0.25)

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
pt.plot(1e-3*t, inputRate(1e-3*t-0.5, **P), ax=ax, no_set=True)
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

# %% [markdown]
# # Plot

# %%
rate_smoothing = 10. # ms
zoom = [350,4000]

RESULTS = {'Martinotti_example_index':1, # *50* 33, 42, 49, 50
           'Basket_example_index':2} # 31

def load_sim(cellType, RESULTS):
    
    sim = Parallel(\
            filename='../data/detailed_model/grating-demo/GratingSim_demo_%s.zip' % cellType)
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
    RESULTS['t'] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt    

    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
    mean_input_rate = np.mean(sim.Stim[0])*RESULTS['stimFreq_%s' % cellType]
    syn_exc_rates = [np.mean([1e3*len(E)/tstop for E in sim.presynaptic_exc_events[i]]) for i in range(len(seeds))]
    print('           exc syn. rate: %.1f +/- %.1f Hz' % (np.mean(syn_exc_rates), np.std(syn_exc_rates)))
    print('              --> average release proba (of single events): %.2f ' % (np.mean(syn_exc_rates)/mean_input_rate))


def load_example_index(cellType, RESULTS):
    
    sim = Parallel(\
            filename='../data/detailed_model/GratingSim_demo_%s.zip' % cellType)
    sim.load()

    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
    RESULTS['Input_%s' % cellType] = sim.Stim[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
    RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
    RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
        

def plot_sim(cellType, color='k', example_index=None, figsize=(1.2,0.6)):

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
        AX[3].fill_between(RESULTS['t'][::20], 0*RESULTS['t'][::20], RESULTS['rate_%s' % cellType][::20],
                           color=color, lw=0)
        AX[3].plot(RESULTS['t'][::20], 0*RESULTS['t'][::20], color=color, lw=1)
        
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
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=10,Ybar_label='10Hz')
    return fig, AX

#for cellType, color, index in zip(['Martinotti', 'Basket'],
for cellType, color, index in zip(['MartinottiFull', 'BasketFull', 'MartinottinoNMDA','MartinottinoNMDAnoSTP'],
                                  ['tab:orange', 'tab:red', 'tab:purple', 'tab:cyan'],
                                  [1, 9, 1, 1]):
    try:
        load_sim(cellType, RESULTS) 
        RESULTS['%s_example_index' % cellType] = index # change here !
        load_example_index(cellType, RESULTS) 
            
        fig, _ = plot_sim(cellType, color=color, figsize=(0.9,0.5))
    except BaseException as be:
        print(cellType)
        print(be)
#    fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_%s.svg' % cellType)

# %%
fig, ax = pt.figure(figsize=(0.9,1.1))

for cellType, color, index in zip(['MartinottiFull', 'BasketFull', 'MartinottinoNMDA'],
                                  ['tab:orange', 'tab:red', 'tab:purple', 'tab:cyan'],
                                  [1, 9, 1, 1]):
    cond = (RESULTS['t']>0.3e3) & (RESULTS['t']<4e3)
    ax.plot(1e-3*RESULTS['t'][cond]-0.5, RESULTS['rate_%s' % cellType][cond]/np.max(RESULTS['rate_%s' % cellType]),
            color=color, lw=1)

pt.set_plot(ax)


# %% [markdown]
# # Summary Effect

# %%
rate_smoothing = 50. # ms

zoom = [-.2, 3.7]

def load_sim(cellType, suffix):

    rates = []
    for iBranch in range(6):
        sim = Parallel(\
                filename='../data/detailed_model/full-gratings/GratingSim_%s%s_branch%i.zip' % (cellType, suffix, iBranch))
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
        if iBranch==0:
            print(cellType, sim.fetch_quantity_on_grid('stimFreq', return_last=True))

    return np.arange(len(rates[0]))*dt, input, rates    


def plot_sim(cellTypes, suffixs, colors, lines=['-','-','-','-'], Ybar=10):

    fig1, ax1 = pt.figure(figsize=(0.95,1.))
    fig2, ax2 = pt.figure(figsize=(0.95,1.))

    for c in range(len(cellTypes)):
        t, input, rates = load_sim(cellTypes[c], suffixs[c])
        pt.plot(1e-3*t-0.5, np.mean(rates, axis=0), sy=np.std(rates, axis=0), ax=ax1, color=colors[c], lw=1)

        norm_factor = 1./(np.mean(rates, axis=0).max()-np.mean(rates, axis=0).min())
        pt.plot(1e-3*t-0.5, norm_factor*(np.mean(rates, axis=0)-np.mean(rates, axis=0).min()),
                sy = norm_factor*stats.sem(rates, axis=0),
                ax=ax2, color=colors[c], lw=1)
        pt.annotate(ax2, c*'\n'+'%.1fHz' % (0.2/norm_factor), (0, 1),
                    ha='right', va='top', color=colors[c], fontsize=7)
        
    #ax2.fill_between(1e-3*t[1:]-0.5, 0*t[1:], input/np.max(input), color='lightgrey')
    pt.set_plot(ax1, xlim=zoom, xlabel='time (s)', ylabel='rate (Hz)')
    pt.set_plot(ax2, ['bottom'], xlabel='time (s)', xticks=[0,2], xlim=zoom)
    pt.draw_bar_scales(ax2, Xbar=1e-12, Ybar=0.2)
    return fig1, fig2

fig1, fig2 = plot_sim(['Martinotti', 'Martinotti', 'Basket'],
                  ['Full', 'noNMDA', 'Full'],
                  ['tab:orange', 'tab:purple', 'tab:red'])
#fig.savefig('../figures/Temp-Properties-Pred/PV-vs-SST.svg')

# %%
fig2.savefig('/Users/yann/Desktop/fig.svg')

# %% [markdown]
# ## Look for traces

# %%
for cellType, color in zip(['Martinotti'], ['tab:orange']):
    load_sim(cellType, RESULTS) 
    for example_index in range(0, 10):
        RESULTS['%s_example_index' % cellType] = example_index
        load_example_index(cellType, RESULTS) 
        fig, _ = plot_sim(cellType, color=color)

# %% [markdown]
# ## Input Range

# %%
rate_smoothing = 30. # ms


for cellType, suffix, label, color in zip(['Basket', 'Martinotti', 'Martinotti'],
                                          ['Full', 'Full', 'noNMDA'],
                                          ['PV - Full', 'SST - Full', 'SST - no NMDA (AMPA+)'],
                                          ['tab:red', 'tab:orange', 'tab:purple']):
    results = {}

    for iBranch in range(6):
        sim = Parallel(\
                filename='../data/detailed_model/grating-range/GratingSim_%s%sRange_branch%i.zip' % (cellType, suffix, iBranch))
        sim.load()
        sim.fetch_quantity_on_grid('spikes', dtype=list)

 
        sim.fetch_quantity_on_grid('spikes', dtype=list)
        seeds = np.unique(sim.spikeSeed)
        dt = sim.fetch_quantity_on_grid('dt', return_last=True)
        tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
        sim.fetch_quantity_on_grid('Stim', dtype=list)
        if 'traceRate' not in results:
            results['traceRate'] = np.zeros((len(np.unique(sim.stimFreq)), 6, int(tstop/dt)+1)) # create the array
            results['t'] = np.arange(int(tstop/dt)+1)*dt
            results['stimFreq'] = np.unique(sim.stimFreq)
            
        for iSF, SF in enumerate(np.unique(sim.stimFreq)):
        
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iSF]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iSF,iBranch,:] = rate
            results['Stim%i'%iSF] = sim.Stim[0][iSF]
    
    fig, AX = pt.figure(axes=(len(results['stimFreq']), 1), right=4.,
                        figsize=(.8,1), wspace=0., hspace=0., left=0.5, bottom=0.)
    INSETS = []
    for iSF, SF in enumerate(results['stimFreq']):
        pt.plot(results['t'], results['traceRate'][iSF,:,:].mean(axis=0), 
                sy=stats.sem(results['traceRate'][iSF,:,:], axis=0),
                ax=AX[iSF], color=pt.viridis(iSF/(len(results['stimFreq'])-1)))
        INSETS.append(pt.inset(AX[iSF], [0,-0.4,1,0.38]))
        INSETS[-1].fill_between(results['t'][1:], 0*results['t'][1:], results['Stim%i'%iSF], color='lightgray')
        INSETS[-1].axis('off')             
        cond = results['t']>500
        pt.annotate(AX[iSF], '%.1fHz' % np.max(results['traceRate'][iSF,:,:].mean(axis=0)[cond]),
                        (0.5, 1), ha='center', color=pt.viridis(iSF/(len(results['stimFreq'])-1)), fontsize=7)
    pt.set_common_ylims(AX); pt.set_common_ylims(INSETS)
    for ax in pt.flatten(AX):
        pt.set_plot(ax, ['left'] if ax==AX[0] else [], ylabel='firing (Hz)' if ax==AX[0] else '')
    pt.draw_bar_scales(AX[-1], loc='bottom-right', Xbar=200, Xbar_label='200ms', Ybar=1e-12)
        
    pt.bar_legend(AX[-1], X=range(len(results['stimFreq'])),
                  ticks_labels=['%.1f' % f for f in results['stimFreq']],
                  colorbar_inset={'rect': [1.2, -0.3, 0.15, 1.6], 'facecolor': None},
                  label='stimFreq (Hz)',
                  colormap=pt.viridis)
    fig.suptitle(label, color=color)

#func('Martinotti', 'Full', 'tab:orange')

# %%

rate_smoothing = 30. # ms

results = {}
sim = Parallel(\
        filename='../data/detailed_model/GratingSim_demo_MartinottiAmpaRationoNMDA.zip')
sim.load()

sim.fetch_quantity_on_grid('spikes', dtype=list)
seeds = np.unique(sim.spikeSeed)
dt = sim.fetch_quantity_on_grid('dt', return_last=True)
tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
sim.fetch_quantity_on_grid('Stim', dtype=list)

results['traceRate'] = np.zeros((len(np.unique(sim.AMPAboost)), int(tstop/dt)+1)) # create the array
results['t'] = np.arange(int(tstop/dt)+1)*dt
results['AMPAboost'] = np.unique(sim.AMPAboost)
    
for iSF, SF in enumerate(np.unique(sim.AMPAboost)):

    # compute time-varying RATE !
    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
    for k, spikes in enumerate(\
        [np.array(sim.spikes[k][iSF]).flatten() for k in range(len(seeds))]):
        spikes_matrix[k,(spikes/dt).astype('int')] = True
    rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                  int(rate_smoothing/dt))
    results['traceRate'][iSF,:] = rate
    results['Stim%i'%iSF] = sim.Stim[0][iSF]

fig, AX = pt.figure(axes=(len(results['AMPAboost']), 1), right=4.,
                    figsize=(.8,1), wspace=0., hspace=0., left=0.5, bottom=0.)
INSETS = []
for iSF, SF in enumerate(results['AMPAboost']):
    pt.plot(results['t'], results['traceRate'][iSF,:], ax=AX[iSF], 
                      color=pt.viridis(iSF/(len(results['AMPAboost'])-1)))
    INSETS.append(pt.inset(AX[iSF], [0,-0.4,1,0.38]))
    INSETS[-1].fill_between(results['t'][1:], 0*results['t'][1:], results['Stim%i'%iSF], color='lightgray')
    INSETS[-1].axis('off')             
    
    pt.annotate(AX[iSF], '%.1fHz' % np.max(results['traceRate'][iSF,1:].mean(axis=0)),
                    (0.5, 1), ha='center', color=pt.viridis(iSF/(len(results['AMPAboost'])-1)), fontsize=7)
pt.set_common_ylims(AX); pt.set_common_ylims(INSETS)
for ax in pt.flatten(AX):
    pt.set_plot(ax, ['left'] if ax==AX[0] else [])
pt.draw_bar_scales(AX[-1], loc='bottom-right', Xbar=200, Xbar_label='200ms', Ybar=1e-12)
    
pt.bar_legend(AX[-1], X=results['AMPAboost'],
              ticks_labels=['%.1f' % f for f in results['AMPAboost']],
              colorbar_inset={'rect': [1.2, -0.3, 0.15, 1.6], 'facecolor': None},
              label='AMPAboost',
              colormap=pt.viridis)
fig.suptitle('SST - no NMDA', color='tab:purple')

#func('Martinotti', 'Full', 'tab:orange')

# %%
sim.AMPAboost

# %%
plot_sim(['Martinotti', 'Martinotti', 'Martinotti', 'Martinotti'],
         ['', 'withSTP', 'noNMDA', 'noNMDAwithSTP'],
         ['tab:orange', 'k', 'tab:purple', 'tab:cyan'],
         lines=['-','--', '-', '-'])
#fig.savefig('../figures/Temp-Properties-Pred/SST-models.svg')

# %%
plot_sim(['Basket', 'Basket'],
         ['', 'noNMDAwithSTP'],
         ['tab:red', 'tab:grey'])

# %%
plot_sim(['Basket', 'Basket'],
         ['', 'withSTP'],
         ['tab:red', 'tab:grey'])
