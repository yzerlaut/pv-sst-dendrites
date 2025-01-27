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
# we feed the neuron with spikes drawn from an inhomogeneous Poisson process whose time-varying rate is set by step functions

# %%
import numpy as np
from scipy.ndimage import gaussian_filter1d

import sys
from parallel import Parallel

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt
sys.path.append('../analyz')
from analyz.processing.signanalysis import autocorrel, crosscorrel

# %% [markdown]
# # Test Simulation
# ```
# python step_stim.py --test -c Martinotti\
#         --with_NMDA --with_STP\
#         --with_presynaptic_spikes\
#         --stimFreq 1 --stepAmpFactor 3\
#         --synapse_subsampling 2 --Inh_fraction 0.15\
#         --iBranch 5
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

# %%
fig, ax = pt.figure(figsize=(2,1))
for i in range(1,3):
    results = np.load('%i.npy' % i, allow_pickle=True).item()
    t = np.arange(len(results['Vm_soma']))*results['dt']
    ax.plot(t, results['Vm_soma'])

# %%
python step_stim.py --test -c Martinotti\
        --with_NMDA\
        --with_presynaptic_spikes\
        --stimFreq 1 --stepAmpFactor 3\
        --synapse_subsampling 2 --Inh_fraction 0.15\
        --iBranch 5 --spikeSeed 0

# %% [markdown]
# # Plot

# %%
rate_smoothing = 2. # ms
zoom = [0,3000]

RESULTS = {'Martinotti_example_index':1, # *50* 33, 42, 49, 50
           'Basket_example_index':2} # 31

def load_sim(RESULTS, cellType,
             with_example_index=None):

    sim = Parallel(\
            filename='../data/detailed_model/StepStim_demo_%s.zip' % cellType)
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
    #RESULTS['bgFreqInhFactor_%s' % cellType] = sim.fetch_quantity_on_grid('bgFreqInhFactor', return_last=True)
    RESULTS['t_%s' % cellType] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt    

    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
    mean_input_rate = np.mean(sim.Stim[0])*RESULTS['stimFreq_%s' % cellType]
    syn_exc_rates = [np.mean([1e3*len(E)/tstop for E in sim.presynaptic_exc_events[i]]) for i in range(len(seeds))]
    print('           exc syn. rate: %.1f +/- %.1f Hz' % (np.mean(syn_exc_rates), np.std(syn_exc_rates)))
    print('              --> average release proba (of single events): %.2f ' % (np.mean(syn_exc_rates)/mean_input_rate))

    if '%s_example_index' % cellType in RESULTS:
        sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)
        RESULTS['Input_%s' % cellType] = sim.Stim[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
        RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
        RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
        RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
        
def plot_sim(RESULTS, cellTypes,
             interstim=30, view=[-200, 300],
             color='k',
             figsize=(1.2,0.6)):

    fig, AX = pt.figure(axes_extents=[[(1,1)],[(1,1)],[(1,4)],[(1,2)]],
                        figsize=figsize, left=0, bottom=0., hspace=0.)

    t0 = 0
    for c, cellType in enumerate(cellTypes):
        cond = ((RESULTS['t_%s' % cellType]-(RESULTS['t_%s' % cellType].mean())>view[0]) &\
                    ((RESULTS['t_%s' % cellType]-RESULTS['t_%s' % cellType].mean())<view[1]))
        t = RESULTS['t_%s' % cellType][cond]
        # input
        AX[0].fill_between(t0+t[:-1][::20], 0*t[:-1][::20], RESULTS['Input_%s' % cellType][cond[1:]][::20],
                           color='tab:grey', lw=1)
    
        # Vm
        AX[2].plot(t0+t[::10], RESULTS['Vm_%s' % cellType][cond][::10], color=color, lw=0.5)
        AX[2].plot(t0+t[::100], -60+0*RESULTS['t_%s' % cellType][cond][::100], 'k:', lw=0.3)
    
        # rate
        if RESULTS['rate_%s' % cellType] is not None:
            AX[3].fill_between(t0+t[::20], 0*t[::20], RESULTS['rate_%s' % cellType][cond][::20],
                               color=color, lw=0)
            AX[3].plot(t0+t[::20], 0*t[::20], color=color, lw=1)
            
        # events
        if 'pre_inh_%s' % cellType in RESULTS:
            subsampling = 6 if 'Basket' in cellType else 1 # for display only
            for i, events in enumerate(RESULTS['pre_exc_%s' % cellType]):
                eCond = ((events-RESULTS['t_%s' % cellType].mean())>view[0]) &\
                                ((events-RESULTS['t_%s' % cellType].mean())<view[1])
                if len(events[eCond])>0:
                    e = np.random.choice(events[eCond], np.min([int(len(events[eCond])/subsampling+1),1]), replace=False)
                    AX[1].plot(t0+e, i*np.ones(len(e)), 'o', fillstyle='full', color='g', ms=.3)
            for i, events in enumerate(RESULTS['pre_inh_%s' % cellType]):
                iCond = ((events-RESULTS['t_%s' % cellType].mean())>view[0]) &\
                                ((events-RESULTS['t_%s' % cellType].mean())<view[1])
                if len(events[iCond])>0:
                    e = np.random.choice(events[iCond], np.min([int(len(events[iCond])/subsampling+1),1]), replace=False)
                    AX[1].plot(t0+e, len(RESULTS['pre_exc_%s' % cellType])+i*np.ones(len(e)), 'o', 
                               fillstyle='full', color='r', ms=.3)
        t0 += t[-1]-t[0]+interstim

    pt.set_common_xlims(AX)#, lims=zoom)
    
    pt.draw_bar_scales(AX[0], Xbar=50, Xbar_label='50ms', Ybar=1, Ybar_label='%.0fHz' % (RESULTS['stimFreq_%s' % cellType]))
    #pt.annotate(AX[2], '-60mV ', (zoom[0],-60), xycoords='data', ha='right', va='center')
    pt.draw_bar_scales(AX[2], Xbar=1e-12, Ybar=20,Ybar_label='20mV')
    for ax in AX:
        ax.axis('off')
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=10,Ybar_label='10Hz ')
    return fig, AX
#for cellType, color, index in zip(['Martinotti', 'Basket', 'MartinottiwithSTP', 'MartinottinoNMDA', 'BasketwithSTP'],
#for cellType, color, index in zip(['MartinottilongFull', 'MartinottilongNoSTP', 'MartinottilongNoSTPNoNMDA'],
#                                  ['tab:orange', 'tab:red', 'gold', 'tab:purple', 'tab:red'],
#                                  [0, 0, 0, 0, 0]):
cellTypes, RESULTS = [], {}
for i in np.arange(1,4):
    cellTypes.append('MartinottiNoSTP-Step%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = 1 # change here !
    load_sim(RESULTS, cellTypes[-1]) 
fig, _ = plot_sim(RESULTS, cellTypes, color='tab:orange', figsize=(2.,0.3))
cellTypes, RESULTS = [], {}
for i in np.arange(1,4):
    cellTypes.append('MartinottiNoSTPNoNMDA-Step%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = 1 # change here !
    load_sim(RESULTS, cellTypes[-1]) 
fig, _ = plot_sim(RESULTS, cellTypes, color='tab:purple', figsize=(2.,0.3))
cellTypes, RESULTS = [], {}
for i in np.arange(1,4):
    cellTypes.append('BasketNoSTP-Step%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = 1 # change here !
    load_sim(RESULTS, cellTypes[-1]) 
fig, _ = plot_sim(RESULTS, cellTypes, color='tab:red', figsize=(2.,0.3))
#    fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_%s.svg' % cellType)

# %% [markdown]
# ## Look for traces

# %%
for cellType, color in zip(['Martinotti'], ['tab:orange']):
    load_sim(cellType, RESULTS) 
    for example_index in range(0, 1):
        RESULTS['%s_example_index' % cellType] = example_index
        load_example_index(cellType, RESULTS) 
        fig, _ = plot_sim(cellType, color=color)

# %% [markdown]
# # Summary Effect

# %%
rate_smoothing = 4

def load_sim(results, cellType):

    rates = []
    for iBranch in range(6):

        filename = '../data/detailed_model/StepStim_sim_iBranch%i_%s.zip' % (iBranch, cellType)
        try:
            sim = Parallel(filename=filename)
            sim.load()
    
            sim.fetch_quantity_on_grid('spikes', dtype=list)
            sim.fetch_quantity_on_grid('Stim', dtype=list)
            seeds = np.unique(sim.spikeSeed)
            sim.fetch_quantity_on_grid('dt')
            sim.fetch_quantity_on_grid('tstop')
        
            for iW, W in enumerate(np.unique(sim.stepWidth)):
                for iA, A in enumerate(np.unique(sim.stepAmpFactor)):
                    # compute time-varying RATE !
                    dt, tstop = sim.dt[0][iW][iA], sim.tstop[0][iW][iA]
                    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                    for k, spikes in enumerate(\
                        [np.array(sim.spikes[k][iW][iA]).flatten() for k in range(len(seeds))]):
                        spikes_matrix[k,(spikes/dt).astype('int')] = True
    
                    rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                  int(rate_smoothing/dt))
                    if 'traceRate_Width%i-Amp%i_%s' % (iA, iW, cellType) in results:
                        results['traceRate_Width%i-Amp%i_%s' % (iA, iW, cellType)].append(rate)
                    else:
                        results['traceRate_Width%i-Amp%i_%s' % (iA, iW, cellType)] = [rate]
                    if not 'traceRate_Width%i'%iW in results:
                        results['t_Width%i'%iW] = np.arange(len(rate))*dt
                    results['stim_Width%i-Amp%i_%s' % (iA, iW, cellType)] = sim.Stim[0][iW][iA]
            
            results['stepWidth_%s' % cellType] = np.unique(sim.stepWidth[0])
            results['stepAmpFactor_%s' % cellType] = np.unique(sim.stepAmpFactor[0])
        except BaseException as be:
            print(be)
            print(' Pb with "%s" ' % filename)
    
results = {}

# %%
load_sim(results, 'Martinotti_longFull')
load_sim(results, 'Basket_longNoSTP')
load_sim(results, 'Martinotti_longNoSTP')

# %%
load_sim(results, 'Martinotti_vStepsFull')


# %%
def plot_sim(cellTypes, colors,
             lines=['-','-','-','-'],
             views=[300, 400, 900, 1500],
             Ybar=10):

    fig, AX = pt.figure(axes=(4,4),
                        figsize=(0.9,0.9), left=0, bottom=0., hspace=1., wspace=0.5)
    INSETS = []
    #for ax in pt.flatten(AX):
    #    ax.axis('off')
    for cellType, color, line in zip(cellTypes, colors, lines):
        for iW, W in enumerate(results['stepWidth_%s' % cellType]):
            for iA, A in enumerate(results['stepAmpFactor_%s' % cellType]):
                pt.plot(results['t_Width%i'%iW]-results['t_Width%i'%iW][-1]/2.,
                                np.mean(results['traceRate_Width%i-Amp%i_%s' % (iA, iW, cellType)], axis=0),
                                sy = np.std(results['traceRate_Width%i-Amp%i_%s' % (iA, iW, cellType)], axis=0),
                                color=color, ax=AX[iA][iW])
                if cellType==cellTypes[-1]:
                    inset = pt.inset(AX[iA][iW], [0,1, 1, 0.4])
                    #inset.axis('off')
                    inset.fill_between(results['t_Width%i'%iW][1:]-results['t_Width%i'%iW][-1]/2.,
                                       results['t_Width%i'%iW][1:]*0,
                                       results['stim_Width%i-Amp%i_%s' % (iA, iW, cellType)], color='lightgray', lw=0)
                    pt.set_plot(AX[iA][iW], [], xlim=[-views[iW]/2.,views[iW]/2.])
                    pt.set_plot(inset, [], xlim=[-views[iW]/2.,views[iW]/2.])
                    INSETS.append(inset)
                    if iA==0:
                        pt.annotate(inset, '%ims' % results['stepWidth_%s' % cellType][iW], (0.5,1), va='top', ha='center')
    pt.set_common_ylims(INSETS)
    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        pt.draw_bar_scales(ax, Ybar=Ybar, Ybar_label='%i Hz' % Ybar if ax==AX[0][0] else '',
                           Xbar=100, Xbar_label='100ms' if ax==AX[0][0] else '')
        
    return fig, AX

fig, AX = plot_sim(['Basket_longNoSTP'], ['tab:red'])

# %%
fig, AX = plot_sim(['Martinotti_longFull', 'Martinotti_longNoNMDA'], ['tab:orange', 'tab:purple'])

# %%
fig, AX = plot_sim(['Martinotti_longFull', 'Martinotti_longNoSTP'], ['tab:orange', 'k'])


# %%

def plot_sim(cellTypes, suffixs, colors, lines=['-','-','-','-'], Ybar=10):

    fig, AX = pt.figure(axes_extents=[[(1,6)],[(1,1)]],
                        figsize=(1.4,0.2), left=0, bottom=0., hspace=0.)
    for ax in AX:
        ax.axis('off')
    for cellType, suffix, color, line in zip(cellTypes, suffixs, colors, lines):
        t, input, rates = load_sim(cellType, suffix)
        pt.plot(t, np.mean(rates, axis=0), sy=np.std(rates, axis=0), ax=AX[0], color=color, lw=0)
        AX[0].plot(t, np.mean(rates, axis=0), color=color, linestyle=line)
    AX[1].fill_between(t[1:], 0*t[1:], input, color='lightgrey')
    pt.draw_bar_scales(AX[0], Xbar=50, Xbar_label='50ms', Ybar=Ybar, Ybar_label='%.0fHz' % Ybar)
    pt.set_common_xlims(AX)
    return fig, AX

fig, _ = plot_sim(['Martinotti'], ['longFull',''], ['tab:orange', 'tab:red'])
#fig.savefig('../figures/Temp-Properties-Pred/PV-vs-SST.svg')

# %%
plot_sim(['Martinotti', 'Martinotti', 'Martinotti'],
         ['', 'withSTP', 'noNMDA'],
         ['tab:orange', 'k', 'tab:purple'],
         lines=['-','--', '-'])
#fig.savefig('../figures/Temp-Properties-Pred/SST-models.svg')

# %%
plot_sim(['Basket', 'Basket'],
         ['', 'withSTP'],
         ['tab:red', 'tab:grey'])

# %%
t, input, rates = load_sim('Martinotti', '')

# %%
pt.plot(t, Y=rates)

# %%

# %%
for cellType, color in zip(['Martinotti'], ['tab:orange']):
    load_sim(cellType, RESULTS) 
    for example_index in range(0, 1):
        RESULTS['%s_example_index' % cellType] = example_index
        load_example_index(cellType, RESULTS) 
        fig, _ = plot_sim(cellType, color=color)

# %% [markdown]
# # Input Range

# %%
rate_smoothing = 20. # ms

results = {}
def load_sim(cellType, suffix):

    sim = Parallel(\
            filename='../data/detailed_model/StepStim_demo_%siRange%s.zip' % (cellType, suffix))

    sim.load()

    nSS = len(np.unique(sim.synapse_subsampling))
    nIF = len(np.unique(sim.Inh_fraction))
    nSF = len(np.unique(sim.stimFreq))
    nSA = len(np.unique(sim.stepAmpFactor))
    
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    seeds = np.unique(sim.spikeSeed)

    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

    results['traceRate'] = np.zeros((nSS, nIF, nSF, nSA, int(tstop/dt)+1))

    for iSS, SS in enumerate(np.unique(sim.synapse_subsampling)):
        for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
            for iSF, SF in enumerate(np.unique(sim.stimFreq)):
                for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
                    
                    # compute time-varying RATE !
                    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                    for k, spikes in enumerate(\
                        [np.array(sim.spikes[k][iSS][iIF][iSF][iSA]).flatten() for k in range(len(seeds))]):
                        spikes_matrix[k,(spikes/dt).astype('int')] = True
                    rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                  int(rate_smoothing/dt))
                    results['traceRate'][iSS,iIF,iSF,iSA,:] = rate
                
    results['t'] = np.arange(len(rate))*dt
    results['Inh_fraction'] = np.unique(sim.Inh_fraction[0])
    results['synapse_subsampling'] = np.unique(sim.synapse_subsampling[0])
    results['stimFreq'] = np.unique(sim.stimFreq[0])
    results['stepAmpFactor'] = np.unique(sim.stepAmpFactor[0])
    return results


def plot_sim(results):

    fig, AX = pt.figure(axes=(len(results['Inh_fraction']), len(results['synapse_subsampling'])),
                        figsize=(1,1), right=4., left=0.5, bottom=0., hspace=0., reshape_axes=False)
    for ax in pt.flatten(AX):
        ax.axis('off')

    for iIF, IF in enumerate(np.unique(results['Inh_fraction'])):
        for iSS, SS in enumerate(np.unique(results['synapse_subsampling'])):
            for iSA, SA in enumerate(np.unique(results['stepAmpFactor'])):
                for iSF, SF in enumerate(np.unique(results['stimFreq'])):
    
                    rate = results['traceRate'][iSS,iIF,iSF,iSA,:]
                    AX[iSS][iIF].plot(results['t'], rate,
                                     color=pt.viridis(iSA/(len(results['stepAmpFactor'])-0.99)))
            pt.draw_bar_scales(AX[iSS][iIF], Ybar=5, Ybar_label='5Hz ', Xbar=1e-12)
            if iIF==0:
                pt.annotate(AX[iSS][0], 'ss=%i' % results['synapse_subsampling'][iSS], (-0.3, 0.5), ha='center', rotation=90)
            if iSS==0:
                pt.annotate(AX[0][iIF], 'I=%.2f' % results['Inh_fraction'][iIF], (0.5, 1.1), ha='center')
    pt.bar_legend(AX[0][-1], X=range(len(results['stepAmpFactor'])),
                  ticks_labels=['%.1f' % f for f in results['stepAmpFactor']],
                  colorbar_inset={'rect': [1.2, 0.1, 0.07, 0.9], 'facecolor': None},
                  label='step factor',
                  colormap=pt.viridis)
    return fig, AX

results = load_sim('Martinotti', '')
fig, AX = plot_sim(results)

# %%
rate_smoothing = 20. # ms

results = {}
def load_sim(cellType, suffix):

    sim = Parallel(\
            filename='../data/detailed_model/StepStim_demo_%sRange.zip' % (cellType))

    sim.load()

    nSF = len(np.unique(sim.stimFreq))
    
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    seeds = np.unique(sim.spikeSeed)
    
    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

    results['traceRate'] = np.zeros((nSF, int(tstop/dt)+1))

    for iSF, SF in enumerate(np.unique(sim.stimFreq)):
            
        # compute time-varying RATE !
        spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
        for k, spikes in enumerate(\
            [np.array(sim.spikes[k][iSF]).flatten() for k in range(len(seeds))]):
            spikes_matrix[k,(spikes/dt).astype('int')] = True
        rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                      int(rate_smoothing/dt))
        results['traceRate'][iSF,:] = rate
                
    results['t'] = np.arange(len(rate))*dt
    results['stimFreq'] = np.unique(sim.stimFreq[0])
    return results


def plot_sim(results, color):

    fig, AX = pt.figure(axes=(len(results['stimFreq']), 1),
                        figsize=(1,1), right=4., left=0.5, bottom=0.)

    for iSF, SF in enumerate(np.unique(results['stimFreq'])):
        rate = results['traceRate'][iSF,:]
        AX[iSF].plot(results['t'], rate, color=color)
        AX[iSF].set_title('sF=%.0fHz' % SF)
        #pt.annotate(AX[iSF][0], 'I=%.2f' % results['Inh_fraction'][iIF], (0.5, 1.1), ha='center')
    return fig, AX

results = load_sim('Basket', '')
fig, AX = plot_sim(results, 'tab:red')

# %%
sim.

# %% [markdown]
# # AMPA calib

# %%
rate_smoothing = 5. # ms

results = {}

sim = Parallel(\
        filename='../data/detailed_model/StepStim_demo_MartinottiAMPAcalib.zip')

sim.load()

nSW = len(np.unique(sim.stepWidth))
nAB = len(np.unique(sim.AMPAboost))

sim.fetch_quantity_on_grid('spikes', dtype=list)
sim.fetch_quantity_on_grid('tstop')
sim.fetch_quantity_on_grid('dt')
seeds = np.unique(sim.spikeSeed)

dt = sim.fetch_quantity_on_grid('dt', return_last=True)
tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

for iSW, SW in enumerate(np.unique(sim.stepWidth)):
    for iAB, AB in enumerate(np.unique(sim.AMPAboost)):

        # compute time-varying RATE !
        dt, tstop = sim.dt[0][iSW][iAB], sim.tstop[0][iSW][iAB]
        spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
        for k, spikes in enumerate(\
            [np.array(sim.spikes[k][iSW][iAB]).flatten() for k in range(len(seeds))]):
            spikes_matrix[k,(spikes/dt).astype('int')] = True
        rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                      int(rate_smoothing/dt))
        results['traceRate_SW%i-AB%i' % (iSW, iAB)] = rate
        results['t_SW%i-AB%i' % (iSW, iAB)] = np.arange(len(rate))*dt

fig, AX = pt.figure(axes=(nSW, nAB),
                    figsize=(1,1), right=4., left=0.5, bottom=0., hspace=0., reshape_axes=False)
for ax in pt.flatten(AX):
    ax.axis('off')

for iSW, SW in enumerate(np.unique(sim.stepWidth)):
    for iAB, AB in enumerate(np.unique(sim.AMPAboost)):

        AX[iAB][iSW].fill_between(results['t_SW%i-AB%i' % (iSW, iAB)],0*results['t_SW%i-AB%i' % (iSW, iAB)],
                                  results['traceRate_SW%i-AB%i' % (iSW, iAB)], color='tab:purple')
        if iSW==0:
            pt.annotate(AX[iAB][iSW], 'Boost=%.1f' % AB, (0,0), rotation=90, ha='right', fontsize=6)

pt.set_common_ylims(AX)
pt.set_common_xlims(AX)
for ax in pt.flatten(AX):
    pt.draw_bar_scales(ax, Ybar=5, Ybar_label='5Hz ', Xbar=50, Xbar_label='50ms' if ax==AX[0][0] else '')

# %%
cellType = 'Martinotti'
sim = Parallel(\
            filename='../data/detailed_model/StepStim_demo_%siRange.zip' % cellType)
sim.load()

# %%
#im.fetch_quantity_on_grid('Stim', dtype=list)
sim.fetch_quantity_on_grid('Vm_soma', dtype=list)

# %%
fig, ax = pt.figure(figsize=(2,2))
for i in range(12):
    ax.plot(sim.Vm_soma[i][0][1][0])

# %%
fig, ax = pt.figure(figsize=(2,2))

sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)

Events = sim.presynaptic_exc_events[0][0][0][1]
for i, events in enumerate(Events):
    ax.scatter(events, i+np.ones(len(events)),
                  facecolor='g', edgecolor=None, alpha=.35, s=3)
    
Ivents = sim.presynaptic_inh_events[0][0][0][1]
for i, events in enumerate(Ivents):
    ax.scatter(events, i+len(Events)+np.ones(len(events)),
                  facecolor='r', edgecolor=None, alpha=.35, s=3)


# %%
plt.scatter(sim.spikes[0][0][0][0][0], 'o')

# %%
sim.spikes[0][0][0][3]

# %%
