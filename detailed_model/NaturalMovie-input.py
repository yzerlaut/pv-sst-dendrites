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
# # Response to Natural Movie input
#
# we feed the neuron with spikes from the "visual_coding" dataset.
#
# see [../visual_coding](../visual_coding)

# %%
import numpy as np

import sys, os
from parallel import Parallel

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt
sys.path.append('../analyz')
from analyz.processing.signanalysis import autocorrel, crosscorrel

# %%
RATES = np.load(os.path.join('..', 'data', 'visual_coding', 'RATES_natural_movie_one.npy'),
                allow_pickle=True).item()

fig1, ax = pt.figure(figsize=(1.5,1), left=0.2)

tlim = [-1.1, 6.]
cond = (RATES['time']>tlim[0]) & (RATES['time']<tlim[1])

neg_rates = 0.5*(\
        np.mean(RATES['PV_negUnits'], axis=0)+\
        np.mean(RATES['SST_negUnits'], axis=0))
scaled_neg_rates = (neg_rates-np.mean(neg_rates))/np.std(neg_rates)


pt.annotate(ax, '%.1fHz' % np.std(4.*neg_rates), (0,1),
            ha='right', va='top', color='gray')

ax.fill_between(RATES['time'][cond], 
                np.min(scaled_neg_rates),
                scaled_neg_rates[cond],
                color='gray', lw=0, alpha=.5)

for k, key, pos_color in zip(range(2),
                            ['SST', 'PV'], 
                            ['tab:orange', 'tab:red']):

    pos_rates = np.mean(RATES['%s_posUnits' % key], axis=0)
    pt.annotate(ax, (k+1)*'\n'+'%.1fHz' % np.std(4.*pos_rates), (0,1),
                ha='right', va='top', color=pos_color)
    ax.plot(RATES['time'][cond], (pos_rates[cond]-np.mean(pos_rates))/np.std(pos_rates), 
            color=pos_color)

ax.plot([tlim[1],tlim[1]-1], [8, 8], 'k-')
ax.annotate('1s',(tlim[1]-.5,8.5), ha='center') 
ax.plot(-1*np.ones(2), [4, 8], 'k-')
pt.set_plot(ax, [], xlim=tlim, title='"visual_coding" data')

# %% [markdown]
# # Single trial example
#
# ```
# python natMovie_sim.py --test --with_STP  -c Basket --tstop 500 --synapse_subsampling 20 --with_presynaptic_spikes
# ```

# %%
results = np.load('single_sim.npy', allow_pickle=True).item()

t = np.arange(len(results['Vm_soma']))*results['dt']
fig, AX = pt.figure(axes_extents=[[(1,2)],[(1,1)]],
                    figsize=(3,1), left=0, bottom=0., hspace=0.)
#AX[0].plot(t, results['Vm_dend'], 'k:', lw=0.5, label=' distal\ndendrite')
AX[0].plot(t, results['Vm_soma'], 'tab:brown', label='soma')
AX[0].plot(t, -60+0*t, 'k:')
pt.annotate(AX[0], '-60mV ', (0,-60), xycoords='data', ha='right', va='center')
pt.draw_bar_scales(AX[0], Xbar=100, Xbar_label='100ms', Ybar=10, Ybar_label='10mV')
AX[0].legend(frameon=False, loc=(1, 0.3))
for i, events in enumerate(results['presynaptic_exc_events']):
    AX[1].scatter(events, i*np.ones(len(events)), facecolor='g', edgecolor=None, alpha=.35, s=3)
for i, events in enumerate(results['presynaptic_inh_events']):
    AX[1].scatter(events, len(results['presynaptic_exc_events'])+i*np.ones(len(events)),
                  facecolor='r', edgecolor=None, alpha=.35, s=3)
    
pt.annotate(AX[1], 'Inh.', (0,1), ha='right', va='top', color='r')
pt.annotate(AX[1], 'Exc.', (0,0), ha='right', va='bottom', color='g')

print('\n number of excitatory events: %i \n ' %\
              np.sum([len(E) for E in results['presynaptic_exc_events']]))
pt.set_common_xlims(AX, lims=[t[0], t[-1]])
for ax in AX:
    ax.axis('off')

# %%
from scipy.ndimage import gaussian_filter1d

rate_smoothing = 10. # ms

# (3,4) ok
RESULTS = {'Martinotti_example_index':0,
           'Basket_example_index':0} # 4, 11 ok, 13 good

for cellType in ['Martinotti', 'Basket', 'MartinottinoNMDA', 'MartinottinoSTP']:

    sim = Parallel(\
            filename='../data/detailed_model/natMovieStim_demo_%s.zip' % cellType)
    sim.load()

    sim.fetch_quantity_on_grid('spikes', dtype=list)
    sim.fetch_quantity_on_grid('synapses', dtype=list)

    seeds = np.unique(sim.spikeSeed)

    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
    RESULTS['%s_rate' % cellType] = []
    for i, spikes in enumerate(sim.spikes):
        spikes_matrix[i,(spikes/dt).astype('int')] = True
    RESULTS['rate_%s' % cellType] = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                           int(rate_smoothing/dt))
    
    RESULTS['bgFreqInhFactor_%s' % cellType] = sim.fetch_quantity_on_grid('bgFreqInhFactor', return_last=True)

    sim.fetch_quantity_on_grid('Rate', return_last=True, dtype=np.ndarray)
    RESULTS['Input_%s' % cellType] = sim.Rate[0]

    if '%s_example_index' % cellType in RESULTS:
        sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
        RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
        RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
        RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
        
    RESULTS['t'] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt
    print(cellType, '%.2f Hz' % np.mean(RESULTS['rate_%s' % cellType][RESULTS['t']>2e3]))

# %%
zoom = [100, 4000]

RESULTS['Martinotti_example_index'] = 0 # 23, 30, 39, 41
RESULTS['MartinottinoNMDA_example_index'] = 0 # 23, 30, 39, 41
RESULTS['MartinottinoSTP_example_index'] = 0 # 23, 30, 39, 41
RESULTS['Basket_example_index'] = 0

for cellType, color in zip(['Basket', 'Martinotti', 'MartinottinoNMDA', 'MartinottinoSTP'],
                           ['tab:red', 'tab:orange', 'tab:purple', 'tab:pink']):
    
    sim = Parallel(filename='../data/detailed_model/natMovieStim_demo_%s.zip' % cellType)
    sim.load()
    sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
    sim.fetch_quantity_on_grid('synapses', return_last=True, dtype=list)
    RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
    RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('synapses', dtype=list)
    synapses = sim.synapses[RESULTS['%s_example_index' % cellType]]
    fig, AX = pt.figure(axes_extents=[[(1,1)],[(1,1)],[(1,2)],[(1,1)]],
                        figsize=(1.5,0.8), left=0, bottom=0., hspace=0.)
    # input
    AX[0].fill_between(RESULTS['t'][:-1][::20], 0*RESULTS['t'][:-1][::20], RESULTS['Input_%s' % cellType][::20],
                       color='tab:grey', lw=0)
    # Vm
    AX[2].plot(RESULTS['t'][::10], RESULTS['Vm_%s' % cellType][::10], color=color, lw=0.5)
    AX[2].plot(RESULTS['t'][::100], -60+0*RESULTS['t'][::100], 'k:', lw=0.3)
    # rate
    if RESULTS['rate_%s' % cellType] is not None:
        AX[3].fill_between(RESULTS['t'][::20], 0*RESULTS['t'][::20], RESULTS['rate_%s' % cellType][::20],
                           color=color, lw=0)
        
    # events
    if 'pre_inh_%s'%cellType in RESULTS:
        subsampling = 2 if cellType=='Basket' else 1
        # 1/4 for display
        for i, events in enumerate(RESULTS['pre_exc_%s' % cellType][::subsampling]):
            AX[1].scatter(events, i%len(synapses)+np.zeros(len(events)), facecolor='g', edgecolor=None, alpha=.35, s=.5)
        for i, events in enumerate(RESULTS['pre_inh_%s' % cellType][::subsampling]):
            AX[1].scatter(events,
                       len(RESULTS['pre_exc_%s' % cellType])/subsampling+i*np.ones(len(events)), 
                       facecolor='r', edgecolor=None, alpha=.35, s=.5)

    pt.set_common_xlims(AX, lims=zoom)
    pt.draw_bar_scales(AX[0], Xbar=200, Xbar_label='200ms', Ybar=4,
                       #Ybar_label2='%.0fHz/syn.' % (2*RESULTS['bgFreqInhFactor_%s' % cellType]),
                       Ybar_label='%.0fHz ' % (4))
    pt.annotate(AX[2], '-60mV ', (zoom[0],-60), xycoords='data', ha='right', va='center')
    pt.draw_bar_scales(AX[2], Xbar=1e-12, Ybar=20,Ybar_label='20mV')
    pt.annotate(AX[1], 'Inh.', (0,1), ha='right', va='top', color='r', fontsize=7)
    pt.annotate(AX[1], 'Exc.', (0,0), ha='right', va='bottom', color='g', fontsize=7)
    pt.annotate(AX[1], '%i syn.' % len(synapses), (0,.5), ha='right', va='center', color='k', fontsize=6)
    for ax in AX:
        ax.axis('off')
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=10,Ybar_label='10Hz')
#    fig.savefig('../figures/Figure5/StochProcSim_example_%s.pdf' % cellType)

# %%
fig, ax = pt.figure(figsize=(1.1,0.85))
subsampling = 100
width = 1000
CCs = {}

for cellType, color in zip(['Martinotti', 'Basket', 'MartinottinoNMDA'],
                           ['tab:orange', 'tab:red']):

    # input
    """
    """
    if cellType=='Martinotti':
        cond = RESULTS['t']>1e3
        CCF, time_shift = crosscorrel(RESULTS['Input_%s' % cellType][cond[1:]][::subsampling], 
                              RESULTS['Input_%s' % cellType][cond[1:]][::subsampling], 
                              width, subsampling*RESULTS['dt'])
        """
        CCs['Input_CC'] = CCF
        ax.plot(time_shift/1e3, CCF, 
                linestyle=(0, (1, 0.2)),
                color='silver', lw=4, label='input')
        """
        
    cond = RESULTS['t']>1e3
    CCF, time_shift = crosscorrel(RESULTS['Input_%s' % cellType][cond[1:]][::subsampling], 
                          RESULTS['rate_%s' % cellType][1:][cond[1:]][::subsampling], 
                          width, subsampling*RESULTS['dt'])
    """
    # gaussian fit
    ax.plot(time_shift/1e3, 
            gaussian(time_shift,
                     fit_gaussian_width(time_shift, CCF/np.max(CCF))), '--', lw=0.5)
    """
    
    if not 'noNMDA' in cellType:
        ax.plot(time_shift/1e3, CCF, color=color, lw=1.5)
    CCs['%s_CC' % cellType] = CCF
    
    CCs['time_shift'] = time_shift
    
#ax.legend(loc=(1,1))
pt.set_plot(ax, xlabel='jitter (s)',
            xticks=[-0.9,0,0.9],
            #yticks=[0.,0.5,1.0],
            #ylim=[-0.15,1.08],
            #xlim=[-0.21,0.27], 
            title='model',
            ylabel='corr. coef.')
#fig.savefig('../figures/Figure5/CrossCorrel-Model.pdf')

# %%
fig, ax = pt.figure(figsize=(0.8,0.85))

for k, cellType, color in zip(range(3),
                              ['Basket', 'Martinotti', 'MartinottinoNMDA'],
                              ['tab:red', 'tab:orange', 'tab:purple']):

    i0 = int(len(CCs['time_shift'])/2)
    
    #tau = fit_exponential_decay(CCs['time_shift'][i0:], CCs['%s_CC' % cellType][i0:]/CCs['%s_CC' % cellType][i0])
    tau = fit_gaussian_width(CCs['time_shift'], CCs['%s_CC' % cellType]/np.max(CCs['%s_CC' % cellType]))[0]
    ax.bar([1+k], [tau], color=color)

    if cellType=='Basket':
        tau = fit_gaussian_width(CCs['time_shift'], CCs['Input_CC']/np.max(CCs['Input_CC']))[0]
        ax.bar([0], [tau], color='tab:grey')
    

    #ax11.bar([k], [CCF[int(len(time_shift)/2)]], color=pos_color)
    
    #plt.plot(ts, np.exp(-ts/tau), color=color)
#pt.set_plot(ax, yticks=[0,50,100])

pt.set_plot(ax, ['left'], yticks=[0,50,100],
            title='single seed',
            #ylabel=u'\u00bd' + ' width\n(ms)',
            ylabel='width (ms)')
#fig.savefig('../figures/detailed_model/Widths.svg')

# %% [markdown]
# ## Analysis over different seeds

# %%
from scipy.ndimage import gaussian_filter1d

rate_smoothing = 10. # ms
subsampling = 100

RESULTS = {} # 4, 11 ok, 13 good

for cellType in ['Martinotti', 'Basket', 'MartinottinoNMDA']:

    RESULTS['rate_%s' % cellType] = []
    RESULTS['Input_%s' % cellType] = []
    RESULTS['CC_%s' % cellType] = [] # cross-correl
    RESULTS['AC_%s' % cellType] = [] # auto-correl
    
    for iBranch in range(6):
        
        try:
            sim = Parallel(\
                    filename='../data/detailed_model/tvRateStim_simBranch%i_%s.zip' % (iBranch,cellType))
            sim.load()

            sim.fetch_quantity_on_grid('spikes', dtype=list)
            sim.fetch_quantity_on_grid('Rate', dtype=np.ndarray)

            spikeSeeds = np.unique(sim.spikeSeed)
            stochProcSeeds = np.unique(sim.stochProcSeed)

            dt = sim.fetch_quantity_on_grid('dt', return_last=True)
            tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
            t = np.arange(int(tstop/dt)+1)*dt

            for stochProcSeed in stochProcSeeds:
                cond = sim.stochProcSeed==stochProcSeed

                # fetch input
                RESULTS['Input_%s' % cellType].append(sim.OU[cond][0])
                # compute rate
                spikes_matrix= np.zeros((len(spikeSeeds), int(tstop/dt)+1))
                for i, spikes in enumerate(sim.spikes[cond]):
                    spikes_matrix[i,(spikes/dt).astype('int')] = True
                # store rate
                RESULTS['rate_%s' % cellType].append(1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                                           int(rate_smoothing/dt)))

                tCond = t[1:]>1e3 # remove initial transients

                # compute Cross-Correl
                CCF, time_shift = crosscorrel(RESULTS['Input_%s' % cellType][-1][tCond][::subsampling], 
                                              RESULTS['rate_%s' % cellType][-1][1:][tCond][::subsampling], 
                                              400, subsampling*dt)
                RESULTS['CC_%s' % cellType].append(CCF)

                # compute Auto-Correl
                CCF, time_shift = crosscorrel(RESULTS['Input_%s' % cellType][-1][tCond][::subsampling], 
                                              RESULTS['Input_%s' % cellType][-1][tCond][::subsampling], 
                                              400, subsampling*dt)
                RESULTS['AC_%s' % cellType].append(CCF)
        except BaseException as be:
            print(cellType, 'branch', iBranch, 'no data')

    """
    print('%s, mean rate %.1f +/- %.1f Hz' % (cellType,
                                              np.mean(np.mean(RESULTS['rate_%s' % cellType], axis=0)),
                                              np.std(np.mean(RESULTS['rate_%s' % cellType], axis=0))))
    """
        
    RESULTS['t'] = t
    RESULTS['dt'] = dt
    RESULTS['time_shift'] = time_shift

# %%
from scipy.optimize import minimize

# Lorentzian Fit of decay

def gaussian(t, X):
    #return (1-X[2])*np.exp(-(t-X[1])**2/2./X[0]**2)+X[2]
    return (1-X[2])*(1/(1+(((t-X[1])/X[0])**2)))+X[2]
                 
def fit_gaussian_width(shift, array,
                       min_time=10,
                       max_time=1000):
    #i0 = np.argmax(array)
    def func(X):
        return np.sum(np.abs(gaussian(shift, X)-array))
        #return np.sum(np.abs(gaussian(shift[i0:]-shift[i0], X)-array[i0:]))
    
    res = minimize(func, [3*min_time,0,0],
                   bounds=[[min_time, max_time],
                           [-max_time, max_time],
                           [-1,1]], method='L-BFGS-B')
    return res.x


# %%
from scipy import stats

fig, ax = pt.figure(figsize=(1.1,0.85))
subsampling = 100

for cellType, color in zip(['Martinotti', 'Basket', 'MartinottinoNMDA'],
                           ['tab:orange', 'tab:red', 'tab:purple']):

    cond = RESULTS['t']>1

    # input
    if cellType=='Martinotti':
        pt.plot(RESULTS['time_shift']/1e3, 
                np.nanmean(RESULTS['AC_%s' % cellType], axis=0),
                sy=stats.sem(RESULTS['AC_%s' % cellType], axis=0),
                ax=ax, no_set=True, color='tab:grey')
    
    pt.plot(RESULTS['time_shift']/1e3, 
            np.nanmean(RESULTS['CC_%s' % cellType], axis=0),
            sy=stats.sem(RESULTS['CC_%s' % cellType], axis=0, nan_policy='omit'),
            ax=ax, no_set=True, color=color, lw=0.5)

    ts = RESULTS['time_shift']
    ccc = np.nanmean(RESULTS['CC_%s' % cellType], axis=0)
    ax.plot(ts/1e3, np.max(ccc)*gaussian(ts, 
                             fit_gaussian_width(ts, ccc/np.max(ccc))), lw=3, color=color, alpha=.3)

    
pt.set_plot(ax, xlabel='jitter (s)', 
            ylabel='corr. coef.',
            yticks=[0.,0.5,1.0],
            xlim=[-0.35,0.35],
            xticks=[-0.3,0,0.3])
#fig.savefig('../figures/detailed_model/CrossCorrel.pdf')

# %%
# Gaussian fit to quantify the decay

fig, ax = pt.figure(figsize=(0.8,0.9))

RESULTS['tauAC'] = [] # auto-correl
for k, cellType, color in zip(range(3),
                              ['Basket', 'Martinotti', 'MartinottinoNMDA'],
                              ['tab:red', 'tab:orange', 'tab:purple']):

    
    RESULTS['tauCC_%s' % cellType] = [] # cross-correl

    if k==0:
        for AC in RESULTS['AC_%s' % cellType]:
            tau = fit_gaussian_width(RESULTS['time_shift'],
                                     AC/np.max(AC))[0]
            RESULTS['tauAC'].append(tau)
        print("Input -> half-width= %.1f +/- %.1f ms" % (np.mean(RESULTS['tauAC']),
                                                      stats.sem(RESULTS['tauAC'])))
    ax.bar([0], 
           [np.mean(RESULTS['tauAC'])],
           yerr=[stats.sem(RESULTS['tauAC'])],
           color='tab:grey')
    
    for CC in RESULTS['CC_%s' % cellType]:
        tau = fit_gaussian_width(RESULTS['time_shift'],
                                 CC/np.max(CC))[0]
        RESULTS['tauCC_%s' % cellType].append(tau)
        
    ax.bar([1+k], 
           [np.mean(RESULTS['tauCC_%s' % cellType])],
           yerr=[stats.sem(RESULTS['tauCC_%s' % cellType])],
           color=color)
    print("%s -> half-width= %.1f +/- %.1f ms" % (cellType, np.mean(RESULTS['tauCC_%s' % cellType]), stats.sem(RESULTS['tauCC_%s' % cellType])))

print('')

keys = ['tauAC', 'tauCC_Basket', 'tauCC_Martinotti', 'tauCC_MartinottinoNMDA']

import itertools
for i, j in itertools.product(range(len(keys)), range(len(keys))):
    if i>j:
        print(keys[i], keys[j], ', p=%.0e' % stats.mannwhitneyu(RESULTS[keys[i]], RESULTS[keys[j]]).pvalue)
    
    
pt.set_plot(ax, ['left'], 
            ylabel=u'\u00bd width (s)',
            #ylabel='width (ms)',
            yticks=[0,100], yticks_labels=['0.0', '0.1'])
#fig.savefig('../figures/Figure5/Half-Widths-Summary.pdf')

# %% [markdown]
# ## Compute cross-correlation functions

# %%
subsampling = 20
for c, cellType, color in zip(range(2), ['Martinotti', 'Basket'], ['tab:orange', 'tab:red']):
    RESULTS['%s_CCs' % cellType] = []
    for s in range(len(seeds)):
        CCF, ts = crosscorrel(RESULTS['%s_rates' % cellType][s][1:][::subsampling],
                              RESULTS['StochProc'][s][::subsampling], 
                              1e3, subsampling*dt)
        RESULTS['%s_CCs' % cellType].append(CCF)
    
RESULTS['CC_StochProc'] = []
for sc in RESULTS['StochProc']:
    CCF, ts = crosscorrel(sc[::subsampling], sc[::subsampling],
                          1e3, subsampling*dt)
    RESULTS['CC_StochProc'].append(CCF)
