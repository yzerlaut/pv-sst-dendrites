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


##################################################
## computing trial-averaged firing rates (psth) ##
##################################################
from scipy.ndimage import gaussian_filter1d
rate_smoothing = 10. # ms
def compute_rate_psth(sim, tstop, dt, seeds,
                      rate_smoothing=rate_smoothing):

    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
    for i, spikes in enumerate(sim.spikes.flatten()):
        spikes_matrix[i,(spikes/dt).astype('int')] = True
    return 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                 int(rate_smoothing/dt))


# %% [markdown]
# # Rate dynamics of PV, SST and ~Pyr populations

# %%
# let's compute the average rates that we will need:
RATES = np.load(os.path.join('..', 'data', 'visual_coding', 'RATES_natural_movie_one.npy'),
                allow_pickle=True).item()
subsampling = RATES['time']<100 
dtype = np.float32
Averaged = {
    'time':np.array(RATES['time'][subsampling], dtype=dtype),
    'PV_posUnits':np.array(np.mean(RATES['PV_posUnits'], axis=0)[subsampling], dtype=dtype),
    'PV_negUnits':np.array(np.mean(RATES['PV_negUnits'], axis=0)[subsampling], dtype=dtype),
    'SST_posUnits':np.array(np.mean(RATES['SST_posUnits'], axis=0)[subsampling], dtype=dtype),
    'SST_negUnits':np.array(np.mean(RATES['SST_negUnits'], axis=0)[subsampling], dtype=dtype),
}        
for key in ['PV', 'SST']:
    pos_rates = [np.mean(r) for r in RATES['%s_posUnits' % key]]
    print(' - mean rate of "%s" cells: %.1f +/- %.1f Hz' % (key, np.mean(pos_rates), np.std(pos_rates)))

np.save(os.path.join('..', 'data', 'visual_coding', 'avRATES_natural_movie_one.npy'), 
        Averaged)

# %%
RATES = np.load(os.path.join('..', 'data', 'visual_coding', 'avRATES_natural_movie_one.npy'),
                allow_pickle=True).item()

fig, AX = pt.figure(axes=(2,1), figsize=(1.5,1), left=0.2)

for tlim, ax, label in zip([[-1.1, 6.], [27.9, 35.]], AX,
                           ['onset', 'offset']):
                           
    cond = (RATES['time']>tlim[0]) & (RATES['time']<tlim[1])
    
    neg_rates = 0.5*(RATES['PV_negUnits']+RATES['SST_negUnits'])
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
    
        pos_rates = RATES['%s_posUnits' % key]
        pt.annotate(ax, (k+1)*'\n'+'%.1fHz' % np.std(4.*pos_rates), (0,1),
                    ha='right', va='top', color=pos_color)
        ax.plot(RATES['time'][cond], (pos_rates[cond]-np.mean(pos_rates))/np.std(pos_rates), 
                color=pos_color)
    
    ax.plot([tlim[1],tlim[1]-1], [8, 8], 'k-')
    ax.annotate('1s',(tlim[1]-.5,8.5), ha='center') 
    ax.plot((tlim[0]+.1)*np.ones(2), [4, 8], 'k-')
    pt.set_plot(ax, [], xlim=tlim, title=label)
pt.set_common_ylims(AX)
#fig.savefig('../figures/visual_coding/natMovie-stim-repetition.svg') # ADDED DRAWINGS, DON'T OVERWRITE !

# %% [markdown]
# # Simulations -> Single trial example
#
# ```
# python natMovie_sim.py --test --with_STP  -c Basket --tstop 3000 --synapse_subsampling 2 --Inh_fraction 0.05 --with_presynaptic_spikes
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
# # Simulations -> finding input parameters to get ~15Hz output firing

# %%
results = {}
Nss, Nif = 4, 4
with_traces = False
#for cellType in :
TYPES = ['BasketInputRange', 'BasketInputRange_noSTP',
         'MartinottiInputRange', 'MartinottiInputRange_noSTP', 'MartinottiInputRange_noNMDA', 'MartinottiInputRange_noNMDAnoSTP']
for cellType in TYPES:
    
    results['oRate_%s' % cellType] = np.zeros((6, Nss, Nif))
    results['Inh_fraction_%s' % cellType], results['synapse_subsampling_%s' % cellType] = np.zeros((Nss, Nif)), np.zeros((Nss, Nif))
        
    for iBranch in range(6):

        filename='../data/detailed_model/natMovieStim_simBranch%i_%s.zip' % (iBranch,cellType)
        try:
            #print('processing ', filename, ' [...]')
            sim = Parallel(filename=filename)
            sim.load()

            sim.fetch_quantity_on_grid('spikes', dtype=list)
            seeds = np.unique(sim.spikeSeed)

            dt = sim.fetch_quantity_on_grid('dt', return_last=True)
            tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

            if with_traces and ('traceRate_%s' % cellType not in results):
                results['traceRate_%s' % cellType] = np.zeros((6, Nss, Nif, int(tstop/dt)+1))

            for i, iF in enumerate(np.unique(sim.Inh_fraction)):
                for s, ss in enumerate(np.unique(sim.synapse_subsampling)):
                    
                    results['oRate_%s' % cellType][iBranch,s,i] = 1e3/tstop*\
                                np.mean([len(sim.spikes[n][i][s]) for n in range(len(seeds))])
                    
                    results['Inh_fraction_%s' % cellType][s,i] = sim.Inh_fraction[0][i][s]
                    results['synapse_subsampling_%s' % cellType][s,i] = sim.synapse_subsampling[0][i][s]
                    
                    if with_traces:
                        # compute time-varying RATE !
                        spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                        for i, spikes in enumerate(sim.spikes[:][i][s]):
                            print(spikes.max())
                            spikes_matrix[i,(spikes/dt).astype('int')] = True
                        rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                      int(100./dt))
                        t = np.arange(len(rate))*dt
                        results['traceRate_%s' % cellType][iBranch,s,i,:] = rate
            results['Inh_fraction'] = np.unique(sim.Inh_fraction[0])
            results['synapse_subsampling'] = np.unique(sim.synapse_subsampling[0])
 
        except BaseException as be:
            print(be)
            print(filename, ' not working')

# %%
ticks = np.arange(5)*5
for cellType in TYPES:
    print(cellType)
    fig, ax = pt.figure(figsize=(1., 1.4), right=8)
    Max = np.max(results['oRate_%s' % cellType].mean(axis=0))
    means = results['oRate_%s' % cellType].mean(axis=0)
    means[means>20] = 20
    Ticks = ticks[ticks<=np.max([5,Max])]
    Ticks_labels = [str(t) if t!=20 else '>20' for t in Ticks]
    fig, ax, acb = pt.matrix(means.T, aspect='auto',
                             origin='upper', ax=ax, bar_legend_args={'label':'firing (Hz)', 'bounds':[0, min([20,max([5,Max])])],
                                                                     'ticks':Ticks, 'ticks_labels':Ticks_labels})
    #acb[1].set_ylim([0,11])
    fig.suptitle(cellType.replace('Basket', 'PV -').replace('Martinotti', 'SST -').replace('InputRange','').replace('_',' ')+' model')
    pt.set_plot(ax, 
                xticks=np.linspace(0.4,Nif-1.3,Nif), yticks=np.linspace(0.4,Nss-1.4,Nss),
                #xticks=np.arange(Nss)+0.5,
                #xticks_labels=['%.1f' for f in 100*results['Inh_fraction_%s' % cellType]],
                xticks_labels=['%.1f'%f for f in 100*results['Inh_fraction']],
                yticks_labels=['%.1f'%f for f in 100/results['synapse_subsampling'][::-1]],
                xticks_rotation=60,
                #xlim = [1., 20], ylim=[5,100],
                xlabel='$F_{inh}$ (%)', ylabel='$F_{syn}$ (%)')
    # find the best parameter to reach Fobjective
    iS = np.flatnonzero((means.flatten()>2) & (means.flatten()<15))
    for i in iS:
        print(' Inh_fraction = %.3f, synapse_subsampling=%i -->   output rate : %.1f +/- %.1f Hz ' %\
                    (results['Inh_fraction_%s' % cellType].flatten()[i],
                     results['synapse_subsampling_%s' % cellType].flatten()[i],
                     results['oRate_%s' % cellType].mean(axis=0).flatten()[i],
                     results['oRate_%s' % cellType].std(axis=0).flatten()[i]))
    fig.savefig('../figures/detailed_model/natMovie-input-space-%s.svg' % cellType)

# %% [markdown]
# # Multiple trials to compute PSTH

# %%
#for cellType in ['Martinotti', 'Basket']:
TYPES = ['Basket', 'BasketnoSTP', 'Martinotti', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottinoSTPnoNMDA']
for cellType in TYPES:

    sim = Parallel(\
            filename='../data/detailed_model/natMovieStim_demo_%s.zip' % cellType)
    sim.load()

    
    print('\n', cellType, ' == from params: Inh. Frac.: %.2f, Syn. Subsmpl.: %i' % (\
                                            sim.fetch_quantity_on_grid('Inh_fraction', return_last=True),
                                            sim.fetch_quantity_on_grid('synapse_subsampling', return_last=True)))
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    sim.fetch_quantity_on_grid('synapses', dtype=list)
    sim.fetch_quantity_on_grid('Rate', return_last=True, dtype=np.ndarray)
    #print(' input rate: %.1f  Hz' % (np.mean(sim.Rate[0])))
    seeds = np.unique(sim.spikeSeed)

    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
    
    # compute RATE !
    rate = compute_rate_psth(sim, tstop, dt, seeds)
    
    sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    syn_exc_rates = [np.mean([1e3*len(E)/tstop for E in sim.presynaptic_exc_events[i]]) for i in range(len(seeds))]
    print('           exc syn. rate: %.1f +/- %.1f Hz' % (np.mean(syn_exc_rates), np.std(syn_exc_rates)))
    print('              --> average release proba (of single events): %.2f ' % (np.mean(syn_exc_rates)/np.mean(sim.Rate[0])))
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
        
    t = np.arange(len(rate))*dt
    print('     => output rate: %.2f Hz' % np.mean(rate[t>0.1e3]))


# %%
def show_single_and_trial_average(cellType, RESULTS,
                                  zoom = [450, 4000],
                                  example_index=0,
                                  color='k', 
                                  with_inset=False, figsize=(1.8,0.9)):

    RESULTS['%s_example_index' % cellType] = example_index
    
    sim = Parallel(filename='../data/detailed_model/natMovieStim_demo_%s.zip' % cellType)
    sim.load()

    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
    seeds = np.unique(sim.spikeSeed)

    sim.fetch_quantity_on_grid('Vm_soma', return_last=True, dtype=np.ndarray)
    sim.fetch_quantity_on_grid('synapses', return_last=True, dtype=list)
    RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
    RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('synapses', dtype=list)
    synapses = sim.synapses[RESULTS['%s_example_index' % cellType]]
    sim.fetch_quantity_on_grid('Rate', return_last=True, dtype=np.ndarray)
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    Inh_frac = sim.fetch_quantity_on_grid('Inh_fraction', return_last=True)
    RESULTS['rate_%s' % cellType] = compute_rate_psth(sim, tstop, dt, seeds)
    RESULTS['Input'] = sim.Rate[0]
    RESULTS['t'] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt
    
    fig, AX = pt.figure(axes_extents=[[(1,1)],[(1,1)],[(1,2)],[(1,1)]],
                        figsize=figsize, left=0, bottom=0., hspace=0., right=5)
    # input
    AX[0].fill_between(RESULTS['t'][:-1][::20], 0*RESULTS['t'][:-1][::20], RESULTS['Input'][::20],
                       color='tab:grey', lw=0)
    # Vm
    cond = (RESULTS['t']>zoom[0]) & (RESULTS['t']<zoom[1])
    AX[2].plot(RESULTS['t'][cond][::10], RESULTS['Vm_%s' % cellType][cond][::10], color=color, lw=0.5)
    AX[2].plot(RESULTS['t'][cond][::100], -60+0*RESULTS['t'][cond][::100], 'k:', lw=0.3)

    
    # rate
    AX[3].fill_between(RESULTS['t'][cond][::20], 0*RESULTS['t'][cond][::20], RESULTS['rate_%s' % cellType][cond][::20],
                       color=color, lw=0)
        
    # events
    if 'pre_inh_%s'%cellType in RESULTS:
        subsampling = 4 if cellType=='Basket' else 1
        # 1/4 for display
        for i, events in enumerate(RESULTS['pre_exc_%s' % cellType][::subsampling]):
            events = np.array(events)
            cond = (events>zoom[0]) & (events<zoom[1])
            AX[1].scatter(events[cond], i%len(synapses)+np.zeros(len(events[cond])), facecolor='g', edgecolor=None, alpha=.35, s=.5)
        for i, events in enumerate(RESULTS['pre_inh_%s' % cellType][::subsampling]):
            events = np.array(events)
            cond = (events>zoom[0]) & (events<zoom[1])
            AX[1].scatter(events[cond],
                       len(RESULTS['pre_exc_%s' % cellType])/subsampling+i*np.ones(len(events[cond])), 
                       facecolor='r', edgecolor=None, alpha=.35, s=.5)

    pt.set_common_xlims(AX, lims=zoom)
    pt.draw_bar_scales(AX[0], Xbar=200, Xbar_label='200ms', Ybar=4,
                       #Ybar_label2='%.0fHz/syn.' % (2*RESULTS['bgFreqInhFactor_%s' % cellType]),
                       Ybar_label='%.0fHz ' % (4))
    pt.annotate(AX[2], '-60mV ', (zoom[0],-60), xycoords='data', ha='right', va='center')
    pt.draw_bar_scales(AX[2], Xbar=1e-12, Ybar=20,Ybar_label='20mV ')
    pt.annotate(AX[1], 'Inh.', (0,1), ha='right', va='top', color='r', fontsize=7)
    pt.annotate(AX[1], 'Exc.', (0,0), ha='right', va='bottom', color='g', fontsize=7)
    #pt.annotate(AX[1], '%i syn.' % len(synapses), (0,.5), ha='right', va='center', color='k', fontsize=6)
    for ax in AX:
        ax.axis('off')
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=10,Ybar_label='10Hz ')

    if with_inset:
        inset = pt.inset(fig, [0.85,0.6,0.1,0.3])
        inset2 = pt.inset(fig, [0.85,0.2,0.1,0.3])
        cond, subsampling, width = RESULTS['t']>0.1e3, 100, 1500
        cond = RESULTS['t']>0.1e3
        CCF, time_shift = crosscorrel(RESULTS['Input'][cond[1:]][::subsampling], RESULTS['Input'][cond[1:]][::subsampling], 
                                      width, subsampling*RESULTS['dt'])
        inset.plot(time_shift/1e3, CCF, color='tab:grey', lw=0.5)
        inset2.plot(time_shift/1e3, CCF/np.max(CCF), color='tab:grey', lw=0.5)
        CCF, time_shift = crosscorrel(RESULTS['Input'][cond[1:]][::subsampling], 
                                      RESULTS['rate_%s' % cellType][1:][cond[1:]][::subsampling], 
                                      width, subsampling*RESULTS['dt'])
        inset.plot(time_shift/1e3, CCF, color=color, lw=1.5)
        inset2.plot(time_shift/1e3, CCF/np.max(CCF), color=color, lw=1.5)
        pt.set_plot(inset, ['left'], xlabel='jitter (s)', xticks=[-0.9,0,0.9],
                    xlim=[-0.95,1.2], yticks=[0.,0.5,1.0], ylabel='corr. coef.')
        pt.set_plot(inset2, xlabel='jitter (s)', xticks=[-0.9,0,0.9],
                    xlim=[-0.95,1.2], yticks=[0.,0.5,1.0], ylabel='norm. C.C.')

    return fig, AX

RESULTS = {}
fig, _ = show_single_and_trial_average('Basket', RESULTS, example_index=6, color='tab:red')
fig.savefig('../figures/detailed_model/natMovie-raw-short-PV.svg')
fig, _ = show_single_and_trial_average('Martinotti', RESULTS, example_index=7, color='tab:orange')
fig.savefig('../figures/detailed_model/natMovie-raw-short-SST.svg') # 1 , 6, 7, 9

# %%
TYPES = ['Basket', 'BasketnoSTP', 'Martinotti', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottinoSTPnoNMDA']
COLORS = ['tab:red', 'rosybrown', 'tab:orange', 'tab:purple', 'gold', 'y']
for cellType, color, id in zip(TYPES, COLORS, [6,6,7,7,7,7,7]):
    fig, AX = show_single_and_trial_average(cellType, RESULTS, example_index=id,
                                  color=color, zoom=[0.1e3, 12e3], figsize=(3.,0.8)) #with_inset=True, figsize=(3.3,0.8))
    fig.suptitle(cellType.replace('Basket', 'PV - ').replace('Martinotti', 'SST - '), color=color)
    fig.savefig('../figures/detailed_model/natMovie-raw-long-%s.svg' % cellType)


# %% [markdown]
# ## Compute Half-Widths of Cross-Correlation Functions

# %%
from scipy.optimize import minimize

# Lorentzian Fit of decay

def lorentzian(t, X):
    return 1./(1+(t/X[0])**2)

Func = lorentzian # Change here !
def fit_half_width(shift, array,
                   min_time=10,
                   max_time=2000):
    def to_minimize(X):
        return np.sum(np.abs(Func(shift, X)-array))
    res = minimize(to_minimize, [3*min_time],
                   bounds=[[min_time, max_time]], method='L-BFGS-B')
    return res.x

def norm(trace):
    return (trace-np.min(trace))/(np.max(trace)-np.min(trace))


# %%
fig, ax = pt.figure(figsize=(1.1,0.85))
subsampling = 100
width = 1500
CCs = {}

TYPES = ['Basket', 'BasketnoSTP', 'Martinotti', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottinoSTPnoNMDA']
COLORS = ['tab:red', 'rosybrown', 'tab:orange', 'tab:purple', 'gold', 'y']

for cellType, color in zip(TYPES, COLORS):


    # input
    """
    """
    if 'Input_CC' not in CCs:
        cond = RESULTS['t']>0.1e3
        CCF, time_shift = crosscorrel(RESULTS['Input'][cond[1:]][::subsampling],  RESULTS['Input'][cond[1:]][::subsampling], 
                              width, subsampling*RESULTS['dt'])
        CCs['Input_CC'] = CCF
        #ax.plot(time_shift/1e3, CCF, color='tab:grey', lw=0.5)
        
    cond = RESULTS['t']>0.1e3
    CCF, time_shift = crosscorrel(RESULTS['Input'][cond[1:]][::subsampling], 
                                  RESULTS['rate_%s' % cellType][1:][cond[1:]][::subsampling], 
                          width, subsampling*RESULTS['dt'])
    ax.plot(time_shift/1e3, CCF, color=color, lw=1.5)
    CCs['%s_CC' % cellType] = CCF
    
    CCs['time_shift'] = time_shift
    
#ax.legend(loc=(1,1))
pt.set_plot(ax, xlabel='jitter (s)',
            xticks=[-0.9,0,0.9],
            xlim=[-0.95,1.2],
            #yticks=[0.,0.5,1.0],
            #ylim=[-0.15,1.08],
            #xlim=[-0.21,0.27], 
            title='model',
            ylabel='corr. coef.')
fig.savefig('../figures/detailed_model/natMovie-CrossCorrel-Func-Example.svg')
#fig.savefig('../figures/Figure5/CrossCorrel-Model.pdf')

# %%
TYPES = ['Basket', 'BasketnoSTP', 'Martinotti', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottinoSTPnoNMDA']
COLORS = ['tab:red', 'rosybrown', 'tab:orange', 'tab:purple', 'gold', 'y']
 
fig, ax = pt.figure(figsize=(1.1,0.85))

for k, cellType, color in zip(range(len(TYPES)), TYPES, COLORS):

    i0 = int(len(CCs['time_shift'])/2)
    fit_cond = (CCs['time_shift']>0) 
    
    #tau = fit_exponential_decay(CCs['time_shift'][i0:], CCs['%s_CC' % cellType][i0:]/CCs['%s_CC' % cellType][i0])
    tau = fit_half_width(CCs['time_shift'][fit_cond], norm(CCs['%s_CC' % cellType][fit_cond]))[0]
    ax.bar([1+k], [1e-3*tau], color=color)

    if cellType=='Basket':
        tau = fit_half_width(CCs['time_shift'][fit_cond], norm(CCs['Input_CC'][fit_cond]))[0]
        ax.bar([0], [1e-3*tau], color='tab:grey')
    

    #ax11.bar([k], [CCF[int(len(time_shift)/2)]], color=pos_color)
    
    #plt.plot(ts, np.exp(-ts/tau), color=color)
#pt.set_plot(ax, yticks=[0,50,100])

pt.set_plot(ax, ['left'], title='single seed',
            ylabel=u'\u00bd' + ' width$^{+}$ (s)')

#fig.savefig('../figures/detailed_model/Widths.svg')

# %% [markdown]
# # Simulations over Multiple Branches 

# %%
from scipy.ndimage import gaussian_filter1d

rate_smoothing = 10. # ms
subsampling, tmax = 100, 1500

RESULTS = {} # 4, 11 ok, 13 good

for cellType in TYPES:

    RESULTS['rate_%s' % cellType] = []
    RESULTS['Input_%s' % cellType] = []
    RESULTS['CC_%s' % cellType] = [] # cross-correl
    
    for iBranch in range(6):
        
        try:
            fn = '../data/detailed_model/natMovieStim_simBranch%i_%s.zip' % (iBranch,
                         cellType.replace('Basket', 'BasketFull').replace('Martinotti', 'MartinottiFull'))
            sim = Parallel(filename=fn)
            sim.load()
       
            dt = sim.fetch_quantity_on_grid('dt', return_last=True)
            tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
            seeds = np.unique(sim.spikeSeed)
            
            sim.fetch_quantity_on_grid('Rate', dtype=np.ndarray)
            RESULTS['Input_Rate'] = sim.Rate[0].flatten()[0]
            sim.fetch_quantity_on_grid('spikes', dtype=list)
            RESULTS['rate_%s' % cellType].append(compute_rate_psth(sim, tstop, dt, seeds))
            
            # compute Cross-Correl
            CCF, time_shift = crosscorrel(RESULTS['Input_Rate'][::subsampling], 
                                          RESULTS['rate_%s' % cellType][-1][1:][::subsampling], 
                                          tmax, subsampling*dt)
            RESULTS['CC_%s' % cellType].append(CCF)

            RESULTS['t'] = np.arange(len(RESULTS['rate_%s' % cellType][-1]))*dt

            # compute Auto-Correl
            if 'ACF' not in RESULTS:
                CCF, time_shift = crosscorrel(RESULTS['Input_Rate'][::subsampling], 
                                              RESULTS['Input_Rate'][::subsampling], 
                                              tmax, subsampling*dt)
                RESULTS['ACF'] = CCF
                RESULTS['time_shift'] = time_shift
        except BaseException as be:
            print(cellType, 'branch', iBranch, 'no data', be)
    print(' %s: output rate: %.1f +/- %.1f Hz' % (cellType,
                                                 np.mean(RESULTS['rate_%s' % cellType],axis=1).mean(),
                                                 np.mean(RESULTS['rate_%s' % cellType],axis=1).std()))
    print('   ', np.mean(RESULTS['rate_%s' % cellType],axis=1))


# %%
fig, ax = pt.figure(figsize=(1.,0.85))

#pt.plot(1e-3*RESULTS['time_shift'], np.mean(RESULTS['CC_Martinotti'], axis=0), 
#        sy=np.std(RESULTS['CC_Martinotti'], axis=0),
pt.plot(1e-3*RESULTS['time_shift'], RESULTS['CC_Martinotti'][3], 
        ax=ax, color='tab:orange')
#pt.plot(1e-3*RESULTS['time_shift'], np.mean(RESULTS['CC_Basket'], axis=0),
#        sy=np.std(RESULTS['CC_Basket'], axis=0),
pt.plot(1e-3*RESULTS['time_shift'], RESULTS['CC_Basket'][1],
        sy=np.std(RESULTS['CC_Basket'], axis=0),
        ax=ax, color='tab:red')
pt.set_plot(ax, xlabel='jitter (s)',
            xticks=[-0.9,0,0.9], xlim=[-0.95,1.2],
            #ylim = [-0.35, 0.95],
            ylabel='corr. coef.')

# %%
fig, ax = pt.figure(figsize=(1.1,0.85))

for k, cellType, color in zip(range(len(TYPES)), TYPES, COLORS):
    pt.annotate(ax, cellType.replace('Basket', 'PV - ').replace('Martinotti', 'SST -'), (k+1, 0), rotation=90, xycoords='data', va='top', ha='center', color=color)
    fit_cond = RESULTS['time_shift']>0
    try:
        RESULTS['tau_%s' % cellType] = [fit_half_width(RESULTS['time_shift'][fit_cond], norm(cc)[fit_cond])[0]\
                     for cc in RESULTS['CC_%s' % cellType]]
        ax.bar([1+k], [1e-3*np.nanmean(RESULTS['tau_%s' % cellType])],
               yerr=[1e-3*np.nanstd(RESULTS['tau_%s' % cellType])], color=color)
    except BaseException as be:
        print(cellType)
        print(be)
        
RESULTS['tau_ACF'] = fit_half_width(RESULTS['time_shift'][fit_cond], norm(RESULTS['ACF'])[fit_cond])[0]
ax.bar([0], [1e-3*RESULTS['tau_ACF']], color='tab:grey')
    
pt.set_plot(ax, ['left'], ylabel=u'\u00bd' + ' width$^{+}$ (s)', yticks=[0,0.2,0.4])

fig.savefig('../figures/detailed_model/natMovie-Widths-Summary.svg')

# %%
import itertools
from scipy import stats
keys = TYPES
for i, j in itertools.product(range(len(keys)), range(len(keys))):
    if i>j:
        try:
            print(keys[i], keys[j], ', p=%.0e' % stats.mannwhitneyu(RESULTS['tau_%s' % keys[i]],
                                                                    RESULTS['tau_%s' % keys[j]]).pvalue)
        except BaseException:
            pass

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
