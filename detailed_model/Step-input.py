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
from scipy import stats

import sys
from parallel import Parallel

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt
sys.path.append('../analyz')
from analyz.processing.signanalysis import autocorrel, crosscorrel

# %% [markdown]
# # Test Simulation
# run:
# ```
# python step_stim.py --test -c Martinotti\
#         --with_NMDA\
#         --with_presynaptic_spikes\
#         --interstim 500 --stepWidth 200\
#         --stimFreq 2 --stepAmpFactor 4\
#         --synapse_subsampling 1 --Inh_fraction 0.2\
#         --iBranch 1 --spikeSeed 3
#
# python step_stim.py --test -c Martinotti\
#         --currentDrive 0.08\
#         --with_presynaptic_spikes\
#         --interstim 500 --stepWidth 200\
#         --stimFreq 2 --stepAmpFactor 6\
#         --synapse_subsampling 1 --Inh_fraction 0.2\
#         --iBranch 1 --spikeSeed 3
# ```
#
# then plot with:

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
# # Raw data to illustrate the diverse spiking temporal properties

# %%
def load_sim(RESULTS, cellType,
             rate_smoothing = 3., # ms
             n='1',
             with_example_index=None):

    sim = Parallel(\
            filename='../data/detailed_model/demo-step-%s/StepSim_demo_%s.zip' % (n,cellType))
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
    RESULTS['t_%s' % cellType] = np.arange(len(RESULTS['rate_%s' % cellType]))*dt
    RESULTS['dt'] = dt    
    #print(sim.fetch_quantity_on_grid('currentDrive', return_last=True))
    sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
    sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
    sim.fetch_quantity_on_grid('Stim', return_last=True, dtype=np.ndarray)

    #print(sim.fetch_quantity_on_grid('currentDrive', return_last=True, dtype=list))
    if '%s_example_index' % cellType in RESULTS:
        sim.fetch_quantity_on_grid('Stim', dtype=np.ndarray)
        RESULTS['Input_%s' % cellType] = sim.Stim[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('Vm_soma', dtype=np.ndarray)
        RESULTS['Vm_%s' % cellType] = sim.Vm_soma[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_exc_events', dtype=list)
        RESULTS['pre_exc_%s' % cellType] = sim.presynaptic_exc_events[RESULTS['%s_example_index' % cellType]]
        sim.fetch_quantity_on_grid('presynaptic_inh_events', dtype=list)
        RESULTS['pre_inh_%s' % cellType] = sim.presynaptic_inh_events[RESULTS['%s_example_index' % cellType]]
        
def plot_sim(RESULTS, cellTypes, with_annot=True,
             interstim=50, Tbar=50, view=[-200, 400],
             color='k',
             figsize=(1.2,0.6)):

    fig, AX = pt.figure(axes_extents=[[(1,1)],[(1,1)],[(1,4)],[(1,2)]],
                        figsize=figsize, left=0, bottom=0., hspace=0.)

    t0 = 0
    for c, cellType in enumerate(cellTypes):
        cond = ((RESULTS['t_%s' % cellType]-(RESULTS['t_%s' % cellType].mean())>view[0]) &\
                    ((RESULTS['t_%s' % cellType]-RESULTS['t_%s' % cellType].mean())<view[1]))
        t = RESULTS['t_%s' % cellType][cond]-RESULTS['t_%s' % cellType][cond][0]
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
            subsampling = 5 if 'Basket' in cellType else 1 # for display only
            for i, events in enumerate(RESULTS['pre_exc_%s' % cellType][::subsampling]):
                eCond = ((events-RESULTS['t_%s' % cellType].mean())>view[0]) &\
                                ((events-RESULTS['t_%s' % cellType].mean())<view[1])
                AX[1].plot(t0+events[eCond]-RESULTS['t_%s' % cellType][cond][0],
                           i*np.ones(len(events[eCond])), 'o', fillstyle='full', color='g', ms=.3)
            for i, events in enumerate(RESULTS['pre_inh_%s' % cellType][::subsampling]):
                iCond = ((events-RESULTS['t_%s' % cellType].mean())>view[0]) &\
                                ((events-RESULTS['t_%s' % cellType].mean())<view[1])
                AX[1].plot(t0+events[iCond]-RESULTS['t_%s' % cellType][cond][0],
                           len(RESULTS['pre_exc_%s' % cellType][::subsampling])+i*np.ones(len(events[iCond])),
                           'o', fillstyle='full', color='r', ms=.3)
        t0 += t[-1]-t[0]+interstim

    pt.set_common_xlims(AX)#, lims=zoom)
    
    pt.draw_bar_scales(AX[0], Xbar=Tbar, Xbar_label='%ims'%Tbar if with_annot else '', Ybar=RESULTS['stimFreq_%s' % cellType], ycolor=color,
                       Ybar_label='%.0fHz' % (RESULTS['stimFreq_%s' % cellType]) if with_annot else '')
    #pt.annotate(AX[2], '-60mV ', (zoom[0],-60), xycoords='data', ha='right', va='center')
    pt.draw_bar_scales(AX[2], Xbar=1e-12, Ybar=20,Ybar_label='20mV' if with_annot else '')
    for ax in AX:
        ax.axis('off')
    pt.draw_bar_scales(AX[3], Xbar=1e-12, Ybar=10,Ybar_label='10Hz' if with_annot else '')
    return fig, AX


# %% [markdown]
# ## Short time scale -- no STP

# %%
fig_params = dict(figsize=(1.6,0.3), interstim=50, Tbar=50, view=[-200, 310])

cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), [60, 10, 9]):
    cellTypes.append('BasketnoSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1]) 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:red', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_noSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()
        
X = [137, 198, 142] # same for those two conditions:
cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), X):
    cellTypes.append('MartinottinoSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1]) 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:orange', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_noSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()

cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), X):
    cellTypes.append('MartinottinoSTPnoNMDA%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1]) 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:purple', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_noSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()


# %% [markdown]
# ## Long timescale -- with STP

# %% [markdown]
# ### Test

# %%
i=1
# 1: 11, 33, 0
# 2: 198, 152, 172, 91, 98, 106, 108, 141, 147
# 3: 120, 142, 4, 99, 100, 106, 122, 168

for I in range(0, 100):
    cellTypes, RESULTS = [], {}
    cellTypes.append('MartinottiwiSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = I # change here !
    load_sim(RESULTS, cellTypes[-1], n=2) 
    cellTypes.append('MartinottiwiSTPnoNMDA%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = I # change here !
    load_sim(RESULTS, cellTypes[-1], n=2) 
    fig, _ = plot_sim(RESULTS, cellTypes, color='k', figsize=(1.5,0.3))
    fig.suptitle('%i' % I)
#fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_noSTP_%s.svg' % cellTypes[-1])

# %% [markdown]
# ### Real

# %%
fig_params = dict(figsize=(1.6,0.3), interstim=50, Tbar=100, view=[-400, 600])

cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), [60, 10, 9]):
    cellTypes.append('BasketwiSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1], n='2') 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:red', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_wiSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()
        
X = [11, 198, 142] # same for those two conditions:
cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), X):
    cellTypes.append('MartinottiwiSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1], n='2') 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:orange', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_wiSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()

cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), X):
    cellTypes.append('MartinottiwiSTPnoNMDA%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1], n='2') 
for Annot in [False, True]:
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:purple', with_annot=Annot, **fig_params)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_wiSTP_%s.svg' % cellTypes[-1][:-1])
        plt.close()

# %%
for K in range(100):
    cellTypes, RESULTS = [], {}
    for i, index in zip(np.arange(1,4), [K, K, K]):
        cellTypes.append('BasketwiSTP%i' % i)
        RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
        load_sim(RESULTS, cellTypes[-1], n=2, rate_smoothing=5) 
    fig, _ = plot_sim(RESULTS, cellTypes, color='tab:red', figsize=(1.5,0.3), view=[-350, 400], interstim=50, Tbar=100)

# %%
cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), [2,2,4]):
    cellTypes.append('MartinottiwiSTP%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1], n=2.1) 
fig, _ = plot_sim(RESULTS, cellTypes, color='tab:orange', figsize=(1.6,0.3), view=[-400, 700], interstim=100, Tbar=200)
#fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_wiSTP_%s.svg' % cellTypes[-1])

# %%
cellTypes, RESULTS = [], {}
for i, index in zip(np.arange(1,4), [2,2,4]):
    cellTypes.append('MartinottiwiSTPnoNMDA%i' % i)
    RESULTS['%s_example_index' % cellTypes[-1]] = index # change here !
    load_sim(RESULTS, cellTypes[-1], n=2.5, rate_smoothing=5) 
fig, _ = plot_sim(RESULTS, cellTypes, color='tab:purple', figsize=(1.6,0.3), view=[-400, 700], interstim=100, Tbar=200)
#fig.savefig('../figures/Temp-Properties-Pred/StepSim_example_wiSTP_%s.svg' % cellTypes[-1])

# %% [markdown]
# # Summary Effect

# %%
def load_sim(results, cellType, suffix,
             windows = [[-200,300], [-200,350], [-250,400], [-600,800]],
             rate_smoothing=5):

    rates = []
    results['stepWidth'] = []
    
    for iWidth in range(1, 5):
        for iBranch in range(6):
            
            filename = '../data/detailed_model/vSteps/StepSim_%svSteps%s%i_branch%i.zip' % (cellType, suffix, iWidth, iBranch)
            try:
                sim = Parallel(filename=filename)
                sim.load()
        
                sim.fetch_quantity_on_grid('spikes', dtype=list)
                sim.fetch_quantity_on_grid('Stim', dtype=list)
                seeds = np.unique(sim.spikeSeed)
                sim.fetch_quantity_on_grid('dt')
                sim.fetch_quantity_on_grid('tstop')
                
                for iA, A in enumerate(np.unique(sim.stepAmpFactor)):
                    # compute time-varying RATE !
                    dt, tstop = sim.dt[0][iA], sim.tstop[0][iA]
                    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                    for k, spikes in enumerate(\
                        [np.array(sim.spikes[k][iA]).flatten() for k in range(len(seeds))]):
                        spikes_matrix[k,(spikes/dt).astype('int')] = True
                    
                    rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                  int(rate_smoothing/dt))
                    t = np.arange(len(rate))*dt
                    cond = ((t-t[-1]/2.)>windows[iWidth-1][0]) & ((t-t[-1]/2.)<windows[iWidth-1][1])
            
                    if 'traceRate_Width%i-Amp%i_%s%s' % (iWidth, iA, cellType, suffix) in results:
                        results['traceRate_Width%i-Amp%i_%s%s' % (iWidth, iA, cellType, suffix)].append(rate[cond])
                    else:
                        results['traceRate_Width%i-Amp%i_%s%s' % (iWidth, iA, cellType, suffix)] = [rate[cond]]
                    if not 't_Width%i'%iWidth in results:
                        results['t_Width%i'%iWidth] = t[cond]
                    results['stim_Width%i-Amp%i_%s%s' % (iWidth, iA, cellType, suffix)] = sim.Stim[0][iA][cond[1:]]
                
                results['iBranch'] = range(6)
                results['stepAmpFactor'] = np.unique(sim.stepAmpFactor[0])
                results['stimFreq_%s%s' % (cellType, suffix)] = sim.fetch_quantity_on_grid('stimFreq', return_last=True)
                if iBranch==0:
                    results['stepWidth'].append(sim.fetch_quantity_on_grid('stepWidth', return_last=True))
                    if iWidth==1:
                        print(cellType, suffix, ' --->', results['stimFreq_%s%s' % (cellType, suffix)])
                
            except BaseException as be:
                print(be)
                print(' Pb with "%s" ' % filename)


# %%
results = {}
load_sim(results, 'Martinotti', 'Full')
load_sim(results, 'Martinotti', 'noSTP')
load_sim(results, 'Martinotti', 'noNMDA')
load_sim(results, 'Martinotti', 'noNMDAnoSTP')
load_sim(results, 'Basket', 'Full')
load_sim(results, 'Basket', 'noSTP')


# %%
def make_fig(results, cellTypes, colors,
             alphas=[1,1,1,1,1,1],
             subsamplings=[1,1,1,1,1,1],#10,200,1000],
             Ybar = 10,
             Ybar_inset = 2,
             with_annot=True):
             
    fig, AX = pt.figure(axes=(4,
                              len(results['stepAmpFactor'])),
                        figsize=(.9,0.9), left=0, bottom=0., wspace=0.2, hspace=0.7)
    INSETS = []
    
    for c, cellType, color, alpha, ss in zip(range(len(cellTypes)), cellTypes, colors, alphas, subsamplings):
        for iW, W in enumerate(results['stepWidth']):
            for iA, A in enumerate(results['stepAmpFactor']):
                rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, cellType)])
                pt.plot(results['t_Width%i'%(1+iW)][::ss],
                        np.mean(rate[:,::ss], axis=0), sy = stats.sem(rate, axis=0)[::ss],
                        color=color, ax=AX[iA][iW], alpha=alpha)
                
                if 'Full' in cellType:
                    inset = pt.inset(AX[iA][iW], [0,1, 1, 0.4])
                    inset.axis('off')
                    inset.fill_between(results['t_Width%i'%(1+iW)][::ss],
                                       results['t_Width%i'%(1+iW)][::ss]*0,
                                       results['stim_Width%i-Amp%i_%s' % (iW+1, iA, cellType)][::ss]/results['stimFreq_%s'%cellType],
                                       color='lightgray', lw=0)
                    pt.set_plot(AX[iA][iW], [])
                    pt.set_plot(inset, [])
                    INSETS.append(inset)
                    if (iA==0 and with_annot):
                        pt.annotate(inset, ('%ims' % results['stepWidth'][iW]).replace('000m', ''), (0.5,1), 
                                    color='gray', va='top', ha='center')
                    if (iW==0 and iA==0 and with_annot):
                        pt.annotate(INSETS[0], c*'\n'+'%.1fHz' % results['stimFreq_%s'%cellType],
                                    (0,1), va='top', ha='right', fontsize=6, color=color)
    pt.set_common_ylims(INSETS)
    pt.set_common_ylims(AX)        
    for i, ax in enumerate(pt.flatten(AX)):
        pt.draw_bar_scales(ax, Ybar=Ybar, Ybar_label='%i Hz' % Ybar if (i%4==0 and with_annot) else '',
                           Xbar=100, Xbar_label='', fontsize=7, lw=0.5)
    for ax in INSETS:
        pt.draw_bar_scales(ax, Ybar=1, Xbar=100, Xbar_label='100ms' if (ax==INSETS[0] and with_annot) else '', fontsize=7, lw=0.5)
    return fig


# %%
for Annot in [False, True]:
    fig = make_fig(results,
                   ['MartinottinoNMDAnoSTP', 'MartinottiFull', 'MartinottinoNMDA', 'MartinottinoSTP'],
                   ['tab:cyan', 'tab:orange', 'tab:purple', '#c86464ff'],
                   alphas=[1,1,1,1], Ybar_inset=8, with_annot=Annot)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/Summary2.svg')
        plt.close()

# %%
for Annot in [False, True]:
    fig = make_fig(results,
                   ['MartinottiFull', 'BasketnoSTP', 'BasketFull'],
                   ['tab:orange', 'lightcoral', 'tab:red'],
                   alphas=[.2,1,1], Ybar_inset=8, with_annot=Annot)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/Summary2.svg')
        plt.close()

# %%
from scipy.ndimage import gaussian_filter1d

def make_fig(results, cellTypes, colors,
             iWidths=[0, 0, 1, 2, 3],
             iAmps=[1, 0, 0, 0, 0],
             subsamplings=[1, 1, 1, 1, 1, 1],
             Ybar = 10,
             Ybar_inset = 2,
             with_annot=True):
             
    fig, AX = pt.figure(axes=(len(iWidths), len(cellTypes)),
                        figsize=(.8,0.6), left=0, bottom=0., wspace=0.2, hspace=0)
    INSETS = []
    
    for c, cellType, color in zip(range(len(cellTypes)), cellTypes, colors):
        for i, iW, iA, ss in zip(range(5), iWidths, iAmps, subsamplings):


            rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, cellType)])
            #rate = gaussian_filter1d(rate, sm, axis=1)
            
            t = results['t_Width%i'%(1+iW)]
            stim = results['stim_Width%i-Amp%i_%s' % (iW+1, iA, cellType)]

            mean = np.mean(rate, axis=0)
            sem = stats.sem(rate, axis=0)
            pt.plot(t[::ss], mean[::ss], sy=sem[::ss], color=color, ax=AX[c][i])

            if c==0:
                inset = pt.inset(AX[0][i], [0, 1, 1, 0.4])
                inset.axis('off')
                inset.fill_between(t[::ss], 0*t[::ss],
                                   stim[::ss]/results['stimFreq_%s'%cellType],
                                   color='lightgray', lw=0)
                #pt.set_plot(inset, [], xlim=[views[iW][0],views[iW][1]])
                if with_annot:
                    pt.annotate(inset, ('%ims' % results['stepWidth'][iW]).replace('000m', ''), (0.5,1), 
                                color='gray', ha='center')
                INSETS.append(inset)
            if (i==0 and with_annot):
                pt.annotate(INSETS[0], c*'\n'+'%.1fHz' % results['stimFreq_%s'%cellType],
                            (0,1), va='top', ha='right', fontsize=6, color=color)
    for i in range(5):
        pt.set_common_ylims(AX[i])     
    """
        #pt.set_plot(AX[iA][iW], [], xlim=[views[iW][0],views[iW][1]])
    #pt.set_common_ylims(AX)        
    """
    pt.set_common_ylims(AX)
    pt.set_common_ylims(INSETS)
    for i, ax in enumerate(pt.flatten(AX)):
        pt.draw_bar_scales(ax, Ybar=Ybar, Ybar_label='%i Hz' % Ybar if (ax==AX[-1][0] and with_annot) else '',
                           Xbar=100, Xbar_label='', fontsize=7, lw=0.5)
        ax.axis('off')
    for ax in INSETS:
        pt.draw_bar_scales(ax, Ybar=1, Xbar=100, Xbar_label='100ms' if (ax==INSETS[0] and with_annot) else '', fontsize=7, lw=0.5)
    return fig

for Annot in [False, True]:
    fig = make_fig(results,
                   ['BasketFull', 'BasketnoSTP', 'MartinottinoNMDAnoSTP', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottiFull'],
                   ['tab:red', 'lightcoral', 'tab:cyan', 'tab:purple', '#c86464ff', 'tab:orange'],
                   Ybar_inset=8, with_annot=Annot)
    if not Annot:
        fig.savefig('../figures/Temp-Properties-Pred/Summary3.svg')
        plt.close()


# %% [markdown]
# ## Correlation Coefficient

# %%
def norm(X):
    return (X-X.min())/(X.max()-X.min())

fig, AX = pt.figure(axes=(2,1), figsize=(0.9,1.1), wspace=0.3)
fig2, AX2 = pt.figure(axes=(2,1), figsize=(1,1))
iA=1
Cells = ['BasketFull', 'BasketnoSTP', 'MartinottinoNMDAnoSTP', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottiFull']
Colors = ['tab:red', 'lightcoral', 'tab:cyan', 'tab:purple', '#c86464ff', 'tab:orange']
X = [0, 0.5, 0.75, 1., 1.25, 1.75, 2.5]
W = [0.5, 0.15, 0.15, 0.15, 0.15, 0.5]

for iW, ax, ax2 in zip([0,3], AX, AX2):
    for c in range(6):
        rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])])
        stim = norm(results['stim_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])])
        cond = stim>0
        #coefs = [np.corrcoef(norm(rate[i,:]), norm(stim))[0,1] for i in range(6)] # over branches
        coefs = [np.sqrt(np.mean((norm(rate[i,:])[cond]*norm(stim)[cond])**2)) for i in range(6)] # over branches
        ax.bar([X[c]], [np.mean(coefs)], yerr=[stats.sem(coefs)], color=Colors[c], width=W[c])
        ax2.plot(norm(rate.mean(axis=0)), color=Colors[c])
    ax2.plot(norm(stim), color='grey')
    
pt.set_common_ylims(AX)
for ax, ax2, label in zip(AX, AX2, ['50ms', '1s']):
    pt.set_plot(ax, ['left'], yticks=[0,0.4,0.8],
                yticks_labels=[] if ax==AX[1] else None)

fig.savefig('../figures/Temp-Properties-Pred/Step-Tracking.svg')


# %% [markdown]
# ## Mean Firing

# %%
fig, ax = pt.figure(figsize=(1.5,1.3))

iA=1
Cells = ['BasketFull', 'BasketnoSTP', 'MartinottinoNMDAnoSTP', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottiFull']
Colors = ['tab:red', 'lightcoral', 'tab:cyan', 'tab:purple', '#c86464ff', 'tab:orange']

for c in range(6):
    rate0 = np.array(results['traceRate_Width%i-Amp%i_%s' % (1, iA, Cells[c])])
    stim0 = results['stim_Width%i-Amp%i_%s' % (1, iA, Cells[c])]
    cond0 = stim0>stim0.min()
    resps = []

    for iW in range(4):
        rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])])
        stim = results['stim_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])]
        cond = stim>stim.min()
        resps.append(np.mean(rate[:,cond], axis=1)/np.mean(rate0[:,cond0], axis=1))
        
    pt.scatter(np.arange(4)+0.025*c-0.05, np.mean(resps, axis=1), 
               sy=1.5*stats.sem(resps, axis=1), color=Colors[c], lw=1, ax=ax, ms=3)

pt.set_plot(ax, yscale='log', xticks=range(4), xticks_labels=[], ylim=[0.35,10], yticks_labels=[])

fig.savefig('../figures/Temp-Properties-Pred/Firing-Dependency.svg')

# %% [markdown]
# ## Peak Onset

# %%
fig, ax = pt.figure(figsize=(1.5,1.3))

iA=1
Cells = ['BasketFull', 'BasketnoSTP', 'MartinottinoNMDAnoSTP', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottiFull']
Colors = ['tab:red', 'lightcoral', 'tab:cyan', 'tab:purple', '#c86464ff', 'tab:orange']

for c in range(6):
    resps = []
    for iW in range(4):
        resps.append([])
        rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])])
        stim = results['stim_Width%i-Amp%i_%s' % (iW+1, iA, Cells[c])]
        i0 = np.flatnonzero(stim>stim.min())[0]
        t = results['t_Width%i'%(1+iW)]
        for b in range(6):
            cond = rate[b,:]>.9*np.max(rate[b,:])
            resps[-1].append(np.min(t[np.flatnonzero(cond)])-t[i0])

    pt.scatter(np.arange(4)+0.025*c-0.05, np.mean(resps, axis=1), 
               sy=1.5*stats.sem(resps, axis=1), color=Colors[c], lw=1, ax=ax, ms=3)

ax.plot(np.arange(4), [50, 100, 200, 1e3], '--', color='gray', lw=1)
pt.set_plot(ax, xticks=range(4), xticks_labels=[], yscale='log', yticks_labels=[])

fig.savefig('../figures/Temp-Properties-Pred/Delays.svg')

# %% [markdown]
# ## Cross-Correl

# %%
from analyz.processing.signanalysis import autocorrel, crosscorrel

fig, AX = pt.figure(axes=(2,1), figsize=(0.9,1.1), wspace=0.3)

iA=1

Cells = ['BasketFull', 'BasketnoSTP', 'MartinottinoNMDAnoSTP', 'MartinottinoNMDA', 'MartinottinoSTP', 'MartinottiFull']
Colors = ['tab:red', 'lightcoral', 'tab:cyan', 'tab:purple', '#c86464ff', 'tab:orange']

for iW, ax, window in zip([0,3], AX, [0.5*1e3, 1e3]):
    for cellType, color in zip(['BasketFull', 'MartinottiFull'], ['tab:red', 'tab:orange']):
        CCFs = []
        rate = np.array(results['traceRate_Width%i-Amp%i_%s' % (iW+1, iA, cellType)])
        stim = norm(results['stim_Width%i-Amp%i_%s' % (iW+1, iA, cellType)])
        t = results['t_Width%i'%(1+iW)]
        for b in range(6):
            CC, TS = crosscorrel(stim[::10], rate[b,:][::10], window, 10*(t[1]-t[0]))
            CCFs.append(CC)          
        pt.plot(TS, np.mean(CCFs, axis=0), sy=stats.sem(CCFs, axis=0), color=color, ax=ax, no_set=True)
"""    
pt.set_common_ylims(AX)
for ax, ax2, label in zip(AX, AX2, ['50ms', '1s']):
    pt.set_plot(ax, ['left'], yticks=[0,0.4,0.8],
                yticks_labels=[] if ax==AX[1] else None)
"""

#fig.savefig('../figures/Temp-Properties-Pred/Step-Tracking.svg')

# %% [markdown]
# ## Calibration: Depolarizing Current for no-NMDA condition 

# %%
rate_smoothing = 5. # ms

for cellType, suffix, label, color in zip(['Martinotti', 'Martinotti'],
                                          ['wiSTP', 'noSTP'],
                                          ['SST - wiSTP', 'SST - no NMDA (AMPA+)'],
                                          ['tab:purple', 'tab:purple']):
    results = {}

    for iBranch in range(6):
        sim = Parallel(\
                filename='../data/detailed_model/current-calib3/StepSim_%scurrentCalib%s_branch%i.zip' % (cellType, suffix, iBranch))
        sim.load()

        sim.fetch_quantity_on_grid('spikes', dtype=list)
        seeds = np.unique(sim.spikeSeed)
        dt = sim.fetch_quantity_on_grid('dt', return_last=True)
        tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
        sim.fetch_quantity_on_grid('Stim', dtype=list)
        if 'traceRate' not in results:
            results['traceRate'] = np.zeros((len(np.unique(sim.currentDrive)), 6, int(tstop/dt)+1)) # create the array
            results['t'] = np.arange(int(tstop/dt)+1)*dt
            results['currentDrive'] = np.unique(sim.currentDrive)
            
        for iSF, SF in enumerate(np.unique(sim.currentDrive)):
        
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iSF]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iSF,iBranch,:] = rate
            results['Stim%i'%iSF] = sim.Stim[0][iSF]
    
    fig, AX = pt.figure(axes=(len(results['currentDrive']), 1), right=4.,
                        figsize=(.8,1), wspace=0., hspace=0., left=0.5, bottom=0.)
    INSETS = []
    for iSF, SF in enumerate(results['currentDrive']):
        pt.plot(results['t'], results['traceRate'][iSF,:,:].mean(axis=0), 
                sy=np.std(results['traceRate'][iSF,:,:], axis=0),
                ax=AX[iSF], color=pt.viridis(iSF/(len(results['currentDrive'])-1)))
        INSETS.append(pt.inset(AX[iSF], [0,-0.4,1,0.38]))
        INSETS[-1].fill_between(results['t'][1:], 0*results['t'][1:], results['Stim%i'%iSF], color='lightgray')
        INSETS[-1].axis('off')             
        cond = results['t']>500
        pt.annotate(AX[iSF], '%.1fHz' % np.max(results['traceRate'][iSF,:,:].mean(axis=0)[cond]),
                        (0.5, 1), ha='center', color=pt.viridis(iSF/(len(results['currentDrive'])-1)), fontsize=7)
    pt.set_common_ylims(AX); pt.set_common_ylims(INSETS)
    for ax in pt.flatten(AX):
        pt.set_plot(ax, ['left'] if ax==AX[0] else [], ylabel='firing (Hz)' if ax==AX[0] else '')
    pt.draw_bar_scales(AX[-1], loc='bottom-right', Xbar=200, Xbar_label='200ms', Ybar=1e-12)
        
    pt.bar_legend(AX[-1], X=range(len(results['currentDrive'])),
                  ticks_labels=['%.2f' % f for f in results['currentDrive']],
                  colorbar_inset={'rect': [1.2, -0.3, 0.15, 1.6], 'facecolor': None},
                  label='currentDrive (nA)',
                  colormap=pt.viridis)
    fig.suptitle(label, color=color)

#func('Martinotti', 'Full', 'tab:orange')

# %%

rate_smoothing = 10. # ms

def func(cellType='Martinotti', suffix='Full', color='tab:orange'):
    results = {}
    
    for iBranch in np.arange(6):
        
        sim = Parallel(\
                filename='../data/detailed_model/StepStim_%ssRange%s_branch%i.zip' % (\
                        cellType, suffix, iBranch))
        sim.load()
        
        sim.fetch_quantity_on_grid('spikes', dtype=list)
        seeds = np.unique(sim.spikeSeed)
        dt = sim.fetch_quantity_on_grid('dt', return_last=True)
        tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

        if iBranch==0:
            results['traceRate'] = np.zeros((6, len(np.unique(sim.stimFreq)), int(tstop/dt)+1)) # create the array
            results['t'] = np.arange(int(tstop/dt)+1)*dt
            results['stimFreq'] = np.unique(sim.stimFreq)
            sim.fetch_quantity_on_grid('Stim', dtype=list)
            
        for iSF, SF in enumerate(np.unique(sim.stimFreq)):
        
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iSF]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iBranch, iSF,:] = rate
            if iBranch==0:
                results['Stim%i'%iSF] = sim.Stim[0][iSF]

    fig, AX = pt.figure(axes=(len(results['stimFreq']), 1), right=4.,
                        figsize=(.8,1), wspace=0., hspace=0., left=0.5, bottom=0.)
    INSETS = []
    for iSF, SF in enumerate(results['stimFreq']):
        pt.plot(results['t'], results['traceRate'][:,iSF,:].mean(axis=0),
                sy=stats.sem(results['traceRate'][:,iSF,:], axis=0), ax=AX[iSF], 
                          color=pt.viridis(iSF/(len(results['stimFreq'])-1)))
        INSETS.append(pt.inset(AX[iSF], [0,-0.4,1,0.38]))
        INSETS[-1].fill_between(results['t'][1:], 0*results['t'][1:], results['Stim%i'%iSF], color='lightgray')
        INSETS[-1].axis('off')             
        
        pt.annotate(AX[iSF], '%.1fHz' % np.max(results['traceRate'][:,iSF,1:].mean(axis=0)),
                        (0.5, 1), ha='center', color=pt.viridis(iSF/(len(results['stimFreq'])-1)), fontsize=7)
    pt.set_common_ylims(AX); pt.set_common_ylims(INSETS)
    for ax in pt.flatten(AX):
        pt.set_plot(ax, ['left'] if ax==AX[0] else [])
    pt.draw_bar_scales(AX[-1], loc='bottom-right', Xbar=200, Xbar_label='200ms', Ybar=1e-12)
        
    pt.bar_legend(AX[-1], X=results['stimFreq'],
                  ticks_labels=['%.1f' % f for f in results['stimFreq']],
                  colorbar_inset={'rect': [1.2, -0.3, 0.15, 1.6], 'facecolor': None},
                  label='stimFreq (Hz)',
                  colormap=pt.viridis)
    fig.suptitle('%s - %s' % (cellType, suffix), color=color)

#func('Martinotti', 'Full', 'tab:orange')


# %%
func('Martinotti', 'Full', 'tab:orange')
func('Martinotti', 'noNMDA', 'tab:purple')

# %%
rate_smoothing = 20. # ms

results = {}
def load_sim(cellType, suffix):

    sim = Parallel(\
            filename='../data/detailed_model/StepSim_demo_%siRange%s.zip' % (cellType, suffix))

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


# %%
results = load_sim('Martinotti', '')

fig, AX = pt.figure(axes=(len(results['Inh_fraction']), len(results['synapse_subsampling'])),
                    figsize=(1,1), right=4., left=0.5, bottom=0., hspace=0., reshape_axes=False)
for ax in pt.flatten(AX):
    ax.axis('off')

for iIF, IF in enumerate(np.unique(results['Inh_fraction'])):
    for iSS, SS in enumerate(np.unique(results['synapse_subsampling'])):
        for iSA, SA in enumerate(np.unique(results['stepAmpFactor'])):
            for iSF, SF in enumerate(np.unique(results['stimFreq'])):

                rate = results['traceRate'][iSS,iIF,iSF,iSA,:]
                AX[iSS][iIF].plot(results['t'], rate, color='tab:orange')
                
        pt.draw_bar_scales(AX[iSS][iIF], Ybar=5, Ybar_label='5Hz ', Xbar=1e-12)
        if iIF==0:
            pt.annotate(AX[iSS][0], 'S.S.=%i' % results['synapse_subsampling'][iSS], (-0.3, 0.5), ha='center', rotation=90)
        if iSS==0:
            pt.annotate(AX[0][iIF], 'I.F.=%.2f' % results['Inh_fraction'][iIF], (0.5, 1.1), ha='center')

# %%
rate_smoothing = 20. # ms

results = {}
def load_sim(cellType, suffix):

    sim = Parallel(\
            filename='../data/detailed_model/StepStim_demo_%sRange%s.zip' % (cellType, suffix))

    sim.load()

    nSS = len(np.unique(sim.synapse_subsampling))
    nIF = len(np.unique(sim.Inh_fraction))
    nIF = len(np.unique(sim.Inh_fraction))
    
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    seeds = np.unique(sim.spikeSeed)

    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

    results['traceRate'] = np.zeros((nSS, nIF, int(tstop/dt)+1))

    for iSS, SS in enumerate(np.unique(sim.synapse_subsampling)):
        for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
                    
                    # compute time-varying RATE !
                    spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                    for k, spikes in enumerate(\
                        [np.array(sim.spikes[k][iSS][iIF]).flatten() for k in range(len(seeds))]):
                        spikes_matrix[k,(spikes/dt).astype('int')] = True
                    rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                                  int(rate_smoothing/dt))
                    results['traceRate'][iSS,iIF,:] = rate
                
    results['t'] = np.arange(len(rate))*dt
    results['Inh_fraction'] = np.unique(sim.Inh_fraction[0])
    results['synapse_subsampling'] = np.unique(sim.synapse_subsampling[0])
    return results
    
    
results = load_sim('Basket', 'NoSTP')

fig, AX = pt.figure(axes=(len(results['Inh_fraction']), len(results['synapse_subsampling'])),
                    figsize=(1,1), right=4., left=0.5, bottom=0., hspace=0., reshape_axes=False)
for ax in pt.flatten(AX):
    ax.axis('off')

for iIF, IF in enumerate(np.unique(results['Inh_fraction'])):
    for iSS, SS in enumerate(np.unique(results['synapse_subsampling'])):

        rate = results['traceRate'][iSS,iIF,:]
        AX[iSS][iIF].fill_between(results['t'], 0*results['t'], rate, color='tab:red')
                
        pt.draw_bar_scales(AX[iSS][iIF], Ybar=5, Ybar_label='5Hz ', Xbar=1e-12)
        if iIF==0:
            pt.annotate(AX[iSS][0], 'S.S.=%i' % results['synapse_subsampling'][iSS], (-0.3, 0.5), ha='center', rotation=90)
        if iSS==0:
            pt.annotate(AX[0][iIF], 'I.F.=%.2f' % results['Inh_fraction'][iIF], (0.5, 1.1), ha='center')

# %%
rate_smoothing = 20. # ms

results = {}
sim = Parallel(\
        filename='../data/detailed_model/StepStim_demo_BasketInputRangeNoSTP.zip')
sim.load()
color = 'tab:red'
nSF = len(np.unique(sim.stimFreq))
nIF = len(np.unique(sim.Inh_fraction))
nSA = len(np.unique(sim.stepAmpFactor))

sim.fetch_quantity_on_grid('spikes', dtype=list)
seeds = np.unique(sim.spikeSeed)

dt = sim.fetch_quantity_on_grid('dt', return_last=True)
tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

results['traceRate'] = np.zeros((nSF, nIF, nSA, int(tstop/dt)+1))

for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
        for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
        
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iIF][iSF][iSA]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iSF,iIF,iSA,:] = rate
            
results['t'] = np.arange(len(rate))*dt

fig, AX = pt.figure(axes=(nIF, nSF), right=4.,
                    figsize=(1,.8), wspace=0., hspace=0., left=0.5, bottom=0.)

for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
        for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
            rate = results['traceRate'][iSF,iIF,iSA,:]
            AX[iSF][iIF].plot(results['t'], rate, color=pt.copper_r(iSA/1.5))
            if iIF==0:
                pt.annotate(AX[iSF][0], 'f=%.1fHz' % SF, (-0.5, 0.5), va='center', rotation=90)
            if iSF==(nSF-1):
                pt.annotate(AX[iSF][iIF], 'IF=%.2f' % IF, (0.5, -0.1), ha='center',va='top')
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    pt.draw_bar_scales(ax, Xbar=100, Xbar_label='100ms' if ax==AX[0][-1] else '', Ybar=1e-12)
    pt.draw_bar_scales(ax, loc='bottom-left', Xbar=1e-5, Ybar=20, Ybar_label='20Hz ' if ax==AX[0][0] else '')
    
pt.bar_legend(ax, X=range(2),
              ticks_labels=['%i' % f for f in np.unique(sim.stepAmpFactor)],
              colorbar_inset={'rect': [1.2, 0.1, 0.07, 2], 'facecolor': None},
              label='step factor',
              colormap=pt.mpl.colors.ListedColormap([pt.copper_r(x) for x in [0,0.6]]))

# %%
iBranch = 0

rate_smoothing = 20. # ms

results = {}
sim = Parallel(\
        filename='../data/detailed_model/StepStim_sim_iBranch%i_Basket_InputRangeNoSTP.zip' % iBranch)
sim.load()

color = 'tab:red'
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

for iSS, SS in enumerate(np.unique(sim.synapse_subsampling)):

    fig, AX = pt.figure(axes=(nIF, nSF), right=4.,
                        figsize=(1,.6), wspace=0., hspace=0., left=0.5, bottom=0.)
    fig.suptitle('Syn. Subspl. %i' % SS)
    for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
        for iSF, SF in enumerate(np.unique(sim.stimFreq)):
            for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
                rate = results['traceRate'][iSS,iIF,iSF,iSA,:]
                AX[iSF][iIF].plot(results['t'], rate, color=pt.copper_r(iSA/1.5))
                if iIF==0:
                    pt.annotate(AX[iSF][0], 'f=%.1fHz' % SF, (-0.4, 0.5), va='center', ha='right')
                if iSF==(nSF-1):
                    pt.annotate(AX[iSF][iIF], 'IF=%.2f' % IF, (0.5, -0.1), ha='center',va='top')
    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
        pt.draw_bar_scales(ax, Xbar=100, Xbar_label='100ms' if ax==AX[0][0] else '', Ybar=1e-12)
        pt.draw_bar_scales(ax, loc='bottom-left', Xbar=1e-5, Ybar=20, Ybar_label='20Hz ' if ax==AX[0][0] else '')
        
    pt.bar_legend(ax, X=range(2),
                  ticks_labels=['%i' % f for f in np.unique(sim.stepAmpFactor)],
                  colorbar_inset={'rect': [1.2, 0.1, 0.07, 2], 'facecolor': None},
                  label='step factor',
                  colormap=pt.mpl.colors.ListedColormap([pt.copper_r(x) for x in [0,0.6]]))

# %%

rate_smoothing = 10. # ms

for iBranch in range(6):
    
    results = {}
    sim = Parallel(\
            filename='../data/detailed_model/StepStim_sim_iBranch%i_Martinotti_InputRangeNoSTP.zip' % iBranch)
    sim.load()
    
    color = 'tab:orange'
    nIF = len(np.unique(sim.Inh_fraction))
    nSF = len(np.unique(sim.stimFreq))
    nSA = len(np.unique(sim.stepAmpFactor))
    
    sim.fetch_quantity_on_grid('spikes', dtype=list)
    seeds = np.unique(sim.spikeSeed)
    
    dt = sim.fetch_quantity_on_grid('dt', return_last=True)
    tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)
    
    results['traceRate'] = np.zeros((nIF, nSF, nSA, int(tstop/dt)+1))
    
    for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
        for iSF, SF in enumerate(np.unique(sim.stimFreq)):
            for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
            
                # compute time-varying RATE !
                spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
                for k, spikes in enumerate(\
                    [np.array(sim.spikes[k][iIF][iSF][iSA]).flatten() for k in range(len(seeds))]):
                    spikes_matrix[k,(spikes/dt).astype('int')] = True
                rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                              int(rate_smoothing/dt))
                results['traceRate'][iIF,iSF,iSA,:] = rate

    print('Branch %i, rate = %.1f Hz' % (iBranch+1, np.mean(results['traceRate'])))
    results['t'] = np.arange(len(rate))*dt
    
    fig, AX = pt.figure(axes=(nIF, nSF), right=4.,
                        figsize=(1,.6), wspace=0., hspace=0., left=0.5, bottom=0.)
    for iIF, IF in enumerate(np.unique(sim.Inh_fraction)):
        for iSF, SF in enumerate(np.unique(sim.stimFreq)):
            for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
                rate = results['traceRate'][iIF,iSF,iSA,:]
                AX[iSF][iIF].plot(results['t'], rate, color=pt.copper_r(iSA/1.5))
                if iIF==0:
                    pt.annotate(AX[iSF][0], 'f=%.1fHz' % SF, (-0.4, 0.5), va='center', ha='right')
                if iSF==(nSF-1):
                    pt.annotate(AX[iSF][iIF], 'IF=%.2f' % IF, (0.5, -0.1), ha='center',va='top')
    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
        pt.draw_bar_scales(ax, Xbar=100, Xbar_label='100ms' if ax==AX[0][0] else '', Ybar=1e-12)
        pt.draw_bar_scales(ax, loc='bottom-left', Xbar=1e-5, Ybar=20, Ybar_label='20Hz ' if ax==AX[0][0] else '')
        
    pt.bar_legend(ax, X=range(2),
                  ticks_labels=['%i' % f for f in np.unique(sim.stepAmpFactor)],
                  colorbar_inset={'rect': [1.2, 0.1, 0.07, 2], 'facecolor': None},
                  label='step factor',
                  colormap=pt.mpl.colors.ListedColormap([pt.copper_r(x) for x in [0,0.6]]))

# %%
sim = Parallel(\
            filename='../data/detailed_model/StepStim_sim_Martinotti_InputRangeNoSTP.zip')
sim.load()
rate_smoothing = 10
color = 'tab:orange'
nSF = len(np.unique(sim.stimFreq))
nSA = len(np.unique(sim.stepAmpFactor))
nB = len(np.unique(sim.iBranch))

sim.fetch_quantity_on_grid('spikes', dtype=list)
seeds = np.unique(sim.spikeSeed)

dt = sim.fetch_quantity_on_grid('dt', return_last=True)
tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

results = {'traceRate': np.zeros((nSF, nSA, nB, int(tstop/dt)+1))}

for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
        for iB, B in enumerate(np.unique(sim.iBranch)):
            
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iB][iSF][iSA]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iSF,iSA,iB,:] = rate
results['t'] = np.arange(len(rate))*dt

fig, AX = pt.figure(axes=(nB, nSF), right=4.,
                    figsize=(1,.6), wspace=0., hspace=0., left=0.5, bottom=0.)
for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
        for iB, B in enumerate(np.unique(sim.iBranch)):
            rate = results['traceRate'][iSF,iSA,iB,:]
            AX[iSF][iB].plot(results['t'], rate, color=pt.copper_r(iSA/1.5))
            if iB==0:
                pt.annotate(AX[iSF][0], 'f=%.1f' % SF, (-0.4, 0.5), va='center', ha='right')
            if iB==(nSF-1):
                pt.annotate(AX[iSF][iB], 'Branch #' % (B+1), (0.5, -0.1), ha='center',va='top')
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    pt.draw_bar_scales(ax, Xbar=100, Xbar_label='100ms' if ax==AX[0][0] else '', Ybar=1e-12)
    pt.draw_bar_scales(ax, loc='bottom-left', Xbar=1e-5, Ybar=20, Ybar_label='20Hz ' if ax==AX[0][0] else '')
    
pt.bar_legend(ax, X=range(2),
              ticks_labels=['%i' % f for f in np.unique(sim.stepAmpFactor)],
              colorbar_inset={'rect': [1.2, 0.1, 0.07, 2], 'facecolor': None},
              label='step factor',
              colormap=pt.mpl.colors.ListedColormap([pt.copper_r(x) for x in [0,0.6]]))

# %%
sim = Parallel(\
            filename='../data/detailed_model/StepStim_sim_Basket_InputRangeNoSTP.zip')
sim.load()
rate_smoothing = 10
nSF = len(np.unique(sim.stimFreq))
nSA = len(np.unique(sim.stepAmpFactor))
nB = len(np.unique(sim.iBranch))

sim.fetch_quantity_on_grid('spikes', dtype=list)
seeds = np.unique(sim.spikeSeed)

dt = sim.fetch_quantity_on_grid('dt', return_last=True)
tstop = sim.fetch_quantity_on_grid('tstop', return_last=True)

results = {'traceRate': np.zeros((nSF, nSA, nB, int(tstop/dt)+1))}

for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
        for iB, B in enumerate(np.unique(sim.iBranch)):
            
            # compute time-varying RATE !
            spikes_matrix= np.zeros((len(seeds), int(tstop/dt)+1))
            for k, spikes in enumerate(\
                [np.array(sim.spikes[k][iB][iSF][iSA]).flatten() for k in range(len(seeds))]):
                spikes_matrix[k,(spikes/dt).astype('int')] = True
            rate = 1e3*gaussian_filter1d(np.mean(spikes_matrix, axis=0)/dt,
                                          int(rate_smoothing/dt))
            results['traceRate'][iSF,iSA,iB,:] = rate
results['t'] = np.arange(len(rate))*dt

fig, AX = pt.figure(axes=(nB, nSF), right=4.,
                    figsize=(1,.6), wspace=0., hspace=0., left=0.5, bottom=0.)
for iSF, SF in enumerate(np.unique(sim.stimFreq)):
    for iSA, SA in enumerate(np.unique(sim.stepAmpFactor)):
        for iB, B in enumerate(np.unique(sim.iBranch)):
            rate = results['traceRate'][iSF,iSA,iB,:]
            AX[iSF][iB].plot(results['t'], rate, color=pt.copper_r(iSA/1.5))
            if iB==0:
                pt.annotate(AX[iSF][0], 'f=%.1f' % SF, (-0.4, 0.5), va='center', ha='right')
            if iB==(nSF-1):
                pt.annotate(AX[iSF][iB], 'Branch #' % (B+1), (0.5, -0.1), ha='center',va='top')
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    pt.draw_bar_scales(ax, Xbar=100, Xbar_label='100ms' if ax==AX[0][0] else '', Ybar=1e-12)
    pt.draw_bar_scales(ax, loc='bottom-left', Xbar=1e-5, Ybar=20, Ybar_label='20Hz ' if ax==AX[0][0] else '')
    
pt.bar_legend(ax, X=range(2),
              ticks_labels=['%i' % f for f in np.unique(sim.stepAmpFactor)],
              colorbar_inset={'rect': [1.2, 0.1, 0.07, 2], 'facecolor': None},
              label='step factor',
              colormap=pt.mpl.colors.ListedColormap([pt.copper_r(x) for x in [0,0.6]]))

# %%
dt

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
fig, ax = pt.figure()
pt.bar_legend(ax, X=range(2),
              ticks_labels=['%.1f' % f for f in results['stepAmpFactor']],
              colorbar_inset={'rect': [1.2, 0.1, 0.07, 0.9], 'facecolor': None},
              label='step factor',
              colormap=pt.mpl.colors.ListedColormap(['r', 'g']))

# %%
