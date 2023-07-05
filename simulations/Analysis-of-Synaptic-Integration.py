# ---
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
# # Analysis of Synaptic Integration

# %%
from single_cell_integration import * # code to run the model: (see content below)

import sys
sys.path.append('..')
import plot_tools as pt

# we load the default parameters
Model = load_params('BRT-parameters.json')


# %% [markdown]
# # Distribute synapses with distance-dependent densities

# %%
# distribution as a function of the bias factor !
def get_distr(x, Model, bias=0):
    if bias>=0:
        distrib = 1-bias*x/(Model['tree-length']*1e-6)
    else:
        distrib = -bias*x/(Model['tree-length']*1e-6)
    return distrib/distrib.sum()

def proba_of_branch_locs(BRANCH_LOCS, Model, bias=0):
    return get_distr(SEGMENTS['distance_to_soma'][BRANCH_LOCS], Model, bias=bias)

BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.arange(n*N+1)

# %% [markdown]
# ### Plot

# %%
# select a given dendrite, the longest one !
from nrn.plot import nrnvyz
BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)
vis = nrnvyz(SEGMENTS)

n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.concatenate([np.arange(n+1),
                              1+20*N+np.arange(3*n)])
#BRANCH_LOCS = np.arange(n*N+1)

fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, BRANCH_LOCS, 2, color='tab:cyan')
ax.set_title('n=%i segments' % len(SEGMENTS['name']), fontsize=6)
BRANCH_LOCS = np.array(BRANCH_LOCS, dtype=int)


# %%
Nsynapses = 10

LOCS = {}
BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['branch-number']+1)

for case, bias, seed in zip(['uniform', 'biased'], [0, 1], [6,17]):
    
    np.random.seed(seed)
    LOCS[case] = np.random.choice(BRANCH_LOCS, Nsynapses,
                                  p=proba_of_branch_locs(BRANCH_LOCS, Model, bias=bias))
    
fig, AX = pt.plt.subplots(1, 2, figsize=(4,2))
pt.plt.subplots_adjust(wspace=.2, hspace=.6)
INSETS = []
for c, ax, bias, case in zip(range(2), AX, [0, 1],
                             ['uniform', 'biased']):
    
    vis.plot_segments(ax=ax, color='tab:grey',
                      bar_scale_args=None, diameter_magnification=2.5)
    vis.add_dots(ax, LOCS[case], 5, color='tab:cyan')

    inset = pt.inset(ax, [0.8, 0.7, 0.3, 0.3])
    bins = np.linspace(0, 100e-6, 10)
    inset.hist(SEGMENTS['distance_to_soma'][LOCS[case]], 
               bins=bins, color='tab:cyan')
    inset.plot(bins, get_distr(bins, Model, bias=bias)*Nsynapses, color='r', lw=1)
    pt.set_plot(inset, xticks=[], yticks=[], ylabel='count', xlabel='dist.')
    INSETS.append(inset)
    
pt.set_common_ylims(INSETS) 
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Simulations of Synaptic integration

# %%
# we restart from the default parameters
Model = load_params('BRT-parameters.json')

def PSP(segment_index, results, length=100):
    """ Extract the PSP from the 'single-syn' case simulation """
    t0 = results['start']+segment_index*results['interstim']
    cond = (results['single-syn']['t']>t0) & (results['single-syn']['t']<(t0+length))
    return results['single-syn']['t'][cond]-t0, results['single-syn']['Vm'][cond]-results['single-syn']['Vm'][0]

def run_sim(Model,
            CASES=['bias=0,rNA=0'],
            Nrepeat=5,
            Nsyns=1+np.arange(10)):

    BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['branch-number']+1)
    
    # initialize results with protocol parameters
    results = {'start':50, 'interspike':1.0, 'interstim':100,
               'seed':10, 'repeat':Nrepeat,
               'CASES':CASES,
               'Nsyns':Nsyns}
    
    ###################################
    ####       run simulations     ####
    ###################################
    
    for c, case in enumerate(['single-syn']+CASES):

        results[case] = {'Vm':[]}
        
        np.random.seed(results['seed'])

        if case=='single-syn':

            # 
            results[case]['spike_IDs'] = BRANCH_LOCS
            results[case]['spike_times'] = results['start']+\
                        np.arange(len(BRANCH_LOCS))*results['interstim']

        else:
            # extract and ratio amp-a-to-nmda ratio
            bias = float(case.split(',')[0].replace('bias=', ''))
            Model['qNMDAtoAMPAratio'] = float(case.split('rNA=')[1])
            
            results[case]['spike_times'] = np.empty(0, dtype=int)
            results[case]['spike_IDs'] = np.empty(0, dtype=int)
            
            for repeat in range(results['repeat']):

                for i, Nsyn in enumerate(results['Nsyns']):


                    spike_IDs = np.random.choice(BRANCH_LOCS, Nsyn,
                                                 p=proba_of_branch_locs(BRANCH_LOCS, Model,
                                                                        bias=bias))
                    # spike_IDs = 5+bias*20*np.ones(Nsyn) # REMOVE (just to debug)

                    results[case]['spike_IDs'] = np.concatenate([results[case]['spike_IDs'], spike_IDs])
                    t0 = results['start']+repeat*len(results['Nsyns'])*results['interstim']+i*results['interstim']
                    spike_times = t0+np.arange(Nsyn)*results['interspike']
                    results[case]['spike_times'] = np.concatenate([results[case]['spike_times'],spike_times])

        net, BRT, neuron = initialize(Model, with_network=True)

        stimulation = nrn.SpikeGeneratorGroup(len(BRANCH_LOCS),
                                              np.array(results[case]['spike_IDs'], dtype=int),
                                              results[case]['spike_times']*nrn.ms)
        net.add(stimulation)

        ES = nrn.Synapses(stimulation, neuron,
                           model=EXC_SYNAPSES_EQUATIONS.format(**Model),
                           on_pre=ON_EXC_EVENT.format(**Model),
                           method='exponential_euler')
        for ipre, iseg_post in zip(BRANCH_LOCS, BRANCH_LOCS):
            # synapse ID matches segment ID !!!
            ES.connect(i=ipre, j=iseg_post)
        net.add(ES)

        # recording and running
        M = nrn.StateMonitor(neuron, ('v'), record=[0])
        net.add(M)

        net.run((results['start']+results['interstim']+np.max(results[case]['spike_times']))*nrn.ms)

        results[case]['Vm'] = np.array(M.v[0]/nrn.mV)
        results[case]['t'] = np.array(M.t/nrn.ms)

    ######################################
    ###  build the linear prediction  ####
    ######################################

    for case in CASES:
        # we reconstruct the linear pred from individual PSP
        results[case]['Vm-linear-pred'] = Model['EL']+0*results[case]['Vm']
        for spk_ID, spk_time in zip(results[case]['spike_IDs'], results[case]['spike_times']):
            t_psp, psp = PSP(spk_ID, results)
            i0 = np.flatnonzero(results[case]['t']>=spk_time)[0]
            results[case]['Vm-linear-pred'][i0:i0+len(t_psp)] += psp

    ###############################
    ###  average over repeats  ####
    ###############################

    N_1repeat = int(len(results['Nsyns'])*results['interstim']/Model['dt'])
    for case in CASES:
        for key in ['Vm', 'Vm-linear-pred']:
            Vms = []
            for repeat in range(results['repeat']):
                i0 = repeat*N_1repeat
                Vms.append(results[case][key][i0:i0+N_1repeat])
            results[case][key+'-trial-average'] = np.mean(Vms, axis=0)
        results[case]['t-trial-average'] = results[case]['t'][:N_1repeat]


    ###############################
    ###    response curves     ####
    ###############################

    N_1stim = int(len(results['Nsyns'])*results['interstim']/Model['dt'])
    for case in CASES:
        for key in ['Vm', 'Vm-linear-pred']:
            # init summary resp
            for resp in ['mean', 'max']:
                results[case]['%s-%s-evoked' % (key, resp)] = np.zeros(len(results['Nsyns']))
            # loop over syn act. levels
            for i, Nsyn in enumerate(results['Nsyns']):
                t0 = i*results['interstim']+results['start'] # window starting at stim onset
                cond = (results[case]['t-trial-average']>t0) &\
                    (results[case]['t-trial-average']<(t0+results['interstim']))
                for resp, func in zip(['mean', 'max'], [np.mean, np.max]):
                    results[case]['%s-%s-evoked' % (key, resp)][i] = func(\
                                    results[case][key+'-trial-average'][cond]-results['single-syn']['Vm'][0])
                    
    return results

results = run_sim(Model, CASES=['bias=0,rNA=0', 'bias=1,rNA=0', 'bias=0,rNA=2.5'])

#np.save('results.npy', results)

# %%
def epsilon(results,
            resp='max',
            evaluate_on='full_curve'):
    if evaluate_on=='full_curve':
        num = np.mean(results[case]['Vm-%s-evoked' % resp])
        denom = np.mean(results[case]['Vm-linear-pred-%s-evoked' % resp])
        return num/denom
    elif evaluate_on=='end_point':
        return results[case]['Vm-%s-evoked' % resp][-1]/results[case]['Vm-linear-pred-%s-evoked' % resp][-1]

resp = 'max'
fig, AX = pt.plt.subplots(1, len(results['CASES']),
                          figsize=(1.5*len(results['CASES']), 1))
pt.plt.subplots_adjust(wspace=1.1)
for c, case in enumerate(results['CASES']):
    for key in ['Vm', 'Vm-linear-pred']:
        AX[c].plot(np.concatenate([[0],results['Nsyns']]),
                   np.concatenate([[0],results[case]['%s-%s-evoked' % (key, resp)]]))
    AX[c].set_title('%s\n$\epsilon=$%.1f%%' % (case, 100*epsilon(results,
                                                                 resp=resp,
                                                                 evaluate_on='end_point')))
    pt.set_plot(AX[c], ylabel='%s depol. (mV)' % resp, xlabel='n$_{syn}$')

# %% [markdown]
# ### raw data

# %%
fig, AX = pt.plt.subplots(len(results['CASES']), figsize=(6, 0.6*len(results['CASES'])))
pt.plt.subplots_adjust(hspace=0.1)
for c, case in enumerate(results['CASES']):
    cond = results[case]['t']>0 #1e4#4*len(results['Nsyns'])*results['interstim']
    AX[c].plot(results[case]['t'][cond], results[case]['Vm'][cond])
    AX[c].plot(results[case]['t'][cond], results[case]['Vm-linear-pred'][cond], ':', lw=0.4)
    pt.annotate(AX[c], case, (1.,0))
pt.set_common_ylims(AX)
inset = pt.inset(AX[0], [0.05, 1.2, 0.9, 0.7])
inset.plot(results['single-syn']['t'], results['single-syn']['Vm'])
pt.draw_bar_scales(inset, Xbar=100, Xbar_label='100ms', Ybar=1, Ybar_label='1mV', remove_axis=True)
pt.annotate(inset, 'single-syn', (1.,0))


# %% [markdown]
# ### average over different patterns

# %%
fig, AX = pt.plt.subplots(len(results['CASES']), figsize=(6, 0.6*len(results['CASES'])))
pt.plt.subplots_adjust(hspace=0.1)
COLORS = ['tab:purple', 'tab:brown', 'tab:olive', 'c']
for c, case in enumerate(results['CASES']):
    AX[c].plot(results[case]['t-trial-average'], 
               results[case]['Vm-trial-average'], alpha=.8, color=COLORS[c])
    AX[c].plot(results[case]['t-trial-average'], 
               results[case]['Vm-linear-pred-trial-average'], ':', lw=0.5, color=COLORS[c]) 
    pt.annotate(AX[c], case, (1.,0), color=COLORS[c])


# %% [markdown]
# ### summary fig

# %%
fig, AX = pt.plt.subplots(len(results['CASES']),
                          figsize=(6, 0.6*len(results['CASES'])))
pt.plt.subplots_adjust(hspace=0.1)

COLORS = ['dimgrey', 'tab:brown', 'tab:olive']
INSETS = []
for c, case in enumerate(results['CASES']):
    AX[c].plot(results[results['CASES'][0]]['t-trial-average'],
               results[results['CASES'][0]]['Vm-linear-pred-trial-average'], 
               ':', lw=0.5, color=COLORS[0])
    AX[c].plot(results[case]['t-trial-average'], results[case]['Vm-trial-average'], 
               alpha=.8, color=COLORS[c])
    pt.set_plot(AX[c], [])
    
    inset = pt.inset(AX[c], (1.1, 0.2, 0.1, 0.8))
    inset.plot(np.concatenate([[0],results['Nsyns']]),
               np.concatenate([[0],results[case]['Vm-max-evoked']]),
               alpha=.8, color=COLORS[c])
    inset.plot(np.concatenate([[0],results['Nsyns']]),
               np.concatenate([[0], results[results['CASES'][0]]['Vm-linear-pred-max-evoked']]),
               ':', lw=0.5, color=COLORS[0])
    
    pt.set_plot(inset, xticks=[0, 5, 10], xticks_labels=[], fontsize=7)#, yticks=[0,10,20])
    INSETS.append(inset)

pt.draw_bar_scales(AX[0], Xbar=20, Xbar_label='20ms', Ybar=5, Ybar_label='5mV ')
pt.set_common_ylims(AX)
pt.set_common_ylims(INSETS)
pt.set_plot(INSETS[2], xticks=[0, 5, 10], #yticks=[0,10,20],
            ylabel=30*' '+'max. depol. (mV)', xlabel='n$_{syn}$', fontsize=7)


# %%
#pt.plt.show()

# %%
fig, ax = pt.plt.subplots(1)
resp = np.linspace(90, 30, 20)
ax.bar(np.arange(len(resp)), resp)
pt.set_plot(ax, ylabel='efficacy $\epsilon$ (%)', xlabel='dist. to soma', xticks=[], yticks=[0, 30, 60, 90])
ax2 = ax.twinx()
distrib = np.linspace(1, 0, len(resp))
ax2.plot(np.arange(len(resp)), distrib/distrib.sum(), color='orange')
pt.annotate(ax, r'prox. biased:$\langle \epsilon \rangle$=%.1f%%' % np.mean(resp*distrib/distrib.mean()), (1, 0.3), color='orange')
distrib = np.ones(len(resp))
ax2.plot(np.arange(len(resp)), distrib/distrib.sum(), color='pink')
pt.annotate(ax, r'uniform:       $\langle \epsilon \rangle$=%.1f%%' % np.mean(resp*distrib/distrib.mean()), (1, 0.1), color='pink')
ax2.axis('off');

# %%
