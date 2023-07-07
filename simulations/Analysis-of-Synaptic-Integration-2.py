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
# # Analysis of Synaptic Integration

# %%
from single_cell_integration import * # code to run the model: (see content below)

import sys
sys.path.append('..')
import plot_tools as pt

# we load the default parameters
Model = load_params('BRT-parameters.json')


# %% [markdown]
#
# # Distribute synapses with distance-dependent densities

# %%
# distribution as a function of the bias factor !
def get_distr(x, Model, bias=0):
    if bias>0:
        distrib = 2/Model['tree-length']*(1-bias*x/Model['tree-length'])
    else:
        distrib = 1/Model['tree-length']+0*x
    return distrib

def proba_of_branch_locs(BRANCH_LOCS, Model, bias=0):
    return get_distr(SEGMENTS['distance_to_soma'][BRANCH_LOCS], Model, bias=bias)

def find_locations(Nsynapses, Model, bias=0):
    x = np.linspace(0, Model['tree-length'], 10000)
    proba_integ = np.cumsum(get_distr(x, Model, bias=bias)[::-1])/len(x)*Model['tree-length']
    proba_increment = 1./Nsynapses
    locs = [0]
    for i in range(Nsynapses-1):
        iL = np.flatnonzero((proba_integ[locs[-1]:]-proba_integ[locs[-1]])>proba_increment)[0]
        locs.append(iL+locs[-1])
    xlocs = Model['tree-length']-np.array([x[l] for l in locs])[::-1]
    return np.clip(xlocs-Model['tree-length']/Nsynapses/2, 0, np.inf)

def find_synaptic_locations(Nsynapses, Model, BRANCH_LOCS, bias=0):
    
    locs = find_locations(Nsynapses, Model, bias=bias)
    
    BRT, neuron = initialize(Model)
    SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

    SYN_LOCS = []
    for l in locs:
        SYN_LOCS.append(BRANCH_LOCS[np.argmin((l-1e6*SEGMENTS['distance_to_soma'][BRANCH_LOCS])**2)])
        
    return SYN_LOCS
    
Nsynapses = 20
x = np.linspace(0, Model['tree-length'], 1000)

fig, AX = pt.plt.subplots(1, 2, figsize=(3.5,1))
pt.plt.subplots_adjust(wspace=0.8)


for ax, bias, title in zip(AX, [0,1], ['uniform', 'biased']):
    ax.plot(x, np.cumsum(get_distr(x, Model, bias=bias))/len(x)*Model['tree-length'], 'k-')
    #ax.plot(x, get_distr(x, Model, bias=bias)*Model['tree-length']/2, color='k', lw=3, alpha=0.4)
    locs = find_locations(Nsynapses, Model, bias=bias)
    for l in locs:
        ax.plot([l,l], np.arange(2), color='tab:brown', lw=0.2)
    pt.set_plot(ax, xlabel='dist. from soma', ylabel='cum. proba', title=title+10*' ')
    
    inset = pt.inset(ax, [0.8, 1.2, 0.3, 0.25])
    inset.plot(x, get_distr(x, Model, bias=bias)*Model['tree-length']/2, 'k-')
    pt.set_plot(inset, xticks=[], yticks=[], ylabel='proba', xlabel='dist.', ylim=[-0.1,1.1])


# %% [markdown]
# ### Plot

# %%
# select a given dendrite, the longest one !
from nrn.plot import nrnvyz

BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)
vis = nrnvyz(SEGMENTS)

n, N = Model['nseg_per_branch'], Model['branch-number']
#BRANCH_LOCS = np.concatenate([np.arange(n+1),
#                              1+Model['nseg_per_branch']*N+np.arange(3*n)])
BRANCH_LOCS = np.arange(n*N+1)

fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey', bar_scale_args={'Ybar':20, 'Ybar_label':'20um ', 'Xbar':1e-12})
vis.add_dots(ax, BRANCH_LOCS, 2, color='tab:cyan')
#vis.add_dots(ax, BRANCH_LOCS[SYN_LOCS], 4, color='tab:red')
ax.set_title('n=%i segments' % len(SEGMENTS['name']), fontsize=6)
BRANCH_LOCS = np.array(BRANCH_LOCS, dtype=int)


# %%
Nsynapses = 40
Ncluster = 6

def find_cluster_syn(loc, Ncluster, LOCS):
    
    i0 = np.argmin((SEGMENTS['distance_to_soma'][LOCS]*1e6-loc)**2)
    iend = min([len(LOCS), i0+int(Ncluster/2)+1])
    istart = iend-Ncluster
    return LOCS[istart:iend]


n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.arange(n*N+1)

LOCS = {}

for case, bias, seed in zip(['uniform', 'biased'], [0, 1], [6,11]):
    
    LOCS[case] = find_synaptic_locations(Nsynapses, Model, BRANCH_LOCS, bias=bias)
    
fig, AX = pt.plt.subplots(2, 2, figsize=(3,3))
pt.plt.subplots_adjust(wspace=0, hspace=.2)

for Ax, loc in zip(AX, [30, 170]):
    
    for c, ax, bias, case in zip(range(2), Ax, [0, 1],
                                 ['uniform', 'biased']):
        
        cluster = find_cluster_syn(loc, Ncluster, LOCS[case])
        
        ax.set_title('cluster @ %.0f$\mu$m\n%s distrib.' % (loc, case), fontsize=7)
        vis.plot_segments(ax=ax, color='grey',
                          bar_scale_args=None, diameter_magnification=2.5)
        vis.add_dots(ax, LOCS[case], 3, color='tab:cyan')
        vis.add_dots(ax, cluster, 12, color='tab:purple')

fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Simulations of Synaptic integration

# %%
# we restart from the default parameters
Model = load_params('BRT-parameters.json')

def PSP(segment_index, results, length=40):
    """ Extract the PSP from the 'single-syn' case simulation """
    t0 = results['start']+segment_index*results['interstim']
    cond = (results['single-syn']['t']>t0) & (results['single-syn']['t']<(t0+length))
    return results['single-syn']['t'][cond]-t0, results['single-syn']['Vm'][cond]-results['single-syn']['Vm'][0]

def run_sim(Model,
            CASES=['loc=50,bias=0,rNA=0'],
            Nrepeat=1, 
            Ntot_synapses=40,
            Nsyns=1+np.arange(3)*6):

    BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['branch-number']+1)
    
    # initialize results with protocol parameters
    results = {'start':10, 'interspike':0.5, 'interstim':50,
               'seed':10, 'repeat':Nrepeat,
               'CASES':CASES, 'Nsyns':Nsyns}
    
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
            loc = float(case.split('oc=')[1].split(',')[0]) # extract loc=XX,
            bias = float(case.split('ias=')[1].split(',')[0]) # extract bias=XX,
            Model['qNMDAtoAMPAratio'] = float(case.split('rNA=')[1]) # extract rNA=XX
            
            SYN_LOCS = find_synaptic_locations(Ntot_synapses, Model, BRANCH_LOCS, bias=bias)

            results[case]['spike_times'] = np.empty(0, dtype=int)
            results[case]['spike_IDs'] = np.empty(0, dtype=int)
            
            for repeat in range(results['repeat']):

                for i, Nsyn in enumerate(results['Nsyns']):
                    
                    # from proximal toward distal or from distal toward proximal
                    spike_IDs = find_cluster_syn(loc, Nsyn, SYN_LOCS)
                    results[case]['spike_IDs'] = np.concatenate([results[case]['spike_IDs'], spike_IDs])
                    t0 = results['start']+repeat*len(results['Nsyns'])*results['interstim']+\
                                    i*results['interstim']
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
            results[case]['%s-maxs-evoked' % key] = []
            for repeat in range(results['repeat']):
                results[case]['%s-maxs-evoked' % key].append([])
                i0 = repeat*N_1repeat
                Vms.append(results[case][key][i0:i0+N_1repeat])
                for i, Nsyn in enumerate(results['Nsyns']):
                    t0 = repeat*results['interstim']*len(results['Nsyns'])+\
                            i*results['interstim']+results['start'] # window starting at stim onset
                    cond = (results[case]['t']>t0) & (results[case]['t']<(t0+results['interstim']))
                    results[case]['%s-maxs-evoked' % key][-1].append(\
                            np.max(results[case][key][cond]-results['single-syn']['Vm'][0]))
            results[case]['%s-maxs-evoked' % key] = np.array(results[case]['%s-maxs-evoked' % key])
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

results = run_sim(Model, CASES=['loc=30,bias=0,rNA=0','loc=30,bias=1,rNA=0',
                                'loc=170,bias=0,rNA=0','loc=170,bias=1,rNA=0'])

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
    
fig, AX = pt.plt.subplots(1, len(results['CASES']),
                          figsize=(1.5*len(results['CASES']), 1))
pt.plt.subplots_adjust(wspace=1.1)
for c, case in enumerate(results['CASES']):
    AX[c].plot(results['Nsyns'],
               100*results[case]['Vm-%s-evoked' % resp]/results[case]['Vm-linear-pred-%s-evoked' % resp])

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
COLORS = [pt.tab10(i) for i in range(10)]
for c, case in enumerate(results['CASES']):
    AX[c].plot(results[case]['t-trial-average'], 
               results[case]['Vm-trial-average'], alpha=.8, color=COLORS[c])
    AX[c].plot(results[case]['t-trial-average'], 
               results[case]['Vm-linear-pred-trial-average'], ':', lw=0.5, color=COLORS[c]) 
    pt.annotate(AX[c], case, (1.,0), color=COLORS[c])


# %% [markdown]
# ### summary fig

# %%
trial_pick = 0

fig, AX = pt.plt.subplots(len(results['CASES']),
                          figsize=(4, 0.4*(1+len(results['CASES']))))
pt.plt.subplots_adjust(hspace=0.1, right=.8)

COLORS = ['dimgrey', 'tab:brown', 'dimgrey', 'tab:brown',]
INSETS = []

for c, case in enumerate(results['CASES']):
    AX[c].fill_between(results[results['CASES'][0]]['t-trial-average'],
                       0*results[results['CASES'][0]]['t-trial-average']+Model['EL'],
               results[results['CASES'][c]]['Vm-linear-pred-trial-average'], lw=0, alpha=.2, color=COLORS[c])
    AX[c].plot(results[case]['t-trial-average'], results[case]['Vm-trial-average'], 
               alpha=.8, color=COLORS[c])
    pt.set_plot(AX[c], [])
    
    inset = pt.inset(AX[c], (1.1, 0.05, 0.1, 0.85))

    inset.plot(np.concatenate([[0],results['Nsyns']]),
               np.concatenate([[0],results[case]['Vm-max-evoked']]),
               alpha=.8, color=COLORS[c], lw=1.5)
    inset.fill_between(np.concatenate([[0],results['Nsyns']]), 
                       0*np.concatenate([[0],results['Nsyns']]),
                       np.concatenate([[0], results[results['CASES'][c]]['Vm-linear-pred-max-evoked']]),
                       alpha=0.1, color=COLORS[c], lw=0)
    pt.scatter(results['Nsyns'], np.mean(results[case]['Vm-maxs-evoked'], axis=0), 
               sy=np.std(results[case]['Vm-maxs-evoked'], axis=0),
               ax=inset, color=COLORS[c], lw=0.5, ms=1)
    
    pt.set_plot(inset,  xticks_labels=[] if c<3 else None, fontsize=7, num_yticks=2)
    INSETS.append(inset)

    pt.draw_bar_scales(AX[c], Xbar=20 if c==0 else 1e-12,
                       Xbar_label='20ms' if c==0 else '', Ybar=5, Ybar_label='5mV ')
    
#pt.set_common_ylims(AX)
#pt.set_common_ylims(INSETS)
pt.set_plot(INSETS[-1], num_yticks=2,
            ylabel=30*' '+'max. depol. (mV)', xlabel='n$_{syn}$', fontsize=7)

fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))


# %%
COLORS = ['dimgrey', 'tab:brown', 'dimgrey', 'tab:brown',]
fig, ax = pt.plt.subplots(1, 2, figsize=(1.5,1))
pt.plt.subplots_adjust(wspace=1.5)
for k in range(2):
    for c, case in enumerate(results['CASES'][2*k:2*k+2]):
        E = results[case]['Vm-maxs-evoked'][0,-1]/results[case]['Vm-linear-pred-maxs-evoked'][0,-1]
        ax[k].bar([c], [100*E], color=COLORS[c], alpha=.7)
    pt.set_plot(ax[k], xticks=[], yticks=np.arange(4)*(30-10*k), ylabel='efficacy $\epsilon$ (%)')
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %%
