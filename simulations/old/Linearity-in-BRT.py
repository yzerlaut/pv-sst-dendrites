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

import matplotlib.pylab as plt
sys.path.append('../')
import plot_tools as pt

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz

# %% [markdown]
# ## Load morphology

# %%
Model = {
    #################################################
    # ---------- MORPHOLOGY PARAMS  --------------- #
    #################################################
    'branch-number':4, #
    'tree-length':400.0, # [um]
    'soma-radius':10.0, # [um]
    'root-diameter':1.5, # [um]
    'nseg_per_branch': 10,
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 2.5, # [pS/um2] = [S/m2] # NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 1., # [uF/cm2] NEURON default
    "Ri": 150., # [Ohm*cm]
    "EL": -70, # [mV]
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'qAMPA':1.,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDAtoAMPAratio': 0,
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'dt':0.025,# [ms]
    'seed':1, #
    'interstim':250, # [ms]
    'interspike':5, #[ms]
    #################################################
    # ---------- MG-BLOCK PARAMS  ----------------- #
    #################################################
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
}

def double_exp_normalization(T1, T2):
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

Model['nAMPA'] = double_exp_normalization(Model['tauRiseAMPA'],Model['tauDecayAMPA'])    
Model['nNMDA'] = double_exp_normalization(Model['tauRiseNMDA'],Model['tauDecayNMDA'])

# %% [markdown]
# ## Set Equations

# %%
# cable theory:
Equation_String = '''
Im = + ({gL}*siemens/meter**2) * (({EL}*mV) - v) : amp/meter**2
Is = gE * (({Ee}*mV) - v) : amp (point current)
gE : siemens'''

# synaptic dynamics:

# -- excitation (NMDA-dependent)
EXC_SYNAPSES_EQUATIONS = '''dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
                            gAMPA = ({qAMPA}*nS)*{nAMPA}*(gDecayAMPA-gRiseAMPA) : siemens
                            gNMDA = ({qAMPA}*{qNMDAtoAMPAratio}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-v_post/({V0NMDA}*mV))) : siemens
                            gE_post = gAMPA+gNMDA : siemens (summed)'''
ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'

# %% [markdown]
# ## Distributions of distance to soma across synaptic locations

# %%
BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                Nbranch=Model['branch-number'],
                                branch_length=1.0*Model['tree-length']/Model['branch-number'],
                                soma_radius=Model['soma-radius'],
                                root_diameter=Model['root-diameter'],
                                Nperbranch=Model['nseg_per_branch'],
                                random_angle=0)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT, without_axon=True)

# %%
fig, [ax0, ax] = pt.plt.subplots(1, 2, figsize=(4,1.3))
vis = nrnvyz(SEGMENTS)
vis.plot_segments(ax=ax0, color='tab:grey')
ax0.annotate('\n   dendrite', (0,0), xycoords='axes fraction', fontsize=6, color='tab:blue')
vis.plot_segments(ax=ax0, color='tab:blue')
ax.hist(1e6*SEGMENTS['distance_to_soma'], density=True,
        bins=np.linspace(0, Model['tree-length'],
                        Model['nseg_per_branch']*Model['branch-number']+1))
ax.set_xlabel('path dist. to soma ($\mu$m)')
ax.set_ylabel('density')
ax.set_yticks([]);


# %% [markdown]
# ## Distribute synapses on a single dendritic branch

# %%
def find_singleBranch_locs(SEGMENTS, with_fig=False):
    # select a given dendrite, the longest one !
    iEndDendrite = np.argmax(SEGMENTS['distance_to_soma'])
    SETS, i = [SEGMENTS['name'][iEndDendrite]], 0
    while (i<10) and len(SETS[-1].split('.'))>1:
        new_name =  '.'.join(SETS[-1].split('.')[:-1])
        SETS.append(new_name)
        i+=1
    BRANCH_LOCS = []
    for i, name in enumerate(SEGMENTS['name']):
        if name in SETS:
            BRANCH_LOCS.append(i)

    if with_fig:
        fig, ax = pt.plt.subplots(1, figsize=(2,2))
        vis = nrnvyz(SEGMENTS)
        vis.plot_segments(ax=ax, color='tab:grey')
        vis.add_dots(ax, BRANCH_LOCS, 2)
        ax.set_title('n=%i segments' % len(SEGMENTS['name']), fontsize=6)
        
    return np.array(BRANCH_LOCS, dtype=int)

BRANCH_LOCS =find_singleBranch_locs(SEGMENTS)


# %%
x = np.linspace(0, 400, 12)
digitized = np.digitize(1e6*SEGMENTS['distance_to_soma'][1:], bins=x)
area = [np.sum(SEGMENTS['area'][1:][digitized==d]/nrn.um**2) for d in np.unique(digitized)]
plt.plot(x[1:], area)
pt.set_plot(plt.gca(), xlabel='dist. to soma', ylabel='sum area')
#plt.plot(1e6*SEGMENTS['distance_to_soma'][1:], SEGMENTS['area'][1:]/nrn.um**2, 'o')

# %%
#distally_biased = get_biased_distrib(-1)

# %%

# getting the different distributions
x = np.linspace(SEGMENTS['distance_to_soma'][BRANCH_LOCS].min(),
                SEGMENTS['distance_to_soma'][BRANCH_LOCS].max(),
                len(BRANCH_LOCS))

def get_biased_distrib(factor):
    """ bias factor should be between 0 and 1"""
    distrib = 0.5+factor/2.-factor*(x-x.min())/(x.max()-x.min())
    return distrib/np.sum(distrib) #np.trapz(distrib, x=1e6*x)
    
uniform = get_biased_distrib(0)
proximally_biased = get_biased_distrib(1)


Nsynapses, LOCS = 20, {}
np.random.seed(4)

digitized_dist = np.digitize(SEGMENTS['distance_to_soma'][BRANCH_LOCS],
                             bins=x, right=True)

for case, proba in zip(['uniform', 'proximally-biased'],
                       [uniform, proximally_biased]):
    
    LOCS[case] = np.random.choice(BRANCH_LOCS,
                                  Nsynapses, p=proba, replace=False)
    
LOCS['single-syn'] = BRANCH_LOCS # to have them all

fig, AX = pt.plt.subplots(1, 2, figsize=(3,1.5))
pt.plt.subplots_adjust(left=0.1, bottom=0.05)
INSETS = []

for c, y, case in zip(range(2),
                      [uniform, proximally_biased],
                      ['uniform', 'proximally-biased']):
    
    AX[c].axis('off')
    vis.plot_segments(ax=AX[c], color='tab:grey',
                      bar_scale_args={'Ybar':100,'Ybar_label':'100$\\mu$m ','Xbar': 1e-10} if c==1 else\
                                     {'Ybar':1e-10,'Xbar': 1e-10})
    vis.add_dots(AX[c], LOCS[case], 6)
    INSETS.append(pt.inset(AX[c], [0.8, 0.8, 0.3, 0.3]))
    INSETS[-1].hist(1e6*SEGMENTS['distance_to_soma'][LOCS[case]], 
                    alpha=.4, bins=10, color='tab:red')
    AX[c].annotate(case, (0.5, 1.3), xycoords='axes fraction', ha='center', fontsize=8)

INSETS[0].plot([0,400], [2,2], '-', color='tab:red', lw=2)
INSETS[1].plot([0,400], [4,0], '-', color='tab:red', lw=2)

pt.set_common_xlims(INSETS)

for ax, y in zip(INSETS,
                 [uniform, proximally_biased]):
    pt.set_plot(ax, xlabel='dist. ($\mu$m)',
                yticks=[], ylabel='density', xticks=[0, 400], fontsize=6,
                ylim = [-0.2, 6],
                xlim_enhancement=0, ylim_enhancement=0, tck_outward=1, tck_length=2)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'temp.svg'))

# %% [markdown]
# ## Synaptic integration with "uniform" and "biased" distributions

# %% [markdown]
# ## Equation and Parameters

# %%
results = {'Nstim':5, 'Nrepeat':5, 'interstim':200}

results['events'] = 20+np.arange(results['Nstim'])*results['interstim']
results['Nsyns'] = 1+np.arange(results['Nstim'])*2

for case in ['single-syn', 
             'uniform',
             'proximally-biased']:

    results[case] = {'Vm':[]}
    
    for repeat in range(results['Nrepeat'] if (case!='single-syn') else 1):
        
        np.random.seed(repeat)

        neuron = nrn.SpatialNeuron(morphology=BRT,
                                   model=Equation_String.format(**Model),
                                   method='euler',
                                   Cm=Model['cm'] * nrn.uF / nrn.cm ** 2,
                                   Ri=Model['Ri'] * nrn.ohm * nrn.cm)

        neuron.v = Model['EL']*nrn.mV # Vm initialized to E

        if 'single-syn' in case:
            spike_times = np.arange(len(LOCS['single-syn']))*results['interstim']
            spike_IDs = np.arange(len(LOCS['single-syn']))
        else:
            spike_IDs, spike_times, synapses = np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)
            
            for e, ns in zip(results['events'], results['Nsyns']):
                s = np.random.choice(np.arange(len(LOCS[case])),
                                     ns, replace=False) # we pick in LOCS[case] !
                spike_times = np.concatenate([spike_times,
                    e+np.arange(len(s))*nrn.defaultclock.dt/nrn.ms])
                spike_IDs = np.concatenate([spike_IDs, np.array(s, dtype=int)])

        results[case]['spike_times_%i'%repeat] = spike_times
        results[case]['spike_IDs_%i'%repeat] = spike_IDs
        
        stimulation = nrn.SpikeGeneratorGroup(len(LOCS[case]),
                                              np.array(spike_IDs, dtype=int),
                                              spike_times*nrn.ms)

        ES = nrn.Synapses(stimulation, neuron,
                          model=EXC_SYNAPSES_EQUATIONS.format(**Model),
                          on_pre=ON_EXC_EVENT.format(**Model),
                          method='exponential_euler')

        for ipre, iseg_post in enumerate(LOCS[case]): # connect spike IDs to a location given by LOCS[case]
            ES.connect(i=ipre, j=iseg_post)
            
        # recording and running
        M = nrn.StateMonitor(neuron, ('v'), record=[0])
        nrn.run((300+np.max(spike_times))*nrn.ms)
        results[case]['Vm'].append(np.array(M.v[0]/nrn.mV))
        
        results[case]['t'] = np.array(M.t/nrn.ms)

# np.save('results.npy', results)
pt.plt.plot(np.mean(results['uniform']['Vm'], axis=0))

# %%
pt.plt.plot(results['single-syn']['Vm'][0])

# %%
results['uniform']


# %%
# results = np.load('results.npy', allow_pickle=True).item()

# build linear prediction from single-syn
def build_linear_pred_trace(results,
                            kernel_window=200, # ms
                            verbose=False):

    # build the set of single-synaptic responses
    results['single-syn-kernel'] = {}
    for i, e in zip(results['single-syn']['spike_IDs_0'],
                    results['single-syn']['spike_times_0']):

        t_cond = (results['single-syn']['t']>e) &  (results['single-syn']['t']<e+kernel_window)
        results['single-syn-kernel'][str(i)] = results['single-syn']['Vm'][0][t_cond]
        results['single-syn-kernel'][str(i)] -= results['single-syn-kernel'][str(i)][0]
    
    # build a dictionary with the individual linear-pred responses
    for case in ['uniform', 'proximally-biased']:
    
        results['%s-linear-pred' % case] = {'Vm':[], 't':results[case]['t']}
        linear_pred = []

        for repeat in range(results['Nrepeat']):
            # re-building the patterns
            linear_pred = np.array(0*results[case]['t']+Model['EL'])
            k=0
            for i, e in zip(results[case]['spike_IDs_%i'%repeat],
                            results[case]['spike_times_%i'%repeat]):

                i0 = np.flatnonzero(results[case]['t']>e)[0] # & (results[case]['t']<(e+160))
                N=len(results['single-syn-kernel'][str(i)])
                linear_pred[i0:i0+N] += results['single-syn-kernel'][str(i)] 
            results['%s-linear-pred' % case]['Vm'].append(linear_pred)

    return results, linear_pred

results, linear_pred = build_linear_pred_trace(results)
pt.plt.plot(np.mean(results['uniform-linear-pred']['Vm'], axis=0))


# %%
for case in ['uniform', 'proximally-biased', 
             'uniform-linear-pred', 'proximally-biased-linear-pred']:

    results[case]['depol'] = []
    results[case]['depol-sd'] = []

    for event in results['events']:

        t_cond = (results[case]['t']>event) & (results[case]['t']<=event+100)

        imax = np.argmax(np.mean(results[case]['Vm'], axis=0)[t_cond])
        results[case]['depol'].append(np.mean(results[case]['Vm'], axis=0)[t_cond][imax]-Model['EL'])
        results[case]['depol-sd'].append(np.std(results[case]['Vm'], axis=0)[t_cond][imax])

fig2, AX = pt.plt.subplots(2, figsize=(4,2.4))
pt.plt.subplots_adjust(right=.99, hspace=0.5)

fig3, AX1 = pt.plt.subplots(1, 3, figsize=(5.2,1.3))
pt.plt.subplots_adjust(left=0.25, right=.9, wspace=1.)

AX1[0].set_ylabel('EPSP peak (mV)')
AX1[0].set_xlabel(' $N_{synapses}$ ')
for ax in AX1[1:]:
    ax.set_xlabel('expected EPSP (mV)   ')
    ax.set_ylabel('observed EPSP (mV)')
for ax, ax1, case, color in zip(AX, AX1[1:], ['uniform', 'proximally-biased'],
                                ['tab:blue', 'tab:green']):
    ax.plot(results[case]['t'], np.mean(results[case]['Vm'],axis=0), color=color, label='real')
    ax.plot(results[case]['t'], np.mean(results[case+'-linear-pred']['Vm'], axis=0), ':', color=color, label='linear')
    ax.set_ylabel('$V_m$ (mV)')
    ax.set_xlabel('time (ms)')
    ax.set_title(case, color=color)
    pt.draw_bar_scales(ax, Ybar=2, Ybar_label='2mV ', Xbar=50, Xbar_label='50ms', remove_axis='both')
    AX1[0].plot(results['Nsyns'], results[case]['depol'], 'o-', color=color, label=case, lw=1, ms=2)
    ax1.plot([0]+results[case+'-linear-pred']['depol'], [0]+results[case]['depol'],
             'o-', color=color, label=case, lw=1, ms=2)
    ax1.plot(results[case+'-linear-pred']['depol'], results[case+'-linear-pred']['depol'], 'k:', lw=0.5)
    ax.legend(loc=(0.9,0.2), frameon=False, fontsize=7)
AX1[0].set_xticks([1,5,9])

#fig2.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', '2.svg'))
#fig3.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', '3.svg'))

# %%
