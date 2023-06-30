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

# %%
import sys, os

import numpy as np

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils import plot_tools as pt

# %% [markdown]
# # Morphology Parameters

# %%
Model = {
    #################################################
    # ---------- MORPHOLOGY PARAMS  --------------- #
    #################################################
    'branch-number':4, #
    'tree-length':400.0, # [um]
    'soma-radius':10.0, # [um]
    'root-diameter':1.0, # [um]
    'diameter-reduction-factor':0.5, # 0.5**(2/3), # Rall branching rule
    'nseg_per_branch': 10,
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 5, # [pS/um2] = [S/m2] # NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 1., # [uF/cm2] NEURON default
    "Ri": 150., # [Ohm*cm]
    "EL": -70, # [mV]
    #################################################
    # ---------- MG-BLOCK PARAMS  ----------------- #
    #################################################
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'qAMPA':1.,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'qNMDAtoAMPAratio': 2.5,
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'dt':0.025,# [ms]
    'seed':1, #
    'interstim':250, # [ms]
}    

def double_exp_normalization(T1, T2):
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

Model['nAMPA'] = double_exp_normalization(Model['tauRiseAMPA'],Model['tauDecayAMPA'])    
Model['nNMDA'] = double_exp_normalization(Model['tauRiseNMDA'],Model['tauDecayNMDA'])

# %%
BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                Nbranch=Model['branch-number'],
                                branch_length=1.0*Model['tree-length']/Model['branch-number'],
                                soma_radius=Model['soma-radius'],
                                root_diameter=Model['root-diameter'],
                                diameter_reduction_factor=Model['diameter-reduction-factor'],
                                Nperbranch=Model['nseg_per_branch'],
                                random_angle=0)

SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

# %%
fig, [ax0, ax] = pt.plt.subplots(1, 2, figsize=(4,1.3))
vis = nrnvyz(SEGMENTS)
vis.plot_segments(ax=ax0, color='tab:blue')
ax.hist(1e6*SEGMENTS['distance_to_soma'], density=True,
        bins=np.linspace(0, Model['tree-length'],
                        Model['nseg_per_branch']*Model['branch-number']+1))
ax.set_xlabel('path dist. to soma ($\mu$m)')
ax.set_ylabel('density')
ax.set_yticks([]);

# %% [markdown]
# # Equations

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
# ## Distribute synapses on a single dendritic branch

# %%
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

fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, BRANCH_LOCS, 2, color='tab:cyan')
ax.set_title('n=%i segments' % len(SEGMENTS['name']), fontsize=6)
BRANCH_LOCS = np.array(BRANCH_LOCS, dtype=int)


# %%
x = np.linspace(SEGMENTS['distance_to_soma'][BRANCH_LOCS].min(),
                SEGMENTS['distance_to_soma'][BRANCH_LOCS].max(), len(BRANCH_LOCS))

def get_distr(x, bias=0):
    distrib = 1-bias*(x-np.min(x))/(np.max(x)-np.min(x))
    return distrib/distrib.sum()


# %%
Nsynapses = 40

LOCS = {}

digitized_dist = np.digitize(SEGMENTS['distance_to_soma'][BRANCH_LOCS],
                             bins=x, right=True)

for case, bias, seed in zip(['uniform', 'proximally-biased'], [0, 1], [11,15]):
    
    np.random.seed(seed)
    proba = get_distr(x, bias=bias)
    LOCS[case] = np.random.choice(np.arange(len(x)), Nsynapses,
                                  p=proba)
    
LOCS['single-syn'] = BRANCH_LOCS # to have them all

fig, AX = pt.plt.subplots(1, 2, figsize=(4,2))
pt.plt.subplots_adjust(wspace=.2, hspace=.6)

for c, ax, bias, case in zip(range(2), AX, [0, 1],
                                   ['uniform', 'proximally-biased']):

    
    vis.plot_segments(ax=ax, color='tab:grey', bar_scale_args=None, diameter_magnification=2.5)
    vis.add_dots(ax, LOCS[case], 4, color='tab:cyan')

    inset = pt.inset(ax, [0.8, 0.7, 0.3, 0.3])
    bins = np.linspace(x.min(), x.max(), 10)
    inset.hist(SEGMENTS['distance_to_soma'][LOCS[case]], 
               bins=bins, color='tab:cyan')
    inset.plot(bins, get_distr(bins, bias=bias)*Nsynapses, color='r', lw=1)
    #pt.set_plot(inset, xticks=[], yticks=[], xlim=[x.min(), x.max()])
    inset.set_xticks([]);inset.set_yticks([0, 10])
    inset.set_xlim([-10e-6, 410e-6])
    inset.set_ylim([-1, 12])
    
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Simulations of Synaptic integration

# %% [markdown]
# ## Equation and Parameters

# %%
# simulation
nrn.defaultclock.dt = 0.1*nrn.ms
# passive
gL = 1.5*nrn.siemens/nrn.meter**2
EL = -70*nrn.mV                
Es = 0*nrn.mV                  
# synaptic
taus = 5.*nrn.ms
w = 0.4*nrn.nS
# equation
eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gs * (Es - v) : amp (point current)
gs : siemens
'''


results = {'Nstim':5, 'Nrepeat':20, 'interstim':200}
results['events'] = 20+np.arange(results['Nstim'])*results['interstim']
results['Nsyns'] = 1+np.arange(results['Nstim'])*2

for case in ['single-syn-uniform', 'uniform', 'biased', 'single-syn-biased']:

    results[case] = {'Vm':[]}
    
    for repeat in range(results['Nrepeat']):
        
        np.random.seed(repeat)

        neuron = nrn.SpatialNeuron(morphology=BRT,
                                   model=eqs,
                                   Cm= 1 * nrn.uF / nrn.cm ** 2,    
                                   Ri= 200 * nrn.ohm * nrn.cm)
        neuron.v = EL

        if 'single-syn' in case:
            spike_times = np.arange(Nsynapses)*results['interstim']
            spike_IDs = np.arange(Nsynapses)
        else:
            spike_IDs, spike_times, synapses = np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)
            for e, ns in zip(results['events'], results['Nsyns']):
                s = np.random.choice(np.arange(Nsynapses), ns, replace=False)
                spike_times = np.concatenate([spike_times,
                    e+np.arange(len(s))*nrn.defaultclock.dt/nrn.ms])
                spike_IDs = np.concatenate([spike_IDs, np.array(s, dtype=int)])

        results[case]['spike_times_%i'%repeat] = spike_times
        results[case]['spike_IDs_%i'%repeat] = spike_IDs

        stimulation = nrn.SpikeGeneratorGroup(len(LOCS['single-syn']),
                                              np.array(spike_IDs, dtype=int),
                                              spike_times*nrn.ms)

        ES = nrn.Synapses(stimulation, neuron,
                           model='''dg/dt = -g/taus : siemens (clock-driven)
                                    gs_post = g : siemens (summed)''',
                           on_pre='g += w',
                           method='exponential_euler')

        for ipre, iseg_post in enumerate(LOCS[case.replace('single-syn-', '')]): # connect spike IDs to a given location
            ES.connect(i=ipre, j=iseg_post)

        # recording and running
        M = nrn.StateMonitor(neuron, ('v'), record=[0])
        nrn.run((200+np.max(spike_times))*nrn.ms)
        results[case]['Vm'].append(np.array(M.v[0]/nrn.mV))
        
        results[case]['t'] = np.array(M.t/nrn.ms)

# np.save('results.npy', results)

pt.plt.plot(np.mean(results['uniform']['Vm'], axis=0))

# %%
LOCS.keys()


# %%
# results = np.load('results.npy', allow_pickle=True).item()

# build linear prediction from single-syn
def build_linear_pred_trace(results):

    # build a dicionary with the individual responses
    
    for case in ['uniform', 'biased']:
    
        results['%s-linear-pred' % case] = {'Vm':[], 't':results[case]['t']}
        linear_pred = []
        results['%s-single-syn-kernel' % case] = []

        for repeat in range(results['Nrepeat']):

            # building single synapse kernel resp to build the linear resp
            results['%s-single-syn-kernel' % case] = {}
            for i, e in zip(results['single-syn-%s'%case]['spike_IDs_%i'%repeat],
                            results['single-syn-%s'%case]['spike_times_%i'%repeat]):

                t_cond = (results['single-syn-%s' % case]['t']>e) &  (results['single-syn-%s' % case]['t']<e+150)
                results['%s-single-syn-kernel' % case][str(i)] = results['single-syn-%s' % case]['Vm'][repeat][t_cond]
                results['%s-single-syn-kernel' % case][str(i)]-= results['%s-single-syn-kernel' % case][str(i)][0]

            # then re-building the patterns
            linear_pred = np.array(0*results[case]['t']-70)
            k=0
            for i, e in zip(results[case]['spike_IDs_%i'%repeat],
                            results[case]['spike_times_%i'%repeat]):

                i0 = np.flatnonzero(results[case]['t']>e)[0] # & (results[case]['t']<(e+160))
                N=len(results['%s-single-syn-kernel' % case][str(i)])
                linear_pred[i0:i0+N] += results['%s-single-syn-kernel' % case][str(i)] 
            results['%s-linear-pred' % case]['Vm'].append(linear_pred)

    return results, linear_pred

results, linear_pred = build_linear_pred_trace(results)
pt.plt.plot(np.mean(results['uniform-linear-pred']['Vm'], axis=0))


# %%
for case in ['uniform', 'biased', 'uniform-linear-pred', 'biased-linear-pred']:

    results[case]['depol'] = []
    results[case]['depol-sd'] = []

    for event in results['events']:

        t_cond = (results[case]['t']>event) & (results[case]['t']<=event+100)

        imax = np.argmax(np.mean(results[case]['Vm'], axis=0)[t_cond])
        results[case]['depol'].append(np.mean(results[case]['Vm'], axis=0)[t_cond][imax]+70)
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
for ax, ax1, case, color in zip(AX, AX1[1:], ['uniform', 'biased'], ['tab:blue', 'tab:green']):
    ax.plot(results[case]['t'], np.mean(results[case]['Vm'],axis=0), color=color, label='real')
    ax.plot(results[case]['t'], np.mean(results[case+'-linear-pred']['Vm'],axis=0), ':', color=color, label='linear')
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
pt.plt.plot(results['single-syn-uniform']['Vm'].flatten()[::10])

# %%
results['single-syn-uniform']

# %%
