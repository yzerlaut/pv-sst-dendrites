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
# # Input Impedance Properties of the Model
#     > Characterizing the parameter dependency of the input impedance spatial profile 
#

# %%
from single_cell_integration import initialize, load_params # code to run the model: (see content below)
import sys
sys.path.append('../')
from neural_network_dynamics import nrn, utils
import plot_tools as pt
import matplotlib.pylab as plt
import numpy as np

# %%
import pandas as pd
data = pd.read_csv('../data/SSTDendrites.csv', sep=';')
expected, real= [], []
iKey = 2
while iKey<len(data.keys()):
    array = np.array(data[data.keys()[iKey]], dtype=float)
    if len(array[np.isfinite(array)])>0:
        expected.append(np.array(data['Expected '][np.isfinite(array)], dtype=float))
        real.append(array[np.isfinite(array)])
    iKey += 2

# %%
fig, ax = pt.figure(figsize=(1.2,1.8), right=5)

means = np.array([np.mean(e) for e in expected])

for e, r, m in zip(expected, real, means):
    ax.plot(e, r, '-', lw=0.5, ms=0.5,color=pt.copper(1-(m-means.min())/(means.max()-means.min())))

pt.set_plot(ax, xlabel='expected EPSP (mV)', ylabel='recorded EPSP (mV)   ',
            xticks=3*np.arange(5), yticks=3*np.arange(5))
pt.bar_legend(ax, colormap=pt.copper_r, orientation='horizontal',
              colorbar_inset={'rect': [0.1, 1.2, 0.8, 0.08], 'facecolor': None})
pt.save(fig)

# %% [markdown]
# ### Locations where to simulate/record along the dendritic tree

# %%
Model = load_params('BRT-parameters.json')
BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

vis = pt.nrnvyz(SEGMENTS)
n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.concatenate([np.arange(n+1),
                              1+40*N+np.arange(3*n)])
#BRANCH_LOCS = np.arange(n*N+1)
fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
#vis.add_dots(ax, BRANCH_LOCS, 1)
BL = BRANCH_LOCS[10::20]
COLORS = [plt.cm.gist_earth((i+1)/(len(BL)+1)) for i in range(len(BL))]
vis.add_dots(ax, BL, 10, color=COLORS)

fig.savefig('../figures/reduced_model/morpho-with-stim-loc.svg')

# %%
BL

# %%
Model

# %% [markdown]
# ### Run Input Impedance Profile Characterization

# %%
Model['qAMPA'] = 0.25
Model['gL'] = 1.
syn = {
    'qAMPA':Model['qAMPA'],# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':Model['qAMPA']*4,# [nS] # NMDA-AMPA ratio=2.7
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauRiseGABA':0.5,# [ms] Destexhe et al. 1998
    'tauDecayGABA':5,# [ms] Destexhe et al. 1998
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
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

syn['nAMPA'] = double_exp_normalization(syn['tauRiseAMPA'],syn['tauDecayAMPA'])    
syn['nNMDA'] = double_exp_normalization(syn['tauRiseNMDA'],syn['tauDecayNMDA'])    


EXC_SYNAPSES_EQUATIONS = '''dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
                            gAMPA = ({qAMPA}*nS)*{nAMPA}*(gDecayAMPA-gRiseAMPA) : siemens
                            gNMDA = ({qNMDA}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-v_post/({V0NMDA}*mV))) : siemens
                            gE_post = gAMPA+gNMDA : siemens (summed)'''.format(**syn)
ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'


results = {'Nsyn':6, 'interstim':300}

#VALUES, key = [0.8, 0.9, 1, 1.1, 1.2], 'root-diameter'
#VALUES, key = [100, 125, 150, 175, 200], 'Ri'
VALUES, key = [0.6, 0.65, 0.7, 0.75, 0.8], 'diameter-reduction-factor'

for k, value in enumerate(VALUES):

    l, loc = 3, BL[3]
    
    case=str(k)
    results[case] = {}
    Model[key] = value
    net, BRT, neuron = None, None, None
    stimulation, ES, M = None, None, None
    
    net, BRT, neuron = initialize(Model, with_network=True)


    spike_IDs, spike_times, synapses = np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)
    for e in range(results['Nsyn']):
        s = np.zeros(e+1)
        spike_times = np.concatenate([spike_times,
                                        (1+e)*results['interstim']+\
                                          np.arange(len(s))*nrn.defaultclock.dt/nrn.ms])
        spike_IDs = np.concatenate([spike_IDs, np.array(s, dtype=int)])

    results[case]['spike_times'] = spike_times
    results[case]['spike_IDs'] = spike_IDs

    stimulation = nrn.SpikeGeneratorGroup(1,
                                          np.array(spike_IDs, dtype=int),
                                          spike_times*nrn.ms)
    net.add(stimulation)
    ES = nrn.Synapses(stimulation, neuron,
                       model=EXC_SYNAPSES_EQUATIONS,
                       on_pre=ON_EXC_EVENT,
                       method='exponential_euler')

    ES.connect(i=0, j=loc)
    net.add(ES)
    # recording and running
    M = nrn.StateMonitor(neuron, ('v'), record=[0, loc])
    net.add(M)
    nrn.run((1.1*results['interstim']+np.max(spike_times))*nrn.ms)
    results[case]['Vm'] = np.array(M.v[0]/nrn.mV)
    results[case]['Vm-loc'] = np.array(M.v[1]/nrn.mV)
    
    results[case]['t'] = np.array(M.t/nrn.ms)

# np.save('results.npy', results)

# %%
def build_linear_pred_trace(results):

    # build a dicionary with the individual responses
    k=0
    while str(k) in results:

        case=str(k)

        results[case]['depol-real'] = []
        results[case]['depol-linear'] = []
    
        results['%s-linear-pred' % case] = {'Vm':np.ones(len(results[case]['t']))*results[case]['Vm'][-1],
                                            'Vm-loc':np.ones(len(results[case]['t']))*results[case]['Vm-loc'][-1],
                                            't':results[case]['t']}

        # single syn kernel
        t_cond = (results[case]['t']>results['interstim']) &  (results[case]['t']<(2*results['interstim']))
        results['%s-single-syn-kernel' % case] = results[case]['Vm'][t_cond]-results[case]['Vm'][t_cond][0]
        results['%s-single-syn-kernel-loc' % case] = results[case]['Vm-loc'][t_cond]-results[case]['Vm-loc'][t_cond][0]

        for e in range(results['Nsyn']):
            start = ((1+e)*results['interstim'])
            i0 = np.flatnonzero(results[case]['t']>start)[0] 
            N=len(results['%s-single-syn-kernel' % case])            
            results['%s-linear-pred' % case]['Vm'][i0:i0+N] += (1+e)*results['%s-single-syn-kernel' % case]
            results['%s-linear-pred' % case]['Vm-loc'][i0:i0+N] += (1+e)*results['%s-single-syn-kernel-loc' % case]
            # compute max depol
            results[case]['depol-real'].append(np.max(results[case]['Vm'][i0:i0+N])-\
                                                   results[case]['Vm'][i0])
            results[case]['depol-linear'].append(np.max(results['%s-linear-pred' % case]['Vm'][i0:i0+N])-\
                                                   results['%s-linear-pred' % case]['Vm'][i0])
        k+=1
    return results


# %%
results = build_linear_pred_trace(results)
k=1
pt.plt.plot(results[str(k)]['Vm-loc'], label='real')
pt.plt.plot(results[str(k)+'-linear-pred']['Vm-loc'], 'r:', label='linear')
pt.plt.legend(loc=(1,1))

# %%
length = 300
fig, AX = pt.figure(axes=(len(VALUES),1), figsize=(0.9,1.3), wspace=0.03, left=0)
for k in range(len(VALUES)):
    for nsyn in range(1, results['Nsyn']):
        tstart = nsyn*results['interstim']
        tCond = (results[case]['t']>tstart) & (results[case]['t']<(length+tstart))
        case=str(k)
        AX[k].plot(1.1*length+results[case]['t'][tCond]-results[case]['t'][tCond][0],
                   results[case]['Vm-loc'][tCond], color=pt.viridis(k/(len(VALUES)-1)), lw=0.5)
        AX[k].plot(results[case]['t'][tCond]-results[case]['t'][tCond][0],
                   results[case+'-linear-pred']['Vm-loc'][tCond], ':', color=pt.viridis(k/(len(VALUES)-1)), lw=0.5)
    AX[k].axis('off')
AX[2].plot(results[case]['t'][tCond]-results[case]['t'][tCond][0], 0*results[case]['t'][tCond], 'k:', lw=0.5)
pt.set_common_ylims(AX)   
pt.draw_bar_scales(AX[0], Xbar=20, Xbar_label='20ms',
                   Ybar=10, Ybar_label='10mV ')

# %%
length = 150
fig, AX = pt.figure(axes=(1,1), figsize=(1.4,1.3), wspace=0.03, left=0)
k=0
for nsyn in range(1, results['Nsyn']):
    case=str(k)
    tstart = nsyn*results['interstim']
    tCond = (results[case]['t']>tstart) & (results[case]['t']<(length+tstart))
    AX.plot(1.1*length+results[case]['t'][tCond]-results[case]['t'][tCond][0],
               results[case]['Vm'][tCond], color='k', lw=0.5)
    AX.plot(results[case]['t'][tCond]-results[case]['t'][tCond][0],
               results[case+'-linear-pred']['Vm'][tCond], ':', color='k', lw=0.5)
    AX.axis('off')
pt.draw_bar_scales(AX, Xbar=20, Xbar_label='20ms',
                   Ybar=2, Ybar_label='2mV ')
pt.save(fig)

# %%
fig, ax = pt.figure(figsize=(1.2,1.8), right=5)

COLORS = [plt.cm.copper_r(i/(len(VALUES))) for i in range(len(VALUES))]

for k in range(len(VALUES)):
    case=str(k)
    ax.plot(results[case]['depol-linear'][1:], results[case]['depol-real'][1:],
                'o-', color=COLORS[k])
ax.plot(ax.get_xlim(), ax.get_xlim(), 'k:', lw=0.5)
    

pt.set_plot(ax, xlabel='expected EPSP (mV)', ylabel='recorded EPSP (mV)   ')

pt.bar_legend(ax, X=VALUES,
              colormap=pt.copper_r, orientation='horizontal',
              colorbar_inset={'rect': [-0.1, 1.2, 1.2, 0.08], 'facecolor': None})
pt.save(fig)

# %%
