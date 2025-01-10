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
sys.path.append('../../')
from neural_network_dynamics import nrn, utils
import plot_tools as pt
import matplotlib.pylab as plt
import numpy as np

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

# %% [markdown]
# ### Run Input Impedance Profile Characterization

# %%
Model['qAMPA'] = 0.23
Model['gL'] = 1.
Model['cm'] = 1.
syn = {
    'qAMPA':Model['qAMPA'],# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':Model['qAMPA']*5,# [nS] # NMDA-AMPA ratio=2.7
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


results = {'Nsyn':10, 'interstim':300}

for l, loc in enumerate(BL):
    
    case=str(loc)
    results[case] = {}
    
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
def build_linear_pred_trace(results, BL):

    # build a dicionary with the individual responses
    
    for l, loc in enumerate(BL):
        
        case=str(loc)

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
            
    return results


# %%
results = build_linear_pred_trace(results, BL)
pt.plt.plot(results[str(BL[3])]['Vm-loc'], label='real')
pt.plt.plot(results[str(BL[3])+'-linear-pred']['Vm-loc'], 'r:', label='linear')
pt.plt.legend(loc=(1,1))

# %%
length = 300
fig, AX = pt.figure(axes=(4,1), figsize=(0.9,1.3), wspace=0.03, left=0)
for l, loc in enumerate(BL):
    for nsyn in range(1, results['Nsyn']):
        tstart = nsyn*results['interstim']
        tCond = (results[case]['t']>tstart) & (results[case]['t']<(length+tstart))
        case=str(loc)
        AX[l].plot(1.1*length+results[case]['t'][tCond]-results[case]['t'][tCond][0],
                   results[case]['Vm-loc'][tCond], color=COLORS[l], lw=0.5)
        AX[l].plot(results[case]['t'][tCond]-results[case]['t'][tCond][0],
                   results[case+'-linear-pred']['Vm-loc'][tCond], ':', color=COLORS[l], lw=0.5)
    AX[l].axis('off')
AX[2].plot(results[case]['t'][tCond]-results[case]['t'][tCond][0], 0*results[case]['t'][tCond], 'k:', lw=0.5)
pt.set_common_ylims(AX)   
pt.draw_bar_scales(AX[0], Xbar=20, Xbar_label='20ms',
                   Ybar=10, Ybar_label='10mV ')

# %%
length = 150
fig, AX = pt.figure(axes=(1,1), figsize=(1.4,1.3), wspace=0.03, left=0)
l, loc = l , BL[3]
for nsyn in range(1, results['Nsyn']):
    tstart = nsyn*results['interstim']
    tCond = (results[case]['t']>tstart) & (results[case]['t']<(length+tstart))
    case=str(loc)
    AX.plot(1.1*length+results[case]['t'][tCond]-results[case]['t'][tCond][0],
               results[case]['Vm'][tCond], color='k', lw=0.5)
    AX.plot(results[case]['t'][tCond]-results[case]['t'][tCond][0],
               results[case+'-linear-pred']['Vm'][tCond], ':', color='k', lw=0.5)
    AX.axis('off')
pt.draw_bar_scales(AX, Xbar=20, Xbar_label='20ms',
                   Ybar=2, Ybar_label='2mV ')
#fig.savefig(os.path.join(os.path.expanduser('~'), 'fig.svg'))

# %%
fig, ax = pt.figure(figsize=(1.1,1.4))
import os
l, loc = 3, BL[3]
case=str(loc)
ax.plot(results[case]['depol-linear'][1:], results[case]['depol-real'][1:],
            'o-', color='k')
ax.plot(results[case]['depol-linear'][1:], results[case]['depol-linear'][1:],
            ':', color='k')
pt.set_plot(ax, xlabel='expected peak EPSP  (mV)  ', 
            xticks=[2,6,10],yticks=[2,6,10],
            ylabel='observed EPSP (mV)    ')
fig.savefig(os.path.join(os.path.expanduser('~'), 'fig.svg'))

# %%
