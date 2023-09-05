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
# # Model Presentation

# %%
from single_cell_integration import * # code to run the model: (see content below)

import sys
sys.path.append('..')
import plot_tools as pt

# %% [markdown]
# ## Equations for cellular and synaptic integration

# %%
# cat single_cell_integration.py

# %% [markdown]
# ## Default Model Parameters
#
# saved as a json file:

# %%
# cat BRT-parameters.json

# %% [markdown]
# ## Load parameters and build the associated morphological model

# %%
# we load the default parameters
Model = load_params('BRT-parameters.json')

# %%
# build and plot the associated morphology;
from nrn.plot import nrnvyz
BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)
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
# # AMPA vs NMDA synaptic events

# %%
dist_loc = 35
t0, space, interstim = 20, 250, 5 # ms
spikes = t0+np.arange(10)*interstim

fig, AX = pt.plt.subplots(1, 2, figsize=(1.9,1.4))

for r, NA_ratio in enumerate([0, 2.5]):
    
    cModel = Model.copy()
    cModel['qNMDAtoAMPAratio'] = NA_ratio
    
    BRT, neuron = initialize(cModel)
    
    # stimulation
    spike_times = np.concatenate([i*space+spikes for i in range(2)]) # ms
    spike_IDs = np.zeros(len(spike_times), dtype=int)
    stimulation = nrn.SpikeGeneratorGroup(1, spike_IDs,
                                          np.array(spike_times)*nrn.ms)
    ES = nrn.Synapses(stimulation, neuron,
                       model=EXC_SYNAPSES_EQUATIONS.format(**cModel),
                       on_pre=ON_EXC_EVENT.format(**cModel),
                       method='exponential_euler')

    ES.connect(i=0, j=dist_loc) # 0 is dist
    #net.add(ES)

    # recording
    M = nrn.StateMonitor(neuron, ('v'),
                         record=[0, dist_loc]) # monitor soma+prox+loc

    for b, bg_current in enumerate([0, 80]):
        
        # running
        neuron.I[dist_loc] = bg_current*nrn.pA
        nrn.run(space*nrn.ms)
        neuron.I = 0*nrn.pA
        
        # plot cond
        cond = (M.t/nrn.ms>(t0-10+b*space)) & (M.t/nrn.ms<(t0-10+(1+b)*space-50))
        AX[r].plot((M.t[cond]-M.t[cond][0])/nrn.ms, M.v[1,cond]/nrn.mV, color='tab:grey')
           
pt.set_common_ylims(AX)
pt.set_common_xlims(AX)
pt.draw_bar_scales(AX[0], Xbar=50, Xbar_label='50ms', Ybar=1e-12)
pt.set_plot(AX[0], ['left'], yticks=[-70, -50, -30], xticks=[])
pt.set_plot(AX[1], ['left'], yticks=[-70, -50, -30], yticks_labels=[], xticks=[])
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Synaptic integration in the proximal and distal segments

# %% [markdown]
# ## Stimulation and Recording locations

# %%
prox_loc = 3
dist_loc = 36

from nrn.plot import nrnvyz # requires: %run ../src/single_cell_integration.py
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)
vis = nrnvyz(SEGMENTS)

n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.concatenate([np.arange(n+1),
                              1+20*N+np.arange(3*n)]),
fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, [prox_loc], 10, color='r')
vis.add_dots(ax, [dist_loc], 10, color='b')
pt.annotate(ax, 'proximal location\n(%i $\mu$m from soma)' % (1e6*SEGMENTS['distance_to_soma'][prox_loc]),
            (1,1), va='top', color='r', fontsize='small')
pt.annotate(ax, '\n\n\ndistal location\n(%i $\mu$m from soma)' % (1e6*SEGMENTS['distance_to_soma'][dist_loc]),
            (1,1), va='top', color='b', fontsize='small')
#fig.savefig('../figures/ball-and-rall-tree.svg')

# %% [markdown]
# ## Showing integrative properties
#
# --> (AMPA only)

# %%

def run_charact(Model,
                LOCS = [0, 4, 20, 33, 40],
                start_at=5, # ms
                space=90, # ms
                interstim=1.0, # ms
                single_sequence_delay=30, # ms
                pulse_amp = 20, # pA
                Nrepeat=10,
                full_output=False):
    """
    a synaptic barrage stimulation in a proximal and then in adistal location
    """

    net, BRT, neuron = initialize(Model, with_network=True)
    
    spike_times = np.empty(0, dtype=int)
    spike_IDs = np.empty(0, dtype=int)
    for d in range(len(LOCS)):
        # loop over prox/dist stimulation
        spike_times = np.concatenate([spike_times, [start_at+d*space]])
        spike_IDs = np.concatenate([spike_IDs, [d]])
        spike_times = np.concatenate([spike_times,\
                spike_times[-1]+single_sequence_delay+interstim*np.arange(Nrepeat)])
        spike_IDs = np.concatenate([spike_IDs, d*np.ones(Nrepeat)])

    stimulation = nrn.SpikeGeneratorGroup(len(LOCS), spike_IDs,
                                          np.array(spike_times)*nrn.ms)
    net.add(stimulation)
    
    ES = nrn.Synapses(stimulation, neuron,
                       model=EXC_SYNAPSES_EQUATIONS.format(**Model),
                       on_pre=ON_EXC_EVENT.format(**Model),
                       method='exponential_euler')
    for i, d in enumerate(LOCS):
        ES.connect(i=i, j=d) 
    net.add(ES)

    # recording
    M = nrn.StateMonitor(neuron, ('v'),
                         record=LOCS) # monitor soma+prox+loc
    net.add(M)
    
    # running
    # --event only first
    neuron.I = 0*nrn.pA
    net.run((start_at+len(LOCS)*space)*nrn.ms)
    
    # --then current step
    neuron.I[0] = pulse_amp*nrn.pA
    net.run(100*nrn.ms)
    neuron.I[0] = 0*nrn.pA
    
    results = {'start_at':start_at, 'space':space, 'Nrepeat':Nrepeat,
               'interstim':interstim, 'single_sequence_delay':single_sequence_delay,
               'LOCS':LOCS}
    for i, loc in enumerate(LOCS):
        results['lin_pred_%s' % loc] = np.ones(len(M.t))*Model['EL']
        
    # build_linear pred. trace from the single EPSP
    for d, stim in enumerate(LOCS):
        t0 = start_at+d*space
        cond = (M.t/nrn.ms>t0) & (M.t/nrn.ms<(t0+single_sequence_delay))
        # first compute single EPSPs
        for i, loc in enumerate(LOCS):
            results['EPSP_at_%s' % loc] = M.v[i,:][cond]/nrn.mV-Model['EL']
            results['lin_pred_%s' % loc][cond] += results['EPSP_at_%s' % loc]
            # then build linear pred
            for k in range(Nrepeat):
                iK = int((t0+single_sequence_delay+k*interstim)/Model['dt'])
                results['lin_pred_%s' % loc][iK:iK+np.sum(cond)] += results['EPSP_at_%s' % loc]
    results['end_tstim'] = start_at+2*space

    # input res from current input at soma
    results['Rinput_soma'] = 1e3*(M.v[0,-1]/nrn.mV-Model['EL'])/pulse_amp
    
    for i, loc in enumerate(LOCS):
        results['Vm_%s' % loc] = M.v[i,:]/nrn.mV
    results['t'] = M.t/nrn.ms
        
    net.remove(neuron)
    net, Ms, stimulation, ES, neuron = None, None, None, None, None
    
    return results

Model = load_params('BRT-parameters.json')

results = run_charact(Model, full_output=True)

# %%
COLORS = ['grey' for i in range(len(results['LOCS']))]

fig, AX = pt.plt.subplots(len(results['LOCS']), len(results['LOCS']),
                          figsize=(1.1*len(results['LOCS']),0.8*len(results['LOCS'])))

epsilons = []

pt.plt.subplots_adjust(hspace=0, wspace=0.1)
for d, stim in enumerate(results['LOCS']):
    t0 = results['start_at']+d*results['space']-10
    cond = (results['t']>t0) & (results['t']<(t0+results['space']))
    for l, label in enumerate(results['LOCS']):
        AX[l][d].plot(results['t'][cond], results['Vm_%s' % label][cond],
                      label=label, color=COLORS[l], lw=0.7)
        AX[l][d].plot(results['t'][cond], results['lin_pred_%s' % label][cond],
                      '--', lw=0.4, label=label, color=COLORS[l])
        AX[l][d].set_xlim([t0, t0+results['space']])
        
    inset = pt.inset(AX[d][d], (0, -0.1, 1, .1))
    inset.plot(np.ones(2)*(results['start_at']+d*results['space']), np.arange(2), 'r-', lw=0.5)
    for k in range(results['Nrepeat']):
        inset.plot(np.ones(2)*(results['start_at']+d*results['space']+\
                               results['single_sequence_delay']+k*results['interstim']), np.arange(2), 'r-', lw=0.5)
    inset.set_xlim([t0, t0+results['space']])
    inset.axis('off')
    efficacy = 100*np.max(results['Vm_%s' % label][cond]-results['Vm_%s' % label][cond][0])/\
        np.max(results['lin_pred_%s' % label][cond]-results['lin_pred_%s' % label][cond][0])
    epsilons.append(efficacy)
    
    AX[0][d].set_title('stim.@%.1f$\mu$m\n$\epsilon$=%.1f%%' % (\
                            1e6*SEGMENTS['distance_to_soma'][stim], efficacy))
    pt.annotate(AX[d][len(results['LOCS'])-1],
                'rec.@%.1f$\mu$m' % (1e6*SEGMENTS['distance_to_soma'][stim]), (1, 0))
    
    
scale = 5 # mV
for i in range(len(results['LOCS'])):
    for l in range(len(results['LOCS'])):
        pt.draw_bar_scales(AX[l][i], Ybar=scale, Ybar_label='%imV ' % scale,
                           Xbar=1e-12, remove_axis=True, color=COLORS[l])
for l in range(len(results['LOCS'])):
    pt.set_common_ylims(AX[l])
    
"""
# plot scale bars
for d, stim in enumerate(results['LOCS']):
    for l, label in enumerate(results['LOCS']):
        if d==1:
            AX[l][d].annotate(' '+label, (1,0), xycoords='axes fraction', color=COLORS[l])
pt.draw_bar_scales(inset, Xbar=20, Xbar_label='20ms ', Ybar=1e-12)
"""

#fig, ax = pt.plot(1e6*SEGMENTS['distance_to_soma'][results['LOCS']], epsilons, lw=1)
#pt.set_plot(ax, xlabel='dist. from soma', ylabel='suppression (%)', yticks=[30, 60, 90])
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Table of Parameters

# %%
# generate Markdown string from Model
table = "|Name|Value|Unit|\n"
table +="|----|:----|:--:|\n"
template = "|Name|Value|Unit|"
def find_unit(key):
    if ('radius' in key) or ('-diameter' in key) or ('length' in key) or (key[:3]=='tau'):
        return 'um'
    elif (key[:3]=='tau') or (key=='dt'):
        return 'ms'
    elif key[0]=='E':
        return 'mV'
    elif (key=='gL'):
        return 'pS/um^2'
    else:
        return ''
for key in Model:
    table += template.replace('Name',key).replace('Value', str(Model[key])).replace('Unit', find_unit(key))+'\n'
    
print(table)

# %% [markdown]
# |Name|Value|Unit|
# |----|:----|:--:|
# |branch-number|4||
# |tree-length|200.0|um|
# |soma-radius|15.0|um|
# |root-diameter|1.0|um|
# |diameter-reduction-factor|0.7|um|
# |nseg_per_branch|10||
# |gL|2.5|pS/um^2|
# |cm|1.0||
# |Ri|150.0||
# |EL|-70|mV|
# |cMg|1.0||
# |etaMg|0.33||
# |V0NMDA|12.5||
# |Mg_NMDA|1.0||
# |Ee|0|mV|
# |qAMPA|0.5||
# |tauRiseAMPA|0.5|um|
# |tauDecayAMPA|5|um|
# |qNMDAtoAMPAratio|0.0||
# |tauRiseNMDA|3|um|
# |tauDecayNMDA|70|um|
# |dt|0.025||
# |nAMPA|1.435055183349871||
# |nNMDA|1.2030285365713829||

# %%