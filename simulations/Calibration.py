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

sys.path.append('..')
import plot_tools as pt

sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz

# %% [markdown]
# ## Load parameters and morphology

# %%
Model = {
    #################################################
    # ---------- MORPHOLOGY PARAMS  --------------- #
    #################################################
    'branch-number':4, #
    'tree-length':260.0, # [um]
    'soma-radius':10.0, # [um]
    'root-diameter':1.0, # [um]
    'diameter-exponent':2./3.,
    'nseg_per_branch': 10,
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 5, # [pS/um2] = [S/m2] # NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 1., # [uF/cm2] NEURON default
    "Ri": 100., # [Ohm*cm]
    "EL": -70, # [mV]
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'qAMPA':0.5,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'tauRiseAMPA':0.5,# [ms], Destexhe et al. 1998: 0.4 to 0.8 ms
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    ###################################################
    # ---------- SIMULATION PARAMS  ----------------- #
    ###################################################
    'dt':0.025,# [ms]
    'seed':1, #
    'interstim':250, # [ms]
}

# %%
BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                Nbranch=Model['branch-number'],
                                branch_length=1.0*Model['tree-length']/Model['branch-number'],
                                soma_radius=Model['soma-radius'],
                                root_diameter=Model['root-diameter'],
                                diameter_exponent=Model['diameter-exponent'],
                                Nperbranch=Model['nseg_per_branch'],
                                random_angle=0)

SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

# %% [markdown]
# ## Stimulation and Recording locations

# %%
prox_loc = 3
dist_loc = 25

vis = nrnvyz(SEGMENTS)

n, N = Model['nseg_per_branch'], Model['branch-number']
BRANCH_LOCS = np.concatenate([np.arange(n+1),
                              1+20*N+np.arange(3*n)]),
fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, [soma_loc], 10, color='r')
vis.add_dots(ax, [dend_loc], 10, color='b')
pt.annotate(ax, 'proximal location\n(%i $\mu$m from soma)' % (1e6*SEGMENTS['distance_to_soma'][prox_loc]),
            (1,1), va='top', color='r', fontsize='small')
pt.annotate(ax, '\n\n\ndistal location\n(%i $\mu$m from soma)' % (1e6*SEGMENTS['distance_to_soma'][dist_loc]),
            (1,1), va='top', color='b', fontsize='small')
#fig.savefig('../figures/ball-and-rall-tree.svg')

# %% [markdown]
# ## Running Simulation

# %%

def run_charact(Model,
                dend_loc=24, soma_loc=1,
                event_at=20, # ms
                interstim=70, # ms
                pulse_amp = 20, # pA
                Nsyn=10,
                full_output=False):
    """
    one synaptic 
    current in pA, durations in ms
    """

    # simulation params
    nrn.defaultclock.dt = Model['dt']*nrn.ms

    # equation
    eqs='''
    Im = gL * (EL - v) : amp/meter**2
    Is = gs * (Es - v) : amp (point current)
    gs : siemens
    I : amp (point current)
    '''

    # passive
    gL = Model['gL']*nrn.siemens/nrn.meter**2
    EL = Model['EL']*nrn.mV
    Es = Model['Ee']*nrn.mV
    taus = Model['tauDecayAMPA']*nrn.ms
    w = Model['qAMPA']*nrn.nS
    
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                    Nbranch=Model['branch-number'],
                                    branch_length=1.0*Model['tree-length']/Model['branch-number'],
                                    soma_radius=Model['soma-radius'],
                                    root_diameter=Model['root-diameter'],
                                    Nperbranch=Model['nseg_per_branch'])
    
    neuron = nrn.SpatialNeuron(morphology=BRT,
                               model=eqs,
                               Cm= 1 * nrn.uF / nrn.cm ** 2,    
                               Ri= Model['Ri'] * nrn.ohm * nrn.cm)
    neuron.v = EL
    neuron.I = 0*nrn.pA
    neuron.gs = 0*nrn.nS

    spike_times = np.empty(0, dtype=int)
    
    for d in range(2):
        # loop over prox/dist stimulation
        for i in range(1, Nsyn):
            spike_times = np.concatenate([spike_times, d*interstim*Nsyn+\
                    event_at+i*interstim+np.arange(i)*nrn.defaultclock.dt/nrn.ms])

    print(len(spike_times))
    spike_IDs = np.ones(len(spike_times), dtype=int)
    spike_IDs[:int(len(spike_times)/2)] = 0
    
    stimulation = nrn.SpikeGeneratorGroup(2, spike_IDs,
                                          np.array(spike_times)*nrn.ms)
    
    ES = nrn.Synapses(stimulation, neuron,
                       model='''dg/dt = -g/taus : siemens (clock-driven)
                                gs_post = g : siemens (summed)''',
                       on_pre='g += w',
                       method='exponential_euler')

    ES.connect(i=0, j=prox_loc) # 0 is prox
    ES.connect(i=1, j=dist_loc) # 1 is dist

    # recording
    M = nrn.StateMonitor(neuron, ('v'),
                         record=[0, prox_loc, dist_loc]) # monitor soma+prox+loc
    
    # running
    # --event only first
    neuron.I = 0*nrn.pA
    nrn.run((event_at+2*(Nsyn+1)*interstim)*nrn.ms)
    
    # --then current step
    neuron.I[0] = pulse_amp*nrn.pA
    nrn.run(100*nrn.ms)
    neuron.I[0] = 0*nrn.pA
    
    results = {'peaks_EPSP_dist':[], 'peaks_EPSP_prox':[]}
    # event
    for d, stim in enumerate(['prox', 'dist']):
        results['%s_EPSP_soma_peaks' % stim] = []
        for i in range(1, Nsyn):
            t0 = event_at+i*interstim+d*Nsyn*interstim
            cond = (M.t/nrn.ms>t0-1) & (M.t/nrn.ms<(t0+interstim/2.)) 
            
            results['%s_EPSP_soma_peaks' % stim].append(np.max(M.v[0,:][cond]/nrn.mV-Model['EL']))
            
    if full_output:
        cond = (M.t/nrn.ms>event_at-1) & (M.t/nrn.ms<(event_at+interstim/2.))
        results['EPSP_at_soma'] = M.v[0,:][cond]/nrn.mV
        results['EPSP_at_prox'] = M.v[1,:][cond]/nrn.mV
        results['EPSP_at_dist'] = M.v[2,:][cond]/nrn.mV
        
    # input res from current input at soma
    results['Rinput_soma'] = 1e3*(M.v[0,-1]/nrn.mV-Model['EL'])/pulse_amp
    
    
    if full_output:
        results['Vm_soma'] = M.v[0,:]/nrn.mV
        results['Vm_prox'] = M.v[1,:]/nrn.mV
        results['Vm_dist'] = M.v[2,:]/nrn.mV

    M, stimulation, ES, neuron = None, None, None, None
    
    return results

results = run_charact(Model, full_output=True)


# %%
def plot(results):
    
    fig1, AX = pt.plt.subplots(1, 2, figsize=(3,1.3))
    #AX[0].plot(results['EPSP_at_dist'], label='prox')
    #AX[0].plot(results['EPSP_at_prox'], label='dist')
    #AX[0].set_title('resp. fraction: %.0f%%' % (100*results['peaks_EPSP_prox'][0]/results['peaks_EPSP_dist'][0]))
    
    AX[0].set_title('proximal stim.')
    x = np.arange(len(results['prox_EPSP_soma_peaks'])+1)*results['prox_EPSP_soma_peaks'][0]
    AX[0].plot(x, x, 'k--', lw=0.5)
    AX[0].plot(x, list([0]+results['prox_EPSP_soma_peaks']), 'o')

    AX[1].set_title('distal stim.')
    x = np.arange(len(results['dist_EPSP_soma_peaks'])+1)*results['dist_EPSP_soma_peaks'][0]
    AX[1].plot(x, x, 'k--', lw=0.5)
    AX[1].plot(x, list([0]+results['dist_EPSP_soma_peaks']), 'o')
    for ax in AX:
        ax.set_xlabel('expect. EPSP (mV)')
        ax.set_ylabel('obs. EPSP (mV)')
        
    fig2, ax = pt.plt.subplots(1, figsize=(4,2))
    for l, label in enumerate(['soma', 'prox', 'dist']):
        ax.plot(10*l+results['Vm_%s' % label], label=label)
    pt.draw_bar_scales(ax, Ybar=10, Xbar=1e-12, Ybar_label='10mV ', remove_axis=True)
    ax.set_xticks([])
    ax.set_title('$R_{in}$=%.1fM$\Omega$' % results['Rinput_soma'])
    ax.legend(frameon=False, loc=(1,0.3))
    
    return fig1, fig2

plot(results)

# %% [markdown]
# ## Run loop

# %%

# %%
