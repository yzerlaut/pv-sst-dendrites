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
import os, time, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

sys.path.append('../neural_network_dynamics') # repo submodule
import nrn

##########################################################
# -- EQUATIONS FOR THE SYNAPTIC AND CELLULAR BIOPHYSICS --
##########################################################

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


Model = {
    #################################################
    # ---------- MORPHOLOGY PARAMS  --------------- #
    #################################################
    'Nbranch':4, # 
    'branch_length':100, # [um]
    'radius_soma':5, # [um]
    'diameter_root_dendrite':2, # [um]
    ##################################################
    # ---------- BIOPHYSICAL PROPS ----------------- #
    ##################################################
    "gL": 0.5, # [pS/um2] = 10*[S/m2] # FITTED --- Farinella et al. 0.5pS/um2 = 0.5*1e-12*1e12 S/m2, NEURON default: 1mS/cm2 -> 10pS/um2
    "cm": 1., # [uF/cm2] NEURON default
    "Ri": 100., # [Ohm*cm]'
    "EL": -75, # [mV]
    #################################################
    # ---------- SYNAPTIC PARAMS  ----------------- #
    #################################################
    'Ee':0,# [mV]
    'qAMPA':0.3,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDAtoAMPAratio': 2.5,
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


# %%
#########################################
# ---------- SIMULATION   ------------- #
#########################################

def run_sim(Model,
            loc='proximal',
            NstimMax=10,
            t0 = 200,
            verbose=True):

    # Ball and Rall Tree morphology
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(Nbranch=Model['Nbranch'],
                                                         branch_length=Model['branch_length'],
                                                         soma_radius=Model['radius_soma'],
                                                         root_diameter=Model['diameter_root_dendrite'],
                                                         Nperbranch=10)
    neuron = nrn.SpatialNeuron(morphology=BRT,
                               model=Equation_String.format(**Model),
                               method='euler',
                               Cm=Model['cm'] * nrn.uF / nrn.cm ** 2,
                               Ri=Model['Ri'] * nrn.ohm * nrn.cm)
    
    gL = Model['gL']*nrn.siemens/nrn.meter**2
    neuron.v = Model['EL']*nrn.mV # Vm initialized to E

    spike_IDs, spike_times, tstims = [], [], []
    for n in range(1, NstimMax+1):
        tstims.append([t0, t0+n*Model['interspike']+Model['interstim']/2.]) # interval to analyze resp
        for k in range(n):
            spike_times.append(t0+k*Model['interspike'])
        t0+=n*Model['interspike']+Model['interstim']
        
    spike_IDs = np.zeros(len(spike_times)) # one single synaptic loc
    
    # spatial location of the synaptic input
    if loc=='distal' and Model['Nbranch']>1:
        dend_comp = getattr(neuron.root, ''.join(['L' for b in range(Model['Nbranch']-1)]))
    elif loc=='proximal' or Model['Nbranch']==1:
        dend_comp = neuron.root
    else:
        dend_comp = None
        print(' /!\ Location not recognized ! /!\ ')
        
    synapses_loc = [dend_comp[5] for i in range(len(spike_times))] # in the middle
        
    Estim, ES = nrn.process_and_connect_event_stimulation(neuron,
                                                          spike_IDs, spike_times,
                                                          synapses_loc,
                                                          EXC_SYNAPSES_EQUATIONS.format(**Model),
                                                          ON_EXC_EVENT.format(**Model))

    Model['tstop']=t0
    np.random.seed(Model['seed'])
        
    # simulation params
    nrn.defaultclock.dt = Model['dt']*nrn.ms
    t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    

    # recording and running
    Ms = nrn.StateMonitor(neuron, ('v'), record=[0]) # soma
    print()
    Md = nrn.StateMonitor(dend_comp, ('v'), record=[5]) # dendrite, n the middle

    # # Run simulation
    nrn.run(Model['tstop']*nrn.ms)

    # # Analyze somatic response
    stim_number = np.arange(NstimMax+1)
    peak_levels = np.zeros(NstimMax+1)
    for n in range(1, NstimMax+1):
        cond = (t>tstims[n-1][0]) & (t<tstims[n-1][1])
        peak_levels[n] = np.max(np.array(Ms.v/nrn.mV)[0,cond]-Model['EL'])
    
    label = '%s stim, $q_{AMPA}$=%.1fnS, NMDA/AMPA=%.1f, $N_{branch}$=%i, $L_{branch}$=%ium, $D_{root}$=%.1fum     ' % (\
            loc, Model['qAMPA'], Model['qNMDAtoAMPAratio'], Model['Nbranch'], Model['branch_length'], Model['diameter_root_dendrite'])
    output = {'t':np.array(Ms.t/nrn.ms),
              'Vm_soma':np.array(Ms.v/nrn.mV)[0,:],
              'Vm_dend':np.array(Md.v/nrn.mV)[0,:],
              'stim_number':stim_number,
              'peak_levels':peak_levels,
              'label':label,
              'Model':Model}
    
    t, neuron, BRT = None, None, None
    return output

# %%
output = run_sim(Model, loc='proximal', NstimMax=15)


# %%

def plot_signals(output):
    # raw traces
    fig, AX = plt.subplots(2, 1, figsize=(6,2))
    AX[0].set_title(output['label'], fontsize=8)
    plt.subplots_adjust(hspace=0, right=.75)
    cond = output['t']>150
    AX[0].plot(output['t'][cond], output['Vm_dend'][cond], '-', color='k', lw=1)
    AX[0].annotate('dendrite', (0.02, 0.95), va='top', xycoords='axes fraction')
    AX[1].plot(output['t'][cond], output['Vm_soma'][cond], '-', color='k', lw=1)
    AX[1].annotate('soma', (0.02, 0.95), va='top', xycoords='axes fraction')
    xlim, ylim = AX[1].get_xlim(), AX[1].get_ylim()
    for ax in AX:
        ax.set_xticks([])
        ax.set_ylabel('$V_m$ (mV)')
        ax.set_xlim(xlim)
    AX[1].plot(xlim[1]-200*np.arange(2), ylim[1]+np.zeros(2), '-', lw=1, color='gray')
    AX[1].annotate('200ms', (xlim[1], ylim[1]), rotation=90, ha='right', va='top', color='gray')
    # summary
    ax = fig.add_axes([0.85,0.35,0.13,0.43])
    ax.set_title('soma resp.', fontsize=10)
    ax.plot(output['stim_number'], output['peak_levels'], 'ko', ms=3)
    ax.set_xlabel('number stim.\nsynapses')
    ax.set_ylabel('peak depol. (mV)    ')
    return fig

#fig = plot_signals(output)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'example.png'))


# %%
cModel = Model.copy()

cModel['Nbranch'] = 1
cModel['branch_length'] = 0.1
cModel['radius_soma'] = 20
cModel['diameter_root_dendrite'] = 10
cModel['qAMPA'] = 0.1
cModel['qNMDAtoAMPAratio'] = 0.01
       
output = run_sim(cModel, loc='proximal', NstimMax=15)
fig = plot_signals(output)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'example.png'))

# %%
cModel = Model.copy()

cModel['Nbranch'] = 5
cModel['branch_length'] = 100
cModel['radius_soma'] = 5
cModel['diameter_root_dendrite'] = 2
cModel['qAMPA'] = 0.1
cModel['qNMDAtoAMPAratio'] = 2.5
       
output = run_sim(cModel, loc='distal', NstimMax=15)
fig = plot_signals(output)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'low-quantal.png'))

# %%
cModel = Model.copy()

cModel['Nbranch'] = 2
cModel['branch_length'] = 1
cModel['radius_soma'] = 10
cModel['diameter_root_dendrite'] = 1
cModel['qAMPA'] = 0.1
cModel['qNMDAtoAMPAratio'] = 0.5
       
output = run_sim(cModel, loc='distal', NstimMax=15)
fig = plot_signals(output)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'distal-PV-like.png'))

# %%
cModel = Model.copy()

cModel['Nbranch'] = 5
cModel['branch_length'] = 100
cModel['radius_soma'] = 5
cModel['diameter_root_dendrite'] = 2
cModel['qAMPA'] = 0.2
cModel['qNMDAtoAMPAratio'] = 2.5
       
output = run_sim(cModel, loc='proximal', NstimMax=15)
fig = plot_signals(output)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'proximal-SST-like.png'))

# %%
cModel = Model.copy()

cModel['Nbranch'] = 5
cModel['branch_length'] = 100
cModel['radius_soma'] = 5
cModel['diameter_root_dendrite'] = 5
cModel['qAMPA'] = 0.2
cModel['qNMDAtoAMPAratio'] = 0.5
       
output = run_sim(cModel, loc='proximal', NstimMax=15)
fig = plot_signals(output)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'proximal-PV-like.png'))

# %%
