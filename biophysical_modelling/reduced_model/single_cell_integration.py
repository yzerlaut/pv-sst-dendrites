import numpy as np
import sys, pathlib, os

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[2],
                             'neural_network_dynamics'))
import nrn
from utils import params

#######################################
#####     Cellular Integration    #####
#######################################

# cable theory:
Equation_String = '''
Im = + ({gL}*siemens/meter**2) * (({EL}*mV) - v) : amp/meter**2
Is = gE * (({Ee}*mV) - v) : amp (point current)
I : amp (point current)
gE : siemens'''

def initialize(Model, with_network=False):

    # simulation params
    nrn.defaultclock.dt = Model['dt']*nrn.ms

    # Ball and Rall Tree morphology
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                    Nbranch=Model['branch-number'],
                                    branch_length=1.0*Model['tree-length']/Model['branch-number'],
                                    soma_radius=Model['soma-radius'],
                                    root_diameter=Model['root-diameter'],
                                    diameter_reduction_factor=Model['diameter-reduction-factor'],
                                    Nperbranch=Model['nseg_per_branch'])
    
    # morphological model
    neuron = nrn.SpatialNeuron(morphology=BRT,
                               model=Equation_String.format(**Model),
                               method='euler',
                               Cm=Model['cm'] * nrn.uF / nrn.cm ** 2,
                               Ri=Model['Ri'] * nrn.ohm * nrn.cm)
    
    # model initialization
    neuron.v = Model['EL']*nrn.mV # Vm initialized to E
    neuron.gE = 0.*nrn.uS
    neuron.I = 0.*nrn.pA

    if with_network:
        net = nrn.Network(nrn.collect())
        net.add(neuron)
        net.store("start")
        return net, BRT, neuron

    else:
        return BRT, neuron


#######################################
#####    Synaptic Integration     #####
#######################################

# normalization factor of synaptic inputs
def double_exp_normalization(T1, T2):
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

# -- excitation (NMDA-dependent)
EXC_SYNAPSES_EQUATIONS = '''dgRiseAMPA/dt = -gRiseAMPA/({tauRiseAMPA}*ms) : 1 (clock-driven)
                            dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
                            dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
                            dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
                            gAMPA = ({qAMPA}*nS)*{nAMPA}*(gDecayAMPA-gRiseAMPA) : siemens
                            gNMDA = ({qAMPA}*{qNMDAtoAMPAratio}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-v_post/({V0NMDA}*mV))) : siemens
                            gE_post = gAMPA+gNMDA : siemens (summed)'''

ON_EXC_EVENT = 'gDecayAMPA += 1; gRiseAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'

def load_params(filename):

    Model = params.load(filename) # neural_network_dynamics/utils/params.py

    Model['nAMPA'] = double_exp_normalization(Model['tauRiseAMPA'],Model['tauDecayAMPA'])    
    Model['nNMDA'] = double_exp_normalization(Model['tauRiseNMDA'],Model['tauDecayNMDA'])

    return Model

if __name__=='__main__':

    Model = load_params('BRT-parameters.json')

    BRT, neuron = initialize(Model)
    nrn.run(100*nrn.ms)
    BRT, neuron = initialize(Model)
    nrn.run(100*nrn.ms)

    
