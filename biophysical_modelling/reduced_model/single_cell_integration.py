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

# -- excitation 
EXC_SYNAPSES_EQUATIONS = '''dgAMPA/dt = -gAMPA/({tauAMPA}*ms) : siemens (clock-driven)
                            gE_post = gAMPA : siemens (summed)'''

ON_EXC_EVENT = 'gAMPA += {qAMPA}*nS'

def load_params(filename):

    Model = params.load(filename) # neural_network_dynamics/utils/params.py

    return Model

if __name__=='__main__':

    Model = load_params('BRT-parameters.json')

    BRT, neuron = initialize(Model)
    nrn.run(100*nrn.ms)
    BRT, neuron = initialize(Model)
    nrn.run(100*nrn.ms)

    
