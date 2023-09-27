from parallel import Parallel
from PV_template import *

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


def train(freq,
          tstop=1.):
    """
    Poisson spike train from a given frequency (homogeneous process)
    """

    spikes = np.cumsum(\
            np.random.exponential(1./freq, int(2*tstop*freq)))

    return spikes[spikes<tstop]


def run_sim(cellType='Basket', 
            iBranch=0,
            # bg Stim props
            bgStimFreq = 1e-3, 
            bgStimSeed = 10, 
            # synapse shuffling
            synShuffled=False,
            synShuffleSeed=0,
            # biophysical props
            NMDAtoAMPA_ratio=0,
            ampa_weight=1e-3, # uS
            # sim props
            filename='single_sim.npy',
            with_Vm=False,
            dt= 0.025,
            tstop = 1000):

    ######################################################
    ##   simulation preparation  #########################
    ######################################################

    # create cell
    if cellType=='Basket':
        ID = '864691135396580129_296758' # Basket Cell example
        cell = PVcell(ID=ID, debug=False)
        cell.check_that_all_dendritic_branches_are_well_covered(verbose=False)
    elif cellType=='Martinotti':
        pass
    else:
        raise Exception(' cell type not recognized  !')
        cell = None

    # shuffle synapses
    if synShuffled:
        np.random.seed(synShuffleSeed+bgStimSeed) # co-shuffling by default 
        synapses = np.random.choice(cell.branches['branches'][iBranch],
                                    len(cell.branches['synapses'][iBranch]))
    else:
        synapses = cell.branches['synapses'][iBranch]

    # prepare presynaptic spike trains
    np.random.seed(bgStimSeed)
    TRAINS = []
    for i, syn in enumerate(synapses):
        TRAINS.append(train(bgStimFreq, tstop=tstop))
        

    ######################################################
    ##   true simulation         #########################
    ######################################################
    AMPAS, NMDAS = [], []
    ampaNETCONS, nmdaNETCONS = [], []
    STIMS, VECSTIMS = [], []

    for i, syn in enumerate(synapses):

        np.random.seed(syn)

        # need to avoid x=0 and x=1, to allow ion concentrations variations in NEURON
        x = np.clip(cell.SEGMENTS['NEURON_segment'][syn], 
                1, cell.SEGMENTS['NEURON_section'][syn].nseg-2)\
                        /cell.SEGMENTS['NEURON_section'][syn].nseg

        AMPAS.append(\
                h.CPGLUIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

        if NMDAtoAMPA_ratio>0:
            NMDAS.append(\
                    h.NMDAIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

        VECSTIMS.append(h.VecStim())
        STIMS.append(h.Vector(TRAINS[i]))

        VECSTIMS[-1].play(STIMS[-1])

        ampaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
        ampaNETCONS[-1].weight[0] = ampa_weight

        if NMDAtoAMPA_ratio>0:
            nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
            nmdaNETCONS[-1].weight[0] = ampa_weight*NMDAtoAMPA_ratio


    t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
    Vm = h.Vector()

    # Vm rec
    Vm.record(cell.soma[0](0.5)._ref_v)

    # spike count
    apc = h.APCount(cell.soma[0](0.5))

    # run
    h.finitialize(cell.El)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS = None, None
    ampaNETCONS, nmdaNETCONS = None, None
    STIMS, VECSTIMS = None, None

    # save the output
    output = {'output_rate': float(apc.n*1e3/tstop),
              'dt': dt, 'tstop':tstop}
    if with_Vm:
        output['Vm'] = np.array(Vm)
    np.save(filename, output)


if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="script description",
                                   formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', "-- cellType",\
                        help="""
                        cell type, either:
                        - Basket
                        - Martinotti
                        """, default='Basket')
    
    parser.add_argument("--branch", help="branch ID (-1 means all)", type=int, default=-1)
    # synaptic shuffling
    parser.add_argument("--synShuffled",
                        help="shuffled synapses uniformly ? (False=real) ", action="store_true")
    parser.add_argument("--synShuffleSeed",
                        help="seed for synaptic shuffling", type=int, default=0)
    # stimulation shuffling
    parser.add_argument("--bgShuffleSeed",
                        help="seed for background stim", type=int, default=1)

    parser.add_argument("-wVm", "--with_Vm", help="store Vm", action="store_true")

    args = parser.parse_args()

    sim = Parallel(\
        filename='../../data/detailed_model/Basket_bgStim_sim.zip')

    sim.build({'iBranch':range(3),
               'bgStimSeed': range(10, 13),
               'bgStimFreq': np.array([5e-4, 1e-3, 5e-3]),
               'synShuffled':[True, False]})
    sim.run(run_sim,
            single_run_args={'cellType':'Basket', 'with_Vm':True}) 

