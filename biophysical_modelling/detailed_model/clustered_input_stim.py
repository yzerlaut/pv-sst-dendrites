import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.append('../..')
import plot_tools as pt

# from parallel import Parallel
from PV_template import *

def find_clustered_input(cell, 
                         iBranch,
                         subsampling_fraction=0.05,
                         synSubsamplingSeed=2,
                         distance=200,
                         nCluster=10,
                         synShuffled=False,
                         synShuffleSeed=10,
                         ax=None, syn_color='r',
                         bins = np.linspace(0, 280, 30),
                         with_plot=False):

                        
    np.random.seed(synShuffleSeed)
    # full synaptic distrib:
    if synShuffled:
        # spatially uniformly shuffled
        synapses = np.random.choice(\
            cell.branches['branches'][iBranch], 
            len(cell.branches['synapses'][iBranch]))
    else:
        # true distrib
        synapses = cell.branches['synapses'][iBranch]
    branch = cell.branches['branches'][iBranch]

    # subsampling
    np.random.seed(synSubsamplingSeed)
    Nsubsampling = int(len(cell.branches['synapses'][iBranch])*\
                        subsampling_fraction)
    synapses = np.random.choice(synapses, Nsubsampling)

    # finding the closest to the distance point
    # using the distance to soma as a distance measure
    iMins = np.argsort((distance-1e6*cell.SEGMENTS['distance_to_soma'][synapses])**2)

    cluster_synapses = synapses[iMins[:nCluster]]

    if ax is None and with_plot:
        fig, ax = plt.subplots(1, figsize=(3,3))

    if with_plot:
        vis = pt.nrnvyz(cell.SEGMENTS)

        vis.plot_segments(cond=(cell.SEGMENTS['comp_type']!='axon'),
                      bar_scale_args={'Ybar':1e-9, 'Xbar':1e-9},
                      ax=ax)
        ax.scatter(1e6*cell.SEGMENTS['x'][cluster_synapses],
                   1e6*cell.SEGMENTS['y'][cluster_synapses],
                   s=5, color=syn_color)

        inset = pt.inset(ax, [-0.1, 0.7, .35, .17])

        # synapses = cell.branches['synapses'][iBranch]
        hist, be = np.histogram(1e6*cell.SEGMENTS['distance_to_soma'][synapses],
                                bins=bins)
        inset.bar(be[1:], hist, width=be[1]-be[0], edgecolor='tab:grey', color='w')
        hist, be = np.histogram(1e6*cell.SEGMENTS['distance_to_soma'][cluster_synapses],
                                bins=bins)
        inset.bar(be[1:], hist, width=be[1]-be[0], color=syn_color)
        pt.set_plot(inset, xticks=[0,200], 
                    title = '%i%% subset' % (100*subsampling_fraction),
                    ylabel='syn.\ncount', xlabel='dist ($\mu$m)', fontsize=7)

    return cluster_synapses


def build_linear_pred(Vm, dt, t0, ISI, delay, nCluster):
    t = np.arange(len(Vm))*dt
    # extract single EPSPs
    sEPSPS = []
    for i in range(nCluster):
        tstart = t0+i*ISI
        cond = (t>tstart) & (t<(tstart+ISI))
        sEPSPS.append(Vm[cond]-Vm[cond][0])
    # compute real responses
    tstart = t0+nCluster*ISI
    cond = (t>tstart) & (t<(tstart+ISI))
    real = Vm[cond]
    # then linear pred
    linear = np.ones(np.sum(cond))*real[0]
    t = np.arange(len(real))*dt
    for i, epsp in enumerate(sEPSPS):
        cond = (t>i*delay)
        linear[cond] += epsp[:np.sum(cond)]

    return real, linear



def run_sim(cellType='Basket', 
            iBranch=0,
            # cluster props
            nCluster=10,
            distance=200,
            delay=5,
            # synapse subsampling
            subsampling_fraction=0.05,
            synSubsamplingSeed=2,
            # synapse shuffling
            synShuffled=False,
            synShuffleSeed=0,
            # biophysical props
            NMDAtoAMPA_ratio=0,
            ampa_weight=1e-3, # uS
            # sim props
            t0=200,
            ISI=200,
            filename='single_sim.npy',
            dt= 0.025):

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

    synapses = find_clustered_input(cell, 
                                    iBranch,
                                    subsampling_fraction=subsampling_fraction,
                                    synSubsamplingSeed=synSubsamplingSeed,
                                    distance=distance,
                                    nCluster=nCluster,
                                    synShuffled=synShuffled,
                                    synShuffleSeed=synShuffleSeed)

    # prepare presynaptic spike trains
    # 1) single events
    TRAINS, tstop = [], t0
    for i, syn in enumerate(synapses):
        TRAINS.append([tstop])
        tstop += ISI
    # 2) grouped event
    for i, syn in enumerate(synapses):
        TRAINS[i].append(tstop+i*delay)
    tstop += 2*ISI

    ######################################################
    ##  single event simulation  #########################
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


    Vm_soma, Vm_dend = h.Vector(), h.Vector()

    # Vm rec
    Vm_soma.record(cell.soma[0](0.5)._ref_v)
    Vm_dend.record(cell.SEGMENTS['NEURON_section'][syn](x)._ref_v) # last one in the loop

    # spike count
    apc = h.APCount(cell.soma[0](0.5))

    # run
    h.finitialize(cell.El)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS = None, None
    ampaNETCONS, nmdaNETCONS = None, None
    STIMS, VECSTIMS = None, None

    # compute the linear prediction
    real_soma, linear_soma = build_linear_pred(np.array(Vm_soma),
                                               dt, t0, ISI, delay, nCluster)
    real_dend, linear_dend = build_linear_pred(np.array(Vm_dend),
                                               dt, t0, ISI, delay, nCluster)

    t = np.arange(len(real))*dt

    # save the output
    output = {'output_rate': float(apc.n*1e3/tstop),
              'real_soma':real_soma, 'real_dend':real_dend,
              'linear_soma':linear_soma, 'linear_dend':linear_dend,
              'Vm_soma': np.array(Vm_soma),
              'Vm_dend': np.array(Vm_dend),
              'dt': dt, 't0':t0, 'ISI':ISI,
              'nCluster':nCluster, 'delay':delay,
              'tstop':tstop}

    np.save(filename, output)


if __name__=='__main__':

    # run_sim()

    """
    ID = '864691135396580129_296758' # Basket Cell example
    cell = PVcell(ID=ID, debug=False)
    index = 0
    find_clustered_input(cell, 0, with_plot=True)
    find_clustered_input(cell, 0,
            synShuffled=True, with_plot=True)
    plt.show()
    """

    # import argparse
    # # First a nice documentation 
    # parser=argparse.ArgumentParser(description="script description",
                                   # formatter_class=argparse.RawTextHelpFormatter)

    # parser.add_argument('-c', "-- cellType",\
                        # help="""
                        # cell type, either:
                        # - Basket
                        # - Martinotti
                        # """, default='Basket')
    
    # parser.add_argument("--branch", help="branch ID (-1 means all)", type=int, default=-1)
    # # synaptic shuffling
    # parser.add_argument("--synShuffled",
                        # help="shuffled synapses uniformly ? (False=real) ", action="store_true")
    # parser.add_argument("--synShuffleSeed",
                        # help="seed for synaptic shuffling", type=int, default=0)
    # # stimulation shuffling
    # parser.add_argument("--bgShuffleSeed",
                        # help="seed for background stim", type=int, default=1)

    # parser.add_argument("-wVm", "--with_Vm", help="store Vm", action="store_true")

    # args = parser.parse_args()

    sim = Parallel(\
        filename='../../data/detailed_model/Basket_clusteredStim_sim.zip')

    # sim.build({'iBranch':range(3),
               # 'bgStimSeed': range(10, 13),
               # 'bgStimFreq': np.array([5e-4, 1e-3, 5e-3]),
               # 'synShuffled':[True, False]})
    # sim.run(run_sim,
            # single_run_args={'cellType':'Basket', 'with_Vm':True}) 
