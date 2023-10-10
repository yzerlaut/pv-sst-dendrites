from cell_template import *
from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

def run_sim(cellType='Basket', 
            iBranch=0,
            from_uniform=False,
            # cluster props
            # -- if from distance intervals
            distance_intervals=[],
            iDistance=-1,
            # -- if from distance points
            nCluster=None,
            distance=100,
            # synapse subsampling
            synSubsamplingFraction=0.03,
            synSubsamplingSeed=2,
            # biophysical props
            NMDAtoAMPA_ratio=0,
            # sim props
            interspike=2,
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
        params_key='BC'
    elif cellType=='Martinotti':
        ID = '864691135571546917_264824' # Martinotti Cell example
        params_key='MC'
    else:
        raise Exception(' cell type not recognized  !')
    cell = Cell(ID=ID, params_key=params_key)

    if from_uniform:
        full_synapses = cell.set_of_synapses_spatially_uniform[iBranch]
    else:
        full_synapses = cell.set_of_synapses[iBranch]                      

    # prepare presynaptic spike trains
    # 1) single events
    TRAINS, tstop = [], t0
    for i, syn in enumerate(synapses):
        TRAINS.append([tstop])
        tstop += ISI
    # 2) grouped events
    for i, syn in enumerate(synapses):
        TRAINS[i].append(tstop+i*interspike)
    tstop += ISI

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
        ampaNETCONS[-1].weight[0] = cell.params['%s_qAMPA'%params_key]

        if NMDAtoAMPA_ratio>0:
            nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
            nmdaNETCONS[-1].weight[0] = NMDAtoAMPA_ratio*cell.params['%s_qAMPA'%params_key]

    Vm_soma, Vm_dend = h.Vector(), h.Vector()

    # Vm rec
    Vm_soma.record(cell.soma[0](0.5)._ref_v)
    if len(synapses)>0:
        Vm_dend.record(cell.SEGMENTS['NEURON_section'][synapses[-1]](0.5)._ref_v) # last one in the loop
    else:
        Vm_dend.record(cell.dend[0](0.5)._ref_v) # root dend


    # spike count
    apc = h.APCount(cell.soma[0](0.5))

    ######################################################
    ##   simulation run   ################################
    ######################################################

    h.finitialize(cell.El)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS = None, None
    ampaNETCONS, nmdaNETCONS = None, None
    STIMS, VECSTIMS = None, None

    # compute the linear prediction
    real_soma, linear_soma = build_linear_pred(np.array(Vm_soma),
                                               dt, t0, ISI, interspike, len(synapses))
    real_dend, linear_dend = build_linear_pred(np.array(Vm_dend),
                                               dt, t0, ISI, interspike, len(synapses))

    t = np.arange(len(real_soma))*dt

    # save the output
    output = {'output_rate': float(apc.n*1e3/tstop),
              'real_soma':real_soma, 'real_dend':real_dend,
              'peak_efficacy_soma':efficacy(real_soma, linear_soma,
                                            based_on='peak'),
              'integral_efficacy_soma':efficacy(real_soma, linear_soma,
                                                based_on='integral'),
              'peak_efficacy_dend':efficacy(real_dend, linear_dend,
                                            based_on='peak'),
              'integral_efficacy_dend':efficacy(real_dend, linear_dend,
                                                based_on='integral'),
              'linear_soma':linear_soma,
              'linear_dend':linear_dend,
              'Vm_soma': np.array(Vm_soma),
              'Vm_dend': np.array(Vm_dend),
              'dt': dt, 't0':t0, 'ISI':ISI,'interspike':interspike,
              'distance_inervals':distance_intervals,
              'synapses':synapses,
              'tstop':tstop}

    np.save(filename, output)


if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="script description",
                                   formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', "--cellType",\
                        help="""
                        cell type, either:
                        - Basket
                        - Martinotti
                        """, default='Basket')
    
    # cluster props
    parser.add_argument("--Proximal", type=float, nargs=2, default=(20,60))
    parser.add_argument("--Medial", type=float, nargs=2, default=(90,130))
    parser.add_argument("--Distal", type=float, nargs=2, default=(160,200))
    parser.add_argument("--synSubsamplingFraction", type=float, default=5e-2)
    parser.add_argument("--interspike", type=float, default=1.0)

    # parser.add_argument("--synSubsamplingSeed", type=int, default=5)
    # parser.add_argument("--NsynSubsamplingSeed", type=int, default=1)
    parser.add_argument("--iDistance", help="min input", type=int, default=1)
    # Branch number
    parser.add_argument("--iBranch", type=int, default=0)
    parser.add_argument("--nBranch", type=int, default=6)
    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")
    parser.add_argument("--NMDAtoAMPA_ratio", type=float, default=2.0)

    parser.add_argument("-wVm", "--with_Vm", help="store Vm", action="store_true")
    parser.add_argument("--suffix", help="suffix for saving", default='')

    parser.add_argument("-t", "--test", help="test func", action="store_true")
    args = parser.parse_args()

    distance_intervals = [args.Proximal, args.Distal]

    if args.test:

        params = dict(cellType=args.cellType,
                      iBranch=args.iBranch,
                      iDistance=args.iDistance,
                      synSubsamplingFraction=args.synSubsamplingFraction,
                      interspike=args.interspike,
                      distance_intervals=distance_intervals)
        run_sim(**params)

    else:

        sim = Parallel(\
            filename='../../data/detailed_model/%s_clusterStim_sim%s.zip' % (args.cellType,
                                                                             args.suffix))

        params = dict(iBranch=np.arange(args.nBranch),
                      iDistance=range(2))

        if args.test_uniform:
            params = dict(from_uniform=[False, True], **params)
        if args.test_NMDA:
            params = dict(NMDAtoAMPA_ratio=[0., args.NMDAtoAMPA_ratio], **params)

        sim.build(params)

        sim.run(run_sim,
                single_run_args=dict(cellType=args.cellType,
                                     synSubsamplingFraction=args.synSubsamplingFraction,
                                     interspike=args.interspike,
                                     distance_intervals=distance_intervals))
