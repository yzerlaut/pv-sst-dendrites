from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


def run_sim(cellType='Basket', 
            iBranch=0,
            nCluster=1,
            nmax=int(1e6),
            # biophysical props
            passive_only=True,
            with_NMDA=False,
            # sim props
            interspike=2,
            t0=200,
            ISI=200,
            filename='single_sim.npy',
            dt= 0.025):

    from cell_template import Cell, h, np
    from synaptic_input import add_synaptic_input
    from clustered_input_stim import find_clustered_input,\
            build_linear_pred, efficacy


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
    cell = Cell(ID=ID, 
                passive_only=passive_only,
                params_key=params_key)

    synapses = cell.set_of_branches[iBranch][::-1][:nmax]

    # prepare presynaptic spike trains
    TRAINS, tstop = [], t0
    for i, syn in enumerate(synapses):
        TRAINS.append(tstop+np.arange(nCluster)*interspike)
        tstop += ISI
    tstop += ISI

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell,\
                                            synapses, 
                                            EI_ratio=0,
                                            with_NMDA=with_NMDA)

    # -- link events to synapses
    for i, syn in enumerate(synapses):

        STIMS.append(h.Vector(TRAINS[i]))
        VECSTIMS[i].play(STIMS[-1])

    Vm_soma, Vm_dend = h.Vector(), h.Vector()

    # -- Vm recording
    Vm_soma.record(cell.soma[0](0.5)._ref_v)
    if len(synapses)>0:
        # last one in the loop
        Vm_dend.record(\
            cell.SEGMENTS['NEURON_section'][synapses[-1]](0.5)._ref_v) 
    else:
        Vm_dend.record(cell.dend[0](0.5)._ref_v) # root dend


    ######################################################
    ##   simulation run   ################################
    ######################################################

    h.finitialize(-70)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS, GABAS = None, None, None
    ampaNETCONS, nmdaNETCONS, gabaNETCONS = None, None, None
    STIMS, VECSTIMS = None, None

    t = np.arange(len(real_soma))*dt

    # save the output
    output = {'Vm_soma': np.array(Vm_soma),
              'Vm_dend': np.array(Vm_dend),
              'dt': dt, 't0':t0, 'ISI':ISI, 'interspike':interspike,
              'distance_inervals':distance_intervals,
              'synapses':synapses,
              'tstop':tstop}

    np.save(filename, output)


if __name__=='__main__':

    import argparse
    import numpy as np

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
    parser.add_argument("--nCluster", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=1)
    parser.add_argument("--interspike", type=float, default=1.0)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=0)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--test_active", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")

    parser.add_argument("--suffix", help="suffix for saving", default='')
    parser.add_argument('-fmo', "--fix_missing_only", help="in scan",\
                        action="store_true")

    parser.add_argument("-t", "--test", help="test func",\
                        action="store_true")
    args = parser.parse_args()

    params = dict(cellType=args.cellType,
                  iBranch=args.iBranch,
                  nmax=args.nmax,
                  nCluster=args.nCluster,
                  interspike=args.interspike)

    if args.test:

        run_sim(**params)

    else:

        sim = Parallel(\
            filename='../../data/detailed_model/Propagation_sim%s_%s.zip' % (args.suffix, args.cellType))

        grid = dict(iBranch=np.arange(args.nBranch),
                    iDistance=range(2),
                    synSubsamplingFraction=[s/100. for s in args.sparsening])

        if args.test_active:
            grid = dict(passive_only=[True, False], **grid)

        if args.test_NMDA:
            grid = dict(with_NMDA=[False, True], **grid)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}),
                fix_missing_only=args.fix_missing_only)
