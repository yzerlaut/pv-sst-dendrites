from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


def find_clustered_input(cell, 
                         iBranch,
                         # cluster props
                         # -- if from distance intervals
                         distance_intervals=[],
                         iDistance=-1,
                         # -- if from distance points
                         nCluster=None,
                         distance=100,
                         from_uniform=False,
                         # synapse sparsening
                         synSubsamplingFraction=0.05,
                         synSubsamplingSeed=3,
                         ax=None, syn_color='r',
                         with_plot=False):

    import numpy as np

    branch = cell.set_of_branches[iBranch]
    
    if from_uniform:
        full_synapses = cell.set_of_synapses_spatially_uniform[iBranch]
    else:
        full_synapses = cell.set_of_synapses[iBranch]                      

    # subsampling
    if synSubsamplingFraction<1:
        np.random.seed(synSubsamplingSeed)
        Nsubsampling = int(len(full_synapses)*synSubsamplingFraction)
        N = int(len(full_synapses)/Nsubsampling)
        # synapses = np.random.choice(full_synapses, Nsubsampling)
        synapses = np.concatenate([full_synapses[::N], [full_synapses[-1]]])
        # synapses = full_synapses[::N]
    else:
        synapses = full_synapses

    # ==== cluster from interval ===
    # ------------------------------
    if iDistance>-1:
        interval = distance_intervals[iDistance]
        cluster_cond = (1e6*cell.SEGMENTS['distance_to_soma'][synapses]>=interval[0]) & \
                (1e6*cell.SEGMENTS['distance_to_soma'][synapses]<interval[1])
        cluster_synapses = synapses[cluster_cond]

    # ==== cluster from distance ===
    # ------------------------------
    # -- finding the closest to the distance point (using the distance to soma metrics)
    if nCluster is not None:
        iSorted = np.argsort((1e6*cell.SEGMENTS['distance_to_soma'][synapses]-distance)**2)
        cluster_synapses = synapses[iSorted[:nCluster]]

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

        bins = np.linspace(\
                0.9*np.min(1e6*cell.SEGMENTS['distance_to_soma'][full_synapses]),
                1.1*np.max(1e6*cell.SEGMENTS['distance_to_soma'][full_synapses]), 20)

        hist, be = np.histogram(1e6*cell.SEGMENTS['distance_to_soma'][full_synapses],
                                bins=bins)
        inset.bar(be[1:], hist, width=be[1]-be[0], edgecolor='tab:grey', color='w')

        bins = np.linspace(0.9*np.min(1e6*cell.SEGMENTS['distance_to_soma'][synapses]),
                           1.1*np.max(1e6*cell.SEGMENTS['distance_to_soma'][synapses]), 15)
        hist, be = np.histogram(1e6*cell.SEGMENTS['distance_to_soma'][synapses],
                                bins=bins)
        inset.bar(be[1:], hist, width=be[1]-be[0], color='tab:grey')

        hist, be = np.histogram(1e6*cell.SEGMENTS['distance_to_soma'][cluster_synapses],
                                bins=bins)
        inset.bar(be[1:], hist, width=be[1]-be[0], color=syn_color)
        pt.set_plot(inset, xticks=[0,200], 
                    yscale='log', yticks=[1,10], yticks_labels=['1', '10'],
                    ylabel='syn. count', xlabel='dist ($\mu$m)', fontsize=7)
        inset.set_title('%i%% subset' % (100*synSubsamplingFraction), 
                        color='tab:grey', fontsize=6)
        pt.annotate(ax, 'n=%i' % len(cluster_synapses), (-0.2,0.2), 
                    fontsize=7, color=syn_color)

        return ax, inset

    else:
        return cluster_synapses


def build_linear_pred(Vm, dt, t0, ISI, interspike, nCluster):
    """
    build the linear prediction from the full stim protocol
    """
    import numpy as np

    t = np.arange(len(Vm))*dt
    # extract single EPSPs
    sEPSPS = []
    for i in range(nCluster):
        tstart = t0+i*ISI
        cond = (t>tstart) & (t<(tstart+ISI))
        sEPSPS.append(Vm[cond]-Vm[cond][0])
    # --- # compute real and linear responses
    tstart = t0+nCluster*ISI
    cond = (t>tstart) & (t<(tstart+ISI))
    real = Vm[cond]
    # then linear pred
    te = np.arange(len(real))*dt
    linear = np.ones(np.sum(cond))*real[0]
    for i, epsp in enumerate(sEPSPS):
        condE = (te>i*interspike)
        linear[condE] += epsp[:np.sum(condE)]
    return real, linear

def efficacy(real, linear,
                based_on='integral'):

    import numpy as np

    if based_on=='peak':
        return 100.*np.max(real-real[0])/np.max(linear-linear[0])
    elif based_on=='integral':
        return 100.*np.sum(real-real[0])/np.sum(linear-linear[0])

def run_sim(cellType='Basket', 
            iBranch=0,
            passive_only=True,
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

    synapses = find_clustered_input(cell, 
                                    iBranch,
                                    from_uniform=from_uniform,
                                    synSubsamplingFraction=synSubsamplingFraction,
                                    synSubsamplingSeed=synSubsamplingSeed,
                                    distance_intervals=distance_intervals,
                                    iDistance=iDistance,
                                    distance=distance,
                                    nCluster=nCluster)

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

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell,\
                                            synapses, 
                                            EI_ratio=0,
                                            boost_AMPA_for_SST_noNMDA=False,
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


    # -- spike count
    apc = h.APCount(cell.soma[0](0.5))

    ######################################################
    ##   simulation run   ################################
    ######################################################

    h.finitialize(-70)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS, GABAS = None, None, None
    ampaNETCONS, nmdaNETCONS, gabaNETCONS = None, None, None
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
              'dt': dt, 't0':t0, 'ISI':ISI, 'interspike':interspike,
              'distance_intervals':distance_intervals,
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
    parser.add_argument("--Proximal", type=float, nargs=2, default=(20,60))
    parser.add_argument("--Distal", type=float, nargs=2, default=(160,200))
    parser.add_argument("--sparsening", help="in percent", type=float, 
                        default=[5], nargs='*')
    parser.add_argument("--interspike", type=float, default=0.5)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=0)
    parser.add_argument("--nBranch", type=int, default=6)
    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")

    parser.add_argument("-wVm", "--with_Vm", help="store Vm", action="store_true")
    parser.add_argument("--suffix", help="suffix for saving", default='')
    parser.add_argument('-fmo', "--fix_missing_only", help="in scan", action="store_true")

    parser.add_argument("-t", "--test", help="test func", action="store_true")
    args = parser.parse_args()

    distance_intervals = [args.Proximal, args.Distal]

    params = dict(cellType=args.cellType,
                  iBranch=args.iBranch,
                  iDistance=1,
                  synSubsamplingFraction=args.sparsening[0]/100.,
                  interspike=args.interspike,
                  distance_intervals=distance_intervals)

    if args.test:

        run_sim(**params)

    else:

        sim = Parallel(\
            filename='../../data/detailed_model/clusterStim_sim%s_%s.zip' % (args.suffix, args.cellType))

        grid = dict(iBranch=np.arange(args.nBranch),
                    iDistance=range(2),
                    synSubsamplingFraction=[s/100. for s in args.sparsening])

        if args.test_uniform:
            grid = dict(from_uniform=[False, True], **grid)

        if args.test_NMDA:
            grid = dict(with_NMDA=[False, True], **grid)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}),
                fix_missing_only=args.fix_missing_only)
