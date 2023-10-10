from cell_template import *
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
    if based_on=='peak':
        return 100.*np.max(real-real[0])/np.max(linear-linear[0])
    elif based_on=='integral':
        return 100.*np.sum(real-real[0])/np.sum(linear-linear[0])

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
    # parser.add_argument("--Medial", type=float, nargs=2, default=(90,130))
    parser.add_argument("--Distal", type=float, nargs=2, default=(160,200))
    parser.add_argument("--synSubsamplingFraction", type=float, default=5e-2)
    parser.add_argument("--sparsening", help="in percent", type=float, default=[5], nargs='*')
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
                      synSubsamplingFraction=args.sparsening[0]/100.,
                      interspike=args.interspike,
                      distance_intervals=distance_intervals)
        run_sim(**params)

    else:

        sim = Parallel(\
            filename='../../data/detailed_model/%s_clusterStim_sim%s.zip' % (args.cellType,
                                                                             args.suffix))

        single_run_args=dict(cellType=args.cellType,
                             interspike=args.interspike,
                             distance_intervals=distance_intervals)

        params = dict(iBranch=np.arange(args.nBranch),
                      synSubsamplingFraction=[s/100. for s in args.sparsening],
                      iDistance=range(2))

        if args.test_uniform:
            params = dict(from_uniform=[False, True], **params)
        if args.test_NMDA:
            params = dict(NMDAtoAMPA_ratio=[0., args.NMDAtoAMPA_ratio], **params)

        sim.build(params)

        sim.run(run_sim,
                single_run_args=single_run_args)
