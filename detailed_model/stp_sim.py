import numpy as np
from parallel import Parallel

import sys, os
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


def run_sim(cellType='Basket', 
            passive_only=True,
            iBranch=0,
            iSyn = 20,
            # stim props
            events = [0.1, 0.2, 0.3],
            seed=0,
            # biophysical props
            with_NMDA=False,
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=False,
            spike_threshold=0.,
            tstop=0.,
            dt= 0.01):

    from cell_template import Cell, h, np, os
    from synaptic_input import add_synaptic_input
    h.dt = dt

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

    synapses = [cell.set_of_synapses[iBranch][iSyn]]

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell, synapses, 
                                                         # release_proba_seed=seed,
                                                         Inh_fraction=0.,
                                                         with_NMDA=with_NMDA)


    # -- background activity 
    TRAINS = [events]

    # -- reordering spike trains
    for i, syn in enumerate(synapses):
        TRAINS[i] = np.sort(TRAINS[i])

    # -- link events to synapses
    for i, syn in enumerate(synapses):

        STIMS.append(h.Vector(TRAINS[i]))
        VECSTIMS[i].play(STIMS[-1])

    Vm_soma = h.Vector()

    # Vm rec
    Vm_soma.record(cell.soma[0](0.5)._ref_v)

    ######################################################
    ##   simulation run   ################################
    ######################################################

    h.finitialize(cell.El)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS, GABAS = None, None, None
    ampaNETCONS, nmdaNETCONS, gabaNETCONS = None, None, None
    STIMS, VECSTIMS = None, None

    # save the output
    Vm = np.array(Vm_soma)
    output = {'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
              'synapses':synapses,
              'Vm_soma':Vm,
              'dt': dt, 
              'tstop':tstop}

    np.save(filename, output)
    return output


if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="script description",
                                   formatter_class=\
                                           argparse.RawTextHelpFormatter)

    parser.add_argument('-c', "--cellType",\
                        help="""
                        cell type, either:
                        - Basket
                        - Martinotti
                        """, default='Basket')
    
    # Branch number
    parser.add_argument("--iBranch", type=int, default=2)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--with_NMDA", action="store_true")

    parser.add_argument("--filename", default='single_sim.npy')
    parser.add_argument("--suffix", help="suffix for saving", default='')
    parser.add_argument('-fmo', "--fix_missing_only",
                        help="in scan", action="store_true")

    parser.add_argument("-t", "--test", help="test func",
                        action="store_true")
    parser.add_argument("--test_with_repeats", help="test func",
                        action="store_true")
    parser.add_argument("-bg_valig", "--background_calibration", 
                        action="store_true")
    parser.add_argument("-wps", "--with_presynaptic_spikes", 
                        action="store_true")

    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tstop", type=float, default=1000.)

    args = parser.parse_args()
     
    params = dict(cellType=args.cellType,
                  iBranch=args.iBranch,
                  with_NMDA=args.with_NMDA,
                  tstop=args.tstop,
                  dt=args.dt)

    if args.test:

        # run with the given params as a test
        print('running test simulation [...]')
        import matplotlib.pylab as plt
        for seed in range(3):
            output = run_sim(events=np.cumsum(100*np.ones(9)),
                             seed=seed,
                             **params)
            plt.plot(np.arange(len(output['Vm_soma']))*output['dt'],\
                    output['Vm_soma'], 'k-')
        plt.show()


    elif args.test_with_repeats:

        sim = Parallel(\
            filename='../data/detailed_model/natMovieStim_demo_%s%s.zip' %\
                                    (args.cellType, args.suffix))

        grid = dict(spikeSeed=np.arange(args.nSpikeSeed))

        if args.test_uniform:
            grid = dict(from_uniform=[False, True], **grid)

        if args.test_NMDA:
            grid = dict(with_NMDA=[False, True], **grid)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}),
                fix_missing_only=args.fix_missing_only)

    else:
   
        # run the simulation with parameter variations

        for b in range(args.nBranch):

            params['iBranch'] = b
            
            sim = Parallel(\
                filename='../data/detailed_model/natMovieStim_simBranch%i_%s.zip' %\
                                (b, args.cellType+args.suffix))

            grid = dict(spikeSeed=np.arange(args.nSpikeSeed))

            if args.test_uniform:
                grid = dict(from_uniform=[False, True], **grid)

            if args.test_NMDA:
                grid = dict(with_NMDA=[False, True], **grid)

            sim.build(grid)

            sim.run(run_sim,
                    single_run_args=\
                        dict({k:v for k,v in params.items() if k not in grid}),
                    fix_missing_only=args.fix_missing_only)
