import numpy as np
from parallel import Parallel

import sys, os
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


def run_sim(cellType='Basket', 
            passive_only=False,
            iBranch=0,
            from_uniform=False,
            # stim props
            with_STP=False,
            Inh_fraction=15./100.,
            synapse_subsampling=5,
            spikeSeed=2,
            # biophysical props
            with_NMDA=False,
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=False,
            spike_threshold=0.,
            tstop=0.,
            dt= 0.01):

    from cell_template import Cell, h, np, os
    from synaptic_input import add_synaptic_input, PoissonSpikeTrain,\
                                    STP_release_filter
    h.dt = dt

    ########################################################
    # Import Natural Movie Spiking Activity
    ########################################################
    RATES = np.load(os.path.join('..', 'data', 'visual_coding', 'avRATES_natural_movie_one.npy'),
                    allow_pickle=True).item()
    t = 1e3*(RATES['time']-RATES['time'][0]) # s to ms
    if tstop<=0.:
        tstop = t[-1]
    # average of PV- and SST- units:
    neg_rates = 0.5*(RATES['PV_negUnits']+RATES['SST_negUnits'])


    ######################################################
    ##   simulation preparation  #########################
    ######################################################

    STP_model = {'P0':1.00, 'P1':1.00, 'dP':0.00, 'tauP':1.0, 'Nmax':1}

    # create cell
    if cellType=='Basket':
        ID = '864691135396580129_296758' # Basket Cell example
        params_key='BC'
        if with_STP:
            STP_model = np.load('../data/detailed_model/PV_stp.npy',
                                allow_pickle=True).item()
        else:
            STP_model = {'P0':0.80, 'P1':0.80, 'dP':0.00, 'tauP':1.0, 'Nmax':1}
    elif cellType=='Martinotti':
        ID = '864691135571546917_264824' # Martinotti Cell example
        params_key='MC'
        if with_STP:
            STP_model = np.load('../data/detailed_model/SST_stp.npy',
                                allow_pickle=True).item()
        else:
            STP_model = {'P0':0.60, 'P1':0.60, 'dP':0.00, 'tauP':1.0, 'Nmax':1}
    else:
        raise Exception(' cell type not recognized  !')

    cell = Cell(ID=ID, 
                passive_only=passive_only,
                params_key=params_key)

    if from_uniform:
        synapses = cell.set_of_synapses_spatially_uniform[iBranch]
        # synapses = np.concatenate([cell.set_of_synapses_spatially_uniform[iBranch]\
                        # for iBranch in range(6)])
    else:
        synapses = cell.set_of_synapses[iBranch]
        # synapses = [cell.set_of_synapses[iBranch]\
                        # for iBranch in range(6)]
    synapses = synapses[::synapse_subsampling]

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell, synapses, 
                                                 Nmax_release=STP_model['Nmax'],
                                                 Inh_fraction=Inh_fraction,
                                                 with_NMDA=with_NMDA)

    # Time-Varying Rate 
    Rate = np.interp(np.arange(0, tstop, dt), 
                                t[t<=tstop],
                                  neg_rates[t<=tstop])

    # -- background activity 
    np.random.seed(spikeSeed)
    TRAINS = [[] for s in range(len(synapses)*STP_model['Nmax'])]
    for i, syn in enumerate(synapses):
        if excitatory[i]:
            # we draw one spike train:
            train_s = np.array(PoissonSpikeTrain(Rate, 
                                    dt=1e-3*dt, tstop=1e-3*tstop)) # Hz,s
            # STP only in excitatory
            N = STP_release_filter(train_s, **STP_model)
            for n in range(1, STP_model['Nmax']+1):
                # we split according to release number ++ train to ** ms **
                TRAINS[i+len(synapses)*(n-1)] += list(1e3*train_s[N==n]) 
        else:
            # GABA -> only single release
            train_s = np.array(PoissonSpikeTrain(Rate, 
                                    dt=1e-3*dt, tstop=1e-3*tstop)) # Hz,s
            TRAINS[i] += list(1e3*train_s) # to ** ms **

    # -- reordering spike trains
    for i in range(len(TRAINS)):
        TRAINS[i] = np.sort(TRAINS[i])

    # -- link events to synapses
    for i in range(len(TRAINS)):
        # print(i, i%len(synapses), excitatory[i%len(synapses)], len(TRAINS[i]))
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
    output = {'Rate':Rate,
              'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
               'Inh_fraction':Inh_fraction,
               'synapse_subsampling':synapse_subsampling,
               'synapses':synapses,
               'dt': dt, 
               'tstop':tstop}

    if not no_Vm:
        output['Vm_soma'] = Vm

    if with_presynaptic_spikes:

        # inhibitory
        output['presynaptic_inh_events'] = [TRAINS[i]\
                for i in range(len(excitatory)) if not excitatory[i]]

        # excitatory, need to loop over N
        output['presynaptic_exc_events'] = [[]\
                for i in range(len(synapses)) if excitatory[i]]
        for e, i in enumerate(np.arange(len(synapses))[excitatory]):
            for n in range(STP_model['Nmax']):
                for k in range(n+1):
                    output['presynaptic_exc_events'][e] +=\
                                        list(TRAINS[i+n*len(synapses)])

    np.save(filename, output)


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
    
    # bg stim props
    parser.add_argument("--Inh_fraction", type=float, 
                        nargs='*', default=[15./100.])
    parser.add_argument("--synapse_subsampling", type=int, 
                        nargs='*', default=[5])
    parser.add_argument("--nStochProcSeed", type=int, default=2)
    parser.add_argument("--spikeSeed", type=int, default=1)
    parser.add_argument("--nSpikeSeed", type=int, default=8)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=2)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--with_NMDA", action="store_true")
    parser.add_argument("--from_uniform", action="store_true")
    parser.add_argument("--with_STP", action="store_true")

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
    parser.add_argument("--no_Vm", action="store_true")

    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tstop", help='in ms, switch to 0 for full sim.',
                        type=float, default=1000.)

    args = parser.parse_args()
     
    params = dict(cellType=args.cellType,
                  passive_only=args.passive,
                  spikeSeed=args.spikeSeed,
                  Inh_fraction=args.Inh_fraction[0],
                  synapse_subsampling=args.synapse_subsampling[0],
                  iBranch=args.iBranch,
                  with_presynaptic_spikes=\
                          args.with_presynaptic_spikes,
                  with_NMDA=args.with_NMDA,
                  from_uniform=args.from_uniform,
                  with_STP=args.with_STP,
                  no_Vm=args.no_Vm,
                  tstop=args.tstop,
                  dt=args.dt)

    if args.test:

        # run with the given params as a test
        print('running test simulation [...]')
        params['filename']=args.filename
        run_sim(**params)

    elif args.test_with_repeats:

        sim = Parallel(\
            filename='../data/detailed_model/natMovieStim_demo_%s%s.zip' %\
                                    (args.cellType, args.suffix))

        grid = dict(spikeSeed=np.arange(args.nSpikeSeed))

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}),
                fix_missing_only=args.fix_missing_only)

    else:
   
        # run the simulation with parameter variations

        for b in range(args.nBranch):

            params['iBranch'] = b

            print('\n running: natMovieStim_simBranch%i_%s.zip \n' %\
                                (b, args.cellType+args.suffix))
            
            sim = Parallel(\
                filename='../data/detailed_model/natMovieStim_simBranch%i_%s.zip' %\
                                (b, args.cellType+args.suffix))

            grid = dict(spikeSeed=np.arange(args.nSpikeSeed),
                        Inh_fraction=args.Inh_fraction,
                        synapse_subsampling=args.synapse_subsampling)

            sim.build(grid)

            sim.run(run_sim,
                    single_run_args=\
                        dict({k:v for k,v in params.items() if k not in grid}),
                    fix_missing_only=args.fix_missing_only)
