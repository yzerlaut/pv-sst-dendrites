import numpy as np
from parallel import Parallel

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


def run_sim(cellType='Basket', 
            passive_only=False,
            iBranch=0,
            from_uniform=False,
            # stim props
            meanStim=2.,
            stdStim=0.75,
            tauStim=50,
            # bg stim
            stimFreq=1e-3,
            bgFreqInhFactor=4.,
            stochProcSeed=1,
            spikeSeed=2,
            # biophysical props
            with_NMDA=False,
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=False,
            spike_threshold=0.,
            tstop=1000.,
            dt= 0.01):

    from cell_template import Cell, h, np
    from synaptic_input import add_synaptic_input, PoissonSpikeTrain
    h.dt = dt

    def OrnsteinUhlenbeck_Process(mu, sigma, tau, 
                                  dt=0.1, tstop=100):
        """ taken from github.com/yzerlaut/analyz """
        diffcoef = 2*sigma**2/tau
        y0, n_steps= mu, int(tstop/dt)
        A = np.sqrt(diffcoef*tau/2.*(1-np.exp(-2*dt/tau)))
        noise, y = np.random.randn(n_steps), np.zeros(n_steps)
        y[0] = y0
        for i in range(n_steps-1):
            y[i+1] = y0 + (y[i]-y0)*np.exp(-dt/tau)+A*noise[i]
        return y

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

    if from_uniform:
        synapses = cell.set_of_synapses_spatially_uniform[iBranch]
        # synapses = np.concatenate([cell.set_of_synapses_spatially_uniform[iBranch]\
                        # for iBranch in range(6)])
    else:
        synapses = cell.set_of_synapses[iBranch]
        # synapses = [cell.set_of_synapses[iBranch]\
                        # for iBranch in range(6)]

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell, synapses, 
                                                       with_NMDA=with_NMDA)

    # Ornstein-Uhlenbeck Time-Varying Rate (clipped to positive-values)
    np.random.seed(stochProcSeed)
    OU = np.clip(\
            OrnsteinUhlenbeck_Process(meanStim,
                                      stdStim, 
                                      tauStim,
                                      dt, tstop),
                0, np.inf)

    # -- background activity 
    np.random.seed(spikeSeed)
    TRAINS = []
    for i, syn in enumerate(synapses):
        if excitatory[i]:
            TRAINS.append(list(PoissonSpikeTrain(stimFreq*OU, 
                                                 dt=dt,
                                                 tstop=tstop)))
        else:
            TRAINS.append(list(PoissonSpikeTrain(\
                                    bgFreqInhFactor*stimFreq*OU,
                                                 dt=dt,
                                                 tstop=tstop)))

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
    output = {'OU':OU,
              'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
              'dt': dt, 
              'synapses':synapses,
              'stimFreq':stimFreq,
              'bgFreqInhFactor':bgFreqInhFactor,
              'tstop':tstop}

    if not no_Vm:
        output['Vm_soma'] = Vm

    if with_presynaptic_spikes:
        output['presynaptic_exc_events'] = [TRAINS[i]\
                for i in range(len(excitatory)) if excitatory[i]]
        output['presynaptic_inh_events'] = [TRAINS[i]\
                for i in range(len(excitatory)) if not excitatory[i]]

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
    parser.add_argument("--stimFreq", type=float, default=1e-2)
    parser.add_argument("--bgFreqInhFactor", type=float, default=1.)
    parser.add_argument("--stochProcSeed", type=int, default=1)
    parser.add_argument("--nStochProcSeed", type=int, default=2)
    parser.add_argument("--spikeSeed", type=int, default=1)
    parser.add_argument("--nSpikeSeed", type=int, default=8)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=2)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--with_NMDA", action="store_true")
    parser.add_argument("--from_uniform", action="store_true")

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
    parser.add_argument("--tstop", type=float, default=5000.)

    args = parser.parse_args()
     
    params = dict(cellType=args.cellType,
                  passive_only=args.passive,
                  stochProcSeed=args.stochProcSeed,
                  spikeSeed=args.spikeSeed,
                  stimFreq=args.stimFreq,
                  bgFreqInhFactor=args.bgFreqInhFactor,
                  iBranch=args.iBranch,
                  with_presynaptic_spikes=\
                          args.with_presynaptic_spikes,
                  with_NMDA=args.with_NMDA,
                  from_uniform=args.from_uniform,
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
            filename='../data/detailed_model/tvRateStim_demo_%s%s.zip' %\
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
                filename='../data/detailed_model/tvRateStim_simBranch%i_%s.zip' %\
                                (b, args.cellType+args.suffix))

            grid = dict(stochProcSeed=np.arange(args.nStochProcSeed),
                        spikeSeed=np.arange(args.nSpikeSeed))

            if args.test_uniform:
                grid = dict(from_uniform=[False, True], **grid)

            if args.test_NMDA:
                grid = dict(with_NMDA=[False, True], **grid)

            sim.build(grid)

            sim.run(run_sim,
                    single_run_args=\
                        dict({k:v for k,v in params.items() if k not in grid}),
                    fix_missing_only=args.fix_missing_only)
