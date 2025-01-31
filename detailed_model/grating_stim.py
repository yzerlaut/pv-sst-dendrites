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
            stimFreq=1,
            stepAmpFactor=4.,
            spikeSeed=2,
            with_STP=False,
            # biophysical props
            with_NMDA=False,
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=False,
            spike_threshold=0.,
            dt= 0.01):

    from cell_template import Cell, h, np
    from synaptic_input import add_synaptic_input, PoissonSpikeTrain,\
                                    STP_release_filter
    from scipy.special import erf

    tstop = 7e3 # -2s:5s

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
            STP_model = {'P0':0.30, 'P1':0.30, 'dP':0.00, 'tauP':1.0, 'Nmax':1}
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
                                            Nmax_release=STP_model['Nmax'],
                                            with_NMDA=with_NMDA)

    t = np.arange(int(tstop/dt))*dt


    # Grating Stim Function:
    def sigmoid(x, width=0.1):
        return (1+erf(x/width))/2.

    def signal(x, t1=0, t2=0, t3=0, t4=0, w1=0, w2=0, w3=0, w4=0):
        y = sigmoid(x-t1, w1)*sigmoid(-(x-t2), w2)+\
                0.35*(sigmoid(x-t3, w3)*sigmoid(-(x-t4), w4))
        return y/y.max()

    P = np.load('../data/detailed_model/grating-stim-input-params.npy', allow_pickle=True).item()
    Stim = stimFreq*(1+stepAmpFactor*signal(t*1e-3-2, **P))

    # -- background activity 
    np.random.seed(spikeSeed)
    TRAINS = [[] for s in range(len(synapses)*STP_model['Nmax'])]
    for i, syn in enumerate(synapses):
        if excitatory[i]:
            # we draw one spike train:
            train_s = np.array(PoissonSpikeTrain(Stim,
                                    dt=1e-3*dt, tstop=1e-3*tstop)) # Hz,s
            # STP only in excitatory
            N = STP_release_filter(train_s, **STP_model)
            for n in range(1, STP_model['Nmax']+1):
                # we split according to release number ++ train to ** ms **
                TRAINS[i+len(synapses)*(n-1)] += list(1e3*train_s[N==n]) 
        else:
            # GABA -> only single release
            train_s = np.array(PoissonSpikeTrain(Stim,
                                    dt=1e-3*dt, tstop=1e-3*tstop)) # Hz,s
            TRAINS[i] += list(1e3*train_s) # to ** ms **

    # -- reordering spike trains
    for i in range(len(TRAINS)):
        TRAINS[i] = np.sort(TRAINS[i])

    # -- link events to synapses
    for i in range(len(TRAINS)):
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
    output = {'Stim':Stim,
              'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
              'dt': dt, 
              'synapses':synapses,
              'stimFreq':stimFreq,
              'stepAmpFactor':stepAmpFactor,
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

    args = parser.parse_args()
     
    params = dict(cellType=args.cellType,
                  passive_only=args.passive,
                  spikeSeed=args.spikeSeed,
                  stimFreq=args.stimFreq,
                  iBranch=args.iBranch,
                  with_presynaptic_spikes=\
                          args.with_presynaptic_spikes,
                  with_NMDA=args.with_NMDA,
                  with_STP=args.with_STP,
                  no_Vm=args.no_Vm,
                  dt=args.dt)

    if args.test:

        # run with the given params as a test
        print('running test simulation [...]')
        params['filename']=args.filename
        run_sim(**params)

    elif args.test_with_repeats:

        sim = Parallel(\
            filename='../data/detailed_model/GratingStim_demo_%s%s.zip' %\
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

        for i in range(args.nBranch):

            params['iBranch'] = i
            
            sim = Parallel(\
                filename='../data/detailed_model/GratingStim_sim_iBranch%i_%s_%s.zip' %\
                                        (i, args.cellType, args.suffix))

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
