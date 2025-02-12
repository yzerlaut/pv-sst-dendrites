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
            stimFreq=1e-3,
            interstim=500.,
            stepWidth=50.,
            stepAmpFactor=2.,
            Inh_fraction=20./100.,
            synapse_subsampling=1,
            # bg stim
            spikeSeed=2,
            # bgFreqInhFactor=4.,
            # biophysical props
            with_STP=False,
            with_NMDA=False,
            AMPAboost=0,
            currentDrive=0, # in nA
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=True,
            spike_threshold=0.,
            dt= 0.01):

    from cell_template import Cell, h, np, ms
    from synaptic_input import add_synaptic_input, PoissonSpikeTrain,\
                                    STP_release_filter

    tstop = 2*interstim+stepWidth

    trialSeed = int(spikeSeed * (\
            ( stimFreq*stepWidth*stepAmpFactor*(iBranch+1) ) ) )%1000000 

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

    # add optional modified AMPA boost:
    if AMPAboost>0:
        cell.params['%s_qAMPAonlyBoost'%cell.params_key] = AMPAboost

    if currentDrive>0:
        ic = h.IClamp(cell.soma[0](0.5))
        ic.amp = currentDrive
        ic.dur =  1e9 * ms
        ic.delay = 0 * ms

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
                                            with_NMDA=with_NMDA,
                                            seed=trialSeed)

    # Step Function
    t = np.arange(int(tstop/dt))*dt
    Stim = np.ones(len(t))*stimFreq
    # +step 
    cond = (t>interstim) & (t<(interstim+stepWidth))
    Stim[cond] *= stepAmpFactor

    # -- synaptic activity 
    TRAINS = [[] for s in range(len(synapses)*STP_model['Nmax'])]
    for i, syn in enumerate(synapses):
        if excitatory[i]:
            # we draw one spike train:
            train_s = np.array(PoissonSpikeTrain(Stim,
                                    dt=1e-3*dt, tstop=1e-3*tstop,
                                    seed=trialSeed+1000+i)) # Hz,s
            # STP only in excitatory
            N = STP_release_filter(train_s, 
                                   seed=trialSeed+2000+i,
                                   **STP_model)
            for n in range(1, STP_model['Nmax']+1):
                # we split according to release number ++ train to ** ms **
                TRAINS[i+len(synapses)*(n-1)] += list(1e3*train_s[N==n]) 
        else:
            # GABA -> only single release
            # train_s = np.array(PoissonSpikeTrain(bgFreqInhFactor*Stim, # REMOVED
            train_s = np.array(PoissonSpikeTrain(Stim, 
                                    dt=1e-3*dt, tstop=1e-3*tstop,
                                    seed=trialSeed+3000+i)) # Hz,s
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

    Vm = np.array(Vm_soma)
    # save the output
    output = {'Stim':Stim,
              'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
              'dt': dt, 
              'synapses':synapses,
              'Inh_fraction':Inh_fraction,
              'stimFreq':stimFreq,
              'stepWidth':stepWidth, 
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
                        """, default='Martinotti')
    
    # stim props
    parser.add_argument("--Inh_fraction", type=float, 
                        nargs='*', default=[20./100.])
    parser.add_argument("--synapse_subsampling", type=int, 
                        nargs='*', default=[1])
    parser.add_argument("--stimFreq", type=float, 
                        nargs='*', default=[1.0])
    parser.add_argument("--stepWidth", type=float, 
                        nargs='*', default=[200.])
    parser.add_argument("--stepAmpFactor", type=float, 
                        nargs='*', default=[3.])
    parser.add_argument("--AMPAboost", type=float, 
                        nargs='*', default=[0])
    parser.add_argument("--currentDrive", type=float, 
                        nargs='*', default=[0])
    parser.add_argument("--spikeSeed", type=int, default=1)
    parser.add_argument("--nSpikeSeed", type=int, default=0)
    parser.add_argument("--interstim", type=float, default=500)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=5)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--with_NMDA", action="store_true")
    parser.add_argument("--from_uniform", action="store_true")
    parser.add_argument("--with_STP", action="store_true")

    parser.add_argument('-f', "--filename", default='single_sim.npy')
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
                  Inh_fraction=args.Inh_fraction[0],
                  stimFreq=args.stimFreq[0],
                  stepWidth=args.stepWidth[0], 
                  stepAmpFactor=args.stepAmpFactor[0],
                  synapse_subsampling=args.synapse_subsampling[0],
                  AMPAboost=args.AMPAboost[0],
                  currentDrive=args.currentDrive[0],
                  iBranch=args.iBranch,
                  with_presynaptic_spikes=\
                          args.with_presynaptic_spikes,
                  with_NMDA=args.with_NMDA,
                  with_STP=args.with_STP,
                  from_uniform=args.from_uniform,
                  interstim=args.interstim,
                  no_Vm=args.no_Vm,
                  dt=args.dt)

    if args.test:

        # run with the given params as a test
        print('running test simulation [...]')
        params['filename']=args.filename
        run_sim(**params)

    elif args.test_with_repeats:

        sim = Parallel(\
            filename='../data/detailed_model/StepSim_demo_%s%s.zip' %\
                                    (args.cellType, args.suffix))

        grid = dict(spikeSeed=np.arange(args.nSpikeSeed))
        for key in ['synapse_subsampling', 'Inh_fraction', 'stimFreq',
                    'stepWidth', 'stepAmpFactor', 'currentDrive']:
            if len(getattr(args, key))>1:
                grid[key] = getattr(args, key)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}),
                fix_missing_only=args.fix_missing_only)

    else:
   
        # run the simulation with parameter variations

        for iBranch in range(args.nBranch):

            params['iBranch'] = iBranch

            sim = Parallel(\
                filename='../data/detailed_model/StepSim_%s%s_branch%i.zip' %\
                                    (args.cellType, args.suffix, iBranch))

            grid = dict(spikeSeed=np.arange(args.nSpikeSeed))
            for key in ['synapse_subsampling', 'Inh_fraction', 'stimFreq',
                        'stepWidth', 'stepAmpFactor', 
                        'AMPAboost', 'currentDrive']:
                if len(getattr(args, key))>1:
                    grid[key] = getattr(args, key)

            sim.build(grid)

            sim.run(run_sim,
                    single_run_args=\
                        dict({k:v for k,v in params.items() if k not in grid}),
                    fix_missing_only=args.fix_missing_only)
