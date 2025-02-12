import numpy as np
from parallel import Parallel

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt

P = dict(t1=0.2, t2=0.45, t3=0.75, t4=2.1,
         #Amp=0.25,
         w1=0.08, w2=0.3, w3=0.2, w4=0.2)

from scipy.special import erf
# Grating Stim Function:
def sigmoid(x, width=0.1):
    return (1+erf(x/width))/2.

def func(x, t1=0, t2=0, t3=0, t4=0, w1=0, w2=0, w3=0, w4=0, Amp=0):
    y = sigmoid(x-t1, w1)*sigmoid(-(x-t2), w2)+\
            Amp*(sigmoid(x-t3, w3)*sigmoid(-(x-t4), w4))
    return y/y.max()

def input_signal(x, Amp=0.25, P=P):
    return func(x, Amp=Amp, **P)

def run_sim(cellType='Basket', 
            passive_only=False,
            iBranch=0,
            from_uniform=False,
            # stim props
            stimFreq=1,
            stepAmpFactor=1.,
            ampLongLasting=0.25,
            Inh_fraction=20./100.,
            synapse_subsampling=1,
            spikeSeed=2,
            currentDrive=0, # in nA
            # biophysical props
            with_STP=False,
            with_NMDA=False,
            filename='single_sim.npy',
            with_presynaptic_spikes=False,
            no_Vm=False,
            spike_threshold=0.,
            t0=100, # ms
            dt= 0.01):

    from cell_template import Cell, h, np, ms
    from synaptic_input import add_synaptic_input, PoissonSpikeTrain,\
                                    STP_release_filter
    from scipy.special import erf
    from grating_stim import input_signal 

    tstop = t0+2.5e3

    trialSeed = int((70+spikeSeed) * (\
            ( stimFreq*stepAmpFactor*(iBranch**2+1) ) ) )%1000000 

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

    if currentDrive>0:
        ic = h.IClamp(cell.soma[0](0.5))
        ic.amp = currentDrive
        ic.dur =  1e9 * ms
        ic.delay = 0 * ms

    if from_uniform:
        synapses = cell.set_of_synapses_spatially_uniform[iBranch]
                        # for iBranch in range(6)])
    else:
        synapses = cell.set_of_synapses[iBranch]
    synapses = synapses[::synapse_subsampling]

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell, synapses, 
                                            Nmax_release=STP_model['Nmax'],
                                            Inh_fraction=Inh_fraction,
                                            with_NMDA=with_NMDA,
                                            seed=trialSeed)

    t = np.arange(int(tstop/dt))*dt
    # Stim = stimFreq*(1+stepAmpFactor*input_signal(t*1e-3-0.5))
    Stim = stimFreq*stepAmpFactor*\
                input_signal((t-t0)*1e-3, Amp=ampLongLasting)

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

    # save the output
    Vm = np.array(Vm_soma)
    output = {'Stim':Stim,
              'spikes':dt*\
                np.argwhere((Vm[1:]>spike_threshold) &\
                                    (Vm[:-1]<=spike_threshold)),
              'dt': dt, 
              'synapses':synapses,
              'Inh_fraction':Inh_fraction,
              'stimFreq':stimFreq,
              'stepAmpFactor':stepAmpFactor,
              'ampLongLasting':ampLongLasting,
              't0':t0,
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
    
    # stim props
    parser.add_argument("--Inh_fraction", type=float, 
                        nargs='*', default=[20./100.])
    parser.add_argument("--synapse_subsampling", type=int, 
                        nargs='*', default=[1])
    parser.add_argument("--stimFreq", type=float, 
                        nargs='*', default=[1.0])
    parser.add_argument("--stepAmpFactor", type=float, 
                        nargs='*', default=[4.])
    parser.add_argument("--currentDrive", type=float, 
                        nargs='*', default=[0])
    parser.add_argument("--ampLongLasting", type=float, 
                        nargs='*', default=[0.4])
    parser.add_argument("--spikeSeed", type=int, default=1)
    parser.add_argument("--nSpikeSeed", type=int, default=0)

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
                  stepAmpFactor=args.stepAmpFactor[0],
                  ampLongLasting=args.ampLongLasting[0],
                  synapse_subsampling=args.synapse_subsampling[0],
                  currentDrive=args.currentDrive[0],
                  iBranch=args.iBranch,
                  with_presynaptic_spikes=\
                          args.with_presynaptic_spikes,
                  with_NMDA=args.with_NMDA,
                  with_STP=args.with_STP,
                  from_uniform=args.from_uniform,
                  no_Vm=args.no_Vm,
                  dt=args.dt)

    if args.test:

        # run with the given params as a test
        print('running test simulation [...]')
        params['filename']=args.filename
        run_sim(**params)

    elif args.test_with_repeats:

        sim = Parallel(\
            filename='../data/detailed_model/GratingSim_demo_%s%s.zip' %\
                                    (args.cellType, args.suffix))

        grid = dict(spikeSeed=np.arange(args.nSpikeSeed))
        for key in ['synapse_subsampling', 'Inh_fraction', 
                    'stimFreq', 'stepAmpFactor', 
                    'ampLongLasting', 'currentDrive']:
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
                filename='../data/detailed_model/GratingSim_%s%s_branch%i.zip' %\
                                    (args.cellType, args.suffix, iBranch))

            grid = dict(spikeSeed=np.arange(args.nSpikeSeed))
            for key in ['synapse_subsampling', 'Inh_fraction', 
                        'stimFreq', 'stepAmpFactor', 
                        'ampLongLasting', 'currentDrive']:
                if len(getattr(args, key))>1:
                    grid[key] = getattr(args, key)

            sim.build(grid)

            sim.run(run_sim,
                    single_run_args=\
                        dict({k:v for k,v in params.items() if k not in grid}),
                    fix_missing_only=args.fix_missing_only)
