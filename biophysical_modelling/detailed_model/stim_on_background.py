from cell_template import *
from synaptic_input import add_synaptic_input
from synaptic_input import PoissonSpikeTrain as train
from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

def run_sim(cellType='Basket', 
            passive_only=False,
            iBranch=0,
            from_uniform=False,
            # stim props
            nStimRepeat=2,
            stimSeed=3,
            nCluster=[10],
            interspike=2,
            t0=200,
            ISI=300,
            # bg stim
            bgStimFreq=1e-3,
            bgFreqInhFactor=4.,
            bgStimSeed=1,
            # biophysical props
            with_NMDA=False,
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
    cell = Cell(ID=ID, 
                passive_only=passive_only,
                params_key=params_key)
    cell.params_key = params_key

    if from_uniform:
        synapses = cell.set_of_synapses_spatially_uniform[iBranch]
    else:
        synapses = cell.set_of_synapses[iBranch]                      

    tstop=t0+nStimRepeat*ISI*len(nCluster)

    # build synaptic input
    AMPAS, NMDAS, GABAS,\
       ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
        STIMS, VECSTIMS, excitatory = add_synaptic_input(cell, synapses, 
                                                         with_NMDA=with_NMDA)

    # prepare presynaptic spike trains
    # -- background activity 
    np.random.seed(10+8**bgStimSeed)
    TRAINS = []
    for i, syn in enumerate(synapses):
        if excitatory[i]:
            TRAINS.append(list(train(bgStimFreq, tstop=tstop)))
        else:
            TRAINS.append(list(train(bgFreqInhFactor*bgStimFreq, tstop=tstop)))

    # -- stim evoked activity 
    np.random.seed(stimSeed)
    for n in range(nStimRepeat):
        for c, nC in enumerate(nCluster):
            picked = np.random.choice(synapses[excitatory],
                                      nC) # in stim in excitatory syn.
            for i, syn in enumerate(picked):
                TRAINS[np.flatnonzero(synapses==syn)[0]].append(t0+\
                                                        n*len(nCluster)*ISI+\
                                                        c*ISI+\
                                                        i*interspike)
    # -- reoardering spike trains
    for i, syn in enumerate(synapses):
        TRAINS[i] = np.sort(TRAINS[i])



    for i, syn in enumerate(synapses):

        STIMS.append(h.Vector(TRAINS[i]))
        VECSTIMS[i].play(STIMS[-1])

    Vm_soma, Vm_dend = h.Vector(), h.Vector()

    # Vm rec
    Vm_soma.record(cell.soma[0](0.5)._ref_v)
    Vm_dend.record(cell.SEGMENTS['NEURON_section'][synapses[-1]](0.5)._ref_v)

    # spike count
    apc = h.APCount(cell.soma[0](0.5))

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
    output = {'output_rate': float(apc.n*1e3/tstop),
              'Vm_soma': np.array(Vm_soma),
              'Vm_dend': np.array(Vm_dend),
              'nCluster':nCluster,
              'nStimRepeat':nStimRepeat,
              'dt': dt, 't0':t0, 'ISI':ISI,
              'interspike':interspike,
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
    
    # bg props
    parser.add_argument("--bgStimFreq", type=float, default=1e-3)
    parser.add_argument("--bgFreqInhFactor", type=float, default=4.)
    parser.add_argument("--bgStimSeed", type=float, default=1)

    # stim props
    parser.add_argument("--nStimRepeat", type=int, default=2)
    parser.add_argument("--interspike", type=float, default=0.5)
    parser.add_argument("--ISI", type=float, default=200)
    parser.add_argument("--nCluster", type=int, nargs='*', 
                        default=np.arange(20)*5)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=0)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")
    parser.add_argument("--passive", action="store_true")

    parser.add_argument("--suffix", help="suffix for saving", default='')

    parser.add_argument("-t", "--test", help="test func", action="store_true")
    args = parser.parse_args()

    params = dict(cellType=args.cellType,
                  passive_only=args.passive,
                  bgFreqInhFactor=args.bgFreqInhFactor,
                  iBranch=args.iBranch,
                  ISI=args.ISI,
                  bgStimFreq=args.bgStimFreq,
                  bgStimSeed=args.bgStimSeed,
                  nStimRepeat=args.nStimRepeat,
                  nCluster=args.nCluster,
                  interspike=args.interspike)

    if args.test:

        # run with the given params as a test
        run_sim(**params)

    else:
   
        # run the simulation with parameter variations

        sim = Parallel(\
            filename='../../data/detailed_model/%s_StimOnBg_sim%s.zip' % (args.cellType,
                                                                          args.suffix))

        grid = dict(iBranch=np.arange(args.nBranch))

        if args.test_uniform:
            grid = dict(from_uniform=[False, True], **grid)
        if args.test_NMDA:
            grid = dict(with_NMDA=[False, True], **grid)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}))
