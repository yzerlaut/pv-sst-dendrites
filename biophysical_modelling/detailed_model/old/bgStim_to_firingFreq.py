from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


def run_sim(cellType='Basket', 
            iBranch=0,
            # bg Stim props
            bgStimFreq = 1e-3, 
            bgStimSeed = 10, 
            # spread synapses uniformly:
            from_uniform=False,
            # biophysical props
            with_NMDA=False,
            # sim props
            filename='single_sim.npy',
            with_Vm=False,
            dt= 0.025,
            tstop = 1000):

    # need to import module for the parallelization
    from cell_template import Cell, h
    from synaptic_input import PoissonSpikeTrain
    import numpy as np

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

    # synaptic distribution
    if from_uniform:
        # uniform
        synapses = cell.set_of_synapses_spatially_uniform[iBranch]
    else:
        # real 
        synapses = cell.set_of_synapses[iBranch]

    # prepare presynaptic spike trains
    np.random.seed(10+8**bgStimSeed)
    TRAINS = []
    for i, syn in enumerate(synapses):
        TRAINS.append(PoissonSpikeTrain(bgStimFreq, tstop=tstop))

    ######################################################
    ##   true simulation         #########################
    ######################################################
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

        if with_NMDA:
            NMDAS.append(\
                    h.NMDAIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

        VECSTIMS.append(h.VecStim())
        STIMS.append(h.Vector(TRAINS[i]))

        VECSTIMS[-1].play(STIMS[-1])

        ampaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
        ampaNETCONS[-1].weight[0] = cell.params['%s_qAMPA'%params_key]

        if with_NMDA:
            nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], NMDAS[-1]))
            nmdaNETCONS[-1].weight[0] = cell.params['%s_NAR'%params_key]*\
                                            cell.params['%s_qAMPA'%params_key]

    t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
    Vm_soma, Vm_dend = h.Vector(), h.Vector()

    # Vm rec @ soma
    Vm_soma.record(cell.soma[0](0.5)._ref_v)
    # Vm rec @ dend
    syn = cell.set_of_branches[iBranch][-5] # 
    Vm_dend.record(cell.SEGMENTS['NEURON_section'][syn](0.5)._ref_v)

    # spike count
    apc = h.APCount(cell.soma[0](0.5))
    apc.thresh = -35

    # run
    h.finitialize(-70)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS = None, None
    ampaNETCONS, nmdaNETCONS = None, None
    STIMS, VECSTIMS = None, None

    # save the output
    output = {'output_rate': float(apc.n*1e3/tstop),
              'dt': dt,
              'tstop':tstop}

    if with_Vm:
        output['Vm_soma'] = np.array(Vm_soma)
        output['Vm_dend'] = np.array(Vm_dend)

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
    
    # Input Range
    parser.add_argument("--Fmin", help="min input", type=float, default=2e-4)
    parser.add_argument("--Fmax", help="max input", type=float, default=1.1e-2)
    parser.add_argument("--nF", help="N input", type=int, default=10)
    parser.add_argument("--logF", help="test func", action="store_true")

    # Input Seed Variations
    parser.add_argument("--nSeed", type=int, default=1)

    # Branch number
    parser.add_argument("--iBranch", type=int, default=0)
    parser.add_argument("--nBranch", type=int, default=6)

    # Testing Conditions
    parser.add_argument("--test_uniform", action="store_true")
    parser.add_argument("--test_NMDA", action="store_true")

    # sim props
    parser.add_argument("--tstop", type=float, default=1000.)
    parser.add_argument("-t", "--test", help="test func", action="store_true")

    args = parser.parse_args()

    # default params
    params = dict(cellType=args.cellType,
                  iBranch=0,
                  bgStimFreq=args.Fmin,
                  bgStimSeed=1,
                  tstop=args.tstop,
                  with_Vm=True)

    if args.test:
        run_sim(**params)
    else:
        sim = Parallel(\
            filename='../../data/detailed_model/%s_bgStim_sim.zip' % args.cellType)

        if args.logF:
            F = np.logspace(np.log10(args.Fmin), np.log10(args.Fmax), args.nF)
        else:
            F = np.linspace(args.Fmin, args.Fmax, args.nF)
           
        grid = dict(bgStimFreq=F,
                    iBranch=np.arange(args.nBranch),
                    bgStimSeed=1+np.arange(args.nSeed)*5)

        if args.test_uniform:
            grid= dict(from_uniform=[False, True], **grid)
        if args.test_NMDA:
            grid = dict(with_NMDA=[False, True], **grid)

        sim.build(grid)

        sim.run(run_sim,
                single_run_args=\
                    dict({k:v for k,v in params.items() if k not in grid}))
