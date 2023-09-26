from PV_template import *

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt


def train(freq,
          tstop=1.):

    spikes = np.cumsum(\
            np.random.exponential(1./freq, int(2*tstop*freq)))

    return spikes[spikes<tstop]


def run_sim(cell, synapses,
            dt= 0.025,
            tstop = 1000,
            ampa_weight=1e-3, # uS
            NMDAtoAMPA_ratio=0,
            stim_freq = 0.1):


    AMPAS, NMDAS = [], []
    ampaNETCONS, nmdaNETCONS = [], []
    STIMS, VECSTIMS = [], []

    for syn in synapses:

        np.random.seed(syn)

        # need to avoid x=0 and x=1, to allow ion concentrations variations in NEURON
        x = np.clip(cell.SEGMENTS['NEURON_segment'][syn], 
                1, cell.SEGMENTS['NEURON_section'][syn].nseg-2)\
                        /cell.SEGMENTS['NEURON_section'][syn].nseg

        AMPAS.append(\
                h.CPGLUIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))
        NMDAS.append(\
                h.NMDAIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

        VECSTIMS.append(h.VecStim())
        STIMS.append(h.Vector(train(stim_freq,
                                    tstop=tstop)))

        VECSTIMS[-1].play(STIMS[-1])

        ampaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
        ampaNETCONS[-1].weight[0] = ampa_weight

        nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
        nmdaNETCONS[-1].weight[0] = ampa_weight*NMDAtoAMPA_ratio


    t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
    Vm = h.Vector()

    Vm.record(cell.soma[0](0.5)._ref_v)

    # run
    h.finitialize(cell.El)
    for i in range(int(tstop/dt)):
        h.fadvance()

    AMPAS, NMDAS = None, None
    ampaNETCONS, nmdaNETCONS = None, None
    STIMS, VECSTIMS = None, None

    return np.arange(len(Vm))*dt, np.array(Vm)


if __name__=='__main__':

    if sys.argv[-1]=='run':

        with_Vm = True

        ID = '864691135396580129_296758' # Basket Cell example
        cell = PVcell(ID=ID, debug=False)
        cell.check_that_all_dendritic_branches_are_well_covered()

        # spike count
        apc = h.APCount(cell.soma[0](0.5))

        # shuffle synapses
        cell.branches['synapses_shuffled'] = []
        for iBranch in range(len(cell.branches['branches'])):
            cell.branches['synapses_shuffled'].append(\
                    np.random.choice(cell.branches['branches'][iBranch],
                        len(cell.branches['synapses'][iBranch])))

        RESULTS = {'dt':0.025,
                   'branches':range(len(cell.branches['synapses']))[:2],
                   'freqs': np.logspace(-3.5, 0, 4),
                   'tstop':1000}

        for synapses in ['synapses', 'synapses_shuffled']:
            print(' o %s ' % synapses)
            for iBranch in RESULTS['branches']:
                print('     branch #%i ' % (1+iBranch))
                for freq in RESULTS['freqs']:
                    print('         freq=%.1e Hz' % freq)
                    apc.n = 0 # reset spike count
                    _, Vm = run_sim(cell, cell.branches[synapses][iBranch],
                                    dt=RESULTS['dt'],
                                    tstop=RESULTS['tstop'],
                                    stim_freq=freq)
                    if with_Vm:
                        RESULTS['Vm_%s_%i_%.1e' % (synapses, iBranch+1, freq)] = Vm

                    RESULTS['Rate_%s_%i_%.1e' % (synapses, iBranch+1, freq)] = \
                            apc.n*1e3/RESULTS['tstop'] # rates in Hz

        np.save('../../data/detailed_model/spikingFreq_vs_stim_PV_relationship.npy',
                RESULTS)

    else:

        RESULTS = np.load(\
                '../../data/detailed_model/spikingFreq_vs_stim_PV_relationship.npy',
                allow_pickle=True).item()

        fig, ax = plt.subplots()
        for synapses in ['synapses', 'synapses_shuffled']:
            for iBranch in RESULTS['branches']:
                for freq in RESULTS['freqs']:
                    Vm = RESULTS['Vm_%s_%i_%.1e' % (synapses, iBranch+1, freq)]
                    ax.plot(np.arange(len(Vm))*RESULTS['dt'], Vm)

        # ax.axis('off')
        # pt.draw_bar_scales(ax, loc='top-right',
                           # Xbar=100, Xbar_label='100ms',
                           # Ybar=10, Ybar_label='10mV')

        plt.show()


