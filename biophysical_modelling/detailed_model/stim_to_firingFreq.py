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
        x = cell.SEGMENTS['NEURON_segment'][syn]/cell.SEGMENTS['NEURON_section'][syn].nseg
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

        ID = '864691135396580129_296758' # Basket Cell example
        cell = PVcell(ID=ID, debug=False)
        cell.check_that_all_dendritic_branches_are_well_covered()

        # shuffle synapses
        cell.branches['synapses_shuffled'] = []
        for iBranch in range(len(cell.branches['branches'])):
            cell.branches['synapses_shuffled'].append(\
                    np.random.choice(cell.branches['branches'][iBranch],
                        len(cell.branches['synapses'][iBranch])))

        RESULTS = {'dt':0.025,
                   'branches':range(len(cell.branches['synapses'])),
                   'freqs': np.logspace(-3.5, 0, 15),
                   'tstop':1000}

        for synapses in ['synapses', 'synapses_shuffled']:
            print(' o %s ' % synapses)
            for iBranch in RESULTS['branches']:
                print('     branch #%i ' % iBranch)
                for freq in RESULTS['freqs']:
                    print('         freq=%.1e Hz' % freq)
                    _, Vm = run_sim(cell, cell.branches[synapses][iBranch],
                                    dt=RESULTS['dt'],
                                    tstop=RESULTS['tstop'],
                                    stim_freq=freq)
                    RESULTS['Vm_%s_%i_%.3f' % (synapses, iBranch, freq)] = Vm

        np.save('../../data/detailed_model/spikingFreq_vs_stim_PV_relationship.npy',
                RESULTS)

    else:

        RESULTS = np.load(\
                '../../data/detailed_model/spikingFreq_vs_stim_PV_relationship.npy',
                allow_pickle=True).item()

        fig, ax = plt.subplots()
        for synapses in ['synapses', 'synapses_shuffled']
            for iBranch in RESULTS['branches']:
                for freq in RESULTS['freqs']:
                    Vm = RESULTS['Vm_%s_%i_%.3f' % (synapses, iBranch, freq)]
                    ax.plot(np.arange(len(Vm))*RESULTS['dt'], Vm)

        # ax.axis('off')
        # pt.draw_bar_scales(ax, loc='top-right',
                           # Xbar=100, Xbar_label='100ms',
                           # Ybar=10, Ybar_label='10mV')

        plt.show()


