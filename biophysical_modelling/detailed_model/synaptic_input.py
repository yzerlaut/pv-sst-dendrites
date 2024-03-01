from neuron import h
import numpy as np

def PoissonSpikeTrain(freq,
                      dt=None,
                      tstop=1.):
    """
    Poisson spike train from a given frequency (homogeneous process)
    """

    if dt is not None and (type(freq) in [np.ndarray, np.array, list]):
        spikes = []
        rdms = np.random.uniform(0, 1, size=len(freq))
        for i, rdm in enumerate(rdms):
            if rdm<(freq[i]*dt):
                spikes.append(i*dt)
        return spikes

    elif type(freq) in [float]:
        spikes = np.cumsum(\
                np.random.exponential(1./freq, int(2.*tstop*freq)+10))
        return spikes[spikes<tstop]

    else:
        print('\n missing input --> no spikes generated ! \n')


def add_synaptic_input(cell, synapses,
                       with_NMDA=False,
                       EI_ratio=20./100.):

    AMPAS, NMDAS, GABAS = [], [], []
    ampaNETCONS, nmdaNETCONS, gabaNETCONS = [], [], []
    STIMS, VECSTIMS = [], []

    excitatory = np.random.choice([True, False],
                                  len(synapses),
                                  p=[1.-EI_ratio, EI_ratio])

    for i, syn in enumerate(synapses):

        np.random.seed(syn)

        VECSTIMS.append(h.VecStim())

        # need to avoid x=0 and x=1, to allow ion concentrations variations in NEURON
        x = np.clip(cell.SEGMENTS['NEURON_segment'][syn], 
                1, cell.SEGMENTS['NEURON_section'][syn].nseg-2)\
                        /cell.SEGMENTS['NEURON_section'][syn].nseg

        if excitatory[i]:
            # excitatory synapses

            AMPAS.append(\
                    h.CPGLUIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

            if with_NMDA:
                NMDAS.append(\
                        h.NMDAIN(x, sec=cell.SEGMENTS['NEURON_section'][syn]))


            ampaNETCONS.append(h.NetCon(VECSTIMS[-1], AMPAS[-1]))
            ampaNETCONS[-1].weight[0] = cell.params['%s_qAMPA'%cell.params_key]

            if with_NMDA:
                nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], NMDAS[-1]))
                nmdaNETCONS[-1].weight[0] = cell.params['%s_NAR'%cell.params_key]*\
                                                cell.params['%s_qAMPA'%cell.params_key]

            GABAS.append(None)
            gabaNETCONS.append(None)

        else:
            # inhibitory synapses

            GABAS.append(\
                    h.GABAain(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

            gabaNETCONS.append(h.NetCon(VECSTIMS[-1], GABAS[-1]))
            gabaNETCONS[-1].weight[0] = cell.params['%s_qGABA'%cell.params_key]

            AMPAS.append(None)
            NMDAS.append(None)
            ampaNETCONS.append(None)
            nmdaNETCONS.append(None)

    return AMPAS, NMDAS, GABAS,\
            ampaNETCONS, nmdaNETCONS, gabaNETCONS,\
            STIMS, VECSTIMS, excitatory

if __name__=='__main__':

    train = PoissonSpikeTrain(10., tstop=10.)
    print(1./np.mean(np.diff(train)))

    dt = 1e-3
    tstop = 10.
    t = np.arange(int(tstop/dt))*dt
    train = PoissonSpikeTrain(0*t+10., tstop=tstop, dt=dt)
    print(1./np.mean(np.diff(train)))



