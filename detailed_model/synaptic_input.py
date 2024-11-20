from neuron import h
import numpy as np
from math import comb # binomial coefficient

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

def STP_release_filter(pre_spikes,
                       P0 = 1.0, # proba at 0-frequency
                       P1 = 1.0, # proba at oo-frequency
                       dP = 0.0, # proba increment
                       tauP = 1.0, # seconds
                       Nmax=1):
    """
    model of spike timing dynamics
        see Synaptic-Dynamics.ipynb notebook
    """
    # build the time-varing release probability:
    P = np.ones(len(pre_spikes))*P0 # initialized
    for i in range(len(pre_spikes)-1):
        P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/(abs(P1-P0)+1e-4) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
    # build the probabilities of each number of vesicles ([!!] from Nmax to 1 [!!] ) :
    Ps = np.cumsum([ (comb(Nmax, n) * P**n * (1-P)**(Nmax-n)) for n in np.arange(Nmax, 0, -1)], axis=0)
    # draw random numbers:
    R = np.random.uniform(0, 1, size=len(pre_spikes))
    # find number of vesicles released:
    N = np.sum((Ps/R>1).astype(int), axis=0)
    return N

def add_synaptic_input(cell, synapses,
                       with_NMDA=False,
                       Nmax_release = 1,
                       boost_AMPA_for_SST_noNMDA=True,
                       Inh_fraction=20./100.):
    """
    add AMPA, NMDA and GABA synapses to a given cell

    if Nmax>1
    it adds other synapses with double (Nmax>=2), triple (Nmax>=3), ...
        vesicular synaptic release only on AMPA and NMDA ! 
    """

    AMPAS, NMDAS, GABAS = [], [], []
    ampaNETCONS, nmdaNETCONS, gabaNETCONS = [], [], []
    STIMS, VECSTIMS = [], []

    excitatory = np.random.choice([True, False],
                                  len(synapses),
                                  p=[1.-Inh_fraction, Inh_fraction])

    for nVesicles in range(1, Nmax_release+1):
        
        for i, syn in enumerate(synapses):

            np.random.seed(syn*(1+nVesicles))

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
                if (cell.params_key=='MC') and (not with_NMDA)\
                        and boost_AMPA_for_SST_noNMDA:
                    # boosted ampa weight
                    ampaNETCONS[-1].weight[0] = cell.params['%s_qAMPA'%cell.params_key]*\
                                    cell.params['%s_qAMPAonlyBoost'%cell.params_key]*\
                                    nVesicles
                else:
                    # regular ampa weight
                    ampaNETCONS[-1].weight[0] = cell.params['%s_qAMPA'%cell.params_key]*\
                                                    nVesicles

                if with_NMDA:
                    nmdaNETCONS.append(h.NetCon(VECSTIMS[-1], NMDAS[-1]))
                    nmdaNETCONS[-1].weight[0] = cell.params['%s_NAR'%cell.params_key]*\
                                                    cell.params['%s_qAMPA'%cell.params_key]*\
                                                    nVesicles

                GABAS.append(None)
                gabaNETCONS.append(None)

            elif nVesicles==1:
                # inhibitory synapses

                GABAS.append(\
                        h.GABAain(x, sec=cell.SEGMENTS['NEURON_section'][syn]))

                gabaNETCONS.append(h.NetCon(VECSTIMS[-1], GABAS[-1]))
                gabaNETCONS[-1].weight[0] = cell.params['%s_qGABA'%cell.params_key]

                AMPAS.append(None)
                NMDAS.append(None)
                ampaNETCONS.append(None)
                nmdaNETCONS.append(None)

            else:
                # nothing, no multi-vesicular of GABAergic synapses
                GABAS.append(None)
                gabaNETCONS.append(None)
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



