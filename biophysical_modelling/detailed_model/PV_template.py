import sys, pathlib, os

from neuron import h
from neuron.units import ms
import numpy as np

h.load_file("stdlib.hoc")
h.load_file("import3d.hoc")

soma_pas = 2.7*1./7600. # raised match ~100MOhm input resistance
soma_Nafin = 0.045
soma_kdrin = 0.036
soma_Kslowin = 0.000725 
soma_hin = 0.00001 
soma_kapin = 0.0032
soma_can = 0.0003
soma_cat = 0.002
soma_kctin =0.0001 
soma_kcain =0.020
v_init = -70.

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.markdown_tables import read_table

class PVcell:

    params = read_table(\
            os.path.join(\
            pathlib.Path(__file__).resolve().parent,
            'README.md'))

    def __init__(self,
                 proximal_limit=100,
                 ID = '864691135396580129_296758', # Basket Cell example
                 debug=False):

        self.load_morphology(ID)

        self.SEGMENTS = np.load("morphologies/%s/segments.npy" % ID,
                                allow_pickle=True).item()
        self.branches = np.load("morphologies/%s/dendritic_branches.npy" % ID,
                                allow_pickle=True).item()

        self.label_compartments(proximal_limit, verbose=debug)

        self.insert_mechanisms_and_properties(debug=debug)

        self.map_SEGMENTS_to_NEURON()

        self.El = v_init

    def load_morphology(self, ID):
        cell = h.Import3d_SWC_read()
        cell.input("morphologies/%s/%s.swc" % (ID,ID))
        i3d = h.Import3d_GUI(cell, False)
        i3d.instantiate(self)

        self.soma_position = [self.soma[0].x3d(1),
                              self.soma[0].y3d(1),
                              self.soma[0].z3d(1)]

    def distance_to_soma(self, sec):
        # icenter = int(sec.nseg/2)+1
        # return np.sqrt(\
                # (sec.x3d(icenter)-self.soma_position[0])**2+\
                # (sec.y3d(icenter)-self.soma_position[1])**2+\
                # (sec.z3d(icenter)-self.soma_position[2])**2)
        # return h.distance(self.soma[0](0.5), int(sec.nseg/2)+1)
        return h.distance(self.soma[0](0.5), sec(1))


    def label_compartments(self, proximal_limit,
                           verbose=False):

        self.compartments = {'soma':[], 'proximal':[], 'distal':[], 'axon':[]}

        for sec in self.all:
            # loop over all compartments
            if 'soma' in sec.name():
                self.compartments['soma'].append(sec)
            elif 'axon' in sec.name():
                self.compartments['axon'].append(sec)
            else:
                # dendrites
                distS = self.distance_to_soma(sec)
                if distS<proximal_limit:
                    self.compartments['proximal'].append(sec)
                else:
                    self.compartments['distal'].append(sec)

        if verbose:
            for comp in self.compartments:
                print(comp, ' n=%i comp. ' % len(self.compartments[comp]))


    def insert_mechanisms_and_properties(self,
                                         debug=False):

        # SOMA
        for sec in self.compartments['soma']:
            # cable
            if not debug:
                sec.nseg = sec.n3d()

            # cable props
            sec.cm = params['BC_soma_cm']
            sec.Ra = params['BC_soma_Ra']
            # passive current
            sec.insert('pas')
            sec.g_pas = params['BC_soma_gPas']
            sec.e_pas = params['BC_ePas']
            # sodium channels
            sec.insert('Nafx')
            # sec.gnafbar_Nafx= soma_Nafin*0.6*5
            sec.gnafbar_Nafx= params['BC_soma_gNa']
            # potassium channels
            sec.insert('kdrin')
            sec.gkdrbar_kdrin = params['BC_soma_gKdrin']
            # 
            sec.insert('IKsin')
            # sec.gKsbar_IKsin= soma_Kslowin
            sec.gKsbar_IKsin = params['BC_soma_gKslowin']
            #
            sec.insert('hin')
            # sec.gbar_hin=soma_hin
            sec.gbar_hin = params['BC_soma_gHin']
            # 
            sec.insert('kapin')
            # sec.gkabar_kapin=soma_kapin
            sec.gkabar_kapin = params['BC_soma_gKapin']
            #
            sec.insert('kctin')
            sec.gkcbar_kctin = params['BC_soma_gKctin']
            #
            sec.insert('kcain')
            sec.gbar_kcain = params['BC_soma_gKcain']
            #
            sec.insert('cadynin')

        # AXON
        for sec in self.compartments['axon']:
            # if not debug:
                # sec.nseg = sec.n3d()
            sec.cm=1.2
            sec.Ra=172
            #
            sec.insert('pas')
            sec.g_pas = soma_pas*7600./281600.
            print(sec.g_pas, self.params['BC_gSoma_pas'])
            sec.e_pas = v_init
            #
            sec.insert('Nafx')
            sec.gnafbar_Nafx=soma_Nafin*0.6*25
            # 				                                                                    
            sec.insert('kdrin')
            sec.gkdrbar_kdrin=soma_kdrin*3


        # PROX DEND
        for sec in self.compartments['proximal']:
            # cable
            if not debug:
                sec.nseg = sec.n3d()

            # cable props
            sec.cm = params['BC_prox_cm']
            sec.Ra = params['BC_prox_Ra']
            # passive current
            sec.insert('pas')
            sec.g_pas = params['BC_prox_gPas']
            sec.e_pas = params['BC_ePas']
            # sodium channels
            sec.insert('Nafx')
            # sec.gnafbar_Nafx= soma_Nafin*0.4
            sec.gnafbar_Nafx= params['BC_prox_gNa']
            # potassium channels
            sec.insert('kdrin')
            sec.gkdrbar_kdrin = params['BC_prox_gKdrin']
            # 
            sec.insert('IKsin')
            # sec.gkdrbar_kdrin=0*0.018*0.5
            sec.gKsbar_IKsin = params['BC_prox_gKslowin']
            # 
            sec.insert('kapin')
            # sec.gkabar_kapin=soma_kapin*0.2                                            
            sec.gkabar_kapin = params['BC_prox_gKapin']
            #
            sec.insert('can')
            sec.gcabar_can = params['BC_prox_gCan']
            #
            sec.insert('cat')
            # sec.gcatbar_cat=soma_cat*0.1
            sec.gcatbar_cat = params['BC_prox_gCat']
            #
            sec.insert('cal')
            # sec.gcalbar_cal=0.00003
            sec.gcalbar_cat = params['BC_prox_gCal']
            #
            sec.insert('cadynin')


        # DISTAL DEND
        for sec in self.compartments['distal']:
            # cable
            if not debug:
                sec.nseg = sec.n3d()
            sec.cm = params['BC_dist_cm']
            sec.Ra = params['BC_dist_Ra']
            # passive current
            sec.insert('pas')
            sec.g_pas = params['BC_dist_gPas']
            sec.e_pas = params['BC_ePas']
            # sodium channel
            sec.insert('Nafx')
            sec.gnafbar_Nafx=0*soma_Nafin*0.4*0.8
            # potassium channel
            sec.insert('kdrin')
            sec.gkdrbar_kdrin=0*0.018*0.5
            # 
            sec.insert('kadin')
            sec.gkabar_kadin=1.8*0.001
            # 
            sec.insert('can')
            sec.gcabar_can = soma_can
            #
            sec.insert('cat')
            sec.gcatbar_cat=soma_cat*0.1
            #
            sec.insert('cal')
            sec.gcalbar_cal=0.00003
            #
            sec.insert('cadynin')

        for sec in self.all:
            sec.v = v_init

        
        h.ko0_k_ion = 3.82 #  //mM
        h.ki0_k_ion = 140  #  //mM  

    def map_SEGMENTS_to_NEURON(self):
        """
        mapping based on position in 3d space

        only on somatic and dendritic compartments
        otherwise can be confused by some axons crossing the dendrites
        (because the NEURON conversion of the swc to compartements/sections 
            make it hard to recover the true xyz coordinates)
        """
        cond = (self.SEGMENTS['comp_type']=='dend') | (self.SEGMENTS['comp_type']=='soma')

        self.SEGMENTS['NEURON_section'] = np.empty(len(self.SEGMENTS['x']),
                                                   dtype=object)
        self.SEGMENTS['NEURON_segment'] = np.empty(len(self.SEGMENTS['x']),
                                                   dtype=object)
        iMins = []
        # for sec in self.all:
        for sec in self.compartments['proximal']+self.compartments['distal']+self.compartments['soma']:
            for iseg in range(sec.nseg):
                try:
                    D = np.sqrt((1e6*self.SEGMENTS['x']-sec.x3d(iseg))**2+\
                                (1e6*self.SEGMENTS['y']-sec.y3d(iseg))**2+\
                                (1e6*self.SEGMENTS['z']-sec.z3d(iseg))**2)
                    # print(np.sqrt(np.min(D)))
                    iMin = np.argmin(D[cond])
                    iMin2 = np.arange(len(cond))[cond][iMin]
                    self.SEGMENTS['NEURON_section'][iMin2] = sec
                    self.SEGMENTS['NEURON_segment'][iMin2] = iseg

                except BaseException as be:
                    print(be)
                    print('PB with: ', iseg, sec)


    def check_that_all_dendritic_branches_are_well_covered(self, 
                                                           show=False,
                                                           verbose=True):

        no_section_cond = self.SEGMENTS['NEURON_section']==None

        if verbose:
            for ib, branch in enumerate(self.branches['branches']):
                print('branch #%i :' % (ib+1), 
                        np.sum(self.SEGMENTS['NEURON_section'][branch]!=None), 
                        '/', len(branch))
        if show:

            import matplotlib.pylab as plt

            for bIndex, branch in enumerate(self.branches['branches']):

                branch_cond = np.zeros(len(self.SEGMENTS['x']), dtype=bool)
                branch_cond[branch] = True

                cond = branch_cond 
                plt.scatter(1e6*self.SEGMENTS['x'][cond], 1e6*self.SEGMENTS['y'][cond],
                            s=0.1, color=plt.cm.tab10(bIndex))
                cond = branch_cond & no_section_cond
                plt.scatter(1e6*self.SEGMENTS['x'][cond], 1e6*self.SEGMENTS['y'][cond],
                            color='r', s=4)

            plt.title('before fix !')
            plt.show()

        # insure that all segments that are on a branch have a matching location
        # --> we take the next one
        for bIndex, branch in enumerate(self.branches['branches']):

            branch_cond = np.zeros(len(self.SEGMENTS['x']), dtype=bool)
            branch_cond[branch] = True

            for i in np.arange(len(branch_cond))[branch_cond & no_section_cond]:
                if (i<len(branch_cond)) and (self.SEGMENTS['NEURON_section'][i+1] is not None):
                    self.SEGMENTS['NEURON_section'][i] = self.SEGMENTS['NEURON_section'][i+1]
                    self.SEGMENTS['NEURON_segment'][i] = self.SEGMENTS['NEURON_segment'][i+1]

        if show:

            no_section_cond = self.SEGMENTS['NEURON_section']==None
            for bIndex, branch in enumerate(self.branches['branches']):

                branch_cond = np.zeros(len(self.SEGMENTS['x']), dtype=bool)
                branch_cond[branch] = True

                cond = branch_cond 
                plt.scatter(1e6*self.SEGMENTS['x'][cond], 1e6*self.SEGMENTS['y'][cond],
                            s=0.1, color=plt.cm.tab10(bIndex))
                cond = branch_cond & no_section_cond
                plt.scatter(1e6*self.SEGMENTS['x'][cond], 1e6*self.SEGMENTS['y'][cond],
                            color='r', s=4)

            plt.title('after fix !')
            plt.show()



if __name__=='__main__':

    cell = PVcell(debug=False)

    # cell.check_that_all_dendritic_branches_are_well_covered(show=True)

    # ID = '864691135571546917_264824' # Martinotti
    # cell = PVcell(ID=ID, debug=False)
    # cell.check_that_all_dendritic_branches_are_well_covered(show=True)

    """
    n = 0
    for sec in cell.all:
        n += sec.nseg-2
    print(n)
    
    ic = h.IClamp(cell.soma[0](0.5))
    ic.amp = 0. 
    ic.dur =  1e9 * ms
    ic.delay = 0 * ms

    dt, tstop = 0.025, 500

    t_stim_vec = h.Vector(np.arange(int(tstop/dt))*dt)
    Vm = h.Vector()

    Vm.record(cell.soma[0](0.5)._ref_v)

    h.finitialize()

    for i in range(int(50/dt)):
        h.fadvance()

    for i in range(1, 11):

        ic.amp = i/10.
        for i in range(int(100/dt)):
            h.fadvance()
        ic.amp = 0
        for i in range(int(100/dt)):
            h.fadvance()

    import matplotlib.pylab as plt
    plt.figure(figsize=(9,3))
    plt.plot(np.arange(len(Vm))*dt, np.array(Vm))
    plt.show()
    """
