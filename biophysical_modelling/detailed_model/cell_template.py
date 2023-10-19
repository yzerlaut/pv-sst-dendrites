import sys, pathlib, os

from neuron import h
from neuron.units import ms
import numpy as np

h.load_file("stdlib.hoc")
h.load_file("import3d.hoc")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.markdown_tables import read_table


BRANCH_COLORS = ['tab:cyan', 'tab:pink', 'tab:blue',
                 'tab:purple', 'tab:green', 'tab:olive']

class Cell:

    params = read_table(\
            os.path.join(\
            pathlib.Path(__file__).resolve().parent,
            'README.md'))

    def __init__(self,
                 proximal_limit=100,
                 ID = '864691135396580129_296758', # Basket Cell example
                 params_key='BC',
                 passive_only=False,
                 with_axon=False,
                 debug=False):

        self.load_morphology(ID,
                             with_axon=with_axon)

        self.SEGMENTS = np.load("morphologies/%s/segments.npy" % ID,
                                allow_pickle=True).item()

        self.preprocess_branches(ID)

        self.label_compartments(proximal_limit, verbose=debug)

        self.insert_mechanisms_and_properties(params_key,
                                              passive_only=passive_only,
                                              debug=debug)
        self.params_key = params_key

        self.map_SEGMENTS_to_NEURON()

        if not debug:
            self.check_that_all_dendritic_branches_are_well_covered(verbose=False)

        self.El = self.params[params_key+'_ePas']

    def preprocess_branches(self, ID):
        """
        loading the branches and set of synapses per branch
        also ordering them by distance to soma for later processing
        """
        branches = np.load("morphologies/%s/dendritic_branches.npy" % ID,
                           allow_pickle=True).item()

        self.set_of_branches, self.set_of_synapses= [], []

        for branch, synapses in zip(branches['branches'],
                                    branches['synapses']):

            iSorted = np.argsort(self.SEGMENTS['distance_to_soma'][branch])
            self.set_of_branches.append(np.array(branch)[iSorted])
            iSorted = np.argsort(self.SEGMENTS['distance_to_soma'][synapses])
            self.set_of_synapses.append(np.array(synapses)[iSorted])

        # --------------------------------------------- #
        #       redistribute synapses uniformly         #
        # --------------------------------------------- #
        self.set_of_synapses_spatially_uniform = []
        for branch, synapses in zip(branches['branches'],
                                    branches['synapses']):
            # spatially uniform from histogram equalization
            distBranch = self.SEGMENTS['distance_to_soma'][branch]
            Rel = np.array((distBranch-distBranch.min())/\
                                (distBranch.max()-distBranch.min())*len(synapses),
                                dtype='int')
            uSynapses, bIndex = [], 0
            while len(uSynapses)<len(synapses):
                if (len(uSynapses)>Rel[bIndex]) and (bIndex<(len(Rel)-1)):
                    bIndex +=1 
                uSynapses.append(branch[bIndex])
            self.set_of_synapses_spatially_uniform.append(np.array(uSynapses))

    def load_morphology(self, ID,
                        with_axon=True):

        cell = h.Import3d_SWC_read()
        if with_axon:
            # we just read the full morphology file
            cell.input("morphologies/%s/%s.swc" % (ID,ID))
        else:
            cell.input("morphologies/%s/%s_no_axon.swc" % (ID,ID))

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

        self.compartments = {'soma':[], 'prox':[],
                             'dist':[], 'axon':[]}

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
                    self.compartments['prox'].append(sec)
                else:
                    self.compartments['dist'].append(sec)

        if verbose:
            for comp in self.compartments:
                print(comp, ' n=%i comp. ' % len(self.compartments[comp]))


    def insert_mechanisms_and_properties(self,
                                         params_key,
                                         passive_only=False,
                                         with_axon=False,
                                         debug=False):

        for LOC in ['soma', 'axon', 'prox', 'dist']:

            for sec in self.compartments[LOC]:

                # cable
                if not debug:
                    sec.nseg = sec.n3d()

                # cable props
                sec.cm = self.params['%s_%s_cm' % (params_key, LOC)]
                sec.Ra = self.params['%s_%s_Ra' % (params_key, LOC)]
                # passive current
                sec.insert('pas')
                sec.g_pas = self.params['%s_%s_gPas' % (params_key, LOC)]
                sec.e_pas = self.params['%s_%s_ePas' % (params_key, LOC)]

                # -------- ACTIVE PROPS --------- #V
                if not passive_only:

                    for mech in ['Nafx', 'Kdrin', 'Kslowin', 
                                 'Hin', 
                                 'Kapin', 'Kapin',
                                 'Kctin', 'Kcain',
                                 'Can', 'Cat', 'Cal']:

                        gKey = '%s_%s_g%s' % (params_key, LOC, mech)

                        if gKey in self.params:

                            sec.insert(mech)
                            getattr(sec, 'gbar_%s' % mech) = self.params[gKey]

                    # + Calcium Decay Dynamics
                    sec.insert('cadynin')

            # initialize
            for sec in self.all:
                sec.v = self.params[params_key+'_ePas']
        
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
            for ib, branch in enumerate(self.set_of_branches):
                print('branch #%i :' % (ib+1), 
                        np.sum(self.SEGMENTS['NEURON_section'][branch]!=None), 
                        '/', len(branch))
        if show:

            import matplotlib.pylab as plt

            for bIndex, branch in enumerate(self.set_of_branches):

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
        for bIndex, branch in enumerate(self.set_of_branches):

            branch_cond = np.zeros(len(self.SEGMENTS['x']), dtype=bool)
            branch_cond[branch] = True

            for i in np.arange(len(branch_cond))[branch_cond & no_section_cond]:
                if (i<len(branch_cond)) and (self.SEGMENTS['NEURON_section'][i+1] is not None):
                    self.SEGMENTS['NEURON_section'][i] = self.SEGMENTS['NEURON_section'][i+1]
                    self.SEGMENTS['NEURON_segment'][i] = self.SEGMENTS['NEURON_segment'][i+1]

        if show:

            no_section_cond = self.SEGMENTS['NEURON_section']==None
            for bIndex, branch in enumerate(self.set_of_branches):

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

    ID = '864691135571546917_264824' # Martinotti
    # ID = '864691135396580129_296758' # Basket

    # cell = Cell(ID=ID, debug=True)
    # cell.check_that_all_dendritic_branches_are_well_covered(show=True)

    cell = Cell(ID=ID, with_axon=False)
    print(cell.compartments['axon'])

    """
    ic = h.IClamp(cell.soma[0](0.5))
    ic.amp = 0. 
    ic.dur =  1e9 * ms
    ic.delay = 0 * ms

    dt= 0.01

    Vm = h.Vector()
    Vm.record(cell.soma[0](0.5)._ref_v)

    h.finitialize()

    for i in range(int(20/dt)):
        h.fadvance()

    for i in range(1, 3):

        ic.amp = i/10.
        for i in range(int(50/dt)):
            h.fadvance()
        ic.amp = 0
        for i in range(int(50/dt)):
            h.fadvance()

    import matplotlib.pylab as plt
    plt.figure(figsize=(9,3))
    plt.plot(np.arange(len(Vm))*dt, np.array(Vm))
    plt.show()
    """

