"""
USING THE `multiprocessing` MODULE TO PARALLELIZE FUNCTION EXECUTION

using `dill` to serialize python closures by subclassing the mp.Process 
See:
https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
"""

import dill
import multiprocessing as mp
import numpy as np
from itertools import product
import zipfile
import os
import pathlib
import tempfile

class DillProcess(mp.Process):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        # Save the target function as bytes, using dill:
        self._target = dill.dumps(self._target)  

    def run(self):
        if self._target:
            # Unpickle the target function before executing:
            self._target = dill.loads(self._target)    
            # Execute the target function:
            self._target(*self._args, **self._kwargs)  

class Parallel:
    """
    a class to run parallel simulations of
    an arbitrary function:

        `single_run_func(a=0, b='long', c=False, ...
                         filename='test.npy')`

    over a multi-dimentsional grid of parameters

    and put all results into a zip file
    """

    def __init__(self,
                 scan_seed=1,
                 filename='data.zip',
                 temp_folder=tempfile.gettempdir()):

        np.random.seed(scan_seed)
        self.temp_folder=temp_folder
        self.PARAMS_SCAN = None
        self.has_grid = False

        self.filename = filename
        self.basename = os.path.basename(filename).replace('.zip', '')
        self.scan_file = os.path.join(self.temp_folder,
                                        self.basename+'_scan.npy')

    def build_filename_single_sim(self, KEYS, VAL):

        FN = self.basename

        for key, val in zip(KEYS, VAL):
            if type(val) in [float, np.float32, np.float64]:
                FN += '_'+key+'_%.3e'%val
            else:
                FN += '_'+key+'_'+str(val)

        # not too mix up results of repeated/successive sims:
        # FN += '_'+str(np.random.randint(100000))+'.npy' 
        FN += '.npy' 
        return FN


    def build(self, DICT):
        """ 
        build the set of simulations with their filenames !
        """
        self.KEYS = [str(k) for k in DICT.keys()]
        self.PARAMS_SCAN = {'filenames':[], 'keys':self.KEYS}
        VALUES = []
        for key in self.KEYS:
            self.PARAMS_SCAN[key] = []
            VALUES.append(DICT[key])

        for VAL in product(*VALUES):
            # params for each sim
            for key, val in zip(self.KEYS, VAL):
                self.PARAMS_SCAN[key].append(val)
            # with a given filename
            self.PARAMS_SCAN['filenames'].append(\
                    self.build_filename_single_sim(self.KEYS, VAL))

    def check_success(self):
        success = True
        for filename in self.PARAMS_SCAN['filenames']:
            if not os.path.isfile(os.path.join(self.temp_folder, filename)):
                # if it doesn't exists !
                success = False
        return success

    def run_scan(self, single_run_func,
            single_run_args={},
            parallelize=True,
            fix_missing_only=False,
            Nmax_simultaneous_processes=None):

        if self.PARAMS_SCAN is not None:

            zf = zipfile.ZipFile(self.filename,
                                 mode='w')

            if parallelize:
                PROCESSES = []
                # Define an output queue
                output = mp.Queue()
            else:
                output = None

            def run_func(i, output):
                import os
                params = {}
                for key in self.KEYS:
                    params[key] = self.PARAMS_SCAN[key][i]
                params['filename'] = os.path.join(self.temp_folder,
                                                  self.PARAMS_SCAN['filenames'][i])
                single_run_func(**params, **single_run_args)
            
            for i, FN in enumerate(self.PARAMS_SCAN['filenames']):
                if parallelize:
                    if fix_missing_only:
                        if not os.path.isfile(os.path.join(self.temp_folder, FN)): # if it doesn't exists !
                            print('running configuration ', FN)
                            PROCESSES.append(DillProcess(target=run_func, args=(i, output)))
                        else:
                            print('configuration DONE: ', FN)
                    else:
                        PROCESSES.append(DillProcess(target=run_func, args=(i, output)))
                else:
                    run_func(i, 0)
             
            if parallelize:
                if Nmax_simultaneous_processes is None:
                    Nmax_simultaneous_processes = int(mp.cpu_count())
                print('parallelizing %i processes over %i cores' % (len(PROCESSES),\
                        Nmax_simultaneous_processes))

                # Run processes
                for i in range(len(PROCESSES)//Nmax_simultaneous_processes+1):
                    print(' - running multiprocessing loop: %i/%i [...]' % (i+1, len(PROCESSES)//Nmax_simultaneous_processes))
                    for p in PROCESSES[Nmax_simultaneous_processes*i:Nmax_simultaneous_processes*(i+1)]:
                        p.start()
                    # # Exit the completed processes
                    for p in PROCESSES[Nmax_simultaneous_processes*i:Nmax_simultaneous_processes*(i+1)]:
                        p.join()
                    # print('   n=%i/%i' % (i*len(PROCESSES), len(PROCESSES)))

            # CHECK SUCCESS !
            success = self.check_success()
            if success:
                # write all single sim files in the zip file
                for FN in self.PARAMS_SCAN['filenames']:
                    print('writing', FN, ' in zip folder')
                    zf.write(os.path.join(self.temp_folder, FN), arcname=FN)

            # add the scan metadata to the zip
            np.save(self.scan_file, self.PARAMS_SCAN)
            zf.write(self.scan_file, arcname='scan.npy')

            # close the zip file
            zf.close()

            return success

        else:
            print(' need to build the simulation with the varied parameters ! ')


    def run(self, single_run_func,
            single_run_args={},
            parallelize=True,
            fix_missing_only=False,
            Nmax_simultaneous_processes=None,
            Nmax_check=10):

        # we do a first scan
        success = self.run_scan(single_run_func, 
                    single_run_args=single_run_args,
                    parallelize=parallelize,
                    fix_missing_only=fix_missing_only,
                    Nmax_simultaneous_processes=Nmax_simultaneous_processes)

        # we check that it was successfull and re-run N times if needed
        i=0
        while not success and i<Nmax_check:
            success = self.run_scan(single_run_func, 
                    single_run_args=single_run_args,
                    parallelize=parallelize,
                    fix_missing_only=True,
                    Nmax_simultaneous_processes=Nmax_simultaneous_processes)

            print('\n\n'+20*'-'+' Success Check #%i: %s'%(i+1, success)+20*'-'+'\n\n')
            i+=1


    def unzip(self):

        zf = zipfile.ZipFile(self.filename, mode='r')

        for fn in self.PARAMS_SCAN['filenames']:
            data = zf.read(fn)
            FN = os.path.join(self.temp_folder, os.path.basename(fn)) 
            with open(FN, 'wb') as f:
                f.write(data)

        zf.close()


    def load(self,
             unzip=True,
             with_data=False):

        zf = zipfile.ZipFile(self.filename, mode='r')
     
        data = zf.read('scan.npy')
        with open(self.scan_file, 'wb') as f:
            f.write(data)

        zf.close()

        self.PARAMS_SCAN = np.load(self.scan_file,
                               allow_pickle=True).item()
        self.keys = self.PARAMS_SCAN['keys'] 

        if unzip:
            self.unzip()

        if with_data:
            self.DATA = []
            for fn in self.PARAMS_SCAN['filenames']:
                data = np.load(os.path.join(self.temp_folder, fn),
                               allow_pickle=True).item()
                self.DATA.append(data)

        zf.close()

        self.build_grid()

    def build_grid(self):

        self.VALUES = [np.sort(np.unique(self.PARAMS_SCAN[k])) for k in self.keys]
       
        for key, array in zip(self.keys,
                              np.meshgrid(*[self.VALUES[iK] for iK in range(len(self.keys))],
                                          indexing='ij')):
            setattr(self, key, array)

        self.has_grid = True

        # now filenames array from list of simulations
        self.filenames = np.empty(getattr(self, self.keys[0]).shape, dtype=object)
        for i, fn in enumerate(self.PARAMS_SCAN['filenames']):
            cond = np.ones(self.filenames.shape, dtype=bool)
            for key in self.keys:
                cond = cond & (getattr(self, key)==self.PARAMS_SCAN[key][i])
            self.filenames[cond] = fn

    def fetch_quantity_on_grid(self, key,
                               dtype=float,
                               return_last=False):

        setattr(self, key, np.empty(self.filenames.shape,
                                    dtype=dtype))

        for iKs in product(*[range(len(X)) for X in self.VALUES]):

            getattr(self, key)[iKs] = np.load(\
                                os.path.join(self.temp_folder, self.filenames[iKs]),
                                allow_pickle=True).item()[key]
        if return_last:
            return getattr(self, key)[iKs]
            
    def get(self, key,
            params={}):
        
        if self.has_grid:

            cond = np.ones(getattr(self,key).shape, dtype=bool)

            for k, val in params.items():

                cond = cond & (getattr(self, k)==val)

            return getattr(self, key)[cond]

        else:

            print(' need to call "build_grid" before ! ')
            return None
        
    
if __name__=='__main__':

    import sys
    import os

    ##################################################
    # the "running_sim_func" should look like that  ##
    ##################################################
    def single_run_func(SEED=0, 
                        a=0,
                        b=1,
                        c=3,
                        filename='test.npy',
                        delay=0.1):
        import time, os
        import numpy as np

        """
        it should have the "filename" argument at least
        """
        time.sleep(delay)
        np.random.seed(a)
        np.save(filename, {'x':np.arange(10),
                           'scalar_output':np.random.randn()})

    sim = Parallel(filename=\
            os.path.join(tempfile.gettempdir(), 'data.zip'))

    if sys.argv[-1]=='load':

        sim.load(unzip=True, with_data=True)
        sim.fetch_quantity_on_grid('scalar_output')

        print(sim.get('scalar_output', {'SEED':1,  'a':6, 'b':False}))
        # print(sim.scalar_output[0,:,0])

    else:
        # means run !
     
        # build the scan over parameters
        sim.build({'SEED': np.arange(3), 
                   'a':np.arange(5, 8),
                   'b':[True, False]})

        import time
        start_time = time.time()
        print('-----------------------------------')
        print(' Without parallelization')
        sim.run(single_run_func,
                parallelize=False)
        print("--- %s seconds ---" % (time.time() - start_time))        
        print('-----------------------------------')


        start_time = time.time()
        print(' With parallelization')
        sim.run(single_run_func,
                parallelize=True)
        print("--- %s seconds ---" % (time.time() - start_time))        

        print('')
        print('open now with: `python parallel.py load` ')
