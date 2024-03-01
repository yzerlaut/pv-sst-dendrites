import os, sys
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import plot_tools as pt

class spikingResponse:
    
    def __init__(self, stim_table, spike_times, t,
                 filename=None):
        
        if filename is not None:
            
            self.load(filename)
            
        else:
            
            self.build(stim_table, spike_times, t)
            
        
    def build(self, stim_table, spike_times, t):
        
        duration = np.mean(stim_table.duration) # find stim duration
        self.t = t
        
        self.time_resolution = self.t[1]-self.t[0]

        self.spike_matrix = np.zeros( (len(stim_table.index.values),
                                       len(self.t)) , dtype=bool)
        self.keys = ['spike_matrix', 't']

        for key in stim_table:
            
            setattr(self, key, np.array(getattr(stim_table, key)))
            self.keys.append(key)

        for trial_idx, trial_start in enumerate(stim_table.start_time.values):

            in_range = (spike_times > (trial_start + self.t[0])) * \
                       (spike_times < (trial_start + self.t[-1]))

            binned_times = ((spike_times[in_range] -\
                             (trial_start + self.t[0])) / self.time_resolution).astype('int')
            self.spike_matrix[trial_idx, binned_times] = True       
            
    def get_rate(self,
                 cond=None,
                 smoothing=5e-3):
        if cond is None:
            cond = np.ones(self.spike_matrix.shape[0], dtype=bool)
            
        iSmooth = int(smoothing/self.time_resolution)
        if iSmooth>=1:
            return gaussian_filter1d(self.spike_matrix[cond,:].mean(axis=0) / self.time_resolution,
                                     iSmooth)
        else:
            return self.spike_matrix[cond,:].mean(axis=0) / self.time_resolution
    
    def save(self, filename):
        D = {'time_resolution':self.time_resolution, 'keys':self.keys}
        for key in self.keys:
            D[key] = getattr(self, key)
        np.save(filename, D)
    
    def load(self, filename):
        D = np.load(filename, allow_pickle=True).item()
        for key in D['keys']:
            setattr(self, key, np.array(D[key]))
        self.keys = D['keys']
        self.time_resolution = D['time_resolution']
        
    def plot(self, 
             cond = None,
             ax1=None, ax2=None,
             smoothing=5e-3, 
             trial_subsampling=1,
             color='k', ms=1):

        if cond is None:
            cond = np.ones(self.spike_matrix.shape[0], dtype=bool)

        if not (ax1 is not None and ax2 is not None):
            fig = plt.figure(figsize=(1.2,2))
            plt.subplots_adjust(left=0.1, top=0.8, right=0.95)
            ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
        else:
            fig = None
            
        for i, t in enumerate(\
                        np.arange(self.spike_matrix.shape[0])[cond][::trial_subsampling]):
            spike_cond = self.spike_matrix[t,:]==1
            ax1.plot(self.t[spike_cond],
                     self.spike_matrix[t,:][spike_cond]+i, '.', ms=ms, color=color)
            
        ax2.fill_between(self.t, 0*self.t, self.get_rate(cond=cond, smoothing=smoothing), color=color)
        ax1.set_ylabel('trial #')
        ax2.set_ylabel('rate (Hz)')
        ax2.set_xlabel('time (s)')
        pt.set_common_xlims([ax1,ax2])

        return fig, [ax1, ax2]


def crosscorrel(Signal1, Signal2, tmax, dt):
    """
    argument : Signal1 (np.array()), Signal2 (np.array())
    returns : np.array()
    take two Signals, and returns their crosscorrelation function 

    CONVENTION:
    --------------------------------------------------------------
    when the peak is in the future (positive t_shift)
    it means that Signal2 is delayed with respect to Signal 1
    check with:
    ```
    t = np.linspace(0,1,200)
    Signal1, Signal2 = 0*t, 0*t
    Signal1[(t>0.4) & (t<0.6)] = 1. # first
    Signal2[(t>0.5) & (t<0.7)] = 1. # second
    # compute
    CCF, time_shift = crosscorrel(Signal1, Signal2, 1, t[1]-t[0])
    # plot
    plt.plot(time_shift, CCF, color='tab:red')
    ```
    --------------------------------------------------------------
    """
    if len(Signal1)!=len(Signal2):
        print('Need two arrays of the same size !!')
        
    steps = int(tmax/dt) # number of steps to sum on
    time_shift = dt*np.concatenate([-np.arange(1, steps)[::-1], np.arange(steps)])
    CCF = np.zeros(len(time_shift))
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal1[:len(Signal1)-i], Signal2[i:])
        CCF[steps-1+i] = ccf[0,1]
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal2[:len(Signal1)-i], Signal1[i:])
        CCF[steps-1-i] = ccf[0,1]
    return CCF, time_shift
