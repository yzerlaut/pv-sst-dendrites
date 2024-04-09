# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

import sys
sys.path.append('..')
import plot_tools as pt
# pt.set_style('dark')
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# %%
data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# %%
sessions = cache.get_session_table()

# %% [markdown]
# # 1) Loading the Optotagging results

# %%
Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'),
                      allow_pickle=True).item()

# %% [markdown]
# # 2) Analyze using the pre-computed visual response metrics

# %%
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

# %%
for k in analysis_metrics.keys():
    print(k)

# %%
from scipy.stats import mannwhitneyu

def show_average_quantity(analysis_metrics,
                          attr = 'g_osi_sg',
                          label = 'OSI'):
    

    fig, ax = plt.subplots(1)
    inset = pt.inset(ax, [1.4, -0.05, 0.3, 0.6])

    # choose attribute and set label


    RESP = {'Others':[]}
    for i, Key in enumerate(['PV', 'SST']):
        RESP[Key] = []
        positive_IDs = np.concatenate(Optotagging[Key+'_positive_units'])
        for unit in positive_IDs:
            if len(getattr(analysis_metrics[analysis_metrics.index==unit], attr).values)>0:
                value = getattr(analysis_metrics[analysis_metrics.index==unit], attr).values[0]
                if np.isfinite(value):
                    RESP[Key].append(value)
        for unit in analysis_metrics[\
                        analysis_metrics.ecephys_structure_acronym.isin(['VISp'])].index.values:
            if (unit not in positive_IDs) and (len(getattr(analysis_metrics[analysis_metrics.index==unit], attr).values)>0):
                value = getattr(analysis_metrics[analysis_metrics.index==unit], attr).values[0]
                if np.isfinite(value):
                    RESP['Others'].append(value)


    COLORS=['lightgrey', 'tab:red', 'tab:orange']
    for i, Key in enumerate(['Others', 'PV', 'SST']):
        pt.annotate(ax, i*'\n'+'n=%i units' % len(RESP[Key]), (1,1), color=COLORS[i], ha='right', va='top')
        ax.hist(RESP[Key], bins=np.linspace(0, 1, 30), color=COLORS[i],
                 label=Key.replace('_sessions', '+ units'), alpha=1 if i==0 else 0.6, density=True)
        pt.violin(RESP[Key], X=[i], COLORS=[COLORS[i]], ax=inset)

    inset.set_xticks([])
    inset.set_yticks([0, 0.5, 1])
    inset.set_ylabel(label)
    inset.set_title('PV vs SST, p=%.1e' % mannwhitneyu(RESP['PV'], RESP['SST']).pvalue)
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_ylabel('density')
    ax.legend(loc=(1.1, 0.8))
    
    return RESP

OSI = show_average_quantity(analysis_metrics, attr = 'g_osi_sg', label = 'OSI')

# %%
DSI = show_average_quantity(analysis_metrics, attr = 'g_dsi_dg', label = 'DSI')

# %% [markdown]
# # 3) Recover the Original Orientation-Dependent PSTH

# %%
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
# %%
def compute_orientation_spike_resp(unit, analysis_metrics,
                                   time_resolution = 1e-3,
                                   t_pre = 50e-3,
                                   t_post = 50e-3):
    """
    orientation response of a given unit
    at its preferred phase and preferred spatial frequency !
    """
    # unit condition in analysis_metrics
    unit_cond = analysis_metrics.index.values==unit
    # get the session for that unit
    session_id = analysis_metrics[unit_cond].ecephys_session_id.values[0]
    session = cache.get_session_data(session_id)

    # get the spikes of that unit
    spike_times = session.spike_times[unit]

    # get the stimulus information
    stim_table = session.get_stimulus_table()
    
    # pick static gratings at preferred freq and preferred phase !
    print('preferred phase:', analysis_metrics[unit_cond].pref_phase_sg.values[0])
    print('preferred spatial freq:', analysis_metrics[unit_cond].pref_sf_sg.values[0])
    
    static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                    (stim_table.orientation!='null') & \
                    (stim_table.spatial_frequency.values.astype('str')==str(analysis_metrics[unit_cond].pref_sf_sg.values[0])) &\
                    (stim_table.phase.astype('str')==str(analysis_metrics[unit_cond].pref_phase_sg.values[0]))
    
    # build the bins for the psth
    duration = np.mean(stim_table[static_gratings].duration) # find stim duration
    bin_edges = np.arange(0-t_pre, duration+t_post, time_resolution)
    time_resolution = np.mean(np.diff(bin_edges))

    spike_matrix = np.zeros( (np.sum(static_gratings),
                              len(bin_edges)), dtype=bool)

    for trial_idx, trial_start in enumerate(stim_table[static_gratings].start_time.values):

        in_range = (spike_times > (trial_start + bin_edges[0])) * \
                   (spike_times < (trial_start + bin_edges[-1]))

        binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
        spike_matrix[trial_idx, binned_times] = True

    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': stim_table[static_gratings].index.values,
            'time_relative_to_stimulus_onset': bin_edges
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset']
    ), stim_table.orientation[static_gratings].values.astype('float')


Key = 'PV'
positive_IDs = np.concatenate(Optotagging[Key+'_positive_units'])
i = np.argmax(OSI[Key])
spikes_matrix, orientations = \
        compute_orientation_spike_resp(positive_IDs[i], analysis_metrics)

# %%
unit = positive_IDs[0]
unit_cond = analysis_metrics.index.values==unit
# get the session for that unit
session_id = analysis_metrics[unit_cond].ecephys_session_id.values[0]
session = cache.get_session_data(session_id)

# get the spikes of that unit
spike_times = session.spike_times[unit]

# get the stimulus information
stim_table = session.get_stimulus_table()
static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                  (stim_table.orientation!='null')


# %%
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
        
        self.time_resolution = np.mean(np.diff(self.t))

        self.spike_matrix = np.zeros( (stim_table.size,
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
                 smoothing=2e-3):
        
        return gaussian_filter1d(self.spike_matrix[:,:].mean(axis=0) / self.time_resolution,
                                int(smoothing/self.time_resolution))
    
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
             smoothing=2, 
             trial_subsampling=10,
             color='k', ms=1):
        
        fig = plt.figure(figsize=(1.2,2))
        plt.subplots_adjust(left=0.1, top=0.8, right=0.95)
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

        for t in range(self.spike_matrix.shape[0])[::trial_subsampling]:
            spike_cond = self.spike_matrix[t,:]==1
            ax1.plot(1e3*self.t[spike_cond],
                     self.spike_matrix[t,:][spike_cond]+t, '.', ms=ms, color=color)
        ax2.fill_between(1e3*self.t, 0*self.t, self.get_rate(), color=color)
        ax1.set_ylabel('trial #')
        ax2.set_ylabel('rate (Hz)')
        ax2.set_xlabel('time (s)')
        pt.set_common_xlims([ax1,ax2])

t = 1e-3*(-100+np.arange(400))
spikeResp = spikingResponse(stim_table[static_gratings], spike_times, t)
spikeResp.plot(trial_subsampling=20)

# %%
rate = gaussian_filter1d(spikeResp.spike_matrix[:,:].mean(axis=0) / spikeResp.time_resolution, 1)

plt.plot(1e3*spikeResp.t, rate) 

# %%
spikeResp.save('../data/visual_coding/test.npy')

# %%
spikeResp = spikingResponse(None, None, None, '../data/visual_coding/test.npy')

# %%
np.unique(spikeResp.orientation)

# %%
plt.plot(1e3*spikeResp.t, spikeResp.spike_matrix.mean(axis=0))


# %%

def show_single_unit_response(spikes_matrix, orientations,
                              smoothing = 10, # bins (i.e. *time_resolution),
                              color='k', ms=1):
    
    time_resolution = np.mean(np.diff(spikes_matrix.time_relative_to_stimulus_onset))

    fig = plt.figure(figsize=(6,2))
    plt.subplots_adjust(left=0.1, top=0.8, right=0.95)
    AX1, AX2 = [], []

    for o, ori in enumerate(np.unique(orientations)):
        ax1 = plt.subplot2grid((5, len(np.unique(orientations))), (0, o), rowspan=3)
        ax2 = plt.subplot2grid((5, len(np.unique(orientations))), (3, o), rowspan=2)

        ax1.set_title('$\\theta$=%.1f$^o$' % ori)
        cond = (orientations==ori)
        for t, trial in enumerate(np.arange(len(orientations))[cond]):
            spike_cond = spikes_matrix[trial,:]==1
            ax1.plot(spikes_matrix.time_relative_to_stimulus_onset.values[spike_cond],
                    spikes_matrix[trial,:][spike_cond]+t, '.', ms=ms, color=color)
        ax2.bar(spikes_matrix.time_relative_to_stimulus_onset.values,
                 gaussian_filter1d(spikes_matrix[cond,:].mean(axis=0) / time_resolution, smoothing), 
                 width=time_resolution, color=color)
        AX1.append(ax1)
        AX2.append(ax2)
        if ax1==AX1[0]:
            ax1.set_ylabel('trial #')
            ax2.set_ylabel('rate (Hz)')
            ax2.set_xlabel(80*' '+'time (s)')
        else:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        ax1.set_xticklabels([])
    pt.set_common_xlims(AX1+AX2)
    for AX in [AX1, AX2]:
        pt.set_common_ylims(AX)
        for ax in AX:
            ax.set_xticks([0, 0.25])
        
    return fig

fig = show_single_unit_response(spikes_matrix, orientations, ms=0.5, color='tab:red')
fig.suptitle('unit %i, OSI=%.2f \n \n' % (positive_IDs[i], OSI[Key][i]), color='k')
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%
# loop over all units
for Key, color in zip(['PV', 'SST'], ['tab:red', 'tab:orange']):
    positive_IDs = np.concatenate(Optotagging[Key+'_positive_units'])
    for k, i in enumerate(np.argsort(OSI[Key])[::-1]):
        fig = show_single_unit_response(\
                *compute_orientation_spike_resp(positive_IDs[i], analysis_metrics),
                ms=0.5, color=color)
        fig.suptitle('unit %i, OSI=%.2f \n \n' % (positive_IDs[i], OSI[Key][i]), color='k')
        fig.savefig(os.path.join('..', 'figures', 'VisualCoding', Key+'-resp', '%i.png' % (k+1)))
        plt.close(fig)

# %%
for k, i in enumerate(np.concatenate([\
                            np.argsort(analysis_metrics.g_osi_sg.values)[::-1][:10],
                            np.argsort(analysis_metrics.g_osi_sg.values)[::-1][::200][1:]])):
    unit = analysis_metrics.index.values[i]
    fig = show_single_unit_response(\
            *compute_orientation_spike_resp(unit, analysis_metrics),
            ms=1, color='k')
    fig.suptitle('unit %i, OSI=%.2f \n \n' % (unit, analysis_metrics.g_osi_sg.values[i]), color='k')
    fig.savefig(os.path.join('..', 'figures', 'VisualCoding', 'Others-resp', '%i.png' % (k+1)))
    plt.close(fig)

# %%
