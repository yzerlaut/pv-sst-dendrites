# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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
import xarray as xr
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# %%
data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# %%
sessions = cache.get_session_table()

# %%
session = cache.get_session_data(sessions.index.values[0])

# %%
V1_units = session.units[session.units.ecephys_structure_acronym == 'VISp']
stim_table = session.get_stimulus_table()

# %%
# get V1 spikes
V1_RASTER = []
for i in V1_units.index:
    V1_RASTER.append(session.spike_times[i])

# %%
# plot raster
t0, duration, subsampling = 1000, 50000, 1000
plt.figure()
for i, spikes in enumerate(V1_RASTER):
    cond = (spikes>t0) & (spikes<(t0+duration))
    plt.plot(spikes[cond][::subsampling], i+0*spikes[cond][::subsampling], 'k.', ms=2)

# %%
static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                    (stim_table.orientation!='null') & \
                    (stim_table.contrast==0.8) # fixed contrast !

mean_duration = np.mean(stim_table.duration[static_gratings].values.astype('float')) # mean stim duration
orientations = stim_table.orientation[static_gratings].values.astype('float')

# %%
units = session.units[session.units.ecephys_structure_acronym.str.match('VIS')]

time_resolution = 1e-3

def ori_response_spike_counts(session, units,
                              bin_edges = np.arange(-0.25, 0.5, time_resolution)):
    
    time_resolution = np.mean(np.diff(bin_edges))
    
    # get the stimulus information
    stim_table = session.get_stimulus_table()
    static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                    (stim_table.orientation!='null') & \
                    (stim_table.contrast==0.8) # fixed contrast !    
    
    
    spike_matrix = np.zeros( (np.sum(static_gratings),
                              len(bin_edges),
                              len(units)) )

    for unit_idx, unit_id in enumerate(units.index.values):

        spike_times = session.spike_times[unit_id]

        for trial_idx, trial_start in enumerate(stim_table[static_gratings].start_time.values):

            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))

            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            spike_matrix[trial_idx, binned_times, unit_idx] = 1

    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': stim_table[static_gratings].index.values,
            'time_relative_to_stimulus_onset': bin_edges,
            'unit_id': units.index.values
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id']
    ), stim_table.orientation[static_gratings].values.astype('float')


spikes_matrix ,orientations = ori_response_spike_counts(session, units)

# %%
import sys
sys.path.append('..')
import plot_tools as pt
from scipy.ndimage import gaussian_filter1d


# %%

def show_single_unit_response(spikes_matrix, unit,
                              smoothing = 10, # bins (i.e. *time_resolution),
                              color='k'):
    
    time_resolution = np.mean(np.diff(spikes_matrix.time_relative_to_stimulus_onset))

    fig = plt.figure(figsize=(8,3))
    AX1, AX2 = [], []

    for o, ori in enumerate(np.unique(orientations)):
        ax1 = plt.subplot2grid((5, len(np.unique(orientations))), (0, o), rowspan=3)
        ax2 = plt.subplot2grid((5, len(np.unique(orientations))), (3, o), rowspan=2)

        ax1.set_title('$\\theta$=%.1f$^o$' % ori, fontsize=8)
        cond = (orientations==ori)
        for t, trial in enumerate(np.arange(len(orientations))[cond]):
            spike_cond = spikes_matrix[trial,:,unit]==1
            ax1.plot(spikes_matrix.time_relative_to_stimulus_onset.values[spike_cond],
                    spikes_matrix[trial,:,unit][spike_cond]+t, '.', ms=1, color=color)
        ax2.bar(spikes_matrix.time_relative_to_stimulus_onset.values,
                 gaussian_filter1d(spikes_matrix[cond,:,unit].mean(axis=0) / time_resolution, smoothing), 
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
        
    return fig

unit = 187
fig = show_single_unit_response(spikes_matrix, unit)

# %%
OSI = []
for unit in units.index:
    OSI.append(analysis_metrics[analysis_metrics.index==unit].g_osi_sg.values[0])
plt.hist(OSI, bins=20)
np.argsort(OSI)[-5:]

# %%

# %%
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

# %%
for k in analysis_metrics1.keys():
    print(k)

# %%
trials = session.optogenetic_stimulation_epochs[(session.optogenetic_stimulation_epochs.duration > 0.009) & \
                                                (session.optogenetic_stimulation_epochs.duration < 0.02)]

units = session.units[session.units.ecephys_structure_acronym.str.match('VIS')]

time_resolution = 0.0005 # 0.5 ms bins

bin_edges = np.arange(-0.01, 0.025, time_resolution)

def optotagging_spike_counts(bin_edges, trials, units):
    
    time_resolution = np.mean(np.diff(bin_edges))

    spike_matrix = np.zeros( (len(trials), len(bin_edges), len(units)) )

    for unit_idx, unit_id in enumerate(units.index.values):

        spike_times = session.spike_times[unit_id]

        for trial_idx, trial_start in enumerate(trials.start_time.values):

            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))

            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            spike_matrix[trial_idx, binned_times, unit_idx] = 1

    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': trials.index.values,
            'time_relative_to_stimulus_onset': bin_edges,
            'unit_id': units.index.values
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id']
    )

da = optotagging_spike_counts(bin_edges, trials, units)


# %%
def plot_optotagging_response(da):

    plt.figure(figsize=(5,10))

    plt.imshow(da.mean(dim='trial_id').T / time_resolution, 
               extent=[np.min(bin_edges), np.max(bin_edges),
                       0, len(units)],
               aspect='auto', vmin=0, vmax=200)    

    for bound in [0.0005, 0.0095]:
        plt.plot([bound, bound],[0, len(units)], ':', color='white', linewidth=1.0)

    plt.xlabel('Time (s)')
    plt.ylabel('Unit #')

    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Mean firing rate (Hz)')
    
plot_optotagging_response(da)

# %%
baseline = da.sel(time_relative_to_stimulus_onset=slice(-0.01,-0.002))
baseline_rate = baseline.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
evoked = da.sel(time_relative_to_stimulus_onset=slice(0.001,0.009))
evoked_rate = evoked.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008

# %%
plt.figure(figsize=(5,5))

plt.scatter(baseline_rate, evoked_rate, s=3)

axis_limit = 150
plt.plot([0,axis_limit],[0,axis_limit], ':k')
plt.plot([0,axis_limit],[0,axis_limit*4], ':r')
plt.xlim([0,axis_limit])
plt.ylim([0,axis_limit])

plt.xlabel('Baseline rate (Hz)')
_ = plt.ylabel('Evoked rate (Hz)')

# %%
pos_cond =  ((evoked_rate / (baseline_rate + 1)) > 4) & (evoked_rate>10)
cre_pos_units = (da.unit_id[pos_cond].values) # add 1 to prevent divide-by-zero errors
cre_pos_units

# %%
