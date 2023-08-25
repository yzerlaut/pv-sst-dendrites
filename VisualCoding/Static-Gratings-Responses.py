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
Optotagging = np.load(os.path.join('..', 'data', 'Optotagging-Results.npy'),
                      allow_pickle=True).item()

# %% [markdown]
# # 2) Analyze using the pre-computed visual response metrics

# %%
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

# %%
from scipy.stats import mannwhitneyu

fig, ax = plt.subplots(1)
inset = pt.inset(ax, [1.4, -0.05, 0.3, 0.6])

OSI = {'Others':[]}
for i, Key in enumerate(['PV', 'SST']):
    OSI[Key] = []
    positive_IDs = np.concatenate(Optotagging[Key+'_positive_units'])
    for unit in positive_IDs:
        OSI[Key].append(analysis_metrics[analysis_metrics.index==unit].g_osi_sg.values[0])
    for unit in analysis_metrics[\
                    analysis_metrics.ecephys_structure_acronym.isin(['VISp'])].index.values:
        if unit not in positive_IDs:
            OSI['Others'].append(analysis_metrics[analysis_metrics.index==unit].g_osi_sg.values[0])

COLORS=['w', 'tab:red', 'tab:orange', 'lightgrey']
for i, Key in enumerate(['Others', 'PV', 'SST']):
    pt.annotate(ax, i*'\n'+'n=%i units' % len(OSI[Key]), (1,1), color=COLORS[i], ha='right', va='top')
    ax.hist(OSI[Key], bins=np.linspace(0, 1, 30), color=COLORS[i],
             label=Key.replace('_sessions', '+ units'), alpha=1 if i==0 else 0.6, density=True)
    pt.violin(OSI[Key], X=[i], COLORS=[COLORS[i]], ax=inset)
inset.set_xticks([])
inset.set_ylabel('OSI')
inset.set_title('PV vs SST, p=%.1e' % mannwhitneyu(OSI['PV'], OSI['SST']).pvalue)
ax.set_xlabel('OSI')
ax.set_ylabel('count')
ax.legend(loc=(1.1, 0.8))

# %%
def find_session_unit_from_index(index, Key):
    i, k = 0, 0
    while index>(k+len(Optotagging[Key.replace('sessions', 'positive_units')][i])):
        k+=len(Optotagging[Key.replace('sessions', 'positive_units')][i])
        i+=1
    return i, index-k
find_session_unit_from_index(8, 'SST_sessions')

# %% [markdown]
# # 3) Recover the Original Orientation-Dependent PSTH

# %%
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
# %%
Key = 'SST_sessions'
session_index = 1
session_id = Optotagging[Key][session_index]
session = cache.get_session_data(session_id)
stim_table = session.get_stimulus_table()

# %%
# get spikes from units
RASTER = []
for i in Optotagging[Key.replace('sessions', 'positive_units')][session_index]:
    RASTER.append(session.spike_times[i])

# %%
# plot raster
t0, duration, subsampling = 1000, 50000, 1000
plt.figure()
for i, spikes in enumerate(RASTER):
    cond = (spikes>t0) & (spikes<(t0+duration))
    plt.plot(spikes[cond][::subsampling], i+0*spikes[cond][::subsampling], '.', ms=2)

# %%
static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                    (stim_table.orientation!='null') & \
                    (stim_table.contrast==0.8) # fixed contrast !

mean_duration = np.mean(\
        stim_table.duration[static_gratings].values.astype('float')) # mean stim duration
orientations = stim_table.orientation[static_gratings].values.astype('float')

# %% 

def compute_orientation_spike_resp(unit, analysis_metrics,
                                   time_resolution = 1e-3,
                                   t_pre = 50e-3, t_post = 50e-3):

    # get the session for that unit
    session_id = analysis_metrics[analysis_metrics.index.values==unit].ecephys_session_id.values[0]
    session = cache.get_session_data(session_id)
    stim_table = session.get_stimulus_table()

    # get the spikes of that unit
    spike_times = session.spike_times[unit]

    # get the stimulus information
    stim_table = session.get_stimulus_table()
    static_gratings = (stim_table.stimulus_name=='static_gratings') &\
                    (stim_table.orientation!='null') & \
                    (stim_table.contrast==0.8) # fixed contrast !    
    

    # build the bins for the psth
    duration = np.mean(stim_table[static_gratings].duration) # find stim duration
    bin_edges = np.arange(0-t_pre, duration+t_post, time_resolution)
    time_resolution = np.mean(np.diff(bin_edges))

    spike_matrix = np.zeros( (np.sum(static_gratings),
                              len(bin_edges)) )

    for trial_idx, trial_start in enumerate(stim_table[static_gratings].start_time.values):

        in_range = (spike_times > (trial_start + bin_edges[0])) * \
                   (spike_times < (trial_start + bin_edges[-1]))

        binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
        spike_matrix[trial_idx, binned_times] = 1

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
        
    return fig

fig = show_single_unit_response(spikes_matrix, orientations, ms=0.5, color='tab:red')
fig.suptitle('unit %i, OSI=%.2f \n \n' % (positive_IDs[i], OSI[Key][i]), color='k')
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

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


    print(i)
