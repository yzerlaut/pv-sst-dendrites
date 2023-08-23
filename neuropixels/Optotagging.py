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

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# %%
data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

# %% [markdown]
# ## Genotypes used for Phototagging

# %%
sessions.full_genotype.value_counts()

# %% [markdown]
# # 1) Identifying PV+ units

# %%
PV_sessions = sessions[(sessions.full_genotype.str.find('Pvalb-IRES-Cre') > -1) & \
                        (sessions.session_type == 'brain_observatory_1.1') & \
                        #(sessions.session_type == 'functional_connectivity') & \
                        (['VISp' in acronyms for acronyms in sessions.ecephys_structure_acronyms])]
PV_sessions

# %%
session = cache.get_session_data(PV_sessions.index.values[0])

# %%
session.optogenetic_stimulation_epochs

# %%
columns = ['stimulus_name', 'duration','level']
session.optogenetic_stimulation_epochs.drop_duplicates(columns).sort_values(by=columns).drop(columns=['start_time','stop_time'])

# %%
trials = session.optogenetic_stimulation_epochs[session.optogenetic_stimulation_epochs.stimulus_name=='fast_pulses']
trials = session.optogenetic_stimulation_epochs[(session.optogenetic_stimulation_epochs.duration > 0.009) & \
                                                (session.optogenetic_stimulation_epochs.duration < 0.02)]
units = session.units[session.units.ecephys_structure_acronym.str.match('VIS')]

def optotagging_spike_counts(trials, units,
                             time_resolution = 1e-3):
    
    duration = np.mean(trials.duration.values)
    bin_edges = np.arange(-duration, 2*duration, time_resolution)
    
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

spikes_matrix = optotagging_spike_counts(trials, units)


# %%
def plot_optotagging_response(spikes_matrix,
                              duration=None):

    time_resolution = np.mean(np.diff(spikes_matrix.time_relative_to_stimulus_onset))
    
    Rates = spikes_matrix.mean(dim='trial_id').T / time_resolution
    plt.imshow(Rates, 
               extent=[np.min(spikes_matrix.time_relative_to_stimulus_onset.values),
                       np.max(spikes_matrix.time_relative_to_stimulus_onset.values),
                       0, spikes_matrix.shape[2]],
               aspect='auto', vmin=0, vmax=np.min([100, 1.2*np.max(Rates)]))

    if duration is not None:
        for bound in [0, duration]:
            plt.plot([bound, bound],[0, len(units)], ':', color='white', linewidth=1.0)
        plt.xticks([0, duration])
        
    plt.xlabel('time (s)')
    plt.ylabel('Unit #')

    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Mean firing rate (Hz)\n(n=%i trials)' % len(spikes_matrix.trial_id.values))
    
plot_optotagging_response(spikes_matrix, duration=0.01)


# %%
def analyze_optotagging_responses(trials, units,
                                  spikes_matrix=None,
                                  inclusion_evoked_factor=4.,
                                  inclusion_min_evoked=5.,
                                  label='',
                                  with_fig=False):
    
    if spikes_matrix is None:
        spikes_matrix = optotagging_spike_counts(trials, units)
        
    duration = np.mean(trials.duration.values)
    dt = np.mean(np.diff(spikes_matrix.time_relative_to_stimulus_onset))

    baseline = spikes_matrix.sel(time_relative_to_stimulus_onset=slice(-duration-2*dt,-2*dt))
    baseline_rate = baseline.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / duration
    evoked = spikes_matrix.sel(time_relative_to_stimulus_onset=slice(2*dt, duration+2*dt))
    evoked_rate = evoked.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / duration
    
    
    inclusion_cond =  ((evoked_rate>inclusion_min_evoked) &\
                       ((evoked_rate / (baseline_rate + 1)) > inclusion_evoked_factor)) # add 1 to prevent divide-by-zero errors
    cre_pos_units = (spikes_matrix.unit_id[inclusion_cond].values) 

    if with_fig:
        fig = plt.figure(figsize=(8, 2.5))
        plt.subplots_adjust(left=0.2, wspace=1)

        plt.subplot2grid((6,6), (0,0), colspan=3, rowspan=6)
        
        pt.annotate(fig, label, (0., 0.5))
            
        plot_optotagging_response(spikes_matrix, duration=duration)
        
        plt.subplot2grid((6,6), (1,4), colspan=2, rowspan=4)
    
        plt.scatter(baseline_rate, evoked_rate, s=1)
        plt.scatter(baseline_rate[inclusion_cond], evoked_rate[inclusion_cond], s=2, color='r')

        axis_limit = 1.2*np.max(evoked_rate)
        plt.plot([0,axis_limit],[0,axis_limit], ':k')
        plt.plot([0,axis_limit],[0,axis_limit*inclusion_evoked_factor], ':r', lw=0.5)
        plt.xlim([0,axis_limit])
        plt.ylim([0,axis_limit])

        plt.title('%i/%i Cre-positive (%.1f%%)' % (np.sum(inclusion_cond), len(inclusion_cond),
                                                   100.*np.sum(inclusion_cond)/len(inclusion_cond)))
        plt.xlabel('Baseline rate (Hz)')
        plt.ylabel('Evoked rate (Hz)')
        
        
    else:
        fig = None
        
    return cre_pos_units, fig
    

    
_ = analyze_optotagging_responses(trials, units, spikes_matrix,
                                  with_fig=True)

# %%
trials = session.optogenetic_stimulation_epochs[\
                (session.optogenetic_stimulation_epochs.stimulus_name=='pulse') & 
                (session.optogenetic_stimulation_epochs.duration > 0.009) &
                (session.optogenetic_stimulation_epochs.duration < 0.02) & 
                (session.optogenetic_stimulation_epochs.level>=2)]
_ = analyze_optotagging_responses(trials, units, spikes_matrix,
                                  label='10ms pulse', with_fig=True)

# %%
trials = session.optogenetic_stimulation_epochs[\
                (session.optogenetic_stimulation_epochs.stimulus_name=='pulse') & 
                (session.optogenetic_stimulation_epochs.duration > 0.003) &
                (session.optogenetic_stimulation_epochs.duration < 0.009) & 
                (session.optogenetic_stimulation_epochs.level>=2)]
_ = analyze_optotagging_responses(trials, units, spikes_matrix,
                                  label='5ms pulse', with_fig=True)

# %%
trials = session.optogenetic_stimulation_epochs[\
        (session.optogenetic_stimulation_epochs.stimulus_name=='fast_pulses') & 
        (session.optogenetic_stimulation_epochs.level>=2)]
_ = analyze_optotagging_responses(trials, units,
                                  label='2.5ms pulses\n  @ 10Hz', with_fig=True)

# %%
trials = session.optogenetic_stimulation_epochs[\
        (session.optogenetic_stimulation_epochs.stimulus_name=='raised_cosine')& 
        (session.optogenetic_stimulation_epochs.level>=2)]
_ = analyze_optotagging_responses(trials, units,
                                  inclusion_evoked_factor=2.,
                                  label='raised cosine', with_fig=True)
