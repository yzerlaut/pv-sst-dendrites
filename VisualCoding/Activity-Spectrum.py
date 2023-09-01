# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
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
sys.path.append('../src')
import fourier_for_real as fourier
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
# # 2) Compute the

# %%
def get_spike_counts(sessionID, positive_units,
                     structure='VISp',
                     dt=5e-3):
    session = cache.get_session_data(sessionID)
    unit_metrics = cache.get_unit_analysis_metrics_for_session(sessionID)
    # time interval
    stims = session.get_stimulus_table()
    tstart= stims[stims.index==0].start_time.values[0]
    tstop = stims[stims.index==(len(stims)-1)].stop_time.values[0]

    t_binned = tstart+np.arange(int((tstop-tstart)/dt))*dt
    positive_units_rate, negative_units_rate = 0*t_binned, 0*t_binned

    # fetch all spike times (need to discard invalid times here)
    spike_times = session.spike_times


    results = {'dt':dt, 'positive_spikes':[], 'negative_spikes':[]}

    structure_cond = [structure in x for x in unit_metrics.ecephys_structure_acronym.values]
    nPos, nNeg = 0., 0.
    for unitID in unit_metrics.index[structure_cond]:
        if unitID in positive_units:
            positive_units_rate[1:] += np.histogram(spike_times[unitID], bins=t_binned)[0]
            nPos += 1.
            results['positive_spikes'].append(spike_times[unitID])
        else:
            negative_units_rate[1:] += np.histogram(spike_times[unitID], bins=t_binned)[0]
            nNeg += 1.
            results['negative_spikes'].append(spike_times[unitID])

    results.update({'t':t_binned,
                   'positive_rate':positive_units_rate/nPos/dt,
                   'negative_rate':negative_units_rate/nNeg/dt})

    return results

index = 4 
results = get_spike_counts(Optotagging['PV_sessions'][index],
                           Optotagging['PV_positive_units'][index])

# %%

def spectrum_fig(results,
                 tlim=[50, 55],
                 freq_range = [0.05, 50],
                 rate_smoothing=50e-3,
                 pos_color='tab:red'):

    fig = plt.figure(figsize=(6,2))
    plt.subplots_adjust(hspace=0.1, wspace=1.2, bottom=0.2)

    # raw data - spikes
    ax = plt.subplot2grid((2,5), (0,0), colspan=3)
    n=0
    for spikes in results['positive_spikes']:
        cond = (spikes>tlim[0]) & (spikes<tlim[1])
        ax.scatter(spikes[cond], n+np.zeros(np.sum(cond)), color=pos_color, s=0.5)
        n+=1
    for spikes in results['negative_spikes']:
        cond = (spikes>tlim[0]) & (spikes<tlim[1])
        ax.scatter(spikes[cond], n+np.zeros(np.sum(cond)), color='tab:grey', s=0.5)
        n+=1

    pt.set_plot(ax, [], xlim=tlim, ylabel='units')


    # raw data - smoothed rates
    ax = plt.subplot2grid((2,5), (1,0), colspan=3)
    smoothing = int(rate_smoothing/results['dt'])
    cond = (results['t']>tlim[0]) & (results['t']<tlim[1])
    ax.plot(results['t'][cond], gaussian_filter1d(results['negative_rate'][cond], smoothing), color='tab:grey')
    ax.plot(results['t'][cond], gaussian_filter1d(results['positive_rate'][cond], smoothing), color=pos_color)
    pt.set_plot(ax, ['left'], xlim=tlim, ylabel='rate (Hz)')
    pt.draw_bar_scales(ax, Xbar=0.5, Xbar_label='0.5s', Ybar=1e-12)

    # spectrum
    ax = plt.subplot2grid((2,5), (0,3), rowspan=2, colspan=2)

    negSpectrum = np.abs(fourier.FT(results['negative_rate'], len(results['t']), results['dt']))
    posSpectrum = np.abs(fourier.FT(results['positive_rate'], len(results['t']), results['dt']))
    freq = fourier.time_to_freq(len(results['t']), results['dt'])

    smoothing = 50

    cond = (freq>freq_range[0]) & (freq<freq_range[1])
    negSpectrum_clean = gaussian_filter1d(negSpectrum[cond], smoothing)
    ax.plot(freq[cond], negSpectrum_clean/negSpectrum_clean[0], color='tab:grey')
    posSpectrum_clean = gaussian_filter1d(posSpectrum[cond], smoothing)
    ax.plot(freq[cond], posSpectrum_clean/posSpectrum_clean[0], color=pos_color)

    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
            ylabel='norm. pow.', yticks=[0.1, 1])

    return fig


fig = spectrum_fig(results)


# %%
index = 11 
results = get_spike_counts(Optotagging['SST_sessions'][index],
                           Optotagging['SST_positive_units'][index])
fig = spectrum_fig(results, pos_color='tab:orange')


# %%                                       
index = 10  
results = get_spike_counts(Optotagging['SST_sessions'][index],
                           Optotagging['SST_positive_units'][index])
fig = spectrum_fig(results, pos_color='tab:orange')


# %%                                       
index = 4 
results = get_spike_counts(Optotagging['PV_sessions'][index],
                           Optotagging['PV_positive_units'][index])
fig = spectrum_fig(results, pos_color='tab:red')


# %% [markdown]
# # Wavelet-Based analysis: spike-triggered wavelet envelope

# %%
from wavelet_transform import my_cwt

# %%
def wavelet_fig(results,
                freqs = np.logspace(-1, 2, 30)
                pos_color = 'tab:red'):

    print('wavelet-transform [...]')
    coefs = my_cwt(results['negative_rate'], freqs, results['dt'])
    envelope = np.abs(coefs)

    fig, ax = plt.subplots(1, figsize=(3,2))

    ax.plot(freqs, np.mean(envelope, axis=1),
            color='tab:grey', label='average')

    print('spike-triggered analysis[...]')
    all_spikes = np.concatenate(results['positive_spikes'])
    spikeTrig_env = np.zeros(len(freqs))
    for s in all_spikes:
        i = np.argmin((results['t']-s)**2)
        spikeTrig_env += envelope[:,i]
    spikeTrig_env /= len(all_spikes)

    print(' done ! ')
    ax.plot(freqs, spikeTrig_env,
            color=pos_color, label='spike triggered')
    ax.legend()
    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
            ylabel='rate envelope (Hz)', yticks=[0.1, 1])




