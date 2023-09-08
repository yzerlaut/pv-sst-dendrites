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
# # 2) Compute the Frequency Spectrum from Fourier Analysis

# %%
def get_spike_counts(sessionID, positive_units,
                     structure='VISp',
                     dt=5e-3, smoothing=20e-3,
                     tstart=-np.inf, 
                     tstop=np.inf):
    
    session = cache.get_session_data(sessionID)
    unit_metrics = cache.get_unit_analysis_metrics_for_session(sessionID)
    # time interval
    stims = session.get_stimulus_table()
    if tstart==-np.inf:
        tstart= stims[stims.index==0].start_time.values[0]
    if tstop==np.inf:
        tstop = stims[stims.index==(len(stims)-1)].stop_time.values[0]

    t_binned = tstart+np.arange(int((tstop-tstart)/dt))*dt
    positive_units_rate, negative_units_rate = 0*t_binned, 0*t_binned

    # fetch all spike times (need to discard invalid times here)
    spike_times = session.spike_times

    results = {'dt':dt, 
               'posUnits_spikes':[], 'negUnits_spikes':[]}

    structure_cond = [structure in x for x in unit_metrics.ecephys_structure_acronym.values]
    nPos, nNeg = 0., 0.
    for unitID in unit_metrics.index[structure_cond]:
        if unitID in positive_units:
            positive_units_rate[1:] += np.histogram(spike_times[unitID], bins=t_binned)[0]
            nPos += 1.
            results['posUnits_spikes'].append(spike_times[unitID])
        else:
            negative_units_rate[1:] += np.histogram(spike_times[unitID], bins=t_binned)[0]
            nNeg += 1.
            results['negUnits_spikes'].append(spike_times[unitID])

    positive_units_rate = gaussian_filter1d(positive_units_rate, int(smoothing/dt))
    negative_units_rate = gaussian_filter1d(negative_units_rate, int(smoothing/dt))
    
    results.update({'t':t_binned,
                   'posUnits_rate':positive_units_rate/nPos/dt,
                   'negUnits_rate':negative_units_rate/nNeg/dt})

    return results

# %%
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
    for spikes in results['posUnits_spikes']:
        cond = (spikes>tlim[0]) & (spikes<tlim[1])
        ax.scatter(spikes[cond], n+np.zeros(np.sum(cond)), color=pos_color, s=0.5)
        n+=1
    for spikes in results['negUnits_spikes']:
        cond = (spikes>tlim[0]) & (spikes<tlim[1])
        ax.scatter(spikes[cond], n+np.zeros(np.sum(cond)), color='tab:grey', s=0.5)
        n+=1

    pt.set_plot(ax, [], xlim=tlim, ylabel='units')

    # raw data - smoothed rates
    ax = plt.subplot2grid((2,5), (1,0), colspan=3)
    smoothing = int(rate_smoothing/results['dt'])
    cond = (results['t']>tlim[0]) & (results['t']<tlim[1])
    ax.plot(results['t'][cond], gaussian_filter1d(results['negUnits_rate'][cond], smoothing), color='tab:grey')
    ax.plot(results['t'][cond], gaussian_filter1d(results['posUnits_rate'][cond], smoothing), color=pos_color)
    pt.set_plot(ax, ['left'], xlim=tlim, ylabel='rate (Hz)')
    pt.draw_bar_scales(ax, Xbar=0.5, Xbar_label='0.5s', Ybar=1e-12)

    # spectrum
    ax = plt.subplot2grid((2,5), (0,3), rowspan=2, colspan=2)

    negSpectrum = np.abs(fourier.FT(results['negUnits_rate'], len(results['t']), results['dt']))
    posSpectrum = np.abs(fourier.FT(results['posUnits_rate'], len(results['t']), results['dt']))
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
index = 4
results = get_spike_counts(Optotagging['PV_sessions'][index],
                           Optotagging['PV_positive_units'][index],
                           tstart=100, tstop=150, 
                           dt=2.5e-3, smoothing=2.5e-3)
#fig = wavelet_fig(results, pos_color='tab:red')

# %%
freqs = np.logspace(-1, 2, 20)
coefs = my_cwt(results['negUnits_rate'], freqs, results['dt'])
pt.time_freq_signal(results['t'], freqs, results['negUnits_rate'], coefs, freq_scale='log')


# %%
def compute_envelopes(results, freqs,
                      verbose=False):
    
    if verbose:
        print('wavelet-transform [...]')
    coefs = my_cwt(results['negUnits_rate'], freqs, results['dt'])
    envelope = np.abs(coefs)
    results['average-envelope'] = np.mean(envelope, axis=1)
    results['freqs'] = freqs
    
    if verbose:
        print('spike-triggered analysis[...]')
        
    # all units together, session data
    all_spikes = np.concatenate(results['posUnits_spikes'])
    iSpikes = np.digitize(all_spikes, results['t'])
    cond = (iSpikes>0) & (iSpikes!=len(results['t']))
    iSpikes = iSpikes[cond]    
    results['spikeTrig-envelope-all-spikes'] = envelope[:,iSpikes].mean(axis=1)
    
    # now unit per unit
    results['spikeTrig-envelope'] = []
    for spikes in results['posUnits_spikes']:
        iSpikes = np.digitize(spikes, results['t'])
        cond = (iSpikes>0) & (iSpikes!=len(results['t']))
        iSpikes = iSpikes[cond]    
        results['spikeTrig-envelope'].append(envelope[:,iSpikes].mean(axis=1))
        
    
def wavelet_fig(results,
                pos_color = 'tab:red'):

    fig, ax = plt.subplots(1, figsize=(3,2))

    ax.plot(results['freqs'], results['average-envelope'],
            color='tab:grey', label='average')

    #ax.plot(freqs, results['spikeTrig-envelope-all-spikes'])

    pt.plot(results['freqs'], np.mean(results['spikeTrig-envelope'], axis=0),
            sy = np.std(results['spikeTrig-envelope'], axis=0), ax=ax,
            color=pos_color, label='spike triggered')
    
    ax.legend()

    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
                ylabel='rate envelope (Hz)', yticks=[0.1, 1])




# %%
compute_envelopes(results, np.logspace(-1, 2, 20))
fig = wavelet_fig(results, pos_color='tab:red')

# %%
RESULTS ={'freqs': np.logspace(-1, 2, 20)}

for key in ['PV', 'SST']:
    RESULTS['%s_average_env' % key] = []
    RESULTS['%s_spikeTrig_env_per_session' % key] = []
    RESULTS['%s_spikeTrig_env_per_unit' % key] = []
    for index in range(len(Optotagging['%s_sessions'%key])):
        print(key, 'session #', index+1)
        results = get_spike_counts(Optotagging['%s_sessions' % key][index],
                                   Optotagging['%s_positive_units' % key][index],
                                   #tstart=50, tstop=150,
                                   dt=2.5e-3, smoothing=2.5e-3)
        if len(results['posUnits_spikes'])>0:
            compute_envelopes(results, RESULTS['freqs'])
            RESULTS['%s_average_env' % key].append(results['average-envelope'])
            RESULTS['%s_spikeTrig_env_per_session' % key].append(results['spikeTrig-envelope-all-spikes'])
            RESULTS['%s_spikeTrig_env_per_unit' % key] += results['spikeTrig-envelope']
            
np.save('../data/spike-triggered-spectrogram-average.npy', RESULTS)

# %%
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_average_env' % key], axis=0),
            sy=np.std(RESULTS['%s_average_env' % key], axis=0),
            color='tab:grey',ax=ax)

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_spikeTrig_env_per_session' % key], axis=0),
            sy=np.std(RESULTS['%s_spikeTrig_env_per_session' % key], axis=0),
            color=color, ax=ax)

    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
                title='%s-cre mice (N=%i sessions)' % (key, len(RESULTS['%s_spikeTrig_env_per_session' % key])),
                ylabel='rate envelope (Hz)' if key=='PV' else '',
                yticks=[0.1, 1])

pt.set_common_ylims(AX)

# %%
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_average_env' % key], axis=0),
            sy=np.std(RESULTS['%s_average_env' % key], axis=0),
            color='tab:grey',ax=ax)

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_spikeTrig_env_per_unit' % key], axis=0),
            sy=np.std(RESULTS['%s_spikeTrig_env_per_unit' % key], axis=0),
            color=color, ax=ax)

    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
                title='%s-cre mice (n=%i units)' % (key, len(RESULTS['%s_spikeTrig_env_per_unit' % key])),
                ylabel='rate envelope (Hz)' if key=='PV' else '',
                yticks=[0.1, 1])

pt.set_common_ylims(AX)

# %%
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, ax = plt.subplots(1, figsize=(3,2))
inset = pt.inset(ax, [1.3, 0.1, 0.3, 0.8])

for i, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):

    RATIOS = [y/x for x, y in zip(RESULTS['%s_average_env' % key], RESULTS['%s_spikeTrig_env' % key])]
    #print(RATIOS)
    pt.plot(RESULTS['freqs'], np.mean(RATIOS, axis=0),
            sy=np.std(RATIOS, axis=0), color=color, ax=ax)

    y = [RESULTS['freqs'][np.argmax(r)] for r in RATIOS]
    pt.violin(y, X=[i], ax=inset, COLORS=[color])
    
pt.annotate(ax, 'PV+ spikes', (0,1), va='top', color='tab:red')
pt.annotate(ax, '\nSST+ spikes', (0,1), va='top', color='tab:orange')

pt.set_plot(ax, 
            xscale='log', 
            #yscale='log', 
            ylabel='env$_{spikeTrig}$ / env$_{average}$',
            xlabel='freq. (Hz)')

pt.set_plot(inset, xticks_labels=[],
            yscale='log', ylim=[RESULTS['freqs'][0], RESULTS['freqs'][-1]],
            ylabel='peak freq. (Hz)')

# %%
