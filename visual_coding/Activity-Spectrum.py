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
from scipy import stats

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
# # 1) Get the spiking data and generate a summary datafile

# %%
## Fetch the Optotagging Results
Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'),
                      allow_pickle=True).item()

# %%
def get_spike_counts(sessionID, positive_units,
                     structure='VISp',
                     stim='all', # either "spontaneous", "natural-movies" or "all"
                     spontaneous_only=True,
                     dt=5e-3, smoothing=20e-3,
                     tstart=-np.inf, 
                     tstop=np.inf):
    
    session = cache.get_session_data(sessionID)
    unit_metrics = cache.get_unit_analysis_metrics_for_session(sessionID)
    # time interval
    stims = session.get_stimulus_table()
    
    if tstart==-np.inf:
        tstart= 0 # stims[stims.index==0].start_time.values[0]
    if tstop==np.inf:
        tstop = stims[stims.index==(len(stims)-1)].stop_time.values[0]

    t_binned = tstart+np.arange(int((tstop-tstart)/dt))*dt
    positive_units_rate, negative_units_rate = 0*t_binned, 0*t_binned

    # fetch all spike times (need to discard invalid times here)
    spike_times = session.spike_times

    results = {'dt':dt, 
               'posUnits_spikes':[], 
               'negUnits_spikes':[]}

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

    # now restricting to specific samples
    print('--- STIM : %s' % stim)
    
    if stim=='spontaneous':
        
        Tcond = np.zeros(len(t_binned), dtype=bool)
        cond = stims.stimulus_name=='spontaneous'
        for t0, t1 in zip(stims[cond].start_time, stims[cond].stop_time):
            if (t1-t0)>120:
                Tcond[int((t0-tstart)/dt):int((t1-tstart)/dt)] = True
                
    elif stim=='natural-movies':
        
        Tcond = np.zeros(len(t_binned), dtype=bool)
        
        for movie in ['one', 'three']:
            x = stims[stims.stimulus_name=='natural_movie_%s' % movie]
            for block in np.unique(x.stimulus_block):
                X = stims[\
                        (stims.stimulus_name=='natural_movie_%s' % movie) &\
                        (stims.stimulus_block==block) ]
                # start and stop of the stimulus block
                t0, t1 = X.start_time.values[0], X.stop_time.values[-1]
                Tcond[int((t0-tstart)/dt):int((t1-tstart)/dt)] = True
                
    else:
        # all by default
        Tcond = np.ones(len(t_binned), dtype=bool)

    # we just clip to spont activity periods, introduce boundary discrepancies, hopefully smalll...
    positive_units_rate = positive_units_rate[Tcond] 
    negative_units_rate = negative_units_rate[Tcond]
    t_binned = np.arange(np.sum(Tcond))*dt
                
    # then smoothing
    positive_units_rate = gaussian_filter1d(positive_units_rate, int(smoothing/dt))
    negative_units_rate = gaussian_filter1d(negative_units_rate, int(smoothing/dt))
        
    results.update({'t':t_binned,
                   'posUnits_rate':positive_units_rate/nPos/dt,
                   'negUnits_rate':negative_units_rate/nNeg/dt})

    return results

# %%
for stim in []:#['all', 'natural-movies', 'spontaneous']:

    DATA ={}
    for key in ['PV', 'SST']:
        DATA[key] = []

        for index in range(len(Optotagging['%s_sessions'%key])):
            print(key, 'session #', index+1)
            results = get_spike_counts(Optotagging['%s_sessions' % key][index],
                                       Optotagging['%s_positive_units' % key][index],
                                       stim=stim,
                                       dt=2.5e-3, smoothing=2.5e-3)

            DATA[key].append(results)            

    np.save('../data/visual_coding/%s-spikes-data.npy' % stim, DATA)

# %% [markdown]
# # 2) Load summary data

# %%
DATA = np.load('../data/visual_coding/spontaneous-spikes-data.npy', allow_pickle=True).item()

# %% [markdown]
# # Wavelet-Based analysis: cross-correl with wavelet

# %%
from wavelet_transform import my_cwt

def spiketimes_to_binary(spikes, t, dt):
    
    binary = 0*t
    cond = spikes<t[-1]
    binary[np.array((spikes[cond]-t[0])/dt, dtype=int)] = 1.
    return binary


def compute_crosscorrel(results, freqs,
                        wavelet_width=2.,
                        verbose=False):
    
    if verbose:
        print('wavelet-transform [...]')
    coefs = np.real(my_cwt(results['negUnits_rate'], freqs, results['dt'],
                           w0=wavelet_width))
    results['freqs'] = freqs
    
    if verbose:
        print('cross-correl analysis[...]')
        
    #for units in ['negUnits', 'posUnits']:
    for units in ['posUnits']: # only for positive units

        # all Positive units together, session data
        all_spikes = np.concatenate(results['%s_spikes' % units])
        binary = spiketimes_to_binary(all_spikes, results['t'], results['dt'])
        results['%s-spike-wavelet-correl_per_session'%units] =\
                    [np.corrcoef(binary, coefs[i, :])[0,1] for i in range(len(freqs))]
        
        # now unit per unit
        results['%s-spike-wavelet-correl'%units] = []
        for spikes in results['%s_spikes' % units]:
            
            binary = spiketimes_to_binary(spikes, results['t'], results['dt'])
            results['%s-spike-wavelet-correl'%units].append(\
                        [np.corrcoef(binary, coefs[i, :])[0,1] for i in range(len(freqs))])
            
    if verbose:
        print(' done !')
            
            
#DATA = np.load('../data/visual-coding-all-spikes-data.npy', allow_pickle=True).item()
#results = DATA['PV'][0]
#np.sum(spiketimes_to_binary(results['posUnits_spikes'][1], results['t'], results['dt'])[:400])
#plt.plot(results['posUnits_spikes'][1][:5], 2*np.ones(5), 'o')
#print(np.min([np.min(x) for x in results['posUnits_spikes']]), results['t'][0])
#print(np.max([np.max(x) for x in results['posUnits_spikes']]), results['t'][-1])

# %%
#for stim in ['natural-movies', 'spontaneous']:
for stim in ['all', 'natural-movies', 'spontaneous']:

    RESULTS ={'freqs': np.logspace(-1, 2, 20)}
    
    DATA = np.load('../data/visual_coding/%s-spikes-data.npy' % stim, allow_pickle=True).item()

    for key in ['PV', 'SST']:
            
        RESULTS['%s_posUnits_spike-wavelet-correl' % key] = []

        for index in range(len(Optotagging['%s_sessions'%key])):
            print(key, 'session #', index+1)

            if len(DATA[key][index]['posUnits_spikes'])>0 and (len(DATA[key][index]['t'])>0):

                compute_crosscorrel(DATA[key][index], RESULTS['freqs'])

                for units in ['posUnits']:
                    #
                    RESULTS['%s_posUnits_spike-wavelet-correl' % key] += \
                                    DATA[key][index]['posUnits-spike-wavelet-correl']

    np.save('../data/visual_coding/%s-spikes-wavelet-correl.npy' % stim, RESULTS)

# %%
#for stim in ['natural-movies', 'spontaneous']:
for stim in ['all', 'natural-movies', 'spontaneous']:
    
    RESULTS = np.load('../data/visual-coding-%s-spikes-wavelet-correl.npy' % stim, allow_pickle=True).item()

    fig, AX = plt.subplots(1, 2, figsize=(6,2))

    for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):
        
        pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_posUnits_spike-wavelet-correl' % key], axis=0),
                sy=np.std(RESULTS['%s_posUnits_spike-wavelet-correl' % key], axis=0),
                color=color,ax=ax)

        pt.set_plot(ax, xscale='log',
                    #yticks=[0.1, 1], 
                    #yscale='log', 
                    xlabel='freq. (Hz)', 
                    title='%s-cre mice (n=%i units)' % (key, len(RESULTS['%s_posUnits_spike-wavelet-correl' % key])),
                    ylabel='wavelet-spike cross-correl' if key=='PV' else '')
        
    pt.annotate(AX[1], stim, (1,0.5), va='center')

    pt.set_common_ylims(AX)

# %%
break_now()


# %% [markdown]
# # 3) Compute the Frequency Spectrum from Fourier Analysis

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

#DATA = np.load('../data/visual-coding-spikes-data-spontaneous.npy', allow_pickle=True).item()
fig = spectrum_fig(DATA['PV'][2])


# %%
fig = spectrum_fig(DATA['SST'][11], pos_color='tab:orange')


# %%
fig = spectrum_fig(DATA['SST'][10], pos_color='tab:orange')


# %%
binary.shape

# %% [markdown]
# # Wavelet-Based analysis: spike-triggered wavelet envelope

# %%
from wavelet_transform import my_cwt

# %%
results = DATA['PV'][4]
freqs = np.logspace(-1, 2, 20)
coefs = my_cwt(results['negUnits_rate'], freqs, results['dt'])
#pt.time_freq_signal(results['t'], freqs, results['negUnits_rate'], coefs, freq_scale='log')

# %%
def spike_trig_mean_envelope(spikes, t, envelope):
    
    iSpikes = np.digitize(spikes, t)
    return envelope[:,iSpikes[(iSpikes>0) & (iSpikes!=len(t))]].mean(axis=1)

def compute_envelopes(results, freqs,
                      seed=0, Nshuffling=10,
                      verbose=False):
    
    np.random.seed(seed) # seed for shuffling
    
    if verbose:
        print('wavelet-transform [...]')
    coefs = my_cwt(results['negUnits_rate'], freqs, results['dt'])
    
    envelope = np.abs(coefs)
    #envelope = np.clip(np.real(coefs), 0, np.inf) # positive 
    
    #results['average-envelope'] = np.mean(envelope, axis=1)
    
    results['freqs'] = freqs
    
    if verbose:
        print('spike-triggered analysis[...]')
        
    #for units in ['negUnits', 'posUnits']:
    for units in ['posUnits']: # only for positive units

        # all Positive units together, session data
        all_spikes = np.concatenate(results['%s_spikes' % units])
        results['%s-spikeTrig-envelope-all-spikes' % units] = \
                spike_trig_mean_envelope(all_spikes, results['t'], envelope)
        # shuffled
        results['%s-shuffled-spikeTrig-envelope-all-spikes' % units] = []
        for k in range(Nshuffling):
            shufSpikes = np.random.uniform(results['t'][0], results['t'][-1], len(all_spikes))
            results['%s-shuffled-spikeTrig-envelope-all-spikes' % units].append(\
                    spike_trig_mean_envelope(shufSpikes, results['t'], envelope))

        # now unit per unit
        results['%s-spikeTrig-envelope'% units] = []
        results['%s-shuffled-spikeTrig-envelope'% units] = []
        for spikes in results['%s_spikes' % units]:
            # real
            iSpikes = np.digitize(spikes, results['t'])
            cond = (iSpikes>0) & (iSpikes!=len(results['t']))
            iSpikes = iSpikes[cond]    
            results['%s-spikeTrig-envelope'% units].append(\
                    spike_trig_mean_envelope(spikes, results['t'], envelope))
            # shuffled
            results['%s-shuffled-spikeTrig-envelope'% units].append([])
            for k in range(Nshuffling):
                shufSpikes = np.random.uniform(results['t'][0], results['t'][-1], len(spikes))
                results['%s-shuffled-spikeTrig-envelope'% units][-1].append(\
                        spike_trig_mean_envelope(shufSpikes, results['t'], envelope))

    
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
def time_freq_signal(results,
                     unit=0,
                     tlim=[100, 120],
                     spike_subsampling=10, space=5, 
                     Tbar=1, rate_bar = 5,
                     color='tab:red',
                     freq_scale = 'log',
                     fig_args=dict(axes_extents=[[[4,1],[1,1]],
                                                 [[4,1],[1,1]],
                                                 [[4,3],[1,3]]],
                                   hspace=0.2, wspace=0.1,
                                   figsize=(0.9,0.7))):
   

    Tcond = (results['t']>tlim[0]) & (results['t']<tlim[1])
    
    Tcond_extended = (results['t']>(tlim[0]-10)) & (results['t']<(tlim[1]+10))
    coefs = my_cwt(results['negUnits_rate'][Tcond_extended], results['freqs'], results['dt'])

    
    fig, AX = pt.figure(**fig_args)
    for i in range(2):
        AX[i][1].axis('off')
    
    # spikes
    n=0
    for spikes in results['negUnits_spikes']:
        Scond = (spikes>tlim[0]) & (spikes<tlim[1])
        AX[0][0].plot(spikes[Scond][::spike_subsampling],
                      n*np.ones(len(spikes[Scond]))[::spike_subsampling],
                      'o', lw=1, ms=0.5, color='tab:grey')
        n+=1
    n+=space
    for spikes in results['posUnits_spikes']:
        Scond = (spikes>tlim[0]) & (spikes<tlim[1])
        AX[0][0].plot(spikes[Scond][::spike_subsampling],
                      n*np.ones(len(spikes[Scond]))[::spike_subsampling],
                      'o', lw=1, ms=0.5, color=color)
        n+=1
    pt.annotate(AX[0][0], '%iunits' % len(results['posUnits_spikes']), (1,1), color=color, va='top')
    pt.annotate(AX[0][0], '\n%iunits' % len(results['negUnits_spikes']), (1,1), color='tab:grey', va='top')
    
    pt.set_plot(AX[0][0], [], xlim=tlim)
    
    AX[-2][0].plot(results['t'][Tcond], results['negUnits_rate'][Tcond], color='tab:grey')
    pt.set_plot(AX[-2][0], [], xlim=tlim)
    pt.draw_bar_scales(AX[-2][0],
                       Xbar=Tbar, Xbar_label='%.1fs' % Tbar,
                       Ybar=rate_bar, Ybar_label='%iHz' % rate_bar)
    # # time frequency power plot
    c = AX[-1][0].contourf(results['t'][Tcond_extended], results['freqs'], coefs[:, :], cmap='PiYG')
    pt.set_plot(AX[-1][0], ['left'],
                ylabel='wavelet freq. (Hz)', 
             ylim=[0.99*freqs[0], freqs[-1]],
             yscale=freq_scale,
             xlim=tlim)

    # # mean power plot over intervals
    AX[-1][1].plot(results['posUnits-spikeTrig-envelope'][unit], results['freqs'],\
                  label='mean', color=color)
    my = np.mean(results['posUnits-shuffled-spikeTrig-envelope'][unit],axis=0)
    sy = np.std(results['posUnits-shuffled-spikeTrig-envelope'][unit], axis=0)
    AX[-1][1].plot(my, results['freqs'], label='mean', color='tab:purple')
    #AX[-1][1].fill_between(mx-sx, freqs, label='mean', color='tab:purple')
    pt.annotate(AX[-1][1], 'real', (0,1), va='top', color=color)
    pt.annotate(AX[-1][1], '\nshuffled', (0,1), va='top', color='tab:purple')
    
    pt.set_plot(AX[-1][1], ['bottom', 'right'],
                title='spike-trig. av.\nspectrogram',
                xlabel=' envelope (Hz)',
                yscale=freq_scale, xscale='log',
                yticks=[1e-1, 1, 1e1, 1e2], yminor_ticks=[],
                ylim=[0.99*freqs[0], freqs[-1]])
    
    return fig, AX

# %%
key, index = 'PV', 4
results = get_spike_counts(Optotagging['%s_sessions' % key][index],
                           Optotagging['%s_positive_units' % key][index],
                           tstart=80, tstop=120,
                           spontaneous_only=False,
                           dt=2.5e-3, smoothing=2.5e-3)
#results = DATA['PV'][4]
freqs = np.logspace(-1, 2, 20)
compute_envelopes(results, freqs)
time_freq_signal(results['t'], results['freqs'], results['negUnits_rate'], coefs, 
                 tlim=[100, 110], Tbar=0.5,
                 freq_scale='log')

# %%
#results = DATA['PV'][4]
#compute_envelopes(results, np.logspace(-1, 2, 20))
time_freq_signal(results, 
                 tlim=[200, 210], Tbar=0.5,
                 freq_scale='log')

# %%
# Generate demo fig

# %%

# %%
RESULTS ={'freqs': np.logspace(-1, 2, 20)}
#DATA = np.load('../data/visual-coding-spikes-data-spontaneous.npy', allow_pickle=True).item()
DATA = np.load('../data/visual-coding-spikes-data.npy', allow_pickle=True).item()

for key in ['PV', 'SST']:
    
    for units in ['posUnits']:
        RESULTS['%s_%s_spikeTrig_env_per_session' % (key, units)] = []
        RESULTS['%s_%s_spikeTrig_env_per_session_shuffled' % (key, units)] = []
        RESULTS['%s_%s_spikeTrig_env_per_unit' % (key, units)] = []
        RESULTS['%s_%s_spikeTrig_env_per_unit_shuffled' % (key, units)] = []
        RESULTS['%s_%s_per_unit_session_id' % (key, units)] = []
        
    for index in range(len(Optotagging['%s_sessions'%key])):
        print(key, 'session #', index+1)
        
        if len(DATA[key][index]['posUnits_spikes'])>0:
            
            compute_envelopes(DATA[key][index], RESULTS['freqs'],
                              Nshuffling=10)
            
            for units in ['posUnits']:
                #
                RESULTS['%s_%s_spikeTrig_env_per_session' % (key, units)].append(\
                                DATA[key][index]['%s-spikeTrig-envelope-all-spikes' % units])
                RESULTS['%s_%s_spikeTrig_env_per_session_shuffled' % (key, units)].append(\
                    np.mean(DATA[key][index]['%s-shuffled-spikeTrig-envelope-all-spikes' % units], axis=0))
                #
                RESULTS['%s_%s_spikeTrig_env_per_unit' % (key, units)] += \
                                DATA[key][index]['%s-spikeTrig-envelope'% units]
                RESULTS['%s_%s_spikeTrig_env_per_unit_shuffled' % (key, units)] += [np.mean(s, axis=0) for s in\
                                DATA[key][index]['%s-shuffled-spikeTrig-envelope'% units]]
                #
                RESULTS['%s_%s_per_unit_session_id' % (key, units)] += list(index*np.ones(\
                                len(DATA[key][index]['%s-spikeTrig-envelope'% units])))


# %%
np.save('../data/spike-triggered-spectrogram-average.npy', RESULTS)

# %%
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):
  
    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_posUnits_spikeTrig_env_per_session' % key], axis=0),
            sy=np.std(RESULTS['%s_posUnits_spikeTrig_env_per_session' % key], axis=0),
            color=color,ax=ax)

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_posUnits_spikeTrig_env_per_session_shuffled' % key], axis=0),
            sy=np.std(RESULTS['%s_posUnits_spikeTrig_env_per_session_shuffled' % key], axis=0),
            color='tab:grey',ax=ax,lw=0.5,alpha=0.5)
    
    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)', 
                title='%s-cre mice (N=%i sessions)' % (key, len(DATA[key])),
                ylabel='rate envelope (Hz)' if key=='PV' else '',
                yticks=[0.1, 1])
    
    pt.annotate(ax, 'all spikes\nmerged', (0,1), va='top', fontsize=6)

pt.set_common_ylims(AX)

# %%
fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):
  
    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_posUnits_spikeTrig_env_per_unit' % key], axis=0),
            sy=np.std(RESULTS['%s_posUnits_spikeTrig_env_per_unit' % key], axis=0),
            color=color,ax=ax)
    
    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_posUnits_spikeTrig_env_per_unit_shuffled' % key], axis=0),
            sy=np.std(RESULTS['%s_posUnits_spikeTrig_env_per_unit_shuffled' % key], axis=0),
            color='tab:grey',ax=ax,lw=0.5,alpha=0.5)
    
    pt.set_plot(ax, xscale='log', yscale='log', xlabel='freq. (Hz)',
                title='%s-cre mice (n=%i units)' % (key, len(RESULTS['%s_posUnits_spikeTrig_env_per_unit' % key])),
                ylabel='rate envelope (Hz)' if key=='PV' else '',
                yticks=[0.1, 1])

pt.set_common_ylims(AX)

# %%

# %%
fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):
  
    RATIOS = []
    for real, shuffled in zip(RESULTS['%s_posUnits_spikeTrig_env_per_unit' % key],
                              RESULTS['%s_posUnits_spikeTrig_env_per_unit_shuffled' % key]):
        RATIOS.append(100*(real-shuffled)/shuffled)
        
    pt.plot(RESULTS['freqs'], np.mean(RATIOS, axis=0), 
            #sy=np.std(RATIOS, axis=0),
            sy=stats.sem(RATIOS, axis=0),
            color=color,ax=ax)
    
    ax.plot(RESULTS['freqs'], 0*RESULTS['freqs'], 'k:', lw=0.5)
    
    pt.set_plot(ax, xscale='log', 
                #yscale='log', 
                xlabel='freq. (Hz)',
                #yticks=[0.1, 1],
                title='%s-cre mice (n=%i units)' % (key, len(RESULTS['%s_posUnits_spikeTrig_env_per_unit' % key])),
                ylabel='var. from shuffled (%)' if key=='PV' else '')

pt.set_common_ylims(AX)

# %%
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, AX = plt.subplots(1, 2, figsize=(6,2))

for key, ax, color in zip(['PV', 'SST'], AX, ['tab:red', 'tab:orange']):

    """
    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_average_env_per_unit' % key], axis=0),
            sy=np.std(RESULTS['%s_average_env_per_unit' % key], axis=0),
            color='tab:grey',ax=ax)
    """

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

    RATIOS = [y/x for x, y in zip(RESULTS['%s_average_env_per_session' % key],
                                  RESULTS['%s_spikeTrig_env_per_session' % key])]
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
RESULTS = np.load('../data/spike-triggered-spectrogram-average.npy', allow_pickle=True).item()

fig, ax = plt.subplots(1, figsize=(3,2))
inset = pt.inset(ax, [1.3, 0.1, 0.3, 0.8])

for i, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):

    pt.plot(RESULTS['freqs'], np.mean(RESULTS['%s_spikeTrig_env_per_unit' % key], axis=0),
            sy=np.std(RESULTS['%s_spikeTrig_env_per_unit' % key], axis=0), color=color, ax=ax)

    #y = [RESULTS['freqs'][np.argmax(r)] for r in RESULTS['%s_spikeTrig_env_per_unit' % key]]
    # center of mass
    y = [np.mean(RESULTS['freqs']*r/np.sum(r)) for r in RESULTS['%s_spikeTrig_env_per_unit' % key]]
    pt.violin(y, X=[i], ax=inset, COLORS=[color])
    
pt.annotate(ax, 'PV+ spikes', (0,1), va='top', color='tab:red')
pt.annotate(ax, '\nSST+ spikes', (0,1), va='top', color='tab:orange')

pt.set_plot(ax, 
            xscale='log', 
            yscale='log', 
            ylabel='env$_{spikeTrig}$ / env$_{average}$',
            xlabel='freq. (Hz)')

pt.set_plot(inset, xticks_labels=[],
            yscale='log', ylim=[RESULTS['freqs'][0], RESULTS['freqs'][-1]],
            ylabel='peak freq. (Hz)')

# %%
len(RESULTS['%s_spikeTrig_env_per_unit' % key])

# %%
