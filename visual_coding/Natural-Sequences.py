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
from scipy import stats

import sys

from analysis import spikingResponse # custom object of trial-aligned spiking reponse
from analysis import pt, crosscorrel # plot_tools and CC-function

import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# just to disable the HDMF cache namespace warnings, REMOVE to see them
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # 1) Loading the Optotagging results

# %%
#Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'), allow_pickle=True).item()
Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results-VISp.npy'), allow_pickle=True).item()

# subsample the negative units to 100 cells per session
np.random.seed(1)
for key in ['PV','SST']:
    for n, nUnits in enumerate(Optotagging[key+'_negative_units']):
        Optotagging[key+'_negative_units'][n] = np.random.choice(nUnits, np.min([len(nUnits),100]), replace=False)

# %% [markdown]
# # 2) Preprocess the stimulus-evoked spikes

# %%
all = True
protocol = 'natural_scenes'

if False:
    # turn True to re-run the preprocessing

    # load data from API
    data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')
    manifest_path = os.path.join(data_directory, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    
    for key in ['PV', 'SST']:
        
        # loop over session
        for sessionID, positive_units, negative_units in zip(Optotagging[key+'_sessions'],
                                                             Optotagging[key+'_positive_units'],
                                                             Optotagging[key+'_negative_units']):
            if all:
                units  = np.concatenate([positive_units, negative_units])
            else:
                units = positive_units
            
            session = cache.get_session_data(sessionID)
            # stimulus infos for that session
            stim_table = session.get_stimulus_table()
            # fetch summary statistics 
            analysis_metrics = cache.get_unit_analysis_metrics_by_session_type(session.session_type)
            
            for unit in units:
                # get the spikes of that unit
                spike_times = session.spike_times[unit]
                if protocol in np.unique(stim_table.stimulus_name):
                    cond = (stim_table.stimulus_name==protocol)
                    duration = int(1e3*np.mean(stim_table[cond].duration))
                    t = 1e-3*np.linspace(-duration/2., 1.5*duration, 200) # 200 points covering pre and post
                    spikeResp = spikingResponse(stim_table[cond], spike_times, t)
                    spikeResp.save(os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit)))

# %% [markdown]
# # 2) Compute the PSTH

# %%
key = 'SST'
protocol = 'natural_scenes'
sessionID = 0
unit = Optotagging[key+'_negative_units'][sessionID][0]

filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
spikeResp = spikingResponse(None, None, None, filename=filename)
spikeResp.plot(cond=spikeResp.frame==1,
               trial_subsampling=1, color='tab:purple')

# %%
# loop over frames to build the time course
RATES = {}
Nframes = 118

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    RATES[key+'_posUnits'] = [[] for f in range(Nframes)]
    RATES[key+'_negUnits'] = [[] for f in range(Nframes)]
    for sessionID in range(len(Optotagging[key+'_sessions'])):
        for u, rates, units, c in zip(range(2),
                               [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                               [Optotagging[key+'_positive_units'][sessionID], Optotagging[key+'_negative_units'][sessionID]],
                               [color, 'tab:grey']):
            for unit in units:
                filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('natural_scenes', unit))
                if os.path.isfile(filename):
                    spikeResp = spikingResponse(None, None, None, filename=filename)
                    for f in np.arange(Nframes):
                        rates[f].append(spikeResp.get_rate(cond=spikeResp.frame==f))

# %% [markdown]
# # 3) Plot

# %%
time = spikeResp.t
window = [-0.1, 0.26]

fig, AX = pt.figure(axes=(2,2), hspace=0.1)

ImageID = 0
fig.suptitle('Image #%i' % (1+ImageID))
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    for u, rates, c in zip(range(2), 
                                  [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                                  [color, 'tab:grey']):    
        tCond = (time>window[0]) & (time<window[1])
        pt.plot(time[tCond], np.mean(rates[ImageID], axis=0)[tCond], 
                sy=stats.sem(rates[ImageID], axis=0)[tCond], 
                ax=AX[u][k], color=c)
        pt.annotate(AX[u][k], 'n=%i' % len(rates[ImageID]), (1,1), va='top', ha='right', color=c)
        pt.set_plot(AX[u][k], ['left'], ylabel='rate (Hz)') 

# %%
time = spikeResp.t
window = [-0., 0.26]

fig, AX = pt.figure(axes=(1,2), hspace=0.1)
AX2 = [ax.twinx() for ax in AX]
fig.suptitle('average over N=%i images first\nsem over cells\n' % Nframes)
for k, key, color, X in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange'], [AX, AX2]):
    for u, rates, c in zip(range(2), 
                                  [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                                  [color, 'tab:grey']):    
        tCond = (time>window[0]) & (time<window[1])
        pt.plot(time[tCond], 
                np.mean([np.mean([rates[ImageID][c] for ImageID in range(Nframes)], axis=0)\
                                    for c in range(len(rates[0]))], axis=0)[tCond], 
                sy=stats.sem([np.mean([rates[ImageID][c] for ImageID in range(Nframes)], axis=0)\
                                    for c in range(len(rates[0]))], axis=0)[tCond], 
                #sy = stats.sem([np.mean(rates[ImageID], axis=0)[tCond] for ImageID in range(Nframes)], axis=0), 
                #sy=0.*np.std(rates[ImageID], axis=0)[tCond], 
                ax=X[u], color=c)
        pt.annotate(X[u], k*'\n'+'n=%i' % len(rates[ImageID]), (1.2,1.3), va='top', color=c)
#pt.set_plot(AX[u], ['left'], ylabel='rate (Hz)')

# %%
time = spikeResp.t
window = [-0., 0.26]

fig, AX = pt.figure(axes=(1,2), hspace=0.1)
AX2 = [ax.twinx() for ax in AX]
fig.suptitle('average over cells first\nsem over N=%i images\n' % Nframes)
for k, key, color, X in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange'], [AX, AX2]):
    for u, rates, c in zip(range(2), 
                          [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                          [color, 'tab:grey']):    
        tCond = (time>window[0]) & (time<window[1])
        pt.plot(time[tCond], 
                np.mean([np.mean(rates[ImageID], axis=0)[tCond] for ImageID in range(Nframes)], axis=0), 
                sy = stats.sem([np.mean(rates[ImageID], axis=0)[tCond] for ImageID in range(Nframes)], axis=0), 
                #sy=0.*np.std(rates[ImageID], axis=0)[tCond], 
                ax=X[u], color=c)
        pt.annotate(X[u], k*'\n'+'n=%i' % len(rates[ImageID]), (1.2,1.3), va='top', color=c)

# %% [markdown]
# ## 4) Cross-correlation pattern

# %%
time = spikeResp.t
extent = 0.1
CCs = {}
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    CCs[key] = []
    for f in np.arange(Nframes):
        negRate = np.mean(RATES[key+'_negUnits'][f], axis=0)
        _, time_shift = crosscorrel(negRate, negRate, extent, time[1]-time[0])
        CCmean = 0*time_shift
        for posRate in RATES[key+'_posUnits'][f]:
            CCF, time_shift = crosscorrel(negRate, posRate, extent, time[1]-time[0])
            CCmean += CCF/len(RATES[key+'_posUnits'][f])
        CCs[key].append(CCmean) 

# %%
fig, ax = plt.subplots(1) # pt.figure()

pt.plot(time_shift, 
        np.nanmean(CCs['PV'], axis=0), 
        #sy=np.nanstd(CCs['PV'], axis=0), 
        color='tab:red', ax=ax)
ax2 = ax.twinx()
pt.plot(time_shift, 
        np.nanmean(CCs['SST'], axis=0), 
        #sy=np.nanstd(CCs['SST'], axis=0), 
        color='tab:orange', ax=ax2)



# %%
time = spikeResp.t
extent = 0.1
CCs = {}
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    CCs[key] = []
    for f in np.arange(Nframes):
        negRate = np.mean(RATES[key+'_negUnits'][f], axis=0)
        _, time_shift = crosscorrel(negRate, negRate, extent, time[1]-time[0])
        CCmean = 0*time_shift
        for posRate in RATES[key+'_posUnits'][f]:
            CCF, time_shift = crosscorrel(negRate, posRate, extent, time[1]-time[0])
            CCmean += CCF/len(RATES[key+'_posUnits'][f])
        CCs[key].append(CCmean) 

# %%
