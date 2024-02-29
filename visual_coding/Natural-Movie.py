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
from scipy import stats

from analysis import spikingResponse # custom object of trial-aligned spiking reponse

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# just to disable the HDMF cache namespace warnings, REMOVE to see them
import warnings
warnings.filterwarnings("ignore")

# %%
data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

# %% [markdown]
# # 1) Loading the Optotagging results

# %%
#Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'), allow_pickle=True).item()
Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'), allow_pickle=True).item()

# subsample the negative units to 100 cells per session
np.random.seed(1)
for key in ['PV','SST']:
    for n, nUnits in enumerate(Optotagging[key+'_negative_units']):
        Optotagging[key+'_negative_units'][n] = np.random.choice(nUnits, 100, replace=False)

# %% [markdown]
# # 2) Compute the PSTH

# %%
# loop over frames to build the time course
RATES = {}
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    RATES[key+'_posUnits'] = []
    RATES[key+'_negUnits'] = []
    for sessionID in range(len(Optotagging[key+'_sessions'])):
        for u, rates, units, c in zip(range(2), [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                               [Optotagging[key+'_positive_units'][sessionID], Optotagging[key+'_negative_units'][sessionID]],
                               [color, 'tab:grey']):
            for unit in units:
                filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('natural_movie_one_more_repeats', unit))
                if not os.path.isfile(filename):
                    filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('natural_movie_one', unit))
                if os.path.isfile(filename):
                    spikeResp = spikingResponse(None, None, None, filename=filename)
                    rate = np.zeros(len(np.unique(spikeResp.frame)))
                    for i, frame in enumerate(np.unique(spikeResp.frame)):
                        rate[i] = spikeResp.get_rate(cond=spikeResp.frame==frame)[0]
                    rates.append(rate)               

# %% [markdown]
# # 3) Plot

# %%
duration = np.mean(spikeResp.duration)
time = np.arange(len(np.unique(spikeResp.frame)))*duration

fig, AX = pt.figure(axes=(2,2), hspace=0.1, figsize=(1.3,1))

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    for u, rates, c in zip(range(2), 
                                  [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                                  [color, 'tab:grey']):            
        pt.plot(time, np.mean(rates, axis=0), sy=0.*np.std(rates, axis=0), ax=AX[u][k], color=c)
        pt.annotate(AX[u][k], 'n=%i' % len(rates), (1,1), va='top', ha='right', color=c)
        pt.set_plot(AX[u][k], ['left','bottom'] if u else ['left'], 
                    ylabel='rate (Hz)', xlabel='time (s)' if u else '') 


# %%
def crosscorrel(Signal1, Signal2, tmax, dt):
    """
    argument : Signal1 (np.array()), Signal2 (np.array())
    returns : np.array()
    take two Signals, and returns their crosscorrelation function 

    CONVENTION:
    --------------------------------------------------------------
    when the peak is in the past (negative t_shift)
    it means that Signal2 is delayed with respect to Signal 1
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

fig, ax = plt.subplots(1) # pt.figure()
ax.set_xlabel('$\delta$ time (s)')
CCF, time_shift = crosscorrel(np.mean(RATES['PV_negUnits'], axis=0),
                              np.mean(RATES['PV_posUnits'], axis=0),
                              2, time[1]-time[0])
ax.plot(time_shift, CCF, color='tab:red')

CCF, time_shift = crosscorrel(np.mean(RATES['SST_negUnits'], axis=0),
                              np.mean(RATES['SST_posUnits'], axis=0),
                              2, time[1]-time[0])
ax2 = ax.twinx()
ax2.plot(time_shift, CCF, color='tab:orange')
for x in [ax,ax2]:
    x.set_xlim([-2,2])
fig.suptitle('cross-correl. "+" vs "-" units')



# %%
from scipy import stats

fig, ax = plt.subplots(1) # pt.figure()
ax2 = ax.twinx()

ax.set_xlabel('$\delta$ time (s)')

for k, key, color, x in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange'], [ax,ax2]):
    CCs = []
    for rate in RATES['%s_posUnits' % key]:
        CCF, time_shift = crosscorrel(np.mean(RATES['%s_negUnits' % key], axis=0),
                                      rate,
                                      time[-1]/2., time[1]-time[0])
        CCs.append(CCF)
        
    pt.plot(time_shift, np.mean(CCs, axis=0), 
            sy=stats.sem(CCs, axis=0), color=color, ax=x)

ax.set_xlim([-3,3])
fig.suptitle('cross-correl. "+" vs "-" units')

# %%
