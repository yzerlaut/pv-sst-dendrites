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

from analysis import spikingResponse, pt # custom object of trial-aligned spiking reponse

import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# just to disable the HDMF cache namespace warnings, REMOVE to see them
import warnings
warnings.filterwarnings("ignore")

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
key = 'SST'
protocol = 'natural_scenes'
sessionID = 0
unit = Optotagging[key+'_negative_units'][sessionID][1]
#session = cache.get_session_data(Optotagging[key+'_sessions'][sessionID])
#stim_table = session.get_stimulus_table()
#spikeResp = spikingResponse(None, None, None, filename=filename)

filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
spikeResp = spikingResponse(None, None, None, filename=filename)
spikeResp.plot(trial_subsampling=10, color='tab:purple')

# %%
len(np.unique(spikeResp.frame))

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

fig, AX = pt.figure(axes=(2,2), hspace=0.1)

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    for u, rates, c in zip(range(2), 
                                  [RATES[key+'_posUnits'], RATES[key+'_negUnits']],
                                  [color, 'tab:grey']):            
        pt.plot(time, np.mean(rates, axis=0), sy=0.*np.std(rates, axis=0), ax=AX[u][k], color=c)
        pt.annotate(AX[u][k], 'n=%i' % len(rates), (1,1), va='top', ha='right', color=c)
        pt.set_plot(AX[u][k], ['left'], ylabel='rate (Hz)') 


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

CCF, time_shift = crosscorrel(np.mean(RATES['PV_posUnits'], axis=0),
                              np.mean(RATES['PV_negUnits'], axis=0),
                              len(RATES['PV_posUnits']), 1)
ax.plot(time_shift, CCF, color='tab:red')

CCF, time_shift = crosscorrel(np.mean(RATES['SST_posUnits'], axis=0),
                              np.mean(RATES['SST_negUnits'], axis=0),
                              len(RATES['PV_posUnits']), 1)
ax2 = ax.twinx()
ax2.plot(time_shift, CCF, color='tab:orange')


