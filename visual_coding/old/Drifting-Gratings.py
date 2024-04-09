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
Optotagging = np.load(os.path.join('..', 'data', 'visual_coding', 'Optotagging-Results.npy'), allow_pickle=True).item()

# subsample the negative units to 100 cells per session
np.random.seed(1)
for key in ['PV','SST']:
    for n, nUnits in enumerate(Optotagging[key+'_negative_units']):
        Optotagging[key+'_negative_units'][n] = np.random.choice(nUnits, 100, replace=False)

# %% [markdown]
# # 2) Preprocess the stimulus-evoked spikes

# %%
all = True
protocol = 'drifting_gratings'  #'drifting_gratings_75_repeats',  #'drifting_gratings_contrast',

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
# # 3) Plot an example

# %%
key = 'SST'
protocol = 'drifting_gratings_contrast'
sessionID = -1
unit = Optotagging[key+'_negative_units'][sessionID][0]
#session = cache.get_session_data(Optotagging[key+'_sessions'][sessionID])
#stim_table = session.get_stimulus_table()
#spikeResp = spikingResponse(None, None, None, filename=filename)

filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
spikeResp = spikingResponse(None, None, None, filename=filename)
spikeResp.plot(cond=spikeResp.contrast==0.2, color='tab:purple')

# %% [markdown]
# # 4) Contrast dependency

# %%
np.unique(spikeResp.contrast)

protocol = 'drifting_gratings_contrast'
contrasts = [0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]

window = [-0.2, 0.7]


fig, AX = pt.figure(axes=(len(contrasts),4), hspace=0.1)


for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red','tab:orange']):
    pRATES, nRATES = [[] for c in range(len(contrasts))], [[] for c in range(len(contrasts))]
    for u, rates, pUnits, nUnits in zip(range(2), [pRATES, nRATES],
                                         Optotagging[key+'_positive_units'],
                                         Optotagging[key+'_negative_units']):
        for x, RATES, units, c in zip(['pos', 'neg'], [pRATES, nRATES],
                                   [pUnits, nUnits], [color, 'tab:grey']):
            for unit in units:
                filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
                if os.path.isfile(filename):
                    spikeResp = spikingResponse(None, None, None, filename=filename)
                    tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                    rate = spikeResp.get_rate(spikeResp.frame==np.unique(spikeResp.frame)[7])
                    rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                    RATES.append(rate[tCond])
                    print(x, len(spikeResp.t[tCond]), len(rate[tCond]))
    for r, RATES, c in zip(range(2), [pRATES, nRATES], [color, 'tab:grey']):
        pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
                sy=sem(RATES, axis=0),
                #sy=np.std(RATES, axis=0),
                color=c, ax=AX[2*k+r], no_set=True)
        pt.annotate(AX[2*k+r], k*'\n'+'n=%i' % len(RATES), (1,1), va='top', color=c)
        pt.set_plot(AX[2*k+r], title=protocol, ylabel='$\delta$ rate (Hz)')



fig.suptitle('average over N=%i images first\nsem over cells\n' % Nframes)
        tCond = (time>window[0]) & (time<window[1])
        pt.plot(time[tCond], 
                np.mean([np.mean([rates[ImageID][c] for ImageID in range(Nframes)], axis=0)\
                                    for c in range(len(rates[0]))], axis=0)[tCond], 
                sy=stats.sem([np.mean([rates[ImageID][c] for ImageID in range(Nframes)], axis=0)\
                                    for c in range(len(rates[0]))], axis=0)[tCond], 
                #sy = stats.sem([np.mean(rates[ImageID], axis=0)[tCond] for ImageID in range(Nframes)], axis=0), 
                #sy=0.*np.std(rates[ImageID], axis=0)[tCond], 
                ax=AX[2*k+u], color=c)
        pt.annotate(AX[2*k+u], k*'\n'+'n=%i' % len(rates[ImageID]), (1.2,1.3), va='top', color=c)
#pt.set_plot(AX[u], ['left'], ylabel='rate (Hz)')


# %%
X=1
for key in spikeResp.keys[5:]:
    try:
        vals = [k for k in np.unique([str(k) for k in getattr(spikeResp, key)]) if k!='null']
        if len(vals)>0 and key not in ['duration', 'stimulus_condition_id']:
            print(key, '->', vals)
            X *= len(vals)
    except BaseException as be:
        pass
print('\nnumber of condition : %i' % X)
print('number of repeats: ', np.sum(np.array([str(o) for o in spikeResp.orientation])!='null')/X)

# %%
from scipy.stats import sem

window = [-0.1, 0.3]
protocol = 'natural_scenes'
fig, AX = pt.figure(axes=(4,1), wspace=2.)
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    pRATES, nRATES = [], []
    for sessionID, pUnits, nUnits in zip(Optotagging[key+'_sessions'], 
                                         Optotagging[key+'_positive_units'],
                                         Optotagging[key+'_negative_units']):
        for x, RATES, units in zip(['pos', 'neg'], [pRATES, nRATES], [pUnits, nUnits]):
            for unit in units:
                filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
                if os.path.isfile(filename):
                    spikeResp = spikingResponse(None, None, None, filename=filename)
                    tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                    rate = spikeResp.get_rate(spikeResp.frame==np.unique(spikeResp.frame)[7])
                    rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                    RATES.append(rate[tCond])
                    print(x, len(spikeResp.t[tCond]), len(rate[tCond]))
    for r, RATES, c in zip(range(2), [pRATES, nRATES], [color, 'tab:grey']):
        pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
                sy=sem(RATES, axis=0),
                #sy=np.std(RATES, axis=0),
                color=c, ax=AX[2*k+r], no_set=True)
        pt.annotate(AX[2*k+r], k*'\n'+'n=%i' % len(RATES), (1,1), va='top', color=c)
        pt.set_plot(AX[2*k+r], title=protocol, ylabel='$\delta$ rate (Hz)')

# %%
from scipy.stats import sem

window = [-0.01, 0.2]
protocol = 'static_gratings'
fig, AX = pt.figure(axes=(2,1), wspace=3.)
for k, ax, key, color in zip(range(2), AX, ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        sRATES = []
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                sRATES.append(rate[tCond])
        if len(sRATES)>0:
            RATES.append(np.mean(sRATES, axis=0))
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'N=%i sessions' % len(RATES), (1,1), va='top', color=color)
    pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)')

# %%
from scipy.stats import sem

window = [-0.01, 0.15]
protocol = 'static_gratings'
fig, ax = pt.figure()
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'n=%i' % len(RATES), (1,1), va='top', color=color)
pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)')

# %%
from scipy.stats import sem

window = [-0.1, 0.24]

protocol = 'flashes'
fig, AX = pt.figure(axes=(2,1), wspace=3.)
for k, ax, key, color in zip(range(2), AX, ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        sRATES = []
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.color==1.)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                sRATES.append(rate[tCond])
        if len(sRATES)>0:
            RATES.append(np.mean(sRATES, axis=0))
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'N=%i sessions' % len(RATES), (1,1), va='top', color=color)
    pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)', xlabel='time (s)')

# %%
from scipy.stats import sem

window = [-0.4, 2.4]

protocol = 'drifting_gratings'
fig, ax = pt.figure()
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)[tCond]
                rate -= np.mean(rate[spikeResp.t[tCond]<0])
                #rate /= np.max(rate)
                #RATES.append((rate-rate.mean())/rate.std())
                RATES.append(rate)
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'n=%i' % len(RATES), (1,1), va='top', color=color)
pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)', xticks=[0,1,2], xlabel='time (s)')

# %%
from scipy.stats import sem

window = [-0.4, 2.4]

protocol = 'drifting_gratings'
fig, AX = pt.figure(axes=(2,1), wspace=3)
for k, ax, key, color in zip(range(2), AX, ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        sRATES = []
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                sRATES.append(rate[tCond])
        if len(sRATES)>0:
            RATES.append(np.mean(sRATES, axis=0))
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'N=%i sessions' % len(RATES), (1,1), va='top', color=color)
    pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)', xticks=[0,1,2], xlabel='time (s)')

# %%
from scipy.stats import sem

window = [-0.4, 2.4]

protocol = 'drifting_gratings_75_repeats'
fig, ax = pt.figure()
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=(spikeResp.contrast==0.8))[tCond]
                rate -= np.mean(rate[spikeResp.t[tCond]<0])
                #rate /= np.max(rate)
                #RATES.append((rate-rate.mean())/rate.std())
                RATES.append(rate)
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            #sy=np.std(RATES, axis=0),
            color=color, ax=ax, no_set=True)
    pt.annotate(ax, k*'\n'+'n=%i' % len(RATES), (1,1), va='top', color=color)
pt.set_plot(ax, title=protocol, ylabel='$\delta$ rate (Hz)', xticks=[0,1,2])

# %%
from scipy.stats import sem

window = [-0.1, 0.6]

protocol = 'drifting_gratings_contrast'
contrasts = [0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]
fig, AX = pt.figure(axes=(len(contrasts),2), wspace=0, hspace=0.5, figsize=(0.8,1.), top=2.)

fig.suptitle(protocol)
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                for c, contrast in enumerate(contrasts):
                    RATES.append([])
                    rate = spikeResp.get_rate(cond=(spikeResp.contrast==contrast))[tCond]
                    rate -= np.mean(rate[spikeResp.t[tCond]<0])
                    RATES[c].append(rate)
    for c, contrast in enumerate(contrasts):
        pt.plot(spikeResp.t[tCond], np.mean(RATES[c], axis=0), 
                sy=sem(RATES[c], axis=0),
                #sy=np.std(RATES, axis=0),
                color=color, ax=AX[k][c], no_set=True)
    pt.annotate(AX[k][0], k*'\n'+'n=%i' % len(RATES[0]), (1,1), va='top', ha='right', color=color, fontsize=6)

for c, contrast in enumerate(contrasts):
    AX[0][c].set_title('c=%.2f' % contrast)
for ax in AX:
    pt.set_common_ylims(ax)
    for x in ax:
        if x==AX[1][0]:
            pt.set_plot(x, xlabel='time (ms)', ylabel='$\delta$ rate (Hz)')
        elif x==AX[0][0]:
            pt.set_plot(x, ['left'], ylabel='$\delta$ rate (Hz)')
        else:
            pt.set_plot(x, [])

# %%
from scipy.stats import sem

window = [-0.1, 0.24]

fig, AX = pt.figure(axes_extents=[[[3,1],[4,1]],[[3,1],[4,1]]], hspace=0.2, wspace=0.2,figsize=(.5,.9))

protocol = 'flashes'
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.color==1.)
                #rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[k][0], no_set=True)


protocol = 'drifting_gratings'
window = [-0.25, 2.25]

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('drifting_gratings', unit))
            if not os.path.isfile(filename):
                filename = os.path.join('..', 'data', 'visual_coding', key,
                                        '%s_unit_%i.npy' % ('drifting_gratings_75_repeats', unit))
            spikeResp = spikingResponse(None, None, None, filename=filename)
            tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
            rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
            #rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
            RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[k][1], no_set=True)
    pt.annotate(AX[k][1], k*'\n'+'%s,n=%i ' % (key, len(RATES)), (1,1), va='top', ha='right', color=color, fontsize=6)
pt.set_common_ylims(AX[0])
pt.set_common_ylims(AX[1])
AX[0][0].set_title('(short)\nlight flash\n')
AX[0][1].set_title('(long)\ndrifting gratings\n')
for i, ax in enumerate(AX[0]):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', yticks_labels=[] if i else None)
for i, ax in enumerate(AX[1]):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', yticks_labels=[] if i else None)
pt.draw_bar_scales(AX[0][0], Xbar=0.05, Xbar_label='50ms', Ybar=1e-12, loc='top-right')
pt.draw_bar_scales(AX[0][1], Xbar=0.5, Xbar_label='500ms', Ybar=1e-12, loc='top-left')

# %%
from scipy.stats import sem

window = [-0.1, 0.24]

fig, AX = pt.figure(axes_extents=[[[3,1],[4,1]],[[3,1],[4,1]]], hspace=0.2, wspace=0.2,figsize=(.5,1.))

protocol = 'static_gratings'

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate()
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0.])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[k][0], no_set=True)
pt.draw_bar_scales(AX[0][0], Xbar=0.05, Xbar_label='50ms', Ybar=1e-12, loc='top-right')

protocol = 'drifting_gratings'
window = [-0.25, 2.25]

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('drifting_gratings', unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0.])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[k][1], no_set=True)
    pt.annotate(AX[k][1], k*'\n'+'%s,n=%i ' % (key, len(RATES)), (1,1), va='top', ha='right', color=color, fontsize=6)
pt.set_common_ylims(AX[0])
pt.set_common_ylims(AX[1])
for i, ax in enumerate(AX[0]):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', 
                yticks_labels=[] if i else None)
for i, ax in enumerate(AX[1]):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', yticks_labels=[] if i else None)
pt.draw_bar_scales(AX[0][1], Xbar=0.5, Xbar_label='500ms', Ybar=1e-12, loc='top-left')
AX[0][0].set_title('(short)\nstatic gratings\n')
AX[0][1].set_title('(long)\ndrifting gratings\n')


# %%
from scipy.stats import sem

window = [-0.1, 0.24]

fig, AX = pt.figure(axes_extents=[[[3,1],[4,1]]], hspace=0.2, wspace=0.2,figsize=(.6,1.1))

protocol = 'flashes'
for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.color==1.)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[0], no_set=True)
pt.draw_bar_scales(AX[0], Xbar=0.05, Xbar_label='50ms', Ybar=1e-12, loc='top-right')
protocol = 'drifting_gratings'
window = [-0.25, 2.25]

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('drifting_gratings', unit))
            if not os.path.isfile(filename):
                filename = os.path.join('..', 'data', 'visual_coding', key,
                                        '%s_unit_%i.npy' % ('drifting_gratings_75_repeats', unit))
            spikeResp = spikingResponse(None, None, None, filename=filename)
            tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
            rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
            rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0])
            RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[1], no_set=True)
    pt.annotate(AX[1], k*'\n'+'%s,n=%i ' % (key, len(RATES)), (1,1), va='top', ha='right', color=color, fontsize=6)
pt.set_common_ylims(AX)
AX[0].set_title('(short)\nlight flash\n')
AX[1].set_title('(long)\ndrifting gratings\n')
for i, ax in enumerate(AX):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', yticks_labels=[] if i else None)
pt.draw_bar_scales(AX[1], Xbar=0.5, Xbar_label='500ms', Ybar=1e-12, loc='top-left')

# %%
from scipy.stats import sem

window = [-0.1, 0.24]

fig, AX = pt.figure(axes_extents=[[[3,1],[4,1]]], hspace=0.2, wspace=0.2,figsize=(.6,1.1))

protocol = 'static_gratings'

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate()
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0.])
                #rate /= np.max(rate[tCond][(spikeResp.t[tCond]>0.) & (spikeResp.t[tCond]<0.15)])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[0], no_set=True)
pt.draw_bar_scales(AX[0], Xbar=0.05, Xbar_label='50ms', Ybar=1e-12, loc='top-right')

protocol = 'drifting_gratings'
window = [-0.25, 2.25]

for k, key, color in zip(range(2), ['PV', 'SST'], ['tab:red', 'tab:orange']):
    RATES = []
    for sessionID, units in zip(Optotagging[key+'_sessions'], Optotagging[key+'_positive_units']):
        for unit in units:
            filename = os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % ('drifting_gratings', unit))
            if os.path.isfile(filename):
                spikeResp = spikingResponse(None, None, None, filename=filename)
                tCond = (spikeResp.t>window[0]) & (spikeResp.t<window[1])
                rate = spikeResp.get_rate(cond=spikeResp.contrast==0.8)
                rate -= np.mean(rate[tCond][spikeResp.t[tCond]<0.])
                #rate /= np.max(rate[tCond][(spikeResp.t[tCond]>0.) & (spikeResp.t[tCond]<0.15)])
                RATES.append(rate[tCond])
    pt.plot(spikeResp.t[tCond], np.mean(RATES, axis=0), 
            sy=sem(RATES, axis=0),
            color=color, ax=AX[1], no_set=True)
    pt.annotate(AX[1], k*'\n'+'%s,n=%i ' % (key, len(RATES)), (1,1), va='top', ha='right', color=color, fontsize=6)
pt.set_common_ylims(AX)
for i, ax in enumerate(AX):
    pt.set_plot(ax, [] if i else ['left'], ylabel='' if i else '$\delta$ rate (Hz)', 
                yticks_labels=[] if i else None)
pt.draw_bar_scales(AX[1], Xbar=0.5, Xbar_label='500ms', Ybar=1e-12, loc='top-left')
AX[0].set_title('(short)\nstatic gratings\n')
AX[1].set_title('(long)\ndrifting gratings\n')


# %%
