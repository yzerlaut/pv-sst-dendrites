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
sys.path.append('..')
import plot_tools as pt
# pt.set_style('dark')
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

# %%
from scipy.stats import mannwhitneyu

def show_average_quantity(analysis_metrics,
                          attr = 'g_osi_sg',
                          label = 'OSI'):
    

    fig, ax = plt.subplots(1)
    inset = pt.inset(ax, [1.4, -0.05, 0.3, 0.6])

    # choose attribute and set label


    RESP = {'Others':[]}
    for i, Key in enumerate(['PV', 'SST']):
        RESP[Key] = []
        positive_IDs = np.concatenate(Optotagging[Key+'_positive_units'])
        for unit in positive_IDs:
            if len(getattr(analysis_metrics[analysis_metrics.index==unit], attr).values)>0:
                value = getattr(analysis_metrics[analysis_metrics.index==unit], attr).values[0]
                if np.isfinite(value):
                    RESP[Key].append(value)
        for unit in analysis_metrics[\
                        analysis_metrics.ecephys_structure_acronym.isin(['VISp'])].index.values:
            if (unit not in positive_IDs) and (len(getattr(analysis_metrics[analysis_metrics.index==unit], attr).values)>0):
                value = getattr(analysis_metrics[analysis_metrics.index==unit], attr).values[0]
                if np.isfinite(value):
                    RESP['Others'].append(value)


    COLORS=['lightgrey', 'tab:red', 'tab:orange']
    for i, Key in enumerate(['Others', 'PV', 'SST']):
        pt.annotate(ax, i*'\n'+'n=%i units' % len(RESP[Key]), (1,1), color=COLORS[i], ha='right', va='top')
        ax.hist(RESP[Key], bins=np.linspace(0, 1, 30), color=COLORS[i],
                 label=Key.replace('_sessions', '+ units'), alpha=1 if i==0 else 0.6, density=True)
        pt.violin(RESP[Key], X=[i], COLORS=[COLORS[i]], ax=inset)

    inset.set_xticks([])
    inset.set_yticks([0, 0.5, 1])
    inset.set_ylabel(label)
    inset.set_title('PV vs SST, p=%.1e' % mannwhitneyu(RESP['PV'], RESP['SST']).pvalue)
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_ylabel('density')
    ax.legend(loc=(1.1, 0.8))
    
    return RESP

analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
OSI = show_average_quantity(analysis_metrics, attr = 'g_osi_sg', label = 'OSI')

# %%
#analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity')
analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
DSI = show_average_quantity(analysis_metrics, attr = 'g_dsi_dg', label = 'DSI')


# %% [markdown]
# # 2) Compute the PSTH

# %%
# now put in analysis.py

class spikingResponse:
    
    def __init__(self, stim_table, spike_times, t,
                 filename=None):
        
        if filename is not None:
            
            self.load(filename)
            
        else:
            
            self.build(stim_table, spike_times, t)
            
        
    def build(self, stim_table, spike_times, t):
        
        duration = np.mean(stim_table.duration) # find stim duration
        self.t = t
        
        self.time_resolution = self.t[1]-self.t[0]

        self.spike_matrix = np.zeros( (len(stim_table.index.values),
                                       len(self.t)) , dtype=bool)
        self.keys = ['spike_matrix', 't']

        for key in stim_table:
            
            setattr(self, key, np.array(getattr(stim_table, key)))
            self.keys.append(key)

        for trial_idx, trial_start in enumerate(stim_table.start_time.values):

            in_range = (spike_times > (trial_start + self.t[0])) * \
                       (spike_times < (trial_start + self.t[-1]))

            binned_times = ((spike_times[in_range] -\
                             (trial_start + self.t[0])) / self.time_resolution).astype('int')
            self.spike_matrix[trial_idx, binned_times] = True       
            
    def get_rate(self,
                 cond=None,
                 smoothing=5e-3):
        if cond is None:
            cond = np.ones(self.spike_matrix.shape[0], dtype=bool)
            
        iSmooth = int(smoothing/self.time_resolution)
        if iSmooth>=1:
            return gaussian_filter1d(self.spike_matrix[cond,:].mean(axis=0) / self.time_resolution,
                                     iSmooth)
        else:
            return self.spike_matrix[cond,:].mean(axis=0) / self.time_resolution
    
    def save(self, filename):
        D = {'time_resolution':self.time_resolution, 'keys':self.keys}
        for key in self.keys:
            D[key] = getattr(self, key)
        np.save(filename, D)
    
    def load(self, filename):
        D = np.load(filename, allow_pickle=True).item()
        for key in D['keys']:
            setattr(self, key, np.array(D[key]))
        self.keys = D['keys']
        self.time_resolution = D['time_resolution']
        
    def plot(self, 
             cond = None,
             ax1=None, ax2=None,
             smoothing=5e-3, 
             trial_subsampling=1,
             color='k', ms=1):

        if cond is None:
            cond = np.ones(self.spike_matrix.shape[0], dtype=bool)

        if not (ax1 is not None and ax2 is not None):
            fig = plt.figure(figsize=(1.2,2))
            plt.subplots_adjust(left=0.1, top=0.8, right=0.95)
            ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
        else:
            fig = None
            
        for i, t in enumerate(\
                        np.arange(self.spike_matrix.shape[0])[cond][::trial_subsampling]):
            spike_cond = self.spike_matrix[t,:]==1
            ax1.plot(self.t[spike_cond],
                     self.spike_matrix[t,:][spike_cond]+i, '.', ms=ms, color=color)
            
        ax2.fill_between(self.t, 0*self.t, self.get_rate(cond=cond, smoothing=smoothing), color=color)
        ax1.set_ylabel('trial #')
        ax2.set_ylabel('rate (Hz)')
        ax2.set_xlabel('time (s)')
        pt.set_common_xlims([ax1,ax2])

        return fig, [ax1, ax2]


# %% [markdown]
# # 3) Store all stimulus-evoked spikes

# %%
all = True

if True:
    # turn True to re-run the analysis
    for key in ['PV', 'SST']:
        # loop over session
        count = 1

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
                for protocol in [\
                                 #'flashes', 'static_gratings', 'drifting_gratings',
                                 #'drifting_gratings_75_repeats', 
                                 #'drifting_gratings_contrast',
                                 'natural_movie_one_more_repeats',
                                 #'natural_movie_one_shuffled',
                                 #'natural_scenes', 
                                 'natural_movie_one']:
                    if protocol in np.unique(stim_table.stimulus_name):
                        cond = (stim_table.stimulus_name==protocol)
                        duration = int(1e3*np.mean(stim_table[cond].duration))
                        if 'natural_movie_one' in protocol:
                            t = np.linspace(0,1,10)*1e-3*duration # 2 points per frame
                        else:
                            t = 1e-3*np.linspace(-duration/2., 1.5*duration, 200) # 200 points covering pre and post
                        spikeResp = spikingResponse(stim_table[cond], spike_times, t)
                        spikeResp.save(os.path.join('..', 'data', 'visual_coding', key, '%s_unit_%i.npy' % (protocol, unit)))
                        count += 1

# %% [markdown]
# # 4) Plot

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
np.max(spikeResp.duration)

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
