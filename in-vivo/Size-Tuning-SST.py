# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules
import sys, os, pprint, pandas
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

sys.path.append('../src')
from analysis import compute_tuning_response_per_cells
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
sys.path.append('../')
import plot_tools as pt

folder = os.path.join(os.path.expanduser('~'),
                      'CURATED', 'SST-WT-NR1-GluN3-2023')

# %% [markdown]
# ## Build the dataset from the NWB files

# %%
import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

DATASET = scan_folder_for_NWBfiles(folder,
                                   verbose=False)


# %%
# -------------------------------------------------- #
# ----    Pick the session datafiles and sort ------ #
# ----      them according to genotype ------------- #
# -------------------------------------------------- #

def init_summary(DATASET):
    
    SUMMARY = {'WT':{'FILES':[], 'subjects':[]}, 
               'GluN1':{'FILES':[], 'subjects':[]}, 
               'GluN3':{'FILES':[], 'subjects':[]}}

    for i, protocols in enumerate(DATASET['protocols']):

        # select the sessions
        if ('size-tuning' in protocols[0]):
            
            # sort the sessions according to the mouse genotype
            if ('NR1' in DATASET['subjects'][i]) or ('GluN1' in DATASET['subjects'][i]):
                SUMMARY['GluN1']['FILES'].append(DATASET['files'][i])
                SUMMARY['GluN1']['subjects'].append(DATASET['subjects'][i])
            elif ('NR3' in DATASET['subjects'][i]) or ('GluN3' in DATASET['subjects'][i]):
                SUMMARY['GluN3']['FILES'].append(DATASET['files'][i])
                SUMMARY['GluN3']['subjects'].append(DATASET['subjects'][i])
            else:
                SUMMARY['WT']['FILES'].append(DATASET['files'][i])
                SUMMARY['WT']['subjects'].append(DATASET['subjects'][i])
                
    return SUMMARY


# %% [markdown]
# ## Analysis

# %%
from physion.analysis.protocols.size_tuning import center_and_compute_size_tuning

def run_dataset_analysis(DATASET,
                         quantity='dFoF',
                         roi_to_neuropil_fluo_inclusion_factor=1.15,
                         neuropil_correction_factor = 0.7,
                         method_for_F0 = 'sliding_percentile',
                         percentile=5., # percent
                         sliding_window = 300, # seconds
                         Nmax=999, # max datafiles (for debugging)
                         verbose=True):

    SUMMARY = init_summary(DATASET)
    
    SUMMARY['quantity'] = quantity
    SUMMARY['quantity_args'] = dict(roi_to_neuropil_fluo_inclusion_factor=\
                                        roi_to_neuropil_fluo_inclusion_factor,
                                    method_for_F0=method_for_F0,
                                    percentile=percentile,
                                    sliding_window=sliding_window,
                                    neuropil_correction_factor=neuropil_correction_factor)
    
    for key in ['WT', 'GluN1', 'GluN3']:

        for k in ['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES']:
            SUMMARY[key][k] = [] 

        for f in SUMMARY[key]['FILES'][:Nmax]:

            print('analyzing "%s" [...] ' % f)
            data = Data(f, verbose=False)

            if quantity=='dFoF':
                data.build_dFoF(roi_to_neuropil_fluo_inclusion_factor=\
                                        roi_to_neuropil_fluo_inclusion_factor,
                                method_for_F0=method_for_F0,
                                percentile=percentile,
                                sliding_window=sliding_window,
                                neuropil_correction_factor=neuropil_correction_factor,
                                verbose=False)
            elif quantity=='rawFluo':
                data.build_rawFluo(verbose=verbose)
            elif quantity=='neuropil':
                data.build_neuropil(verbose=verbose)            
            else:
                print('quantity not recognized !!')

            #print('-->', data.vNrois)
            radii, size_resps, rois, pref_angles = center_and_compute_size_tuning(data,
                                                                                  imaging_quantity=quantity,
                                                                                  with_rois_and_angles=True,
                                                                                  verbose=False)
            if len(size_resps)>0:
                for k, q in zip(['RESPONSES', 'CENTERED_ROIS', 'PREF_ANGLES'],
                                [size_resps, rois, pref_angles]):
                    SUMMARY[key][k].append(q)

            if len(radii)>0:
                SUMMARY['radii'] = radii
                
    return SUMMARY


# %%
for quantity in ['rawFluo', 'neuropil', 'dFoF']:
    SUMMARY = run_dataset_analysis(DATASET, quantity=quantity, verbose=False)
    np.save('../data/in-vivo/size-tuning-%s-summary.npy' % quantity, SUMMARY)

# %% [markdown]
# ## Varying the preprocessing parameters

# %%

for neuropil_correction_factor in [0.6, 0.7, 0.8, 0.9]:
    # rawFluo
    SUMMARY = run_dataset_analysis(DATASET, quantity='dFoF', 
                                   neuropil_correction_factor=neuropil_correction_factor,
                                   verbose=False)
    np.save('../data/in-vivo/size-tuning-factor-neuropil-%.1f-summary.npy' % neuropil_correction_factor, SUMMARY)
    
for roi_to_neuropil_fluo_inclusion_factor in [1.1, 1.15, 1.2, 1.25, 1.3]:
    # rawFluo
    SUMMARY = run_dataset_analysis(DATASET, 
                                   quantity='dFoF', 
                                   roi_to_neuropil_fluo_inclusion_factor=roi_to_neuropil_fluo_inclusion_factor,
                                   verbose=False)
    np.save('../data/in-vivo/size-tuning-inclusion-factor-neuropil-%.1f-summary.npy' % roi_to_neuropil_fluo_inclusion_factor, SUMMARY)

# %% [markdown]
# ## Quantification & Data visualization

# %%
from scipy.special import erf
from scipy.optimize import minimize

def func(S, X):
    """ fitting function """
    return X[0]*(erf(S/X[1])-X[3]*erf(S/X[2]))

def angle_lin_to_true_angle(angle):
    """
    see:
    https://github.com/yzerlaut/physion/blob/main/notebooks/Visual-Stim-Design.ipynb
    """
    return 180./np.pi*np.arctan(angle/180.*np.pi)

def suppression_index(resp1, resp2):
    resp1 = np.clip(resp1, 1e-2, np.inf)
    return np.clip((resp1-resp2)/resp1, 0, 1)
    
def plot_summary(SUMMARY,
                 KEYS = ['WT', 'GluN1'], 
                 COLORS = ['k', 'tab:blue'],
                 average_by='sessions',
                 xscale='lin', ms=2):
    
    fig, AX = pt.plt.subplots(1, 2, figsize=(5.5,1.))
    AX[0].annotate('average\nover\n%s' % average_by,
                   (-0.8, 0.5), va='center', ha='center', xycoords='axes fraction')
    plt.subplots_adjust(wspace=0.7, right=0.6)

    center_index = 2

    inset = pt.inset(AX[1], [1.6, .2, .4, .8])
    
    stim_size = 2*angle_lin_to_true_angle(SUMMARY['radii']) # radius to diameter, and linear-approx correction
    SIs = []
    
    for i, key, color in zip(range(2), KEYS, COLORS):

        if average_by=='sessions':
            resp = np.array([np.mean(np.clip(r, 0, np.inf), axis=0) for r in SUMMARY[key]['RESPONSES']])
        else:
            resp = np.concatenate([np.clip(r, 0, np.inf) for r in SUMMARY[key]['RESPONSES']])
            
        #resp = np.clip(resp, 0, 100) # CLIP RESPONSIVE TO POSITIVE VALUES
        
        SIs.append([suppression_index(np.mean(r[2:5]), np.mean(r[-3:])) for r in resp])

        # data
        pt.scatter(stim_size, np.mean(resp, axis=0),
                   sy=stats.sem(resp, axis=0),
                   ax=AX[i], color=color, ms=ms)

        # fit
        def to_minimize(x0):
            return np.sum((resp.mean(axis=0)-\
                           func(stim_size, x0))**2)
        res = minimize(to_minimize,
                       [2, 20, 40, 0])
        x = np.linspace(0, stim_size[-1], 100)
        AX[i].plot(x, func(x, res.x), lw=2, alpha=.5, color=color)

        AX[i].set_title(key, color=color)
        if average_by=='sessions':
            inset.annotate(i*'\n'+'\nN=%i %s (%i ROIs, %i mice)' % (len(resp),
                                                average_by, np.sum([len(r) for r in SUMMARY[key]['RESPONSES']]),
                                                len(np.unique(SUMMARY[key]['subjects']))),
                           (0,0), fontsize=7,
                           va='top',color=color, xycoords='axes fraction')
        else:
            inset.annotate(i*'\n'+'\nn=%i %s (%i sessions, %i mice)' % (len(resp),
                                                average_by, len(SUMMARY[key]['RESPONSES']),
                                                                len(np.unique(SUMMARY[key]['subjects']))),
                           (0,0), fontsize=7,
                           va='top',color=color, xycoords='axes fraction')
        
    # suppression index
    for i, key, color in zip(range(2), KEYS, COLORS):
        pt.violin(SIs[i], X=[i], ax=inset, COLORS=[color])
    inset.plot([0,1], 1.05*np.ones(2), 'k-', lw=0.5)
    inset.annotate('p=%.1e' % stats.mannwhitneyu(SIs[0], SIs[1]).pvalue,
                   (0.5, 1.08), ha='center', fontsize=6)
    
    for ax in AX:
        if xscale=='log':
            pt.set_plot(ax, xlabel='stim. size ($^o$)',
                        xscale='log', #xticks=[10,100], xticks_labels=['10', '100'], xlim=[9,109],
                        ylabel='$\delta$ %s' % SUMMARY['quantity'].replace('dFoF', '$\Delta$F/F'))
        else:
            pt.set_plot(ax, xlabel='stim. size ($^o$)',
                        xticks=[0, 40, 80, 120], xticks_labels=['0', '40', '80', 'FF'],
                        ylabel='$\delta$ %s' % SUMMARY['quantity'].replace('dFoF', '$\Delta$F/F'))
    pt.set_common_ylims(AX)
    pt.set_plot(inset, xticks=[], ylabel='suppr. index', yticks=[0, 0.5, 1], ylim=[0, 1.09])
    return fig

SUMMARY = np.load('../data/in-vivo/size-tuning-dFoF-summary.npy', allow_pickle=True).item()
fig = plot_summary(SUMMARY, ['WT', 'GluN1'], ['k', 'tab:blue'], average_by='ROIs')
fig = plot_summary(SUMMARY, ['WT', 'GluN1'], ['k', 'tab:blue'], average_by='sessions')

# %%
fig = plot_summary(SUMMARY, ['WT', 'GluN3'], ['darkgreen', 'mediumseagreen'], average_by='ROIs')
fig.savefig('size-tuning-WT-GluN3-per-ROI.svg')
fig = plot_summary(SUMMARY, ['WT', 'GluN3'], ['darkgreen', 'mediumseagreen'], average_by='sessions')
fig.savefig('size-tuning-WT-GluN3-per-session.svg')

# %%
fig = plot_summary(SUMMARY, ['GluN1', 'GluN3'], ['tab:blue', 'g'], average_by='ROIs')
fig = plot_summary(SUMMARY, ['GluN1', 'GluN3'], ['tab:blue', 'g'], average_by='sessions')

# %%
for quantity in ['rawFluo', 'neuropil', 'dFoF']:
    SUMMARY = np.load('data/%s-summary.npy' % quantity, allow_pickle=True).item()
    fig = plot_summary(SUMMARY, average_by='ROIs')

# %%
for neuropil_correction_factor in [0.6, 0.7, 0.8, 0.9]:
    SUMMARY = np.load('data/factor-neuropil-%.1f-summary.npy' % neuropil_correction_factor,
                      allow_pickle=True).item()
    fig = plot_summary(SUMMARY, average_by='ROIs')
    plt.annotate('Neuropil-factor for substraction: %.1f\n\n' % neuropil_correction_factor,
                 (1,1), xycoords='axes fraction')

# %%
for roi_to_neuropil_fluo_inclusion_factor in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
    SUMMARY = np.load('data/inclusion-factor-neuropil-%.1f-summary.npy' % roi_to_neuropil_fluo_inclusion_factor,
                      allow_pickle=True).item()
    fig = plot_summary(SUMMARY, average_by='ROIs')
    plt.annotate('roiFluo/Neuropil inclusion-factor: %.2f\n' % roi_to_neuropil_fluo_inclusion_factor,
                 (1,1), xycoords='axes fraction')

# %% [markdown]
# # Visualizing some evoked response in single ROI

# %%
import sys, os
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from physion.dataviz.episodes.trial_average import plot_trial_average
sys.path.append('../')
import plot_tools as pt

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)


def cell_tuning_example_fig(filename,
                            contrast=1.0,
                            stat_test_props = dict(interval_pre=[-1,0], 
                                                   interval_post=[1,2],
                                                   test='ttest',
                                                   positive=True),
                            response_significance_threshold = 0.01,
                            Nsamples = 10, # how many cells we show
                            seed=10):
    
    np.random.seed(seed)
    
    data = Data(filename)
    
    protocol_id = data.get_protocol_id('size-tuning-protocol-dep')
    if protocol_id is None:
        protocol_id = data.get_protocol_id('size-tuning-protocol-dep-long')
    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=protocol_id,
                           with_visual_stim=True,
                           verbose=True)
    
    fig, AX = pt.plt.subplots(Nsamples, len(EPISODES.varied_parameters['radius']), 
                          figsize=(7,7))
    plt.subplots_adjust(right=0.75, left=0.1, top=0.97, bottom=0.05, wspace=0.1, hspace=0.8)
    
    for Ax in AX:
        for ax in Ax:
            ax.axis('off')

    for i, r in enumerate(np.random.choice(np.arange(data.vNrois), 
                                           min([Nsamples, data.vNrois]), replace=False)):

        # SHOW trial-average
        plot_trial_average(EPISODES,
                           condition=EPISODES.find_episode_cond(key='angle', value=0),
                           column_key='radius',
                           quantity='dFoF',
                           ybar=1., ybarlabel='1dF/F',
                           xbar=1., xbarlabel='1s',
                           roiIndex=r,
                           with_stat_test=True,
                           stat_test_props=stat_test_props,
                           with_screen_inset=True,
                           AX=[AX[i]], no_set=False)
        AX[i][0].annotate('roi #%i  ' % (r+1), (0,0), ha='right', xycoords='axes fraction')

        # SHOW summary angle dependence
        inset = pt.inset(AX[i][-1], (2.2, 0.2, 1.2, 0.8))

        radii, y, sy, responsive_radii = [], [], [], []
        responsive = False

        for a, radius in enumerate(EPISODES.varied_parameters['radius']):

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=\
                                            EPISODES.find_episode_cond(key='radius',
                                                                       value=radius),
                                                            response_args=dict(quantity='dFoF', roiIndex=r),
                                                            **stat_test_props)

            radii.append(radius)
            y.append(np.mean(stats.y-stats.x))    # means "post-pre"
            sy.append(np.std(stats.y-stats.x))    # std "post-pre"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_radii.append(radius)

        pt.plot(radii, np.array(y), sy=0*np.array(sy), ax=inset)
        inset.plot(radii, 0*np.array(radii), 'k:', lw=0.5)
        inset.set_ylabel('$\delta$ $\Delta$F/F     ', fontsize=7)
        inset.set_xticks([0,50,100])
        if i==(Nsamples-1):
            inset.set_xlabel('radius ($^{o}$)', fontsize=7)

        #SI = selectivity_index(angles, y)
        #inset.annotate('SI=%.2f ' % SI, (0, 1), ha='right', weight='bold', fontsize=8,
        #               color=('k' if responsive else 'lightgray'), xycoords='axes fraction')
        inset.annotate(('responsive' if responsive else 'unresponsive'), (1, 1), ha='right',
                        weight='bold', fontsize=6, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)),
                        xycoords='axes fraction')
        
    return fig

folder = os.path.join(os.path.expanduser('~'),
                      'CURATED', 'SST-WT-NR1-GluN3-2023')
fig = cell_tuning_example_fig(os.path.join(folder, '2023_04_20-16-02-15.nwb'),
                              #os.path.join(folder, SUMMARY['WT']['FILES'][5]),
                             contrast=1)
fig.savefig('size-tuning-WT-examples.svg')

# %%
SUMMARY['WT']['FILES'][5]

# %%
fig = cell_tuning_example_fig(SUMMARY['GluN3']['FILES'][8])
#fig.savefig('size-tuning-GluN3-examples.svg')

# %% [markdown]
# # Visualizing some raw population data

# %%
import sys, os
import numpy as np
import matplotlib.pylab as plt
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings

sys.path.append('../')
import plot_tools as pt

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

data = Data(os.path.join(folder, SUMMARY['WT']['FILES'][5]),
            with_visual_stim=True)
data.init_visual_stim()

tlim = [984,1080]

settings = {'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1, 'subsampling': 1, 'color': 'purple'},
            'Pupil': {'fig_fraction': 2, 'subsampling': 1, 'color': '#d62728'},
             'CaImaging': {'fig_fraction': 4,
              'subsampling': 1,
              'subquantity': 'dF/F',
              'color': '#2ca02c',
              'roiIndices': np.random.choice(np.arange(data.nROIs), 5, replace=False)},
             'CaImagingRaster': {'fig_fraction': 3,
              'subsampling': 1,
              'roiIndices': 'all',
              'normalization': 'per-line',
              'subquantity': 'dF/F'},
             'VisualStim': {'fig_fraction': 0.5, 'color': 'black', 'with_screen_inset':True}}

fig, _ = plot_raw(data, tlim=tlim, settings=settings)
#fig.savefig('raw-data-zoom.svg')

# %%
settings = {'Locomotion': {'fig_fraction': 1, 'subsampling': 2, 'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1, 'subsampling': 2, 'color': 'purple'},
            'Pupil': {'fig_fraction': 2, 'subsampling': 1, 'color': '#d62728'},
             'CaImaging': {'fig_fraction': 4,
              'subsampling': 1,
              'subquantity': 'dF/F',
              'color': '#2ca02c',
              'roiIndices': np.random.choice(np.arange(data.nROIs), 5, replace=False)},
             'CaImagingRaster': {'fig_fraction': 3,
              'subsampling': 1,
              'roiIndices': 'all',
              'normalization': 'per-line',
              'subquantity': 'dF/F'}}

tlim = [900, 1300]
fig, _ = plot_raw(data, tlim=tlim, settings=settings)
#fig.savefig('raw-data-unzoom.svg')

# %%
from physion.dataviz.imaging import show_CaImaging_FOV
data = Data(os.path.join(folder, SUMMARY['WT']['FILES'][5]))
fig, ax = pt.figure(figsize=(2,5))
fig = show_CaImaging_FOV(data, key='meanImg', NL=3, ax=ax)#, roiIndices='all')

# %%
data = Data(os.path.join(folder, SUMMARY['GluN3']['FILES'][8]))
fig, ax = pt.figure(figsize=(2,5))
fig = show_CaImaging_FOV(data, key='meanImg', NL=2.5, ax=ax)#, roiIndices='all')

# %%
