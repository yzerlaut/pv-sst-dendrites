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
# general python modules
import sys, os, pprint, pandas
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

#sys.path.append('./')
from analysis import compute_tuning_response_per_cells
sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
sys.path.append('../')
import plot_tools as pt

folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'SST-WT-NR1-GluN3-2023')

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

# %% [markdown]
# ## Build the dataset from the NWB files

# %%
DATASET = scan_folder_for_NWBfiles(folder,
                                   verbose=False)


# %%
# -------------------------------------------------- #
# ----    Pick the session datafiles and sort ------ #
# ----      them according to genotype ------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# ----    Pick the session datafiles and sort ------ #
# ----      them according to genotype ------------- #
# -------------------------------------------------- #

def init_summary(DATASET):

    SUMMARY = {'WT':{'FILES':[], 'subjects':[]}, 
               'GluN1':{'FILES':[], 'subjects':[]}, 
               'GluN3':{'FILES':[], 'subjects':[]},
               # add a summary for half contrast
               'WT_c=0.5':{'FILES':[]},
               'GluN1_c=0.5':{'FILES':[]},
               'GluN3_c=0.5':{'FILES':[]}}

    for i, protocols in enumerate(DATASET['protocols']):

        # select the sessions with different 
        if ('ff-gratings-8orientation-2contrasts-15repeats' in protocols) or\
            ('ff-gratings-8orientation-2contrasts-10repeats' in protocols):

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
# -------------------------------------------------- #
# ----   Loop over datafiles to compute    --------- #
# ----           the evoked responses      --------- #
# -------------------------------------------------- #

def orientation_selectivity_index(resp_pref, resp_90):
    """                                                                         
     computes the selectivity index: (Pref-Orth)/Pref
     clipped in [0,1] --> because resp_90 can be negative    
    """
    return (resp_pref-np.clip(resp_90, 0, np.inf))/resp_pref

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       positive=True) 
    
def compute_summary_responses(DATASET,
                              quantity='dFoF',
                              roi_to_neuropil_fluo_inclusion_factor=1.15,
                              neuropil_correction_factor = 0.7,
                              method_for_F0 = 'sliding_percentile',
                              percentile=5., # percent
                              sliding_window = 300, # seconds
                              Nmax=999, # max datafiles (for debugging)
                              stat_test_props=dict(interval_pre=[-1.,0],                                   
                                                   interval_post=[1.,2.],                                   
                                                   test='anova',                                            
                                                   positive=True),
                              response_significance_threshold=5e-2,
                              verbose=True):
    
    SUMMARY = init_summary(DATASET)
    
    SUMMARY['quantity'] = quantity
    SUMMARY['quantity_args'] = dict(roi_to_neuropil_fluo_inclusion_factor=\
                                        roi_to_neuropil_fluo_inclusion_factor,
                                    method_for_F0=method_for_F0,
                                    percentile=percentile,
                                    sliding_window=sliding_window,
                                    neuropil_correction_factor=neuropil_correction_factor)
    
    for key in ['WT', 'GluN1']:

        SUMMARY[key]['responses'] = []

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
            
            protocol = 'ff-gratings-8orientation-2contrasts-15repeats' if\
                        ('ff-gratings-8orientation-2contrasts-15repeats' in data.protocols) else\
                        'ff-gratings-8orientation-2contrasts-10repeats'

            t, significant_waveforms = compute_tuning_response_per_cells(data,
                                                                 imaging_quantity=quantity,
                                                                 prestim_duration=4.,
                                                                 contrast=1, # at full contrast
                                                                 protocol_name=protocol,
                                                                 stat_test_props=stat_test_props,
                                                                 response_significance_threshold=response_significance_threshold,
                                                                 return_significant_waveforms=True,
                                                                 verbose=False)

            SUMMARY[key]['responses'] += significant_waveforms
                
    SUMMARY['t'] = t
                  
    return SUMMARY


# %%
quantity = 'dFoF'
SUMMARY = compute_summary_responses(DATASET, quantity=quantity, verbose=False, Nmax=10000)
np.save('../data/in-vivo/%s-ff-gratings-waveforms.npy' % quantity, SUMMARY)

# %%
from scipy.stats import sem

SUMMARY = np.load('../data/in-vivo/%s-ff-gratings-waveforms.npy' % quantity,
                 allow_pickle=True).item()

fig, AX = pt.figure(axes=(2,1), right=10)
peak_time = {}
for i, key, color in zip(range(2), ['WT', 'GluN1'], ['tab:orange', 'tab:purple']):

    pt.plot(SUMMARY['t'], np.mean(SUMMARY[key]['responses'], axis=0),
            sy = sem(SUMMARY[key]['responses'], axis=0), color=color,
            ax=AX[0], no_set=True)
    
    normed = [(resp-np.min(resp))/(np.max(resp)-np.min(resp))\
                  for resp in SUMMARY[key]['responses']]
    cond = (SUMMARY['t']>0) & (SUMMARY['t']<4)
    peak_time[key] = [SUMMARY['t'][cond][np.argmax(n[cond])] for n in normed]
    pt.plot(SUMMARY['t'], np.mean(normed, axis=0), 
            sy = np.std(normed, axis=0), 
            color=color, ax=AX[1], no_set=True)
    pt.annotate(AX[1], i*'\n'+'%s, n=%i' % (key, len(SUMMARY[key]['responses'])), 
                (1,1), va='top', color=color)
pt.set_plot(AX[0], ylabel='$\Delta$F/F', xlabel='time (s)')
pt.set_plot(AX[1], ylabel='norm. $\Delta$F/F', xlabel='time (s)', yticks=[0,1])

# %%
from scipy.stats import ttest_ind
fig, ax = pt.figure(figsize=(.8,1.2))
pt.violin(peak_time['WT'], X=[0], ax=ax, COLORS=['tab:orange'])
pt.violin(peak_time['GluN1'], X=[1], ax=ax, COLORS=['tab:purple'])
ax.plot([0,1], ax.get_ylim()[1]*np.ones(2), 'k')
pt.annotate(ax, pt.from_pval_to_star(\
                ttest_ind(peak_time['WT'], peak_time['GluN1']).pvalue),
            (0.5, ax.get_ylim()[1]), xycoords='data', ha='center')
pt.set_plot(ax, ylabel='time (s)\nof peak resp.', 
            xticks=[0,1], xticks_labels=['WT', 'GluN1'])
#centers_of_mass['GluN1']], X=[0,1])

# %%
