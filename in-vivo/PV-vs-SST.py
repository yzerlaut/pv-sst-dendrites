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
# general python modules
import sys, os, pprint, pandas
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
sys.path.append('../')
import plot_tools as pt

from analysis import compute_tuning_response_per_cells, shift_orientation_according_to_pref

root_folder = os.path.join(os.path.expanduser('~'), 'DATA', 'PV-SST-GCaMP8s-ffGratings-2024')

dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        method_for_F0 = 'sliding_percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
)

stat_test_props=dict(interval_pre=[-1.,0],                                   
                   interval_post=[1.,2.],                                   
                   test='anova',                                            
                   positive=True)

response_significance_threshold=5e-2

# %% [markdown]
# ## Build the dataset from the NWB files

# %%
DATASET = {\
    'PV':scan_folder_for_NWBfiles(os.path.join(root_folder, 'PV', 'NWBs'),
                                  for_protocol='ff-gratings-8orientation-2contrasts-15repeats'),
    'SST':scan_folder_for_NWBfiles(os.path.join(root_folder, 'SST', 'NWBs'),
                                   for_protocol='ff-gratings-8orientation-2contrasts-15repeats'),
}


# %%
# -------------------------------------------------- #
# ----   Loop over datafiles               --------- #
# -------------------------------------------------- #

def orientation_selectivity_index(resp_pref, resp_90):
    """                                                                         
     computes the selectivity index: (Pref-Orth)/Pref
     clipped in [0,1] --> because resp_90 can be negative    
    """
    return (resp_pref-np.clip(resp_90, 0, np.inf))/resp_pref

def compute_summary_responses(DATASET,
                              Nmax=999, # max datafiles (for debugging)
                              verbose=True):
    
    SUMMARY = {}
    
    SUMMARY['dFoF_args'] = dFoF_parameters
    
    for key in ['PV', 'SST']:
        SUMMARY[key] = {'subjects':[], 'tstart':[], 'tstop':[]}
        
        for contrast in [0.5, 1]:
            SUMMARY[key]['RESPONSES_c=%.1f' % contrast], SUMMARY[key]['OSI_c=%.1f' % contrast] = [], []
            SUMMARY[key]['FRAC_RESP_c=%.1f' % contrast], SUMMARY[key]['SIGNIFICANT_c=%.1f' % contrast] = [], []
            
        for f, s in zip(DATASET[key]['files'][:Nmax], DATASET[key]['subjects'][:Nmax]):
            
            data = Data(f, verbose=False)
           
            protocol = None
            if 'ff-gratings-8orientation-2contrasts-15repeats' in data.protocols:
                protocol = 'ff-gratings-8orientation-2contrasts-15repeats'
            elif 'ff-gratings-8orientation-2contrasts-10repeats' in data.protocols:
                protocol = 'ff-gratings-8orientation-2contrasts-10repeats'

            if protocol is not None:
                    
                SUMMARY[key]['subjects'].append(s)
                print('analyzing "%s" [...] ' % f, '    ( mouse: %s )' % s)
                
                data.build_dFoF(**dFoF_parameters, verbose=False)
                data.init_visual_stim()
                
                pCond = data.get_protocol_cond(protocol_id=data.get_protocol_id(protocol_name=protocol))
                # start and stop of the grating protocol (in case of rec. with other protocols)
                SUMMARY[key]['tstart'].append(np.min(data.visual_stim.experiment['time_start'][:len(pCond)][pCond]))
                SUMMARY[key]['tstop'].append(np.max(data.visual_stim.experiment['time_stop'][:len(pCond)][pCond]))
                
                for contrast in [0.5, 1]:
                    
                    responses, significant, shifted_angle = compute_tuning_response_per_cells(data,
                                                                                              imaging_quantity='dFoF',
                                                                                              contrast=contrast,
                                                                                              protocol_name=protocol,
                                                                                              stat_test_props=stat_test_props,
                                                                                              response_significance_threshold=\
                                                                                                          response_significance_threshold,
                                                                                              verbose=False)
                    
                    SUMMARY[key]['RESPONSES_c=%.1f' % contrast].append(responses)
                    SUMMARY[key]['FRAC_RESP_c=%.1f' % contrast].append(np.sum(significant)/len(significant))
                    SUMMARY[key]['SIGNIFICANT_c=%.1f' % contrast].append(significant)
                    SUMMARY[key]['OSI_c=%.1f' % contrast].append([orientation_selectivity_index(r[1], r[5]) for r in responses])
    
                    
    SUMMARY['shifted_angle'] = shifted_angle
    print('done ! ')
    
    return SUMMARY

if False:
    # swith to true to redo the analysis
    SUMMARY = compute_summary_responses(DATASET,
                                        verbose=False)
    np.save('../data/in-vivo/summary-episodes-PV-SST.npy', SUMMARY)

# %% [markdown]
# # Responsiveness to visual stimulation

# %%
from scipy import stats

SUMMARY = np.load('../data/in-vivo/summary-episodes-PV-SST.npy', allow_pickle=True).item()

fig1, AX1 = pt.figure(axes=(2,2),figsize=(1.,1.2))

cases, colors = ['PV', 'SST'], ['tab:red', 'tab:orange']

for i, case in enumerate(cases):
    for c, contrast in enumerate([0.5, 1.0]):
        resp = np.sum([np.sum(r) for r in SUMMARY[case]['SIGNIFICANT_c=%.1f' % contrast]])
        tot = np.sum([len(r) for r in SUMMARY[case]['SIGNIFICANT_c=%.1f' % contrast]])
        pt.pie([100*resp/tot, 100*(1-resp/tot)], ax=AX1[i][c], COLORS=[colors[i], 'lightgrey'])
        pt.annotate(AX1[i][c], '%.1f%%' % (100*resp/tot), (.5,.85), va='top', ha='center',color='w', fontsize=7)
        AX1[i][c].set_title('n=%i/%i\nN=%i sess.' % (resp,tot,len(SUMMARY[case]['SIGNIFICANT_c=%.1f' % contrast])), fontsize=7)

fig2, AX2 = pt.figure(axes=(2,1), figsize=(.6,1.), wspace=5.)
for c, contrast in enumerate([0.5, 1.0]):
    for i, case in enumerate(cases):
        frac = 100*np.mean([r for r in SUMMARY[case]['FRAC_RESP_c=%.1f' % contrast]])
        pt.violin(100*np.array(SUMMARY[case]['FRAC_RESP_c=%.1f' % contrast]),
                  x=i, color=colors[i], ax=AX2[c])
    pt.set_plot(AX2[c], ['left'], ylabel='% responsive', yticks=[0,50,100])
#stats.mannwhitneyu(SUMMARY['PV']['FRAC_RESP_c=1.0'], SUMMARY['KO']['FRAC_RESP_c=1.0'])
#
#fig1.savefig('../figures/in-vivo/PV-SST-summary-responsiveness.svg')
#fig1.savefig('../figures/Figure7/PV-SST-summary-responsiveness.pdf')

# %% [markdown]
# # Temporal Dynamics of Visually-Evoked Responses

# %%
TAU_DECONVOLUTION = 0.8

dFoF_parameters['smoothing'] = 0.5

def compute_summary_responses(DATASET,
                              Nmax=999, # max datafiles (for debugging)
                              verbose=True):
    
    SUMMARY = np.load('../data/in-vivo/summary-episodes-PV-SST.npy', allow_pickle=True).item()

    SUMMARY['dFoF_args'] = dFoF_parameters
    
    for key in ['PV', 'SST']:
        SUMMARY[key]['subjects'] = []
        SUMMARY[key]['tstart'] = []
        SUMMARY[key]['tstop'] = []
        
        for contrast in [0.5, 1]:
            SUMMARY[key]['WAVEFORMS_c=%.1f' % contrast] = []
            
        for f, s, significant in zip(DATASET[key]['files'][:Nmax],
                                     DATASET[key]['subjects'],
                                     SUMMARY[key]['SIGNIFICANT_c=1.0']):
            
            SUMMARY[key]['subjects'].append(s)
            print('analyzing "%s" [...] ' % f)
            data = Data(f, verbose=False)
            
            data.build_dFoF(**dFoF_parameters, verbose=False)
            data.build_Deconvolved(Tau=TAU_DECONVOLUTION)
            data.init_visual_stim()

            protocol = 'ff-gratings-8orientation-2contrasts-15repeats' if\
                        ('ff-gratings-8orientation-2contrasts-15repeats' in data.protocols) else\
                        'ff-gratings-8orientation-2contrasts-10repeats'

            for contrast in [0.5, 1]:
                
                t, significant_waveforms = compute_tuning_response_per_cells(data,
                                                                             prestim_duration=3.5,
                                                                             imaging_quantity='Deconvolved',
                                                                             contrast=contrast,
                                                                             protocol_name=protocol,
                                                                             stat_test_props=stat_test_props,
                                                                             response_significance_threshold=\
                                                                                          response_significance_threshold,
                                                                             force_significant=significant,
                                                                             verbose=False,
                                                                             return_significant_waveforms=True)
                
                SUMMARY[key]['WAVEFORMS_c=%.1f' % contrast].append(significant_waveforms)
                
    SUMMARY['t'] = t
    print('done ! ')
    
    return SUMMARY

if False:
    SUMMARY = compute_summary_responses(DATASET,
                                        verbose=False)
    np.save('../data/in-vivo/summary-deconvolved-PV-SST.npy', SUMMARY)

# %%
SUMMARY = np.load('../data/in-vivo/summary-deconvolved-PV-SST.npy', allow_pickle=True).item()
#SUMMARY['t'] 
fig, AX = pt.figure(axes=(2,1),figsize=(1.1,1.),wspace=0.1, right=7.)

cases, colors = ['SST', 'PV'], ['tab:orange', 'tab:red']

for i, case in enumerate(cases):

    for c, contrast in enumerate([0.5, 1.0]):

        resp = np.array([np.mean(r, axis=0) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])

        pt.plot(SUMMARY['t'], np.mean(resp, axis=0),
                sy=stats.sem(resp, axis=0),
                color=colors[i], ax=AX[c])


for c, contrast in enumerate([0.5, 1.0]):
    #AX[c].fill_between([0,2], [0,0], [1,1], color='lightgray', alpha=.2, lw=0)
    pt.set_plot(AX[c], ['left','bottom'] if c==0 else ['bottom'],
                xlim=[-1,6], xticks=[0,2,4], 
                #yticks=[0, 1], yticks_labels=['0','1'] if c==0 else [],
                #ylim=[0,1], 
                ylabel='deconvolved \n resp. (norm.)' if c==0 else '',
                title='c=%.1f' % contrast,
                xlabel=20*' '+'time from stim. (s)' if c==0 else '')

# %%
SUMMARY = np.load('../data/in-vivo/summary-deconvolved-PV-SST.npy', allow_pickle=True).item()

fig, ax = pt.figure(figsize=(0.9,1.05),wspace=0.1, right=7.)

contrast = 1.0
cases, colors = ['SST', 'PV'], ['tab:orange', 'tab:red']

inset = pt.inset(ax, [1.15, 0., 0.45, 1.0])
fitInterval = [0.8, 1.8]
Levels = {}

for i, case in enumerate(cases):

    resp = np.array([np.mean(r, axis=0) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    nROIs = np.sum([len(r) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast]])
    
    # peak norm:
    peak = np.array([np.max(np.mean(r, axis=0)) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    through = np.array([np.min(np.mean(r, axis=0)) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    resp = np.transpose((resp.T-through)/(peak-through))
    
    pt.plot(SUMMARY['t'], np.mean(np.transpose(resp.T-np.min(resp, axis=1)), axis=0),
            sy=1.5*stats.sem(resp, axis=0),
            color=colors[i], ax=ax)
    pt.annotate(ax, 'N=%i sess. (n=%i ROIs, %i mice)' % (\
                        len(SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast]),
                        nROIs, len(np.unique(SUMMARY[case]['subjects'])))+i*'\n',
                (0.6,1.1), ha='right', color=colors[i], fontsize=6)
    pt.plot(SUMMARY['t'], np.mean(np.transpose(resp.T-np.min(resp, axis=1)), axis=0), 
               sy=stats.sem(resp, axis=0), 
               color=colors[i], ax=inset)
    cond = (SUMMARY['t']>fitInterval[0]) & (SUMMARY['t']<fitInterval[1])
    Levels[case] = np.mean(resp[:,cond], axis=1)


ax.annotate(pt.from_pval_to_star(stats.mannwhitneyu(Levels['PV'], Levels['SST']).pvalue),
            (np.sum(fitInterval)/2., 0.05), ha='center')
        
ax.plot(fitInterval, [0,0], '-', color='tab:grey', lw=4)

pt.set_plot(inset,
            xlim=[-0.15,0.32], xticks=[0, 0.3], xticks_labels=['0', '0.3'],
            yticks=[0,1], yticks_labels=[], ylim=[0,1])
#pt.draw_bar_scales(inset, Xbar=0.2, Xbar_label='0.2s', Ybar=1e-12, fontsize=7, lw=0.5)
    
ax.fill_between([0,2], [0,0], [1,1], color='lightgray', alpha=.2, lw=0)
pt.set_plot(ax, ['left','bottom'],
            xlim=[-0.5,2.6], xticks=[0,1,2], yticks=[0,1], ylim=[0,1],
            ylabel='deconv. resp.\n (peak norm.)')
fig.savefig('../figures/in-vivo/PV-SST-summary-evoked-resp.svg')

# %% [markdown]
# ## Post-Stim Level

# %%
SUMMARY = np.load('../data/in-vivo/summary-deconvolved-PV-SST.npy', allow_pickle=True).item()

contrast = 1.0
cases, colors = ['SST', 'PV'], ['tab:orange', 'tab:red']

Delays = {}
for i, case in enumerate(cases):
    
    resp = np.array([np.mean(r, axis=0) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    nROIs = np.sum([len(r) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast]])

    Delays[case] = [SUMMARY['t'][np.argmax(resp[r,:])] for r in range(resp.shape[0])]
    print(case, ' %.2f +/- %.2f (N=%i sess.) ' % (np.mean(Delays[case]), stats.sem(Delays[case]), resp.shape[0] ) )
    
print('Mann-Whitney: p=%.1e' % stats.mannwhitneyu(Delays['SST'], Delays['PV']).pvalue)

# %% [markdown]
# ## Peak Delay

# %%
SUMMARY = np.load('../data/in-vivo/summary-deconvolved-PV-SST.npy', allow_pickle=True).item()

contrast = 1.0
cases, colors = ['SST', 'PV'], ['tab:orange', 'tab:red']

Delays = {}
for i, case in enumerate(cases):
    
    resp = np.array([np.mean(r, axis=0) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    nROIs = np.sum([len(r) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast]])

    Delays[case] = [SUMMARY['t'][np.argmax(resp[r,:])] for r in range(resp.shape[0])]
    print(case, ' %.2f +/- %.2f (N=%i sess.) ' % (np.mean(Delays[case]), stats.sem(Delays[case]), resp.shape[0] ) )
    
print('Mann-Whitney: p=%.1e' % stats.mannwhitneyu(Delays['SST'], Delays['PV']).pvalue)


# %%
SUMMARY = np.load('../data/in-vivo/summary-deconvolved-PV-SST.npy', allow_pickle=True).item()

fig, ax = pt.figure(figsize=(1.1,1.),wspace=0.1, right=7.)

cases, colors = ['SST', 'PV'], ['tab:orange', 'tab:red']


for i, case in enumerate(cases):
    contrast = 1.0

    resp = np.concatenate([r for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])

    # peak norm:
    peak = np.concatenate([np.max(r, axis=1) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    through = np.concatenate([np.min(r, axis=1) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    resp = np.transpose((resp.T-through)/(peak-through))
    
    # Z-score:
    #norm = np.array([np.std(np.mean(r, axis=0)) for r in SUMMARY[case]['WAVEFORMS_c=%.1f' % contrast] if len(r)>0])
    #resp = np.transpose((resp.T-np.mean(resp,axis=1))/norm)
    
    pt.plot(SUMMARY['t'], np.mean(resp, axis=0),
            sy=stats.sem(resp, axis=0),
            #sy=np.std(resp, axis=0),
            color=colors[i], ax=ax)
    pt.annotate(ax, i*'\n'+'n=%i ROIs' % resp.shape[0],
                (1,1), va='top', ha='right', color=colors[i], fontsize=6)

pt.set_plot(ax, ['left','bottom'] if c==0 else ['bottom'],
            xlim=[-1,6], xticks=[0,2,4], #yticks=[0,1],
            #yticks_labels=['0','1'] if c==0 else [],
            #ylim=[0,1], 
            ylabel='deconvolved \n resp. (norm.)',
            title='c=%.1f' % contrast,
            xlabel='time from stim. (s)')

# %% [markdown]
# # Orientation tuning

# %%
from scipy.optimize import minimize

def func(S, X):
    """ fitting function """
    nS = (S+90)%180-90
    return X[0]*np.exp(-(nS**2/2./X[1]**2))+X[2]

def selectivity_index(resp1, resp2):
    return (resp1-np.clip(resp2, 0, np.inf))/resp1

def generate_comparison_figs(SUMMARY, cases, contrasts,
                             average_by='ROIs',
                             colors=['k', 'tab:blue', 'tab:green'],
                             norm='',
                             ms=1):
    
    fig, [ax,ax2] = plt.subplots(1, 2, figsize=(8, 1))
    plt.subplots_adjust(top=0.9, bottom=0.2, right=0.4)
    inset = pt.inset(ax2, (1.7, 0.2, 0.3, 0.8))

    SIs = []
    for i, key, contrast in zip(range(2), cases, contrasts):

        if average_by=='sessions':
            resp = np.array([np.mean(r, axis=0)\
                             for r in SUMMARY[key]['RESPONSES_c=%.1f' % contrast]])
        else:
            resp = np.concatenate([r\
                            for r in SUMMARY[key]['RESPONSES_c=%.1f' % contrast]])
        resp = np.clip(resp, 0, np.inf) # CLIP RESPONSIVE TO POSITIVE VALUES
        
        if norm!='':
            resp = np.divide(resp, np.max(resp, axis=1, keepdims=True))
            
        SIs.append([selectivity_index(r[1], r[5]) for r in resp])

        # data
        pt.scatter(SUMMARY['shifted_angle']+2*i, np.mean(resp, axis=0),
                   sy=stats.sem(resp, axis=0), ax=ax, color=colors[i], ms=ms)

        
        # fit
        def to_minimize(x0):
            return np.sum((resp.mean(axis=0)-\
                           func(SUMMARY['shifted_angle'], x0))**2)
        
        res = minimize(to_minimize,
                       [0.8, 10, 0.2])
        x = np.linspace(-30, 180-30, 100)
        ax.plot(x, func(x, res.x), lw=2, alpha=.5, color=colors[i])

        # cum. proba plot
        bins = np.linspace(0,1,50)
        hist, _ = np.histogram(SIs[-1], bins=bins)
        ax2.plot(np.mean(SIs[-1])*np.ones(2), [0,1], color=colors[i], lw=0.5)
        ax2.plot(bins[1:], np.cumsum(hist)/hist.sum(), color=colors[i])
        pt.set_plot(ax2, xticks=[0,0.5,1.], xlabel='select. index', 
                    yticks=[0, 0.5, 1], ylabel='cum. frac')
        print(key, 'c=%.1f' % contrast, 'average by:', average_by, ', OSI= %.2f +/- %.2f' % (np.mean(SIs[-1]), stats.sem(SIs[-1])))
        try:
            if average_by=='sessions':
                inset.annotate(i*'\n'+'\nN=%i %s (%i ROIs, %i mice)' % (len(resp),
                        average_by, np.sum([len(r) for r in\
                                            SUMMARY[key]['RESPONSES_c=%.1f' % contrast]]),
                        len(np.unique(SUMMARY[key]['subjects']))),
                               (0,0), fontsize=7,
                               va='top',color=colors[i], xycoords='axes fraction')
            else:
                inset.annotate(i*'\n'+'\nn=%i %s (%i sessions, %i mice)' % (len(resp),
                                average_by, len(SUMMARY[key]['RESPONSES_c=%.1f' % contrast]),
                                                len(np.unique(SUMMARY[key]['subjects']))),
                               (0,0), fontsize=7,
                               va='top',color=colors[i], xycoords='axes fraction')
        except BaseException as be:
            pass
        
    # selectivity index
    for i, key in enumerate(cases):
        pt.violin(SIs[i], X=[i], ax=inset, COLORS=[colors[i]])
    if len(cases)==2:
        inset.plot([0,1], 1.05*np.ones(2), 'k-', lw=0.5)
        inset.annotate('Mann-Whitney: p=%.1e' % stats.mannwhitneyu(SIs[0], SIs[1]).pvalue,
                       (0, 1.09), fontsize=6)
        print('Mann-Whitney: p=%.1e' % stats.mannwhitneyu(SIs[0], SIs[1]).pvalue)
    pt.set_plot(inset, xticks=[], ylabel='select. index', yticks=[0, 0.5, 1], ylim=[0, 1.09])
    

    ylabel=norm+'$\delta$ $\Delta F/F$'
    pt.set_plot(ax, xlabel='angle ($^o$) from pref.',
                ylabel=ylabel,
                #yticks=[0.4,0.6,0.8,1],
                xticks=SUMMARY['shifted_angle'],
                xticks_labels=['%.0f'%s if (i%4==1) else '' for i,s in enumerate(SUMMARY['shifted_angle'])])

    return fig, ax

SUMMARY = np.load('../data/in-vivo/summary-episodes.npy', allow_pickle=True).item()

fig1, ax = generate_comparison_figs(SUMMARY, ['WT', 'KO'], [1., 1.],
                                    average_by='sessions', norm='norm. ',
                                    colors=['darkorange', 'tab:purple'])
fig1.savefig('../figures/in-vivo/orientation-tuning-WT-vs-KO-c=1-SESSIONS.svg')
fig1.savefig('../figures/Figure7/orientation-tuning-WT-vs-KO-c=1-SESSIONS.pdf')
print('')
fig2, ax = generate_comparison_figs(SUMMARY, ['WT', 'KO'], [1., 1.],
                                    average_by='ROIs', norm='norm. ',
                                    colors=['darkorange', 'tab:purple'])
fig2.savefig('../figures/in-vivo/orientation-tuning-WT-vs-KO-c=1-ROIS.svg')
fig2.savefig('../figures/Figure7/orientation-tuning-WT-vs-KO-c=1-ROIS.pdf')
print('')
fig3, ax = generate_comparison_figs(SUMMARY, ['WT', 'WT'], [1., 0.5],
                                    average_by='sessions', norm='norm. ',
                                    colors=['darkorange', 'gold'])
fig3.savefig('../figures/in-vivo/orientation-tuning-WT-c=1-vs-c=05-SESSIONS.svg')
fig3.savefig('../figures/Figure7/orientation-tuning-WT-c=1-vs-c=05-SESSIONS.pdf')
print('')
fig4, ax = generate_comparison_figs(SUMMARY, ['WT', 'WT'], [1., 0.5],
                                    average_by='ROIs', norm='norm. ',
                                    colors=['darkorange', 'gold'])
fig4.savefig('../figures/in-vivo/orientation-tuning-WT-c=1-vs-c=05-ROIS.svg')
fig4.savefig('../figures/Figure7/orientation-tuning-WT-c=1-vs-c=05-ROIS.pdf')
print('')

# %%
