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

# %% [markdown]
# # Simulation of Clustered-Input on Dendritic Segments 

# %%
from cell_template import Cell
from clustered_input_stim import * 

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

# %%
from parallel import Parallel

sim = Parallel(\
        filename='../../data/detailed_model/Basket_clusterStim_sims5S6.zip')

based_on = 'peak_efficacy_soma'
sim.load()
sim.fetch_quantity_on_grid(based_on, dtype=float)

COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
fig, ax = pt.figure(figsize=(1., 1.))
for f, fU in enumerate([False, True]):
    for iDistance in np.unique(sim.iDistance):
        params = {'iDistance':iDistance, 'from_uniform':fU}
        ax.bar([iDistance+0.4*f], [100-np.nanmean(sim.get(based_on, params))],
               yerr=[np.nanstd(sim.get(based_on, params))], color=COLORS[f], width=0.35)
    pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
pt.set_plot(ax, ylabel='suppr. (%)',# ylabel='efficacy (%)',
            #xticks=0.2+np.arange(3), xticks_labels=['prox.', 'med.', 'dist.'],
            xticks=0.2+np.arange(2), xticks_labels=['prox.', 'dist.'],
            xticks_rotation=30)

# %%
based_on = 'integral_efficacy_soma'
for i in np.arange(2,7):
    sim = Parallel(\
            filename='../../data/detailed_model/Basket_clusterStim_sims%i.zip' % i)
    sim.load()
    sim.fetch_quantity_on_grid(based_on, dtype=float)

    COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
    fig, ax = pt.figure(figsize=(1.2, 1.))
    fig.suptitle('sparsening %i%%' % i)
    for f, fU in enumerate([False, True]):
        for iDistance in np.unique(sim.iDistance):
            params = {'iDistance':iDistance, 'from_uniform':fU}
            ax.bar([iDistance+0.4*f], [100-np.mean(sim.get(based_on, params))],
                   yerr=[np.nanstd(sim.get(based_on, params))], color=COLORS[f], width=0.35)
        pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
    pt.set_plot(ax, ylabel='suppr. (%)',# ylabel='efficacy (%)',
                xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
for i in np.arange(2,6):
    sim = Parallel(\
            filename='../../data/detailed_model/Basket_clusterStim_sims%iS2.zip' % i)
    sim.load()
    sim.fetch_quantity_on_grid('peak_efficacy_soma', dtype=float)

    COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
    fig, ax = pt.figure(figsize=(1.2, 1.))
    fig.suptitle('sparsening %i%%' % i)
    for f, fU in enumerate([False, True]):
        for iDistance in np.unique(sim.iDistance):
            params = {'iDistance':iDistance, 'from_uniform':fU}
            ax.bar([iDistance+0.4*f], [100-np.nanmean(sim.get('peak_efficacy_soma', params))],
                   yerr=[np.nanstd(sim.get('peak_efficacy_soma', params))], color=COLORS[f], width=0.35)
        pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
    pt.set_plot(ax, ylabel='suppr. (%)',# ylabel='efficacy (%)',
                xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
for i in np.arange(2,8):
    sim = Parallel(\
            filename='../../data/detailed_model/Basket_clusterStim_sims%iS3.zip' % i)
    sim.load()
    sim.fetch_quantity_on_grid('peak_efficacy_soma', dtype=float)

    COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
    fig, ax = pt.figure(figsize=(1.2, 1.))
    fig.suptitle('sparsening %i%%' % i)
    for f, fU in enumerate([False, True]):
        for iDistance in np.unique(sim.iDistance):
            params = {'iDistance':iDistance, 'from_uniform':fU}
            ax.bar([iDistance+0.4*f], [100-np.nanmean(sim.get('peak_efficacy_soma', params))],
                   yerr=[np.nanstd(sim.get('peak_efficacy_soma', params))], color=COLORS[f], width=0.35)
        pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
    pt.set_plot(ax, ylabel='suppr. (%)',# ylabel='efficacy (%)',
                xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
for i in np.arange(3,6):
    sim = Parallel(\
            filename='../../data/detailed_model/Basket_clusterStim_sims%iS1.zip' % i)
    sim.load()
    sim.fetch_quantity_on_grid('peak_efficacy_soma', dtype=float)

    COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
    fig, ax = pt.figure(figsize=(1.2, 1.))
    fig.suptitle('sparsening %i%%' % i)
    for f, fU in enumerate([False, True]):
        for iDistance in np.unique(sim.iDistance):
            params = {'iDistance':iDistance, 'from_uniform':fU}
            ax.bar([iDistance+0.4*f], [100-np.nanmean(sim.get('peak_efficacy_soma', params))],
                   yerr=[np.nanstd(sim.get('peak_efficacy_soma', params))], color=COLORS[f], width=0.35)
        pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
    pt.set_plot(ax, ylabel='suppr. (%)',# ylabel='efficacy (%)',
                xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
loc, based_on = 'soma', 'peak'

sim.fetch_quantity_on_grid('linear_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('real_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('%s_efficacy_%s' % (based_on, loc), dtype=float)
sim.fetch_quantity_on_grid('synapses', dtype=list)
dt = sim.fetch_quantity_on_grid('dt', return_last=True)

fig, AX = pt.figure(axes=(2*len(np.unique(sim.iDistance)),
                          len(np.unique(sim.iBranch))))

CONDS = ['proximal', 'distal']
COLORS, LABELS = ['tab:red', 'tab:grey'], [' real', ' uniform']
    
for iDistance in np.unique(sim.iDistance):
    for iBranch in np.unique(sim.iBranch):
        params = {'iDistance':iDistance, 'iBranch':iBranch}
        for f, fU in enumerate([False, True]):
            real = sim.get('real_%s' % loc, dict(from_uniform=fU, **params))[0]
            linear = sim.get('linear_%s' % loc, dict(from_uniform=fU, **params))[0]
            t = np.arange(len(real))*dt
            AX[iBranch][2*iDistance+f].plot(t, real, '-', color=COLORS[f])
            AX[iBranch][2*iDistance+f].plot(t, linear, ':', color=COLORS[f], lw=0.5)
            n = len(sim.get('synapses', dict(from_uniform=fU, **params))[0])
            pt.annotate(AX[iBranch][2*iDistance+f], 
                        '$\epsilon$=%.1f%%\n(n=%i)' % (sim.get('%s_efficacy_%s' % (based_on, loc),
                                                      dict(from_uniform=fU, **params))[0], n),
                        (0.5,0), va='top', ha='center', color=COLORS[f], fontsize=6)
            
            pt.set_plot(AX[iBranch][2*iDistance+f], [], xlim=[0,50])
            pt.draw_bar_scales(AX[iBranch][2*iDistance+f], loc='top-right',
                               Xbar=10, Xbar_label='10ms',
                               Ybar=2 if loc=='soma' else 10,
                               Ybar_label='2mV' if loc=='soma' else '10mV')
            if iBranch==0:
                pt.annotate(AX[0][2*iDistance+f], CONDS[iDistance], (0.5,1.2), ha='right')
                pt.annotate(AX[0][2*iDistance+f], LABELS[f], (0.5,1.2), color=COLORS[f])
    
        if iDistance==0:
            pt.annotate(AX[iBranch][iDistance], 'branch #%i' % (iBranch+1),
                        (-1.3,0.3), color=BRANCH_COLORS[iBranch])
    



# %%
from parallel import Parallel

sim = Parallel(\
        filename='../../data/detailed_model/Martinotti_clusterStim_sim.zip')

based_on = 'integral'
sim.load()
sim.fetch_quantity_on_grid('%s_efficacy_soma' % based_on, dtype=float)

COLORS, LABELS = ['tab:orange', 'tab:grey'], ['with-NMDA', 'without']
fig, ax = pt.figure(figsize=(1.2, 1.))
for f, NAr in enumerate([0, np.unique(sim.NMDAtoAMPA_ratio)[1]]):
    for iDistance in np.unique(sim.iDistance):
        params = {'iDistance':iDistance, 'NMDAtoAMPA_ratio':NAr}
        print(NAr, sim.get('%s_efficacy_soma' % based_on, params))
        ax.bar([iDistance+0.4*f], [sim.get('%s_efficacy_soma' % based_on, params).mean()], bottom=50,
               yerr=[sim.get('%s_efficacy_soma' % based_on, params).std()], color=COLORS[f], width=0.35)
    pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
pt.set_plot(ax, ylabel='efficacy (%)',
            xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
loc, based_on = 'soma', 'peak'

sim.fetch_quantity_on_grid('linear_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('real_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('%s_efficacy_%s' % (based_on, loc), dtype=float)
dt = sim.fetch_quantity_on_grid('dt', return_last=True)

fig, AX = pt.figure(axes=(len(np.unique(sim.iDistance)), len(np.unique(sim.iBranch))))

COLORS, LABELS = ['tab:orange', 'tab:grey'], ['with-NMDA', '\nwithout']
for label, color in zip(LABELS, COLORS):                
    pt.annotate(AX[0][-1], label, (1.5,0), va='top', color=color)
    
for iDistance in np.unique(sim.iDistance):
    for iBranch in np.unique(sim.iBranch):
        params = {'iDistance':iDistance, 'iBranch':iBranch}
        for NAr, color in zip([np.unique(sim.NMDAtoAMPA_ratio)[1], 0], COLORS):
            real = sim.get('real_%s' % loc, dict(NMDAtoAMPA_ratio=NAr, **params))[0]
            linear = sim.get('linear_%s' % loc, dict(NMDAtoAMPA_ratio=NAr, **params))[0]
            t = np.arange(len(real))*dt
            AX[iBranch][iDistance].plot(t, real, '-', color=color)
            AX[iBranch][iDistance].plot(t, linear, ':', color=color, lw=0.5)
            pt.annotate(AX[iBranch][iDistance], ('\n' if NAr==0 else'')+\
                        '$\epsilon$=%.1f%%' % sim.get('%s_efficacy_%s' % (based_on, loc),
                                                      dict(NMDAtoAMPA_ratio=NAr, **params))[0],
                        (1,1), va='top', ha='right', color=COLORS[0 if NAr>0 else 1])
            
        if iDistance==0:
            pt.annotate(AX[iBranch][iDistance], 'branch #%i' % (iBranch+1),
                        (-1.3,0.3), color='k')


# %% [markdown]
# # Visualizing the Distant-Dependent Clusters

# %%
# load cell
ID = '864691135396580129_296758' # Basket Cell example
cell = Cell(ID=ID, params_key='BC')

# %%
distance_intervals  = [[20,60], 
                       [160,200]]

BRANCH_COLORS = [plt.cm.tab10(i) for i in [9,6,0,4,2,8]]
def show_cluster(iDistance, label):
    
    props ={'iDistance':iDistance, # 2 -> means "distal" range
            'distance_intervals':distance_intervals,
            'synSubsamplingFraction':5/100.}

    fig, AX = pt.figure(figsize=(1.7,2.5), axes=(2, 6), hspace=0.2, wspace=0., left=0.3)
    
    for iBranch in range(6):
        c, INSETS = BRANCH_COLORS[iBranch], []
        pt.annotate(AX[iBranch][0], 'branch #%i' % (1+iBranch), (-1, 0.5), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[iBranch][0], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[iBranch][1])
        INSETS.append(inset)
        pt.annotate(AX[iBranch][0], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[iBranch][1], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('**%s cluster**,  interval: %s um' % (label, str(distance_intervals[iDistance])))
    return fig


# %%
fig = show_cluster(0, 'proximal')

# %%
fig = show_cluster(1, 'distal')

# %%
