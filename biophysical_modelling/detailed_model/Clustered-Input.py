# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
        filename='../../data/detailed_model/Basket_clusterStim_sim.zip')

sim.load()
sim.fetch_quantity_on_grid('peak_efficacy_soma', dtype=float)

COLORS = ['tab:red', 'tab:grey']
fig, ax = pt.figure(figsize=(1.2, 1.))
for f, fU in enumerate([False, True]):
    for iDistance in np.unique(sim.iDistance):
        print(sim.get('peak_efficacy_soma', params).std())
        params = {'iDistance':iDistance, 'from_uniform':fU}
        ax.bar([iDistance+0.4*f], [sim.get('peak_efficacy_soma', params).mean()],
               yerr=[sim.get('peak_efficacy_soma', params).std()], color=COLORS[f], width=0.35)
pt.set_plot(ax, ylabel='efficacy (%)',
            xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
# load cell
ID = '864691135396580129_296758' # Basket Cell example
cell = Cell(ID=ID, params_key='BC')

# %% [markdown]
# ## Distant-Dependent Clusters

# %%
for label, iDistance in zip(['proximal','interm.','distal'], range(3)):

    props ={'iDistance':iDistance, # 2 -> means "distal" range
            'synSubsamplingFraction':5/100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(3, 2), hspace=0, wspace=0.1)
    
    for iBranch in range(3):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('%s cluster,  interval: %s um' % (label, str(distance_intervals[iDistance])))
    
#fig.savefig('/tmp/1.svg')

# %%
for seed in [2, 4]:

    props ={'iDistance':2, # 2 -> means "distal" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)

# %% [markdown]
# ## Proximal Clusters

# %%
for seed in [2, 4]:

    props ={'iDistance':0, # 0 -> means "proximal" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)

# %% [markdown]
# ## Intermediate Clusters

# %%
for seed in [2, 4]:

    props ={'iDistance':1, # 1 -> means "intermediate" range
            'subsampling_fraction':4./100.}

    fig, AX = pt.figure(figsize=(1.5,2.2), axes=(6, 2), hspace=0, wspace=0.1)
    for iBranch in range(6):
        c, INSETS = plt.cm.tab10(iBranch), []
        AX[0][iBranch].set_title('branch #%i' % (1+iBranch), color=c)
        _, inset = find_clustered_input(cell, iBranch, **props,
                            with_plot=True, ax=AX[0][iBranch], syn_color=c)
        INSETS.append(inset)
        _, inset = find_clustered_input(cell, iBranch, from_uniform=True, **props,
                            with_plot=True, syn_color=c, ax=AX[1][iBranch])
        INSETS.append(inset)
        pt.annotate(AX[0][iBranch], 'real', (-0.3,0.3), bold=True, color=c)
        pt.annotate(AX[1][iBranch], 'uniform', (-0.3,0.3), bold=True, color=c)

        pt.set_common_ylims(INSETS)
    fig.suptitle('sparsening seed #%i' % seed)


# %% [markdown]
# ## Compute Suppression

# %%
results = np.load('single_sim.npy', allow_pickle=True).item()


# %%
def efficacy(real, linear,
                based_on='integral'):
    if based_on=='peak':
        return 100.*np.max(real-real[0])/np.max(linear-linear[0])
    elif based_on=='integral':
        return 100.*np.sum(real-real[0])/np.sum(linear-linear[0])

def show(at='soma', color='k', ax=None,
         xlim=[0,50], Ybar=2, Xbar=5):
    if ax is None:
        fig, ax = pt.figure()
    ax.plot(np.arange(len(results['real_%s'%at]))*results['dt'], results['linear_%s'%at], ':', lw=0.5, color=color)
    ax.plot(np.arange(len(results['real_%s'%at]))*results['dt'], results['real_%s'%at], color=color)
    pt.set_plot(ax, [], xlim=xlim)
    pt.draw_bar_scales(ax,
                       Xbar=Xbar, Xbar_label='%ims'%Xbar,
                       Ybar=Ybar, Ybar_label='%smV'%Ybar,
                       loc='top-right')
    return ax

fig, AX = pt.figure(axes=(2,1))
show(at='soma', ax=AX[0])
AX[0].set_title('$\epsilon$=%.1f%%' % results['integral_efficacy_soma'])
show(at='dend', ax=AX[1], Ybar=10)

# %%
plt.plot(results['linear_soma']-results['linear_soma'][0])
plt.plot(results['real_soma']-results['real_soma'][0])

# %%
