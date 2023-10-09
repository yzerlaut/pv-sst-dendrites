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
        filename='../../data/detailed_model/Basket_clusterStim_sim.zip')

sim.load()
sim.fetch_quantity_on_grid('peak_efficacy_soma', dtype=float)

COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', 'uniform']
fig, ax = pt.figure(figsize=(1.2, 1.))
for f, fU in enumerate([False, True]):
    for iDistance in np.unique(sim.iDistance):
        params = {'iDistance':iDistance, 'from_uniform':fU}
        #pt.scatter([iDistance+0.4*f], [sim.get('peak_efficacy_soma', params).mean()],
        #            sy=[sim.get('peak_efficacy_soma', params).std()], color=COLORS[f], ax=ax, ms=2, lw=2)
        ax.bar([iDistance+0.4*f], [sim.get('peak_efficacy_soma', params).mean()],
               yerr=[sim.get('peak_efficacy_soma', params).std()], color=COLORS[f], width=0.35)
    pt.annotate(ax, f*'\n'+LABELS[f], (1,1), va='top', color=COLORS[f])
pt.set_plot(ax, ylabel='efficacy (%)',
            xticks=0.2+np.arange(3), xticks_labels=['prox.', 'mid.', 'dist.'], xticks_rotation=30)

# %%
loc, based_on = 'soma', 'peak'

sim.fetch_quantity_on_grid('linear_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('real_%s' % loc, dtype=list)
sim.fetch_quantity_on_grid('%s_efficacy_%s' % (based_on, loc), dtype=float)
sim.fetch_quantity_on_grid('synapses', dtype=list)
dt = sim.fetch_quantity_on_grid('dt', return_last=True)

fig, AX = pt.figure(axes=(len(np.unique(sim.iDistance)), len(np.unique(sim.iBranch))))

CONDS = ['proximal', 'interm.', 'distal']
COLORS, LABELS = ['tab:red', 'tab:grey'], ['real', '\nuniform']
for label, color in zip(LABELS, COLORS):                
    pt.annotate(AX[0][-1], label, (1.5,0), va='top', color=color)
    
for iDistance in np.unique(sim.iDistance):
    for iBranch in np.unique(sim.iBranch):
        params = {'iDistance':iDistance, 'iBranch':iBranch}
        for f, fU in enumerate([False, True]):
            real = sim.get('real_%s' % loc, dict(from_uniform=fU, **params))[0]
            linear = sim.get('linear_%s' % loc, dict(from_uniform=fU, **params))[0]
            t = np.arange(len(real))*dt
            AX[iBranch][iDistance].plot(t, real, '-', color=COLORS[f])
            AX[iBranch][iDistance].plot(t, linear, ':', color=COLORS[f], lw=0.5)
            n = len(sim.get('synapses', dict(from_uniform=fU, **params))[0])
            pt.annotate(AX[iBranch][iDistance], ('\n' if f==1 else'')+\
                        '$\epsilon$=%.1f%% (n=%i)' % (sim.get('%s_efficacy_%s' % (based_on, loc),
                                                      dict(from_uniform=fU, **params))[0], n),
                        (1.2,1), va='top', ha='right', color=COLORS[f], fontsize=6)
            
        if iDistance==0:
            pt.annotate(AX[iBranch][iDistance], 'branch #%i' % (iBranch+1),
                        (-1.3,0.3), color='k')
    
        pt.set_plot(AX[iBranch][iDistance], [], xlim=[0,80])
        pt.draw_bar_scales(AX[iBranch][iDistance], Xbar=10, Ybar=2, Xbar_label='10ms', Ybar_label='2mV')
    AX[0][iDistance].set_title(CONDS[iDistance]+'\n')
            

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
            

# %%

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
