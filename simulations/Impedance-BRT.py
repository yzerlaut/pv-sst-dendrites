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
# # Input Impedance Properties of the Model
#     > Characterizing the parameter dependency of the input impedance spatial profile 
#

# %%
from single_cell_integration import * # code to run the model: (see content below)

import sys
sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt

# %% [markdown]
# ### Locations where to simulate/record along the dendritic tree

# %%
from nrn.plot import nrnvyz
Model = load_params('BRT-parameters.json')
BRT, neuron = initialize(Model)
SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

vis = nrnvyz(SEGMENTS)
n, N = Model['nseg_per_branch'], Model['branch-number']
#BRANCH_LOCS = np.concatenate([np.arange(n+1),
#                              1+20*N+np.arange(3*n)])
BRANCH_LOCS = np.arange(n*N+1)
fig, ax = pt.plt.subplots(1, figsize=(2,2))
vis.plot_segments(ax=ax, color='tab:grey')
vis.add_dots(ax, BRANCH_LOCS, 2)
ax.set_title('n=%i segments' % len(BRANCH_LOCS), fontsize=6)
Model['BRANCH_LOCS'] = BRANCH_LOCS # set for later !!


# %%

# %% [markdown]
# ### Run Input Impedance Profile Characterization

# %%
# function to test the spatial-dep
def run_imped_charact(Model,
                      pulse={'amp':1,
                             'duration':200}):
    """
    loop over dendritic locations on the morphology
    
    current in pA, durations in ms
    """

    #print(' ', nrn.collect().pop())

    output = {'loc':[],
              'input_resistance':[],
              'transfer_resistance_to_soma':[]}

    net, BRT, neuron = initialize(Model, with_network=True)
   
    for b in np.arange(Model['nseg_per_branch']*Model['branch-number']+1):
        
        net.restore('start')
        
        # recording
        Ms = nrn.StateMonitor(neuron, ('v'), record=[0, b]) # soma + dend loc
        net.add(Ms)
        
        # run a pre-period for relaxation
        net.run(100*nrn.ms)
        
        # turn on the current pulse
        neuron.I[b] = pulse['amp']*nrn.pA
        # run until relaxation (long pulse)
        net.run(pulse['duration']*nrn.ms)
        # turn off the current pulse
        neuron.I[b] = 0*nrn.pA

        # measure all quantities
        output['input_resistance'].append(\
                1e6*(neuron.v[b]-Model['EL']*nrn.mV)/nrn.volt/pulse['amp']) # 1e6*V/pA = MOhm
        output['transfer_resistance_to_soma'].append(\
                1e6*(neuron.v[0]-Model['EL']*nrn.mV)/nrn.volt/pulse['amp']) # 1e6*V/pA = MOhm
        output['loc'].append(b/len(Model['BRANCH_LOCS'])*Model['tree-length'])

        net.remove(Ms)
        Ms = None
        
    net.remove(neuron)
    net, BRT, neuron = None, None, None
    return output

results = run_imped_charact(Model)
#results = run_imped_charact(Model)

# %%
import copy

def run_params_scan(key, values):
    Model = load_params('BRT-parameters.json')
    n, N = Model['nseg_per_branch'], Model['branch-number']
    Model['BRANCH_LOCS'] = np.concatenate([np.arange(n+1),1+20*N+np.arange(3*n)])
    RESULTS = []
    for i, value in enumerate(values):
        cModel = copy.deepcopy(Model)
        cModel[key] = value
        RESULTS.append(run_imped_charact(cModel))
    np.save('../data/%s-impact.npy' % key,
            {key:values, 'results':RESULTS})

def plot_parameter_variation(key,
                             title='title', 
                             label='label',
                             yscale='log'):

    data = np.load('../data/%s-impact.npy' % key, allow_pickle=True).item()

    fig, AX = plt.subplots(1, 2, figsize=(4, 1.3))
    plt.subplots_adjust(wspace=0.6, right=0.8, left=0.15)

    AX[0].annotate(title, (-0.7, 0.5), rotation=90, xycoords='axes fraction', va='center')
    AX[0].set_title('input resistance')
    AX[1].set_title('transfer res. to soma')

    for i, results in enumerate(data['results']):
        color = plt.cm.viridis_r(i/(len(data[key])-1))
        AX[0].plot(results['loc'], results['input_resistance'], color=color, lw=1.5)
        AX[1].plot(results['loc'], results['transfer_resistance_to_soma'], color=color, lw=1.5)

    for ax in AX:
        pt.set_plot(ax, xlabel='dist. from soma ($\mu$m)', ylabel='M$\Omega$', yscale=yscale)

    inset = pt.inset(AX[1], (1.4, 0.0, 0.1, 1.0))
    pt.bar_legend(fig, X=range(len(data[key])+1),
                  ticks = np.arange(len(data[key]))+0.5,
                  ticks_labels = [str(k) for k in data[key]],
                  colormap=plt.cm.viridis_r, ax_colorbar=inset,
                  label=label)

    return fig


# %% [markdown]
# ## Impact of Branching+Tapering

# %%
run_params_scan('branch-number', [1,2,3,4,5])

# %%
fig = plot_parameter_variation('branch-number',
                               title='Branching+Tapering',
                               label='branch\nnumber')


# %%
#run_params_scan('branch-number', [1,2,3,4])

# %%
# figure for paper
key='branch-number'
data = np.load('../data/%s-impact.npy' % key, allow_pickle=True).item()

fig, AX = pt.plt.subplots(2, figsize=(1., 1.5))
pt.plt.subplots_adjust(hspace=0.1, right=0.8, left=0.15)

for i, results in enumerate(data['results']):
    color = pt.viridis_r(i/(len(data[key])-1))
    AX[0].plot(results['loc'], results['input_resistance'], color=color, lw=1.5)
    AX[1].plot(results['loc'], results['transfer_resistance_to_soma'], color=color, lw=1.5)

pt.set_plot(AX[0], xticks=[0,200,400], yscale='log', ylim=[90, 9700], xticks_labels=[])
pt.set_plot(AX[1], xticks=[0,200,400], xlabel='dist. from soma ($\mu$m)', yscale='log', ylim=[5, 300])

inset = pt.inset(AX[1], (1.4, 0.5, 0.1, 1.5))
pt.bar_legend(fig, X=range(len(data[key])+1),
              ticks = np.arange(len(data[key]))+0.5,
              ticks_labels = [str(k) for k in data[key]],
              colormap=pt.viridis_r, ax_colorbar=inset,
              label='branch number')
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %%
BRANCHES = [1,2,3,4]
fig, AX = plt.subplots(1, len(BRANCHES), figsize=(5, 1.5))

for i, N in enumerate(BRANCHES):
    BRT = nrn.morphologies.BallandRallsTree.build_morpho(\
                                    Nbranch=N,
                                    branch_length=1.0*Model['tree-length']/N,
                                    soma_radius=Model['soma-radius'],
                                    root_diameter=Model['root-diameter'],
                                    Nperbranch=Model['nseg_per_branch'])

    SEGMENTS = nrn.morpho_analysis.compute_segments(BRT)

    vis = nrnvyz(SEGMENTS)
    BRANCH_LOCS = np.arange(Model['nseg_per_branch']*Model['branch-number']+1)
    vis.plot_segments(ax=AX[i], color=plt.cm.viridis_r(i/4),
                      bar_scale_args={'Ybar':100,'Ybar_label':'100$\\mu$m ','Xbar': 1e-10} if i==0 else\
                             {'Ybar':1e-10,'Xbar': 1e-10})
    AX[i].set_title('$N_B$=%i' % N, fontsize=7)
pt.set_common_xlims(AX)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# ## Impact of Tapering (diameter reduction rule)

# %%
run_params_scan('diameter-reduction-factor', [0.3, 0.5, 0.7, 0.9])

# %%
fig = plot_parameter_variation('diameter-reduction-factor',
                               title='Tapering',
                               label='tapering factor')

# %% [markdown]
# ## Impact of Tree Length

# %%
run_params_scan('tree-length', [100,200,400,600])

# %%
fig = plot_parameter_variation('tree-length',
                               title='Tree Length',
                               label='full length\n($\\mu$m)')


# %% [markdown]
# ## Impact of Intracellular Resistance

# %%
run_params_scan('Ri', [25, 50, 100, 150, 200])

# %%
fig = plot_parameter_variation('Ri',
                               title='   Intracellular Resistivity',
                               label='$R_i$\n($\Omega$.cm)')


# %% [markdown]
# ## Impact of Transmembrane Resistance

# %%
run_params_scan('gL', [1, 2.5, 5, 10, 25])

# %%
fig = plot_parameter_variation('gL',
                               title='Membrane Conductance',
                               label='$g_L$\n(pS/$\\mu$m$^2$)')


# %% [markdown]
# ## Impact of Root Diameter

# %%
run_params_scan('root-diameter', [0.6, 0.8, 1, 1.5, 2])

# %%
fig = plot_parameter_variation('root-diameter',
                               title='Tree Root Diameter',
                               label='root diam.\n($\\mu$m)')


# %% [markdown]
# ## Impact of Soma Size

# %%
run_params_scan('soma-radius', [2, 5, 10, 20, 50])

# %%
fig = plot_parameter_variation('soma-radius',
                               title='Soma Size',
                               label='soma radius ($\\mu$m)')


# %% [markdown]
# ## Summary Fig

# %%
KEYS = [\
    'branch-number',
    'tree-length',
    'Ri',
    'gL',
    'root-diameter',
    'soma-radius']

TITLES = [\
    'Branching\n+Tapering',
    'Tree\nLength',
    'Intracellular\n Resistivity',
    'Membrane\nConductance',
    'Root\nDiameter',
    'Soma Size']

LABELS=[\
    'branch\nnumber',
    'full length\n($\\mu$m)',
    '$R_i$\n($\Omega$.cm)',
    '$g_L$\n(pS/$\\mu$m$^2$)',
    'root diam.\n($\\mu$m)',
    'soma radius ($\\mu$m)']

N = len(KEYS)
fig, AXS = plt.subplots(N, 2, figsize=(3.5, 1.4*N))
plt.subplots_adjust(wspace=0.6, hspace=0.5, right=0.8, left=0.2)

AXS[0][0].set_title('input resistance')
AXS[0][1].set_title('transfer resistance')

for key, title, label, AX in zip(KEYS, TITLES, LABELS, AXS):

    data = np.load('../data/%s-impact.npy' % key, allow_pickle=True).item()

    AX[0].annotate(title, (-0.9, 0.5), rotation=90, xycoords='axes fraction',
                   va='center', ha='center')


    for i, results in enumerate(data['results']):
        color = plt.cm.viridis(i/(len(data[key])-1))
        AX[0].plot(results['loc'], results['input_resistance'], color=color, lw=1.5)
        AX[1].plot(results['loc'], results['transfer_resistance_to_soma'], color=color, lw=1.5)

    inset = pt.inset(AX[1], (1.4, 0.0, 0.1, 1.0))
    pt.bar_legend(fig, X=range(len(data[key])+1),
                  ticks = np.arange(len(data[key]))+0.5,
                  ticks_labels = [str(k) for k in data[key]],
                  colormap=plt.cm.viridis, ax_colorbar=inset,
                  label=label)
    for ax in AX:
        ax.set_ylabel('M$\Omega$')
        ax.set_yscale('log')
        ax.set_xticks([0,200,400])
        if key==KEYS[-1]:
            ax.set_xlabel('dist. to soma ($\mu$m)')

#fig.savefig('../figures/Ball-and-Rall-Tree-parameters.svg')


# %%
