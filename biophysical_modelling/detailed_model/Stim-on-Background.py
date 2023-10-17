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
import numpy as np

from cell_template import BRANCH_COLORS
from parallel import Parallel

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

def load_params_from(sim):

    p = {}
    for k in ['dt', 'nStimRepeat', 'ISI', 't0']:
        p[k] = sim.fetch_quantity_on_grid(k, dtype=float, return_last=True) 
    p['nCluster'] = sim.fetch_quantity_on_grid('nCluster', dtype=object, return_last=True)

    return p

# %%
results = np.load('single_sim.npy', allow_pickle=True).item()

t = np.arange(len(results['Vm_soma']))*results['dt']
fig, ax = pt.figure(figsize=(4,2), left=0.2, bottom=0.5)

for i in range(results['nStimRepeat']):
    pt.arrow(ax, [results['t0']+i*results['ISI'], 0, 0, -10], head_width=2, head_length=5, width=0.1)

# ax.plot(t, results['Vm_dend'], lw=0.5)
ax.plot(t, results['Vm_soma'])

# %%
sim = Parallel(\
        filename='../../data/detailed_model/Basket_StimOnBg_simDemo.zip')
sim.load()

def show_Vm_trace(sim, 
                  loc='soma',
                  iBranch=0, 
                  varied_key = 'with_NMDA',
                  plot = {'with-NMDA':{'varied_key':True,
                                       'color':'tab:orange',
                                       'lw':1.0},
                          'without':{'varied_key':False,
                                    'color':'tab:grey',
                                    'lw':0.5}},
                  zoom=None):
    
    sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 
    p = load_params_from(sim)
    
    params = dict(iBranch=iBranch)

    fig, ax = pt.figure(figsize=(2.5,2), left=0.2, bottom=0.5)

    Vms = {}
    for label in plot:
        params[varied_key] = plot[label]['varied_key']
        Vm = sim.get('Vm_%s' % loc, params)[0]
        t = np.arange(len(Vm))*p['dt']
        if zoom is not None:
            Vm = Vm[(t>zoom[0]) & (t<zoom[1])]
            t = t[(t>zoom[0]) & (t<zoom[1])]
        else:
            zoom=[t[0], t[-1]]
        ax.plot(t, Vm, label=label, color=plot[label]['color'])

    for r in range(int(p['nStimRepeat'])):
        for c, nC in enumerate(p['nCluster']):
            tstart = p['t0']+r*len(p['nCluster'])*p['ISI']+c*p['ISI']
            if (tstart>zoom[0]) and (tstart<zoom[1]):
                pt.arrow(ax, [tstart, 0, 0, -10],
                         head_width=4, head_length=5, width=0.1)
                pt.annotate(ax, 'n$_{syn}$=%i' % nC, (tstart, 5), 
                            rotation=90, xycoords='data', fontsize=6, ha='center')
            
    ax.axis('off')
    ax.legend(loc=(1,0.4), frameon=False)
    pt.draw_bar_scales(ax, Xbar=50, Xbar_label='50ms', Ybar=20, Ybar_label='20mV')
    
t0 = 000
show_Vm_trace(sim, 
              iBranch=1, zoom=[t0,t0+2000],
              varied_key='from_uniform', 
              plot = {'real':{'varied_key':False,
                              'color':'tab:red',
                              'lw':1.0},
                      'uniform':{'varied_key':True,
                                 'color':'tab:grey',
                                 'lw':0.5}})

# %%
sim = Parallel(\
        filename='../../data/detailed_model/Martinotti_StimOnBg_simDemo.zip')
sim.load()
    
show_Vm_trace(sim, iBranch=1, zoom=[t0,t0+2000],
              varied_key = 'with_NMDA',
              plot = {'with-NMDA':{'varied_key':True,
                                       'color':'tab:orange',
                                       'lw':1.0},
                      'without':{'varied_key':False,
                                 'color':'tab:grey',
                                 'lw':0.5}})


# %%
def extract_trials(sim, 
                  loc='soma',
                  varied_key = 'with_NMDA',
                  true_false_labels=['with-NMDA', 'without'],
                  pre=-30, post=150):
    
    sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 
    p = load_params_from(sim)
    
    T = np.arange(int(pre/p['dt']), int(post/p['dt']))*p['dt']
    nBranch = len(np.unique(sim.iBranch))
    nStims = len(p['nCluster'])
    VMs, SPIKEs = {}, {}
    
    for l, label in enumerate(true_false_labels):
        VMs[label] = np.zeros((nBranch, nStims, int(p['nStimRepeat']), len(T)))
        SPIKEs[label] = np.zeros((nBranch, nStims, int(p['nStimRepeat']), len(T)), dtype=int)
        
        for iBranch in np.unique(sim.iBranch):
            
            params = {varied_key:(label==true_false_labels[0]),
                      'iBranch':iBranch}

            Vm = sim.get('Vm_%s' % loc, params)[0]
            
            _, VMs[label][iBranch, :, :, :], SPIKEs[label][iBranch, :, :, :] = \
                    trial_alignement(Vm, p, pre=pre, post=post)
            
    return T, VMs, SPIKEs
    
def trial_alignement(Vm, p, 
                     spike_threshold=-20,
                     pre=-30, post=150):
    
    t = np.arange(len(Vm))*p['dt']
    T = np.arange(int(pre/p['dt']), int(post/p['dt']))*p['dt']
    VMs = np.zeros((len(p['nCluster']), int(p['nStimRepeat']), len(T)))
    SPIKEs = np.zeros((len(p['nCluster']), int(p['nStimRepeat']), len(T)), dtype=int)
    for r in range(int(p['nStimRepeat'])):
        for c, nC in enumerate(p['nCluster']):
            tstart = p['t0']+r*len(p['nCluster'])*p['ISI']+c*p['ISI']
            cond = t>=(tstart+T[0])
            VMs[c,r,:] = Vm[cond][:len(T)]
            # count spikes
            iSpks =np.argwhere((VMs[c,r,:][1:]>=spike_threshold) & (VMs[c,r,:][:-1]<spike_threshold))
            SPIKEs[c,r,1:][iSpks] = 1
        
    return T, VMs, SPIKEs


# %%
T, VMs, SPIKEs = extract_trials(sim,
                                varied_key = 'with_NMDA',
                                true_false_labels=['with-NMDA', 'without'])

# %%

fig, AX = pt.figure(axes=(2,VMs['with-NMDA'].shape[0]), figsize=(2,2))
for iBranch in range(VMs['with-NMDA'].shape[0]):
    for l, label, color in zip(range(2), ['without', 'with-NMDA'], ['tab:grey', 'tab:orange']):
        print(VMs[label].shape)
        for c in range(VMs[label].shape[1]):
            for r in range(VMs[label].shape[2]):
                AX[iBranch][l].plot(T, VMs[label][iBranch,c,r,:], color=color)
                spikes = T[SPIKEs[label][iBranch,c,r,:]==1]
                AX[iBranch][l].plot(spikes, 0*spikes, 'o', ms=4)
    #AX[l].set_ylim([-70,-50])
#pt.set_common_ylims(AX)
#ax.legend(loc=(1,0.4), frameon=False)


# %%
T, VMs, SPIKEs = extract_trials(sim,
                                varied_key = 'from_uniform',
                                true_false_labels=['uniform', 'real'])

# %%
fig, AX = pt.figure(axes=(2,VMs['real'].shape[0]), figsize=(2,2))
for iBranch in range(VMs['real'].shape[0]):
    for l, label, color in zip(range(2), ['uniform', 'real'], ['tab:grey', 'tab:red']):
        for c in range(VMs[label].shape[1]):
            print(c/(VMs[label].shape[1]-1))
            AX[iBranch][l].plot(T, VMs[label][iBranch,c,:,:].mean(axis=0),
                                color=pt.get_linear_colormap('lightgrey', color)(c/(VMs[label].shape[1]-1)))
            #for r in range(VMs[label].shape[2]):
            #    AX[iBranch][l].plot(T, VMs[label][iBranch,c,r,:],
            #                        color=pt.get_linear_colormap('lightgrey', color)(r/(VMs[label].shape[2]-1)))
            #    spikes = T[SPIKEs[label][iBranch,c,r,:]==1]
            #    AX[iBranch][l].plot(spikes, 0*spikes, 'o', ms=4)
        AX[iBranch][l].set_ylim([-70,-30])
#pt.set_common_ylims(AX)
#ax.legend(loc=(1,0.4), frameon=False)


# %%
sim = Parallel(\
        filename='../../data/detailed_model/Basket_simpleStim_sim.zip')

loc = 'soma'

sim.load()
sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 

p = {}
for k in ['dt', 'nStimRepeat', 'ISI', 't0']:
    p[k] = sim.fetch_quantity_on_grid(k, dtype=float, return_last=True) 
p['nCluster'] = sim.fetch_quantity_on_grid('nCluster', dtype=object, return_last=True) 
print(p['nCluster'])

params = dict(iBranch=0)

fig, ax = pt.figure(figsize=(3,2), left=0.2, bottom=0.5)

for l, label in enumerate(['real', 'uniform']):
    Vm = sim.get('Vm_%s' % loc, dict(from_uniform=(label=='uniform'), **params))[0]
    ax.plot(np.arange(len(Vm))*p['dt'], Vm, label=label)

for r in range(int(p['nStimRepeat'])):
    for c, nC in enumerate(p['nCluster']):
        pt.arrow(ax, [p['t0']+r*p['ISI']+c*p['nStimRepeat']*p['ISI'], 0, 0, -10],
                 head_width=2, head_length=5, width=0.1)

ax.legend(loc=(1,0.4), frameon=False)



# %%
def trial_alignement(Vm, p, 
                     pre=-30, post=200):
    
    t = np.arange(len(Vm))*p['dt']
    T = np.arange(int(pre/p['dt']), int(post/p['dt']))*p['dt']
    X = []
    for c, nC in enumerate(p['nCluster']):
        X.append([])
        for r in range(int(p['nStimRepeat'])):
            tstart = p['t0']+r*len(p['nCluster'])*p['ISI']+c*p['ISI']
            cond = t>=(tstart+T[0])
            X[c].append(Vm[cond][:len(T)])
        
    return T, np.array(X)


fig, AX = pt.figure(axes=(2,1), figsize=(2,2))
for l, label, color in zip(range(2), ['real', 'uniform'], ['tab:grey', 'tab:orange']):
    Vm = sim.get('Vm_%s' % loc, dict(from_uniform=(label=='uniform'), **params))[0]
    T, X = trial_alignement(Vm, p)
    for c, nC in enumerate(p['nCluster']):
        for r, x in enumerate(X[c]):
            AX[l].plot(T, x, color=plt.cm.tab10(c))
        #AX[l].plot(T, X[c].mean(axis=0), color=color)

pt.set_common_ylims(AX)
#ax.legend(loc=(1,0.4), frameon=False)

# %% [markdown]
# # Calibration of the Background Activity Setting
#
# Generate the simulation data with:
#     
# ```
# python stim_on_background.py --background_calib
# ```

# %%
# load data

sim = Parallel(\
        filename='../../data/detailed_model/StimOnBg_BgCalib.zip')

loc = 'soma'

sim.load()
sim.fetch_quantity_on_grid('Vm_soma', dtype=object) 
sim.fetch_quantity_on_grid('output_rate', dtype=object) 

# %%
sim.keys


# %%
def show_Vm_trace(sim, 
                  cellType='Basket', iBranch=0, ibgStimFreq=0, ibgFreqInhFactor=0,
                  loc='soma',
                  color='k',
                  zoom=None):
    
    sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 
    p = load_params_from(sim)
    
    params = dict(iBranch=iBranch, cellType=cellType,
                  bgStimFreq=np.unique(sim.bgStimFreq)[ibgStimFreq],
                  bgFreqInhFactor=np.unique(sim.bgFreqInhFactor)[ibgFreqInhFactor])

    fig, ax = pt.figure(figsize=(2.5,2), left=0.2, bottom=0.5)

    Vm = sim.get('Vm_%s' % loc, params)[0]
    t = np.arange(len(Vm))*p['dt']
    if zoom is not None:
        Vm = Vm[(t>zoom[0]) & (t<zoom[1])]
        t = t[(t>zoom[0]) & (t<zoom[1])]
    else:
        zoom=[t[0], t[-1]]
    ax.plot(t, Vm, color='k', lw=1)

    for r in range(int(p['nStimRepeat'])):
        for c, nC in enumerate(p['nCluster']):
            tstart = p['t0']+r*len(p['nCluster'])*p['ISI']+c*p['ISI']
            if (tstart>zoom[0]) and (tstart<zoom[1]):
                pt.arrow(ax, [tstart, 0, 0, -10],
                         head_width=4, head_length=5, width=0.1)
                pt.annotate(ax, 'n$_{syn}$=%i' % nC, (tstart, 5), 
                            rotation=90, xycoords='data', fontsize=6, ha='center')
            
    ax.axis('off')
    ax.legend(loc=(1,0.4), frameon=False)
    pt.draw_bar_scales(ax, Xbar=50, Xbar_label='50ms', Ybar=20, Ybar_label='20mV')
    
show_Vm_trace(sim, iBranch=0)

# %%
show_Vm_trace(sim, cellType='Basket', iBranch=0, ibgStimFreq=0, ibgFreqInhFactor=0)

# %%
fig = show_Vm_trace(sim, cellType='Basket', iBranch=1, ibgStimFreq=1, ibgFreqInhFactor=0, zoom=[400,1600])
plt.savefig('fig.svg')

# %%
