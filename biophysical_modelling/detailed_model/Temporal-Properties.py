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


# %%
# results = np.load('single_sim.npy', allow_pickle=True).item()

# t = np.arange(len(results['Vm_soma']))*results['dt']
# fig, ax = pt.figure(figsize=(4,2), left=0.2, bottom=0.5)

# for i in range(results['nStimRepeat']):
    # pt.arrow(ax, [results['t0']+i*results['ISI'], 0, 0, -10], head_width=2, head_length=5, width=0.1)

# ax.plot(t, results['Vm_dend'], lw=0.5)
# ax.plot(t, results['Vm_soma'])

# plt.show()


# %%
sim = Parallel(\
        filename='../../data/detailed_model/Martinotti_simpleStim_sim.zip')

loc = 'soma'

sim.load()
sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 

p = {}
for k in ['dt', 'nStimRepeat', 'ISI', 't0']:
    p[k] = sim.fetch_quantity_on_grid(k, dtype=float, return_last=True) 
p['nCluster'] = sim.fetch_quantity_on_grid('nCluster', dtype=object, return_last=True) 
print(p['nCluster'])

params = dict(iBranch=0)

fig, ax = pt.figure(figsize=(3,1), left=0.2, bottom=0.5)

for l, label in enumerate(['without', 'with-NMDA']):
    Vm = sim.get('Vm_%s' % loc, dict(with_NMDA=(label=='with-NMDA'), **params))[0]
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
            tstart = p['t0']+r*p['ISI']+c*p['nStimRepeat']*p['ISI']
            cond = t>=(tstart+T[0])
            X[c].append(Vm[cond][:len(T)])
        
    return T, np.array(X)


fig, AX = pt.figure(axes=(2,1), figsize=(2,2))
for l, label, color in zip(range(2), ['without', 'with-NMDA'], ['tab:grey', 'tab:orange']):
    Vm = sim.get('Vm_%s' % loc, dict(with_NMDA=(label=='with-NMDA'), **params))[0]
    T, X = trial_alignement(Vm, p)
    for c, nC in enumerate(p['nCluster']):
        AX[l].plot(T, X[c].mean(axis=0), color=color)

pt.set_common_ylims(AX)
#ax.legend(loc=(1,0.4), frameon=False)

# %%


