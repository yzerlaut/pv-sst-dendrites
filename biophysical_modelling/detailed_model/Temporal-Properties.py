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
        filename='../../data/detailed_model/Basket_simpleStim_sim.zip')

loc = 'soma'

sim.load()
sim.fetch_quantity_on_grid('Vm_%s' % loc, dtype=object) 

p = {}
for k in ['dt', 'nStimRepeat', 'ISI', 't0']:
    p[k] = sim.fetch_quantity_on_grid(k, dtype=float, return_last=True) 

params = dict(iBranch=0)

fig, ax = pt.figure(figsize=(4,2), left=0.2, bottom=0.5)

for NAr, label in zip([0, np.unique(sim.NMDAtoAMPA_ratio)[1], 0],
                      ['without', 'with-NMDA']):
    Vm = sim.get('Vm_%s' % loc, dict(NMDAtoAMPA_ratio=NAr, **params))[0]
    ax.plot(np.arange(len(Vm))*p['dt'], Vm, label=label)

for i in range(int(p['nStimRepeat'])):
    pt.arrow(ax, [p['t0']+i*p['ISI'], 0, 0, -10], head_width=2, head_length=5, width=0.1)

ax.legend()
plt.show()

# %%


