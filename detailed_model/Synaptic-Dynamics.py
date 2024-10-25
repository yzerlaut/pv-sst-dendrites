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

# %% [markdown]
# # Modelling the Dynamics of Glutamatergic Synapses (short-term plasticity)
#
# We derive a very simplified model of synaptic dynamics that can capture synaptic depression and synaptic facilitation with stochastic release
#
# #### 1) A time varying dynamics of release probability
#
# We use a very simplified kinetic model where the release probability $p$ follows a first order dynamics given by:
#
# \begin{equation}
# \frac{dp}{dt} = \frac{p_0-p(t)}{\tau_p} - \sum_{spike} \Delta p \cdot \big( p_1 - p(t) \big) \cdot \delta(t-t_{spike})
# \end{equation}
#
# where:
# - $p_0$ is the probability at zero frequency stimulation (infinitely spaced events)
# - $p_1$ is the probability at infinite frequency stimulation (null spaced events)
# - $\Delta p$ is the probability increment
#       - positive for facilitation
#       - negative for depression
#       - N.B. with contraint: $| \Delta p| \leq |p_1 - p_0| $
# - $\tau_p$ is the recovery time constant to go back to $p_0$ level
#
# #### 2) A stochastic release scheme
#
# For a spike event at time $t$, we draw a random number $r$ (from a uniform distribution between 0 and 1) and we release an event if: $r < p(t) $
#
# ### Implementation in an event-based simulation scheme
#
#
#
# ```
# last_spike_time = -np.inf # far in the past
# last_p = p0
#
# def net_receive(new_spike_time):
#
#     Dt = new_spike_time-last_spike_time
#     new_p =  p0+(last_p-p0)*np.exp(-Dt/tau)
#     # -- Draw the Random Number here and test with new_p !! -- #
#     # now update after event
#     new_p += dp*(p1-new_p)
#     # increment variables
#     last_spike = e
#     last_p = new_p
#
#
# ```

# %%
import sys, os
import numpy as np
from scipy import stats

sys.path.append('..')
import plot_tools as pt
import matplotlib.pylab as plt


# %%
def proba_sim(\
            dt=1e-4, # seconds
            tstop = 2, # seconds
            p0 = 0.1, # proba at 0-frequency
            p1 = 0.5, # proba at oo-frequency
            dp = 0.7, # proba increment
            tau = 0.5, # seconds
            interstims=[.1 for i in range(4)]+[1.]):
    
    t = np.arange(int(tstop/dt))*dt
    p = 0*t+p0

    events = np.cumsum(np.array(interstims)/dt, dtype=int)
    for i in range(len(t)-1):
        p[i+1] = p[i] + dt*( (p0-p[i])/tau )
        if i in events:
            p[i+1] += dp*(p1-p[i])

    ps = np.ones(len(events)+1)*p0
    last_spike, last_p = -np.inf, p0
    for i, e in enumerate(events*dt):
        Dt = e-last_spike
        new_p = p0+(last_p-p0)*np.exp(-Dt/tau)
        #print()
        # -- Draw the Random Number here and test with new_p !! -- #
        ps[i] = new_p # we store it       
        # now update after event
        new_p += dp*(p1-new_p)
        # increment variables
        last_spike = e
        last_p = new_p
        
    return (t, p), (events*dt, ps[:-1])

fig, AX = pt.figure(axes=(2,1), figsize=(1.3,1))
for ax, title, p0, p1 in zip(AX, ['depression', 'facilitation'], [0.8, 0.1], [0.1, 0.6]):
    #pt.plot(*proba_sim(p0=p0, p1=p1, interstims=np.ones(100)*1e-2), ax=ax)
    pt.scatter(*proba_sim(p0=p0, p1=p1)[1], color='r', ax=ax)
    pt.plot(*proba_sim(p0=p0, p1=p1)[0], ax=ax)
    pt.set_plot(ax, title=title, ylim=[0,1], ylabel='rel. proba', xlabel='time')
    xlim = ax.get_xlim()
    for p, l in zip([p0, p1], ['$p_0$', '$p_1$']):
        ax.plot(xlim, p*np.ones(2), 'k:', lw=0.5)
        ax.annotate(l, (xlim[1], p), va='center')

# %%
