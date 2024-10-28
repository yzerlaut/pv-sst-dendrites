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
# #### - 1) A time varying dynamics of release probability
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
# #### - 2) A stochastic release scheme
#
# For a spike event at time $t$, we draw a random number $r$ (from a uniform distribution between 0 and 1) and we release an event if: $r < p(t) $
#
# ### Implementation in an event-based simulation scheme
#
#
# ```
#
# def get_release_events(presynaptic_spikes,
#                        P0=0.5, P1=0.5, dP=0.1, tauP=0.2):
#
#     # initialisation
#     last_spike_time, last_p = -np.inf, p0 # adding a pre-spike far in the past
#     P = np.zeros(len(presynaptic_spikes)) # probas at events
#
#     for i, new_spike_time in enumerate(presynaptic_spikes):
#         Dt = new_spike_time-last_spike_time
#         new_p =  P0+(last_p-P0)*np.exp(-Dt/tauP)
#         P[i] = new_p
#         # -- Draw the Random Number here and test with new_p !! -- #
#         # now update after event
#         new_p += dP*(P1-new_p)
#         # increment variables
#         last_spike_time = new_spike_time
#         last_p = new_p
#
#     return presynaptic_spikes[\
#                 np.random.uniform(0, 1, size=len(P))<P]
#
# pre_spikes_array = np.arange(10)*0.1 #
# get_release_events(pre_spikes_array)
#
#
# ```

# %% [markdown]
# # Release Probability Dynamics

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
            dp = 0.8, # proba increment
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


# %% [markdown]
# # Impact on Event Release

# %%
def get_release_events(presynaptic_spikes,
                       P0=0.5, P1=0.5, dP=0.1, tauP=0.2):

    # initialisation
    last_spike_time, last_p = -np.inf, p0 # adding a pre-spike far in the past
    P = np.zeros(len(presynaptic_spikes)) # probas at events

    for i, new_spike_time in enumerate(presynaptic_spikes):
        Dt = new_spike_time-last_spike_time
        new_p =  P0+(last_p-P0)*np.exp(-Dt/tauP)
        P[i] = new_p
        # -- Draw the Random Number here and test with new_p !! -- #
        # now update after event
        new_p += dP*(P1-new_p)
        # increment variables
        last_spike_time = new_spike_time
        last_p = new_p

    return presynaptic_spikes[\
                np.random.uniform(0, 1, size=len(P))<P]


def rough_release_dynamics(events,
            tstop=2., t0=0, dt=1e-3, tau=0.03, Q=1):
    
    t = np.arange(int(tstop/dt))*dt+t0
    rel = 0*t
    iEvents = np.array(events/dt, dtype=int)
    for i in range(len(t)-1):
        rel[i+1] = rel[i] - dt*rel[i]/tau
        if i in iEvents:
            rel[i+1] += Q
    return rel
    
# pre_spikes_array = np.arange(10)*0.1
# pt.plot(rough_release_dynamics(get_release_events(pre_spikes_array)))


# %%
def sim_release(release_proba_params={},
                nTrials = 20,
                freq = 10, # Hz 
                nStim = 7,
                t0 = 0.1,
                tstop = 1.2,
                dt = 1e-3,
                seed=0):

    np.random.seed(seed)
    
    pre_Events = [\
        np.concatenate([t0+np.arange(nStim)/freq, [tstop]]) for i in range(nTrials)]
    
    release_Events = [\
        get_release_events(pre, **release_proba_params) for pre in pre_Events]
    
    fig, AX = pt.figure(axes=(4,1), figsize=(1.2,1.1), wspace=0.6)
    Rs, t = [], -t0+np.arange(1.2*tstop/dt)*dt
    for i in range(nTrials):
        AX[0].scatter(-t0+pre_Events[i], i*np.ones(len(pre_Events[i])), color='lightgray', s=1)
        AX[1].scatter(-t0+release_Events[i], i*np.ones(len(release_Events[i])), color='k', s=1)
        Rs.append(rough_release_dynamics(release_Events[i], tstop=1.2*tstop, dt=dt))
        AX[2].plot(t, i+Rs[-1], color='k', lw=0.5)
    
    for ax, title in zip(AX, ['presynaptic APs', 'release events', 'release dynamics']):
        pt.set_plot(ax, yticks=[], ylabel='trials', xlabel='time (s)', title=title, 
                    ylim=[-0.5,nTrials+.5], xlim=[-t0,1.2*tstop-t0])

    pt.plot(t, np.mean(Rs, axis=0), sy=np.std(Rs, axis=0), ax=AX[3])
    pt.set_plot(AX[3], yticks=[0,1], xlabel='time (s)', title='release average', xlim=[-t0,1.2*tstop-t0]) 
            
    return fig


# %%
for title, release_proba_params in zip(['faithfull', 'stochastic\n(constant)', 'depressing', 'facilitating'], [\
                {'P0':1.0, 'P1':1.0, 'dP':0.0, 'tauP':0.4}, 
                {'P0':0.5, 'P1':0.5, 'dP':0.0, 'tauP':0.4}, 
                {'P0':0.9, 'P1':0.2, 'dP':0.6, 'tauP':0.4}, 
                {'P0':0.1, 'P1':0.4, 'dP':0.15, 'tauP':0.4}]):
    fig = sim_release(release_proba_params=release_proba_params, seed=6)
    pt.annotate(fig, title, (0, 0.5))

# %%
