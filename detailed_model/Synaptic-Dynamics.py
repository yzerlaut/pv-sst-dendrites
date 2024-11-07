# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modelling the Dynamics of Glutamatergic Synapses 
#
# ## --> Short-Term Plasticity + Stochastic Vesicular Release
#
# We derive a very simplified model of synaptic dynamics that can capture synaptic depression and synaptic facilitation with stochastic multi-vesicular release.
#
# ### - 1) A time varying dynamics of release probability
#
# We use a very simplified kinetic model where the release probability $p$ follows a first order dynamics given by:
#
# \begin{equation}
# \frac{dp}{dt} = \frac{p_0-p(t)}{\tau_p} + \sum_{spike} \delta p \cdot \big( p_\infty - p(t) \big) \cdot \delta(t-t_{spike})
# \end{equation}
#
# where:
# - $p_0$ is the probability at zero frequency stimulation (infinitely spaced events)
# - $p_\infty$ is the probability at infinite frequency stimulation (null spaced events)
# - $\delta p$ is a probability increment factor --strictly positive -- with constraint: $ \delta p \leq |p_\infty - p_0| $
# - $\tau_p$ is the recovery time constant to go back to $p_0$ level
#
# *Synaptic depression* corresponds to the setting: $p_0 > t_\infty$. *Synaptic facilitation* corresponds to the setting: $p_0 < t_\infty$.
#
# ### - 2) A stochastic release scheme with multi-release
#
# For a spike event at time $t$, we draw a random number $r$ *from a uniform distribution between 0 and 1* and we release $N$ single-vesicles (up to $N_{max}$) if: 
# $$
# \mathcal{H}(N_{max}-N) \cdot \big({p(t)}\big)^{N} < r \leq \big({p(t)}\big)^{N-1} \qquad \qquad  \forall N \in [1, N_{max}]
# $$
#
# where $\mathcal{H}$ is the right-continuous Heaviside function ( i.e. $\mathcal{H}(0)=1$ and $\mathcal{H}(x<0)=0$ ).
#
# As an example, for $N_{max}=2$ at $t$ where $p(t)=0.5$, this would correspond to:
#
# - failure if $r > 0.5$
# - single-vesicle release for $0.25 \leq r < 0.5$
# - double-vesicle release for $0  \leq r < 0.25$
#
# Or, for $N_{max}=3$ at $t$ where $p(t)=0.1$, this would correspond to:
#
# - failure if $r > 0.1$
# - single-vesicle release for $0.01 \leq r < 0.1$
# - double-vesicle release for $0.001  \leq r < 0.01$
# - triple-vesicle release for $0.0001  \leq r < 0.001$
#
# ## Analytical estimate (usefull for fitting and re-scalings)
#
# For a set of event $ \large\{ t_i \large\} $, the release probability at $p_{(i)}$ at $t=t_i$ is:
#
# $$
# p_{(i)} = p_0 + \Big( p_{(i-1)} + \delta p \cdot \big( p_\infty -p_{(i-1)} \big) - p_0 \Big) \cdot e^{ - \frac{ t_{i} - t_{i-1}} { \tau_p } }
# $$
#
# Therefore the average release $R$ at $t_i$ for a maximum vesicle release of $N_{max}$ with vesicle weight $w$ is:
#
# $$
# R(i) = \sum_{N \in [1,N_{max}]} N \cdot w \cdot \big( ( p_{(i)} )^N - \mathcal{H}(N_{max}-N) * ( p_{(i)} )^{N+1} \big) 
# $$
#
#
# ## Implementation in an event-based simulation scheme
#
#
# ```
#
# def get_release_events(pre_spikes,
#                        P0=0.5, P1=0.5, dP=0.1, tauP=0.2, Nmax=1,
#                        dt_multirelease=1e-3, # each vesicle release is shifted by dt
#                        verbose=False):
#
#     # build the time-varing release probability:
#     P = np.ones(len(pre_spikes))*P0 # initialized
#     for i in range(len(pre_spikes)-1):
#         P[i+1] = P0 + ( P[i]+dP*(P1-P[i]) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
#
#     # draw random numbers:
#     R = np.random.uniform(0, 1, size=len(P))
#
#     # find number of vesicles released:
#     N = np.sum([(P**n/R >1).astype(int) for n in range(Nmax+1)], axis=0)-1
#
#     # build release events:
#     release_events = []
#     for e, n in zip(pre_spikes, N):
#         for i in range(n):
#             release_events.append(e+i*dt_multirelease)
#           
#     return np.array(release_events)
#
# ```

# %%
# general modules
import sys, os
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
# project-specific modules
sys.path.append('..')
import plot_tools as pt
from synaptic_input import PoissonSpikeTrain
from scipy.optimize import minimize
sys.path.append('../analyz')
from analyz.processing.signanalysis import crosscorrel

# %%
SIMS = {\
    'models':['faithfull', 'stochastic', 'depressing', 'facilitating'],
    'faithfull':   {'P0':1.00, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':1},
    'stochastic':  {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':1},
    'depressing':  {'P0':0.90, 'P1':0.20, 'dP':0.60, 'tauP':0.50, 'Nmax':1},
    'facilitating':{'P0':0.05, 'P1':0.60, 'dP':0.15, 'tauP':0.50, 'Nmax':1},
}


# %% [markdown]
# # Release Probability Dynamics

# %%
def proba_sim(events,\
              output='time-course', # or "events"
              dt=1e-4, # seconds
              tstop = 2, # seconds
              P0 = 0.1, # proba at 0-frequency
              P1 = 0.5, # proba at oo-frequency
              dP = 0.8, # proba increment
              tauP = 0.4, # seconds
              Nmax = None): # useless in this function

    if output=='time-course':
        t = np.arange(int(tstop/dt))*dt
        p = 0*t+P0
    
        # computing the proba time-course
        iEvents = np.array(events/dt, dtype=int)
        for i in range(len(t)-1):
            p[i+1] = p[i] + dt*( (P0-p[i])/tauP )
            if i in iEvents:
                p[i+1] += dP*(P1-p[i])
        return t, p

    else:
        # event-based sim
        P = np.ones(len(events))*P0 # init to p0
        for i in range(len(events)-1):
            P[i+1] = P0 + ( P[i]+dP*(P1-P[i]) - P0 )*np.exp( -(events[i+1]-events[i])/tauP )
        return P

events = np.concatenate([0.1+0.1*np.arange(5), [1.4]])
fig, AX = pt.figure(axes=(3,1), figsize=(1.3,1))
for ax, model in zip(AX, ['stochastic', 'depressing', 'facilitating']):
    pt.scatter(events, proba_sim(events, output='events', **SIMS[model]), ms=8, color='k', ax=ax)
    pt.plot(*proba_sim(events, output='time-course', **SIMS[model]), lw=0.3, ax=ax)
    pt.set_plot(ax, title=model, ylim=[0,1], yticks=[0,0.5,1],ylabel='rel. proba', xlabel='time')
    xlim = ax.get_xlim()
    for p, l, va in zip([SIMS[model]['P0'], SIMS[model]['P1']], ['$p_0$', '$p_\infty$'], ['bottom', 'top']):
        ax.plot(xlim, p*np.ones(2), 'k:', lw=0.5)
        ax.annotate(l, (xlim[1], p), va=va)
    AX[1].annotate('$\\delta_p$', (1.15,0.45))
    AX[1].annotate('$\\tau_p$', (1.6,0.4))


# %% [markdown]
# # Impact on Event Release

# %%
def get_release_events(pre_spikes,
                       P0=0.5, P1=0.5, dP=0.1, tauP=0.2, Nmax=1,
                       dt_multirelease=1e-3, # the vesicle release is slightly shifted in time (for the brian2 simulator)
                       verbose=False):

    # build the time-varing release probability:
    P = np.ones(len(pre_spikes))*P0 # initialized
    for i in range(len(pre_spikes)-1):
        P[i+1] = P0 + ( P[i]+dP*(P1-P[i]) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )

    # draw random numbers:
    R = np.random.uniform(0, 1, size=len(P))
    # find number of vesicles released:
    N = np.sum([(P**n/R >1).astype(int) for n in range(Nmax+1)], axis=0)-1
    # build release events:
    release_events = []
    for e, n in zip(pre_spikes, N):
        for i in range(n):
            release_events.append(e+i*dt_multirelease)

    if verbose:
        for e, p, r, n in zip(pre_spikes, P, R, N):
            print('@ %.1fs, p=%.2f, random=%.2f' % (e, p, r))
            print('   -> release :', n, 'event(s)  -- r < p^(N=%i) = %.2f' % (n, p**n))
           
    return np.array(release_events)


def rough_release_dynamics(events,
            tstop=2., t0=0, dt=5e-4, tau=0.02, Q=1):
    
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
np.random.seed(6)
for i in range(1):
    releaseEvents = get_release_events(np.arange(10)*0.1, 
                                       **{'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':3},
                                       verbose=True)
    print('')
    print('==> release events :', releaseEvents)
    print('')


# %%
def sim_release(release_proba_params={},
                nTrials = 10,
                freq = 10, # Hz 
                nStim = 7,
                t0 = 0.1,
                tstop = 1.2,
                dt = 5e-4,
                seed=0):

    np.random.seed(seed)
    
    pre_Events = [\
        np.concatenate([t0+np.arange(nStim)/freq, [tstop]]) for i in range(nTrials)]
    
    release_Events = [\
        get_release_events(pre, **release_proba_params) for pre in pre_Events]
    
    fig, AX = pt.figure(axes=(4,1), figsize=(1.2,1.1), wspace=0.6)
    # analytical prediction:
    AX[3].plot(pre_Events[0]-t0, analytical_estimate(pre_Events[0], release_proba_params=release_proba_params), 'ro:', lw=0.3, ms=2)
    # simulations
    Rs, t = [], -t0+np.arange(1.2*tstop/dt)*dt
    for i in range(nTrials):
        AX[0].scatter(-t0+pre_Events[i], i*np.ones(len(pre_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=15)
        AX[1].scatter(-t0+release_Events[i], i*np.ones(len(release_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=15)
        Rs.append(rough_release_dynamics(release_Events[i], tstop=1.2*tstop, dt=dt))
        AX[2].plot(t, i+Rs[-1], color='k', lw=0.5, alpha=0.7)
    # we add some more
    for i in range(20*nTrials):
        release_Events = get_release_events(pre_Events[i%nTrials], **release_proba_params)
        Rs.append(rough_release_dynamics(release_Events, tstop=1.2*tstop, dt=dt))
    
    for ax, title in zip(AX, ['presynaptic APs', 'release events', 'release dynamics']):
        pt.set_plot(ax, yticks=[], ylabel='trials', xlabel='time (s)', title=title, 
                    ylim=[-0.5,nTrials+.5], xlim=[-t0,1.2*tstop-t0])

    AX[2].plot([t[-1],t[-1]], [-0.5,0.5], color='k', lw=2)
    pt.annotate(AX[2], ' unitary release', (t[-1],-0.5), xycoords='data', fontsize=5, rotation=90)
    pt.annotate(AX[3], 'analytic ', (0,1), fontsize=5, color='r', rotation=90, va='top')
    pt.plot(t, np.mean(Rs, axis=0), sy=np.std(Rs, axis=0), ax=AX[3])
    pt.set_plot(AX[3], yticks=[0,1], xlabel='time (s)', title='release average', xlim=[-t0,1.2*tstop-t0]) 
    pt.annotate(AX[3], '%i trials' % (20*nTrials), (1,1), va='top', ha='right', fontsize=6, rotation=90) 
            
    return fig

def analytical_estimate(events,
                        weight=1,
                        release_proba_params={}):
   Nmax = release_proba_params['Nmax']
   P = proba_sim(events, output='events', **release_proba_params)
   return np.sum([(n*weight)*(P**n-P**(n+1)*np.sign(Nmax-n))\
                          for n in range(1, Nmax+1)], axis=0)


# %%
SIMS = {\
    'faithfull':   {'P0':1.00, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':1},
    'stochastic\nNmax=1':  {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':1},
    'stochastic\nNmax=2':  {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':2},
    'stochastic\nNmax=3':  {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50, 'Nmax':3},
    'depressing':  {'P0':0.90, 'P1':0.20, 'dP':0.60, 'tauP':0.50, 'Nmax':2},
    'facilitating':{'P0':0.05, 'P1':0.60, 'dP':0.15, 'tauP':0.50, 'Nmax':2},
}

for model in SIMS:
    fig = sim_release(release_proba_params=SIMS[model], seed=6)
    pt.annotate(fig, model, (0, 0.5))

# %% [markdown]
# # Model calibration: fitting parameters

# %%
# TO DO
pass

# %% [markdown]
# # Impact on Temporal Signal Integration

# %%
from synaptic_input import PoissonSpikeTrain

def sim_release(release_proba_params={},
                nSyns = 10,
                freq = 1, # Hz 
                mean=5, amp=4,
                tstop = 4.,
                dt = 1e-3,
                seed=1):

    np.random.seed(seed)

    t = np.arange(int(tstop/dt))*dt
    rate = np.clip(mean+amp*np.sin(2*np.pi*freq*t), 0, np.inf)
    
    pre_Events = [np.array(PoissonSpikeTrain(rate, dt=dt, tstop=tstop))\
                          for i in range(nSyns)]
    
    release_Events = [\
        get_release_events(pre, **release_proba_params) for pre in pre_Events]
    
    fig, AX = pt.figure(axes=(5,1), figsize=(1.1,1.), wspace=0.6)
    AX[0].fill_between(t, 0*t, rate, color='lightgray')

    Rs = []
    for i in range(nSyns):
        AX[1].scatter(pre_Events[i], i*np.ones(len(pre_Events[i])), color='lightgray', s=1)
        AX[2].scatter(release_Events[i], i*np.ones(len(release_Events[i])), color='k', s=1)
        Rs.append(rough_release_dynamics(release_Events[i], tstop=tstop, dt=dt))
        AX[3].plot(t, i+Rs[-1], color='k', lw=0.5)
    # we add some more
    for i in range(100*nSyns):
        release_Events = get_release_events(pre_Events[i%nSyns], **release_proba_params)
        Rs.append(rough_release_dynamics(release_Events, tstop=tstop, dt=dt))
    
    for ax, title in zip(AX, ['presynaptic rate', 'presynaptic APs', 'release events', 'release dynamics']):
        pt.set_plot(ax, yticks=None if ax==AX[0] else [], ylabel='Hz' if ax==AX[0] else 'synapses', xlabel='time (s)', title=title, 
                    ylim=[-0.5,nSyns+.5], xlim=[0,tstop])

    pt.plot(t, np.mean(Rs, axis=0), sy=np.std(Rs, axis=0), ax=AX[4])
    pt.set_plot(AX[4], yticks=[0,1], xlabel='time (s)', title='release average', xlim=[0,tstop]) 
            
    return fig

fig = sim_release()


# %%
def sim_release(release_proba_params={},
                nSyns = 100,
                freqs = np.logspace(np.log10(0.04), np.log10(4.), 5), # Hz 
                mean=10, amp=8,
                dt = 1e-3,
                seed=1,
                with_fig=True):

    np.random.seed(seed)

    if with_fig:
        fig, AX = pt.figure(axes=(len(freqs),2), figsize=(1.1,1.), wspace=0.2, hspace=0.2)
    amps = []
    for f, freq in enumerate(freqs):
        Rs = []
        tstop = np.max([5., 3./freq])
        t = np.arange(int(tstop/dt))*dt
        rate = np.clip(mean+amp*np.sin(2*np.pi*freq*t), 0, np.inf)
        if with_fig:
            AX[0][f].fill_between(t, 0*t, rate, color='lightgray')
        for i in range(nSyns):
            release_Events = get_release_events(\
                        np.array(PoissonSpikeTrain(rate, dt=dt, tstop=tstop)),
                                                   **release_proba_params)
            Rs.append(rough_release_dynamics(release_Events, tstop=tstop, dt=dt))
        if with_fig:
            AX[1][f].plot(t[10:], np.mean(Rs, axis=0)[10:], 'k-')
        def to_minimize(x):
            return np.sum((np.mean(Rs, axis=0)[10:]-(x[0]+x[1]*np.sin(2*np.pi*freq*t[10:])))**2)
        res = minimize(to_minimize, [0.2,0.2])
        if with_fig:
            AX[1][f].plot(t, res.x[0]+res.x[1]*np.sin(2*np.pi*freq*t), 'r-', lw=0.5)    
            pt.set_plot(AX[0][f], title="$\\nu$=%.2fHz" % freq,
                        ylabel='rate (Hz)' if f==0 else '', yticks_labels= None if f==0 else [], xticks_labels=[])
            pt.set_plot(AX[1][f], ylabel='release' if f==0 else '', yticks_labels= None if f==0 else [], xlabel='time (s)')
        amps.append(res.x[1])

    if with_fig:
        AX[0][0].plot(t, 0*t+mean, 'k:', lw=0.5)
        AX[0][0].annotate('mean', (0,mean), va='top', fontsize=6)
        AX[0][0].plot(t, 0*t+mean+amp, 'k:', lw=0.5)
        AX[0][0].annotate('amp', (0,mean+amp), va='top', fontsize=6)
    
        pt.set_common_ylims(AX[0])
        pt.set_common_ylims(AX[1])
    else:
        fig = None
            
    return fig, (freqs, amps)

RESPs = {}
for title, release_proba_params in zip(['faithfull', 'stochastic', 'depressing', 'facilitating'], [\
                {'P0':1.0, 'P1':1.0, 'dP':0.0, 'tauP':0.5}, 
                {'P0':0.5, 'P1':0.5, 'dP':0.0, 'tauP':0.5},
                {'P0':0.9, 'P1':0.1, 'dP':0.6, 'tauP':1.0},
                {'P0':0.05, 'P1':0.6, 'dP':0.15, 'tauP':0.5}]):
    fig, (freqs, RESPs[title]) = sim_release(release_proba_params=release_proba_params)
    pt.annotate(fig, title, (0, 1), va='top')


# %%
fig, ax = pt.figure(figsize=(1.5,3))
#ax.plot(freqs, 1.+0*np.array(freqs), 'k:', lw=0.5)
COLORS = ['tab:grey', 'tab:pink', 'tab:red', 'tab:orange']
for i, title in enumerate(RESPs):
    ax.plot(freqs, RESPs[title], 'o-', color=COLORS[i], lw=0.5, ms=2)
    pt.annotate(ax, i*'\n'+title, (1,1), color=COLORS[i], va='top')
pt.set_plot(ax, xscale='log', xlabel='$\\nu$ (Hz)', ylabel='release sinusoid amplitude')

# %% [markdown]
# ## Doing this more systematically by varying the scaling of the presynaptic rate amplitude

# %%
SIMS = {\
    'means':[0.2, 1., 4.0, 4.0, 8.0, 12.0, 20.0, 20.0],
    'amps': [0.18, 0.8, 1.0, 6.0,  6.0,  6.0, 15.0],
    'models':['stochastic', 'depressing', 'facilitating'],
    'stochastic':{'P0':0.5, 'P1':0.5, 'dP':0.0, 'tauP':0.5},
    'depressing':{'P0':0.9, 'P1':0.1, 'dP':0.6, 'tauP':1.0},
    'facilitating':{'P0':0.05, 'P1':0.6, 'dP':0.15, 'tauP':0.5},
    'RESPS':[]}


for i, mean, amp in zip(range(10), SIMS['means'], SIMS['amps']):
    print(' - running $\\nu$=%.1f$\pm$%.1fHz' % (mean, amp))
    RESPs = {}
    for model in SIMS['models']:
        print('    o :', model, ' [...]')
        _, (freqs, RESPs[model]) = sim_release(release_proba_params=SIMS[model], with_fig=False, 
                                               mean=mean, amp=amp,
                                               freqs=np.logspace(-2, 1, 8))        
    SIMS['RESPS'].append(RESPs)

SIMS['freqs'] = freqs
np.save('../data/detailed_model/release-dynamics-varying-mean-amp-with-stp.npy', SIMS)

# %%
SIMS = np.load('../data/detailed_model/release-dynamics-varying-mean-amp-with-stp.npy',
               allow_pickle=True).item()

fig, AX = pt.figure(axes=(len(SIMS['amps']),2), wspace=0.5)
COLORS = ['tab:pink', 'tab:red', 'tab:orange']

for m, model in enumerate(SIMS['models']):
    pt.annotate(AX[0][-1], m*'\n'+model, (1,1), color=COLORS[m], va='top')

    for i, mean, amp in zip(range(10), SIMS['means'], SIMS['amps']):
    
        AX[0][i].plot(SIMS['freqs'], SIMS['RESPS'][i][model], 
                   'o-', color=COLORS[m], ms=2, lw=0.5)
        AX[1][i].plot(SIMS['freqs'], SIMS['RESPS'][i][model]/np.mean(SIMS['RESPS'][i][model]), 
                   'o-', color=COLORS[m], ms=2, lw=0.5)
        
        if m==0:
            AX[0][i].set_title('$\\nu$=%.1f$\pm$%.1fHz' % (mean, amp))
            AX[1][i].plot(SIMS['freqs'], 0*SIMS['freqs']+1, ':', lw=0.5, color='grey')
            
for ax in AX[0]:
    pt.set_plot(ax, xscale='log', 
                ylabel='release' if ax==AX[0][0] else '')
for ax in AX[1]:
    pt.set_plot(ax, xscale='log', xlabel='$\\nu$ (Hz)', 
                ylabel='release var.' if ax==AX[1][0] else '')
pt.set_common_ylims(AX[1])
#pt.set_common_ylims(AX[1])

# %% [markdown]
# --> The effect of STP only appears in the >1Hz presynaptic rate range !!
#
# ---------------------------------------------

# %% [markdown]
# # Effect on Visual Processing Dynamics

# %%
RATES = np.load(os.path.join('..', 'data', 'visual_coding', 'RATES_natural_movie_one.npy'),
                allow_pickle=True).item()
t = RATES['time']-RATES['time'][0] # s 
dt = t[1]-t[0]
rate = 0.5*(\
        np.mean(RATES['PV_negUnits'], axis=0)+\
        np.mean(RATES['SST_negUnits'], axis=0))


# %%
def sim_release(release_proba_params={},
                nSyns = 500,
                seed=1):

    np.random.seed(seed)

    Release = 0*t    
    for i in range(nSyns):
        release_Events = get_release_events(\
                    np.array(PoissonSpikeTrain(rate, dt=dt, tstop=t[-1])),
                                               **release_proba_params)
        Release[1:] += rough_release_dynamics(release_Events, 
                                              tstop=t[-1]+dt, dt=dt)
    Release /= nSyns
            
    return Release

SIMS = {\
    'models':['faithfull', 'stochastic', 'depressing', 'facilitating'],
    'faithfull':    {'P0':1.00, 'P1':1.00, 'dP':0.00, 'tauP':0.50},
    'stochastic':   {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':0.50},
    'depressing':   {'P0':0.90, 'P1':0.10, 'dP':0.60, 'tauP':1.00},
    'facilitating': {'P0':0.05, 'P1':0.60, 'dP':0.15, 'tauP':0.50},
}

for i, model in enumerate(SIMS['models']):
    SIMS['release_%s'%model] = sim_release(release_proba_params=SIMS[model])
np.save('../data/detailed_model/release-dynamics-visual-processing-with-stp.npy', SIMS)

# %%
SIMS = np.load('../data/detailed_model/release-dynamics-visual-processing-with-stp.npy',
               allow_pickle=True).item()
COLORS = ['k', 'tab:pink', 'tab:red', 'tab:orange'] 
cond = (t>0.5) & (t<10)
fig, AX = pt.figure(axes=(1,5), figsize=(2,0.6), hspace=0.4)
for i, model in enumerate(SIMS['models']):
    AX[i+1].plot(t[cond], SIMS['release_%s'%model][cond], color=COLORS[i])
    pt.annotate(AX[i+1], model, (1,.8), color=COLORS[i], va='top')
pt.annotate(AX[0], 'input rate', (1,.8), color='grey', va='top')
AX[0].fill_between(t[cond], 0*t[cond], rate[cond], color='lightgrey')

# %%
SIMS = np.load('../data/detailed_model/release-dynamics-visual-processing-with-stp.npy',
               allow_pickle=True).item()
COLORS = ['k', 'tab:pink', 'tab:red', 'tab:orange']
fig, ax = pt.figure(figsize=(1.1,1.))
for i, model in enumerate(SIMS['models']):
    cond = t>0.8
    CCF, time_shift = crosscorrel(rate[cond], SIMS['release_%s'%model][cond], 1.3, dt)
    ax.plot(time_shift, CCF/np.max(CCF), '--' if model=='stochastic' else '-', color=COLORS[i])
    pt.annotate(ax, i*'\n'+model, (1,1), color=COLORS[i],  va='top')

pt.set_plot(ax, xlabel='shift (s)', ylabel='peak norm.\n corr. coef.', yticks=[0.5,1])

# %%
