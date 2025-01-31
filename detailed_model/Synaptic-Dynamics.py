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
# \frac{dp}{dt} = \frac{p_0-p(t)}{\tau_p} + \sum_{spike} \delta p \cdot 
#             \big( \frac {p_\infty - p(t)}{ | p_\infty - p_0 | } \big) \cdot \delta(t-t_{spike})
# \end{equation}
#
# where:
# - $p_0$ is the probability at zero frequency stimulation (infinitely spaced events)
# - $p_\infty$ is the probability at infinite frequency stimulation (null spaced events)
# - $\delta p$ is a probability increment -- strictly positive -- with constraint: $ | \delta p | \leq |p_\infty - p_0| $
# - $\tau_p$ is the recovery time constant to go back to $p_0$ level
#
# *Synaptic depression* corresponds to the setting: $p_0 > p_\infty$ .
#
# *Synaptic facilitation* corresponds to the setting: $p_0 < p_\infty$ & $\delta p > 0 $.
#
# ### - 2) A stochastic release scheme with multi-release
#
# For a spike event at time $t$, we draw a random number $r$ *from a uniform distribution between 0 and 1* and we release $n$ single-vesicles (up to $N$) if: 
# $$
# \mathcal{H}(N-n) \cdot 
#      \binom{N}{n} \big({p_{(t)}}\big)^{n} \big(1-{p_{(t)}}\big)^{N-n}  
#          < r \leq 
#       \binom{N}{n-1} \big({p_{(t)}}\big)^{n-1} \big(1-{p_{(t)}}\big)^{N-(n-1)} 
# $$
#
# where $\mathcal{H}$ is the right-continuous Heaviside function ( i.e. $\mathcal{H}(0)=1$ and $\mathcal{H}(x<0)=0$ ).
#
# As an example, for $N_{max}=2$ at $t$ where $p(t)=0.5$, this would correspond to:
#
# - double-vesicle release for $0  \leq r < 0.25$ ( --> $p_{n=-} = 0.5^2 = 0.25 $ )
# - single-vesicle release for $0.25 \leq r < 0.75$ ( --> $ p_{n=1} = 2 \cdot 0.5 \cdot (1-0.5) = 0.5$)
# - failure if $r > 0.75$ ( --> $ p_{n=2} = (1-0.5)^2 = 0.25 $)
#
# ## Analytical estimate (usefull for fitting and re-scalings)
#
# For a set of event $ \large\{ t_i \large\} $, the release probability at $p_{(i)}$ at $t=t_i$ is:
#
# $$
# p_{(i)} = p_0 + \Big( p_{(i-1)} + \delta p \cdot \big( \frac{p_\infty -p_{(i-1)}}{ | p_\infty - p_0 | } \big) - p_0 \Big) 
#                         \cdot e^{ - \frac{ t_{i} - t_{i-1}} { \tau_p } }
# $$
#
# Therefore the average release $R$ at $t_i$ for a maximum vesicle release of $N$ with vesicle weight $w$ is:
#
# $$
# R(i) = \sum_{n \in [1,N]} n \cdot w \cdot \binom{N}{n} \big({p_{(t)}}\big)^{n} \big(1-{p_{(t)}}\big)^{N-n}
# $$
#
#
# ## Implementation in an event-based simulation scheme
#
#
# ```
# def get_release_events(pre_spikes,
#                        P0=0.5, P1=0.5, dP=0.1, tauP=0.2, Nmax=1,
#                        verbose=False):
#
#     # build the time-varing release probability:
#     P = np.ones(len(pre_spikes))*P0 # initialized
#     for i in range(len(pre_spikes)-1):
#         P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/(abs(P1-P0)+1e-4) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
#
#     # build the probabilities of each number of vesicles ([!!] from Nmax to 1 [!!] ) :
#     Ps = np.cumsum([ (comb(Nmax, n) * P**n * (1-P)**(Nmax-n)) for n in np.arange(Nmax, 0, -1)], axis=0)
#
#     # draw random numbers:
#     R = np.random.uniform(0, 1, size=len(pre_spikes))
#
#     # find number of vesicles released:
#     N = np.sum((Ps/R>1).astype(int), axis=0)
#
#     # build release events:
#     release_events = []
#     for e, n in zip(pre_spikes, N):
#         release_events += [e for _ in range(n)]
#           
#     return np.array(release_events)
#
# ```

# %%
# general modules
import sys, os
import numpy as np
from math import comb # binomial coefficient
from scipy import stats
import matplotlib.pylab as plt
# project-specific modules
sys.path.append('..')
import plot_tools as pt
from synaptic_input import PoissonSpikeTrain
from scipy.optimize import minimize
sys.path.append('../analyz')
from analyz.processing.signanalysis import crosscorrel, autocorrel
sys.path.append('..')
import plot_tools as pt

# %%
SIMS = {\
    'models':['faithfull', 'stochastic', 'depressing', 'facilitating'],
    'faithfull':   {'P0':1.00, 'P1':1.00, 'dP':0.00, 'tauP':1.0, 'Nmax':1},
    'stochastic':  {'P0':0.50, 'P1':0.50, 'dP':0.00, 'tauP':1.0, 'Nmax':1},
    'depressing':  {'P0':0.90, 'P1':0.20, 'dP':0.40, 'tauP':1.0, 'Nmax':2},
    'facilitating':{'P0':0.1, 'P1':0.60, 'dP':0.25, 'tauP':1.0, 'Nmax':2},
}


# %% [markdown]
# # Release Probability Dynamics

# %%
def proba_sim(events,\
              output='time-course', # or "events"
              dt=1e-4, # seconds
              tstop = 3., # seconds
              P0 = 0.1, # proba at 0-frequency
              P1 = 0.5, # proba at oo-frequency
              dP = 0.8, # proba increment
              tauP = 2.0, # seconds
              Nmax = None): # useless in this function

    if output=='time-course':
        t = np.arange(int(tstop/dt))*dt
        p = 0*t+P0
    
        # computing the proba time-course
        iEvents = np.array(events/dt, dtype=int)
        for i in range(len(t)-1):
            p[i+1] = p[i] + dt*( (P0-p[i])/tauP )
            if i in iEvents:
                p[i+1] += dP*(P1-p[i])/(abs(P1-P0)+1e-4)
        return t, p

    else:
        # event-based sim
        P = np.ones(len(events))*P0 # init to p0
        for i in range(len(events)-1):
            P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/(abs(P1-P0)+1e-4) - P0 )*np.exp( -(events[i+1]-events[i])/tauP )
        return P

events = np.concatenate([0.1+0.2*np.arange(5), [2.4]])
fig, AX = pt.figure(axes=(4,1), figsize=(1.2,1), wspace=1.3)
for ax, model in zip(AX, ['faithfull', 'stochastic', 'depressing', 'facilitating']):
    pt.scatter(events, proba_sim(events, output='events', **SIMS[model]), ms=8, color='k', ax=ax)
    pt.plot(*proba_sim(events, output='time-course', **SIMS[model]), lw=0.3, ax=ax)
    pt.set_plot(ax, title=model, ylim=[0,1.05], yticks=[0,0.5,1],ylabel='rel. proba', xlabel='time')
    xlim = ax.get_xlim()
    for p, l, va in zip([SIMS[model]['P0'], SIMS[model]['P1']], ['$p_0$', '$p_\infty$'], ['bottom', 'top']):
        ax.plot(xlim, p*np.ones(2), 'k:', lw=0.5)
        ax.annotate(l, (xlim[1], p), va=va)
AX[2].annotate(r'$\delta_p \cdot \frac{p_{(t)} - p_0}{ | p_\infty - p_0 | }$', (1.15,0.45))
AX[2].annotate('$\\tau_p$', (1.6,0.4))
fig.savefig('../figures/detailed_model/STPsupp_sketch.svg')

# %% [markdown]
# # Impact on Event Release

# %%
from math import comb

def get_release_events(pre_spikes,
                       P0=0.5, P1=0.5, dP=0.1, tauP=0.2, Nmax=2,
                       verbose=False):

    # build the time-varing release probability:
    P = np.ones(len(pre_spikes))*P0 # initialized
    for i in range(len(pre_spikes)-1):
        P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/(abs(P1-P0)+1e-4) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
    # build the probabilities of each number of vesicles ([!!] from Nmax to 1 [!!] ) :
    Ps = np.cumsum([ (comb(Nmax, n) * P**n * (1-P)**(Nmax-n)) for n in np.arange(Nmax, 0, -1)], axis=0)
    # draw random numbers:
    R = np.random.uniform(0, 1, size=len(pre_spikes))
    # find number of vesicles released:
    N = np.sum((Ps/R>1).astype(int), axis=0)
    # build release events:
    release_events = []
    for e, n in zip(pre_spikes, N):
        release_events += [e for _ in range(n)]

    if verbose:
        for i in range(len(pre_spikes)):
            print('@ %.1fs, p=%.2f, random=%.2f' % (pre_spikes[i], P[i], R[i]))
            print('   -> release :', N[i], 'event(s)')
            if N[i]==0:
                print('            because  r > P(n=%i)=%.2f ' %(N[i]+1,Ps[Nmax-1-N[i],i]))
            elif N[i]==Nmax:
                print('            because  r < P(n=%i)=%.2f ' %(N[i],Ps[Nmax-N[i],i]))
            else:
                print('            because  P(n=%i)=%.2f  < r < P(n=%i)=%.2f :' % (N[i],Ps[Nmax-1-N[i],i], N[i]-1,Ps[Nmax-N[i],i]))
    return np.array(release_events)


def rough_release_dynamics(events,
                           tstop=2., t0=0, dt=5e-4, tau=0.02, Q=1):
    
    t = np.arange(int(tstop/dt))*dt+t0
    rel = 0*t
    iEvents = np.array(events/dt, dtype=int)
    for i in range(len(t)-1):
        rel[i+1] = rel[i] - dt*rel[i]/tau
        # add events if any
        rel[i+1] += Q*len(np.flatnonzero(i==iEvents))
    return rel
    
# pre_spikes_array = np.arange(10)*0.1
# pt.plot(rough_release_dynamics(get_release_events(pre_spikes_array)))


# %% [markdown]
# ### Testing this random scheme

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
                events = np.concatenate([0.1+np.arange(7)/10., [1.1]]),
                tstop=1.3, nTrials=10,
                dt = 5e-4,
                seed=0,
                with_title=True, figsize=(1.1,1.2)):

    np.random.seed(seed)
    
    pre_Events = [events for i in range(nTrials)]
    
    release_Events = [\
        get_release_events(pre, **release_proba_params) for pre in pre_Events]
    
    fig, AX = pt.figure(axes=(4,1), figsize=figsize, wspace=0.6)
    # analytical prediction:
    AX[3].plot(pre_Events[0], analytical_estimate(pre_Events[0], release_proba_params=release_proba_params),
               'o:', color='firebrick', lw=0.3, ms=2)
    # simulations
    Rs, t = [], np.arange(tstop/dt)*dt
    for i in range(nTrials):
        AX[0].scatter(pre_Events[i], i*np.ones(len(pre_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=15)
        AX[1].scatter(release_Events[i], i*np.ones(len(release_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=15)
        Rs.append(rough_release_dynamics(release_Events[i], tstop=tstop, dt=dt))
        AX[2].plot(t, i+Rs[-1], color='k', lw=0.5, alpha=0.7)
    # we add some more
    for i in range(20*nTrials):
        release_Events = get_release_events(pre_Events[i%nTrials], **release_proba_params)
        Rs.append(rough_release_dynamics(release_Events, tstop=tstop, dt=dt))
    
    for ax, title in zip(AX, ['presynaptic APs', 'release events', 'release dynamics']):
        pt.set_plot(ax, yticks=[], ylabel='trials', xlabel='time (s)', title=title if with_title else '', 
                    ylim=[-0.5,nTrials+1.5], xlim=[0,tstop])

    AX[2].plot([t[-1],t[-1]], [-0.5,0.5], color='k', lw=2)
    pt.annotate(AX[2], ' 1 vesicle', (t[-1],0.5), xycoords='data', fontsize=5, rotation=90)
    pt.annotate(AX[3], 'analytic ', (0,1), fontsize=5, color='firebrick', rotation=90, va='top')
    pt.plot(t, np.mean(Rs, axis=0), sy=np.std(Rs, axis=0), ax=AX[3])
    pt.set_plot(AX[3], yticks=[0,1], xlabel='time (s)', ylabel='release (/ves.)',
                title='release average' if with_title else '', xlim=[0,tstop]) 
    pt.annotate(AX[3], '%i trials' % (20*nTrials), (1,1), va='top', ha='right', fontsize=6, rotation=90) 
            
    return fig

def analytical_estimate(events,
                        weight=1,
                        release_proba_params={}):
   Nmax = release_proba_params['Nmax']
   P = proba_sim(events, output='events', **release_proba_params)
   return np.sum([(n*weight)*( comb(Nmax, n) * P**n * (1-P)**(Nmax-n) )\
                          for n in range(Nmax, 0, -1)], axis=0)
    


# %%
SIMS2 = {\
    'faithfull\n$N_{max}$=1':   SIMS['faithfull'].copy(),
    'stochastic\n$N_{max}$=1':  SIMS['stochastic'].copy(),
    'stochastic\n$N_{max}$=2':  SIMS['stochastic'].copy(),
    'stochastic\n$N_{max}$=3':  SIMS['stochastic'].copy(),
    'depressing\n$N_{max}$=2':  SIMS['depressing'].copy(),
    'facilitating\n$N_{max}$=2':SIMS['facilitating'].copy(),
}
SIMS2['stochastic\n$N_{max}$=1']['Nmax'] = 1
SIMS2['stochastic\n$N_{max}$=2']['Nmax'] = 2
SIMS2['stochastic\n$N_{max}$=3']['Nmax'] = 3

for m, model in enumerate(SIMS2):
    fig = sim_release(release_proba_params=SIMS2[model], seed=6, figsize=(1.4,1.1), with_title=(model=='faithfull\n$N_{max}$=1'))
    pt.annotate(fig, model+'\n$p_0$=%.2f, $p_\infty$=%.2f,\n' % (SIMS2[model]['P0'], SIMS2[model]['P1'])+\
                            '$\delta p$=%.2f, $\\tau_p$=%.1fs' % (SIMS2[model]['dP'], SIMS2[model]['tauP']),
    (-0.02, 0.7), ha='center', va='top')
    fig.savefig('../figures/detailed_model/STPsupp_example%i.svg' % (m+1))

# %% [markdown]
# # Model calibration: fitting parameters

# %% [markdown]
# ## SSTs

# %%
from scipy.optimize import minimize


## PAIRED RECORDINGS ##
Pratios = np.loadtxt('../data/paired-recordings/SST-pn-over-p1.csv',
           delimiter=';', converters=lambda s: s.replace(b',', b'.'))
# SST protocol: 9 APs @ 20Hz
time_PR = np.arange(9)*1./20.
## SNIFFER DATA ##
sniffer = {'p_mean':0.19, 'p_sem':0.05, 'n':8}
# SST protocol: 7 APs @ 10Hz
time_SN = np.arange(7)*1./10.

# %%
from scipy.optimize import minimize


fig, ax = pt.figure()
Pratios = np.loadtxt('../data/paired-recordings/SST-pn-over-p1.csv',
           delimiter=';', converters=lambda s: s.replace(b',', b'.'))
pt.plot(Pratios.mean(axis=1), sy=stats.sem(Pratios, axis=1), color='g', ax=ax)
Pratios = np.loadtxt('../data/paired-recordings/SST-pn-over-p1-Ca15mM.csv',
           delimiter=';', converters=lambda s: s.replace(b',', b'.'))
pt.plot(Pratios.mean(axis=1), sy=stats.sem(Pratios, axis=1), color='tab:orange', ax=ax)
pt.annotate(ax, '[Ca2+]=2mM', (0,1), va='top', color='g', fontsize='x-small')
pt.annotate(ax, '\n[Ca2+]=1.5mM', (0,1), va='top', color='tab:orange', fontsize='x-small')
pt.set_plot(ax, xlabel='AP #', ylabel='$p_n$/$p_1$', yticks=np.arange(1,6), title='SST')

# %%
tauP = 1.0
Nmax = 2

def get_probas(pre_spikes, X):
    P0, P1, dP, tauP = X
    P = np.ones(len(pre_spikes))*P0
    for i in range(len(pre_spikes)-1):
        P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/abs((P1-P0)+1e-4) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
    return P

def to_minimize(X):
    if X[2]>abs(X[1]-X[0]):
        # dp > abs(p01-p1) --> forbidden !
        return 1e5
    else:
        Ps_PR = get_probas(time_PR, X) # paired recordings pred.
        Ps_SN = get_probas(time_SN, X) # sniffer protocol pred.
        residual_PR = np.mean((Pratios.mean(axis=1)-Ps_PR/Ps_PR[0])**2)
        residual_SN = abs(np.mean(Ps_SN)-sniffer['p_mean'])
        return residual_PR+np.sign(residual_SN-1e-2)
                

res = minimize(to_minimize, x0=[.8*sniffer['p_mean'], 1.5*sniffer['p_mean'], 0.1, tauP],
               method='Nelder-Mead',
               bounds=[[0.01, Pratios.mean(axis=1).max()*sniffer['p_mean']],
                       [0.01, .8*Pratios.mean(axis=1).max()*sniffer['p_mean']],
                       [0.05,0.5],
                       [tauP, tauP]])

fig, ax = pt.figure(figsize=(1.1,1.2), hspace=0.3, right=10.)
ax2 = pt.inset(ax, [1.6, 0, 0.3, 0.8])
#for i in range(Pratios.shape[1]):
#    ax.scatter(time+np.random.randn()*5e-3, Pratios[:,i], color='k', marker='o', s=0.1)
    
pt.scatter(1+np.arange(9), Pratios.mean(axis=1), sy=stats.sem(Pratios, axis=1), ax=ax, alpha=0.5)
#ax.errorbar(time, np.mean(failures, axis=1), yerr=failures.std(axis=1), fmt='o-',color='k', lw=2, alpha=0.5)
#ax2.errorbar(time, probas, fmt='o-',color='grey', lw=2, alpha=0.5)

Ps = get_probas(time_PR, res.x)
ax.plot(1+np.arange(9), Ps/Ps[0], '-', color='magenta')

ax2.bar([1], [sniffer['p_mean']], yerr=[sniffer['p_sem']], color='dimgrey')
ax2.bar([0], [np.mean(get_probas(time_SN, res.x))], color='magenta')

sf = 'fitted: '
for s, p in zip(['$p_0$', '$p_\infty$', '$\delta p$'], res.x):
    sf += s+'=%.2f, '%p
pt.annotate(ax, sf[:-2], (1.4,1.3), ha='center', fontsize='x-small', color='magenta')
pt.annotate(ax, 'fixed: $N_{max}=%i$, $\\tau_p$=%0ds' % (Nmax, tauP) , (-0.2,1.3), color='magenta', fontsize='x-small')

pt.annotate(ax, 'paired recordings\n%i APs @ 20Hz' % Pratios.shape[0], (0.5,1), va='center', ha='center', fontsize='small')
pt.annotate(ax, 'sniffer data\n7 APs @ 10Hz', (1.7,1.), va='center', ha='center', fontsize='small')
#pt.annotate(ax, '[Ca$^{2+}$]=2mM', (1,0), va='bottom', ha='left', fontsize='xx-small', rotation=90)
pt.annotate(ax2, '[Ca$^{2+}$]=1.5mM', (1,0), va='bottom', ha='left', fontsize='xx-small', rotation=90)
pt.annotate(ax, 'n=%i' % Pratios.shape[1], (0,0.4), fontsize='small')
pt.annotate(ax2, ' n=%i' % sniffer['n'], (1,0), ha='center', xycoords='data', fontsize='small', color='w', rotation=90)

ax.plot([0,9], [1,1], 'k:', lw=0.3)
pt.set_plot(ax, xlabel='$n$ (AP #)', ylabel='$p_n$/$p_1$', xlim=[0.5,9.5], xticks=1+np.arange(3)*4, yticks=[1,3,5])
pt.set_plot(ax2, ylabel=r' $\langle$p$\rangle$ ves. rel.', xticks=[0,1], xticks_labels=['fit', 'data'], xticks_rotation=50)

print(res.x)

fig.savefig('../figures/detailed_model/STPsupp_data_SST.svg')

# %%
SST_model = {'P0':res.x[0], 'P1':res.x[1], 'dP':res.x[2], 'tauP':tauP, 'Nmax':Nmax}
fig = sim_release(events=0.1+np.arange(9)/20., tstop=0.65, release_proba_params=SST_model, seed=0)
fig.savefig('../figures/detailed_model/STPsupp_model_SST.svg')
np.save('../data/detailed_model/SST_stp.npy', SST_model)

# %% [markdown]
# ## PVs

# %%
from scipy.optimize import minimize


## PAIRED RECORDINGS ##
Pratios = np.loadtxt('../data/paired-recordings/PV-pn-over-p1.csv',
           delimiter=';', converters=lambda s: s.replace(b',', b'.'))
# SST protocol: 5 APs @ 20Hz
time_PR = np.arange(5)*1./20.
## SNIFFER DATA ##
sniffer = {'p_mean':0.48, 'p_sem':0.09, 'n':9}
# SST protocol: 7 APs @ 10Hz
time_SN = np.arange(7)*1./10.

# %%
tauP = 1.0
Nmax = 2

def get_probas(pre_spikes, X):
    P0, P1, dP, tauP = X
    P = np.ones(len(pre_spikes))*P0
    for i in range(len(pre_spikes)-1):
        P[i+1] = P0 + ( P[i]+dP*(P1-P[i])/abs((P1-P0)+1e-4) - P0 )*np.exp( -(pre_spikes[i+1]-pre_spikes[i])/tauP )
    return P

def to_minimize(X):
    if X[2]>abs(X[1]-X[0]):
        # dp > abs(p01-p1) --> forbidden !
        return 1e5
    else:
        Ps_PR = get_probas(time_PR, X) # paired recordings pred.
        Ps_SN = get_probas(time_SN, X) # sniffer protocol pred.
        residual_PR = np.mean((Pratios.mean(axis=1)-Ps_PR/Ps_PR[0])**2)
        residual_SN = abs(np.mean(Ps_SN)-sniffer['p_mean'])
        return residual_PR+np.sign(residual_SN-1e-2)
                

res = minimize(to_minimize, x0=[1.1*sniffer['p_mean'], 0.8*sniffer['p_mean'], 0.1, tauP],
               method='Nelder-Mead',
               bounds=[[Pratios.mean(axis=1).min()*sniffer['p_mean'], 1.],
                       [Pratios.mean(axis=1).min()*sniffer['p_mean'], 1.],
                       [0.05,0.5],
                       [tauP, tauP]])

fig, ax = pt.figure(figsize=(1.1,1.2), hspace=0.3, right=10.)
ax2 = pt.inset(ax, [1.6, 0, 0.3, 0.8])
    
pt.scatter(1+np.arange(5), Pratios.mean(axis=1), sy=2*stats.sem(Pratios, axis=1), ax=ax, alpha=0.5)

Ps = get_probas(time_PR, res.x)
ax.plot(1+np.arange(5), Ps/Ps[0], '-', color='magenta')

ax2.bar([1], [sniffer['p_mean']], yerr=[sniffer['p_sem']], color='dimgrey')
ax2.bar([0], [np.mean(get_probas(time_SN, res.x))], color='magenta')
sf = 'fitted: '
for s, p in zip(['$p_0$', '$p_\infty$', '$\delta p$'], res.x):
    sf += s+'=%.2f, '%p
pt.annotate(ax, sf[:-2], (1.4,1.3), ha='center', fontsize='x-small', color='magenta')
pt.annotate(ax, 'fixed: $N_{max}=%i$, $\\tau_p$=%0ds' % (Nmax, tauP) , (-0.2,1.3), color='magenta', fontsize='x-small')

pt.annotate(ax, 'paired recordings\n%i APs @ 20Hz' % Pratios.shape[0], (0.5,1), va='center', ha='center', fontsize='small')
pt.annotate(ax, 'sniffer data\n7 APs @ 10Hz', (1.7,1.), va='center', ha='center', fontsize='small')
#pt.annotate(ax, '[Ca$^{2+}$]=2mM', (1,0), va='bottom', ha='left', fontsize='xx-small', rotation=90)
pt.annotate(ax2, '[Ca$^{2+}$]=1.5mM', (1,0), va='bottom', ha='left', fontsize='xx-small', rotation=90)
pt.annotate(ax, 'n=%i' % Pratios.shape[1], (0,0), fontsize='small')
pt.annotate(ax2, ' n=%i' % sniffer['n'], (1,0), ha='center', xycoords='data', fontsize='small', color='w', rotation=90)
ax.plot([0,9], [1,1], 'k:', lw=0.3)
pt.set_plot(ax, xlabel='$n$ (AP #)', ylabel='$p_n$/$p_1$', xlim=[0.5,5.5], xticks=1+np.arange(3)*2, yticks=range(3),ylim=[0,2])
pt.set_plot(ax2, ylabel=r' $\langle$p$\rangle$ ves. rel.',
            xticks=[0,1], xticks_labels=['fit', 'data'], xticks_rotation=50)

fig.savefig('../figures/detailed_model/STPsupp_data_PV.svg')

# %%
PV_model = {'P0':res.x[0], 'P1':res.x[1], 'dP':res.x[2], 'tauP':tauP, 'Nmax':2}
fig = sim_release(events=0.1+np.arange(9)/20., tstop=0.65, release_proba_params=PV_model, seed=0)
fig.savefig('../figures/detailed_model/STPsupp_model_PV.svg')
np.save('../data/detailed_model/PV_stp.npy', PV_model)


# %%
def sim_release_pretty(release_proba_params={},
                events = np.concatenate([0.1+np.arange(7)/10., [1.1]]),
                tstop=1.3, nTrials=10,
                dt = 5e-4,
                seed=0,
                with_title=True, figsize=(1,1)):

    np.random.seed(seed)
    
    pre_Events = [events for i in range(nTrials)]
    
    release_Events = [\
        get_release_events(pre, **release_proba_params) for pre in pre_Events]
    
    fig, AX = pt.figure(axes=(3,1), figsize=figsize, wspace=0.6)

    # simulations
    Rs, t = [], np.arange(tstop/dt)*dt
    for i in range(nTrials):
        AX[0].scatter(pre_Events[i], i*np.ones(len(pre_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=6)
        AX[1].scatter(release_Events[i], i*np.ones(len(release_Events[i])), facecolor='k', edgecolor=None, alpha=.35, lw=0, s=10)
        Rs.append(rough_release_dynamics(release_Events[i], tstop=tstop, dt=dt))
        AX[2].plot(t, i+Rs[-1], color='k', lw=0.5, alpha=0.7)
    # we add some more
    for i in range(20*nTrials):
        release_Events = get_release_events(pre_Events[i%nTrials], **release_proba_params)
        Rs.append(rough_release_dynamics(release_Events, tstop=tstop, dt=dt))
    
    for ax in AX:
        ax.plot([0,0.05], (nTrials+0.5)*np.ones(2), 'k-', lw=1)
        pt.set_plot(ax, [], ylim=[-0.5,nTrials+1.5], xlim=[0,tstop])
    pt.annotate(AX[0], '50ms ', (0, nTrials+0.5), ha='right', va='center', fontsize=5, xycoords='data')
    pt.annotate(AX[0], 'trials', (0, nTrials/2.), rotation=90, ha='right', va='center', fontsize=5, xycoords='data')

    AX[2].plot([t[-1],t[-1]], [-0.5,0.5], color='k', lw=2)
    pt.annotate(AX[2], '  1 vesicle', (t[-1],0.5), xycoords='data', fontsize=5, rotation=90)
            
    return fig


# %%
PV_model = np.load('../data/detailed_model/PV_stp.npy', allow_pickle=True).item()
fig = sim_release_pretty(events=0.1+np.arange(9)/20., tstop=0.65, release_proba_params=PV_model, seed=1)
fig.savefig('../figures/Figure5/STP_PV.svg')

# %%
SST_model = np.load('../data/detailed_model/SST_stp.npy', allow_pickle=True).item()
fig = sim_release_pretty(events=0.1+np.arange(9)/20., tstop=0.65, release_proba_params=SST_model, seed=3)
fig.savefig('../figures/Figure5/STP_SST.svg')

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
fig.suptitle('release dynamics')

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

pt.set_plot(ax, xlabel='shift (s)', 
            title='(glutamate)\nrelease dynamics',
            ylabel='peak norm.\n corr. coef.', yticks=[0.5,1])


# %%
def sim_release(release_proba_params={},
                nSyns = 200,
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
    'models':['PV', 'SST'],
    'SST':   np.load('../data/detailed_model/SST_stp.npy', allow_pickle=True).item(),
    'PV':   np.load('../data/detailed_model/PV_stp.npy', allow_pickle=True).item(),
}

for i, model in enumerate(SIMS['models']):
    SIMS['release_%s'%model] = sim_release(release_proba_params=SIMS[model])
    
np.save('../data/detailed_model/release-dynamics-visual-processing-fits-with-stp.npy',
        SIMS)

# %%
SIMS = np.load('../data/detailed_model/release-dynamics-visual-processing-fits-with-stp.npy',
               allow_pickle=True).item()

COLORS = ['tab:red', 'tab:orange']
fig, ax = pt.figure(figsize=(1.1,1.))
for i, model in enumerate(SIMS['models']):
    cond = t>0.8
    CCF, time_shift = crosscorrel(rate[cond], SIMS['release_%s'%model][cond], 1.4, dt)
    ax.plot(time_shift, CCF/np.max(CCF), '-', lw=1, color=COLORS[i])
    pt.annotate(ax, i*'\n'+model, (1,1), color=COLORS[i],  va='top')
    
ACF, time_shift = autocorrel(rate[cond], 1.4, dt)
ax.plot(time_shift, ACF/np.max(ACF), '-', lw=0.5, color='grey')

pt.set_plot(ax, xlabel='shift (s)', ylabel='peak norm.\n corr. coef.',
            title='(glutamate)\nrelease dynamics', yticks=[0.5,1])


# %% [markdown]
# # Step Functions

# %%
def get_rate(t, stimFreq=1., stepFactor=3.):
    rate = stimFreq+0*t
    rate[(t>0.5) & (t<1)] *=stepFactor
    return rate 
def sim_release(release_proba_params={},
                stimFreq = 1., stepFactor=3.,
                nSyns = 200, tau=0.02,
                dt = 1e-3):

    tstop = 1.5
    t = np.arange(int(tstop/dt))*dt
    rate = get_rate(t, stimFreq=stimFreq, stepFactor=stepFactor)

    Release = 0*t    
    for i in range(nSyns):
        release_Events = get_release_events(\
                    np.array(PoissonSpikeTrain(rate, dt=dt, tstop=t[-1], seed=i)),
                                               **release_proba_params)
        Release += rough_release_dynamics(release_Events, 
                                              tstop=t[-1]+dt, dt=dt, tau=tau)
    Release /= nSyns
            
    return t, Release

SIMS = {\
    'models':['PV', 'SST'],
    'SST':   np.load('../data/detailed_model/SST_stp.npy', allow_pickle=True).item(),
    'PV':   np.load('../data/detailed_model/PV_stp.npy', allow_pickle=True).item(),
}

for i, model in enumerate(SIMS['models']):
    SIMS['t'], SIMS['release_%s'%model] = sim_release(release_proba_params=SIMS[model])
    
COLORS = ['tab:red', 'tab:orange']
fig, AX = pt.figure(axes=(1,3), figsize=(2,0.6), hspace=0.4)
AX[0].fill_between(SIMS['t'], 0*SIMS['t'], get_rate(SIMS['t']), color='lightgray')
pt.annotate(AX[0], ' input rate (Hz)', (1,1), va='top', ha='right')
pt.draw_bar_scales(AX[0], Xbar=0.1, Ybar=1, Ybar_label='1Hz ')
for i, model in enumerate(SIMS['models']):
    AX[i+1].plot(SIMS['t'], SIMS['release_%s'%model], color=COLORS[i])
    pt.annotate(AX[i+1], model, (1,.8), color=COLORS[i], va='top')
pt.annotate(AX[2], 'release dynamics', (0.5,1), ha='center')
for ax in pt.flatten(AX):
    ax.axis('off')
    pt.draw_bar_scales(ax, Xbar=0.1, Xbar_label='100ms 'if ax==AX[0] else '',
                       Ybar=1 if ax==AX[0] else 1e-12, Ybar_label='1Hz 'if ax==AX[0] else '')
pt.annotate(AX[2], 'glut. release (a.u.)', (0.,0.), ha='right', rotation=90)

# %%
fig, AX = pt.figure(axes=(3,5), figsize=(1,0.8), hspace=0.2, wspace=0.2)

COLORS = ['tab:orange', 'grey']
STP_model = np.load('../data/detailed_model/SST_stp.npy', allow_pickle=True).item()
Static_model = {'P0':0.3, 'P1':0.3, 'dP':0.00, 'tauP':1.0, 'Nmax':1}

for i, freq in enumerate([0.5, 1, 1.5, 2, 2.5]):
    for j, factor in enumerate([2., 3, 4.]):
        t, rate = sim_release(release_proba_params=STP_model,
                              stimFreq=freq, stepFactor=factor, tau=0.05, dt=5e-3, nSyns=500)
        AX[i][j].plot(t, rate, color=COLORS[0])
        t, rate = sim_release(release_proba_params=Static_model,
                              stimFreq=freq, stepFactor=factor, tau=0.05, dt=5e-3, nSyns=500)
        AX[i][j].plot(t, rate, color=COLORS[1])
        AX[i][j].axis('off')
        pt.draw_bar_scales(AX[i][j], Xbar=0.5, Ybar=5e-3)
        if j==0:
            pt.annotate(AX[i][0], 'f=%.1fHz' % freq, (-0.3, 0.5), ha='center', rotation=90)
        if i==0:
            pt.annotate(AX[0][j], 'step x%i' % factor, (0.5, 1.1), ha='center')

# %%
fig, AX = pt.figure(axes=(3,5), figsize=(1,0.8), hspace=0.2, wspace=0.2)

COLORS = ['tab:red', 'grey']
STP_model = np.load('../data/detailed_model/PV_stp.npy', allow_pickle=True).item()
Static_model = {'P0':0.80, 'P1':0.80, 'dP':0.00, 'tauP':1.0, 'Nmax':1}

for i, freq in enumerate([2, 4, 6, 8, 10]):
    for j, factor in enumerate([2.,3, 4.]):
        t, rate = sim_release(release_proba_params=Static_model,
                              stimFreq=freq, stepFactor=factor, tau=0.05, dt=5e-3, nSyns=500)
        AX[i][j].plot(t, rate, color=COLORS[1])
        t, rate = sim_release(release_proba_params=STP_model,
                              stimFreq=freq, stepFactor=factor, tau=0.05, dt=5e-3, nSyns=500)
        AX[i][j].plot(t, rate, color=COLORS[0])
        AX[i][j].axis('off')
        pt.draw_bar_scales(AX[i][j], Xbar=0.5, Ybar=5e-3)
        if j==0:
            pt.annotate(AX[i][0], 'f=%.1fHz' % freq, (-0.3, 0.5), ha='center', rotation=90)
        if i==0:
            pt.annotate(AX[0][j], 'step x%i' % factor, (0.5, 1.1), ha='center')

# %%
