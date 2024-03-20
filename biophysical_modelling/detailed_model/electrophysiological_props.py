import sys, pathlib, os

from scipy.optimize import minimize

from neuron import h
from neuron.units import ms
import numpy as np

from cell_template import *

import sys
sys.path.append('../..')
import plot_tools as pt
import matplotlib.pylab as plt

def FIcurve_sim(cell,
                duration = 150, # ms
                AMPS = np.array([-0.05, 0.15, 0.35, 0.55]),
                dt=0.025):
    
    ic = h.IClamp(cell.soma[0](0.5))
    ic.amp = 0. 
    ic.dur =  1e9 * ms
    ic.delay = 0 * ms

    Vm = h.Vector()
    Vm.record(cell.soma[0](0.5)._ref_v)

    apc = h.APCount(cell.soma[0](0.5))

    h.dt = dt
    h.finitialize(-70)

    for i in range(int(duration/dt)):
        h.fadvance()

    V0 = cell.soma[0](0.5).v # after relaxation

    RATES = []
    for a, amp in enumerate(AMPS):

        ic.amp = amp
        apc.n = 0
        for i in range(int(duration/dt)):
            h.fadvance()
        if a==0:
            # calculate input res.
            Rin = (cell.soma[0](0.5).v-V0)/amp # Mohm

        RATES.append(apc.n*1e3/duration) # rates in Hz
        ic.amp = 0
        for i in range(int(duration/dt)):
            h.fadvance()
            
    return dict(Vm=np.array(Vm), 
                dt=dt, duration=duration,
                Rin=Rin, AMPS=np.array(AMPS),
                RATES=RATES, V0=V0)


def exp_func(t, Tau, A, B):
    return A*(1-np.exp(-t/Tau))+B
    

def FIcurve_plot(results, Vbar=10):

    fig, ax = plt.subplots(figsize=(9,3))

    t = np.arange(len(results['Vm']))*results['dt']
    
    tCond = (t>results['duration']) & (t<2*results['duration'])
    def to_minimize(X):
        return np.abs(\
            exp_func(t[tCond]-results['duration'], *X)-\
                results['Vm'][tCond]).sum()

    
    ax.plot(t[(t>100)], results['Vm'][(t>100)], color='tab:grey')

    res = minimize(to_minimize, [20, 2, results['Vm'][0]])
    ax.plot(t[tCond], exp_func(t[tCond]-results['duration'], *res.x), 'r:', lw=0.5)
    
    # current trace
    I = 0*t
    for a, amp in enumerate(results['AMPS']):
        cond = (t>((2*a+1)*results['duration'])) & (t<((2*a+2)*results['duration']))
        I[cond] = amp
    
    ax.plot(t[(t>100)], results['Vm'].min()-3*Vbar/2+I[(t>100)]/I.max()*Vbar, color='k')

    ax.axis('off')
    pt.draw_bar_scales(ax, loc='top-right',
                       Xbar=100, Xbar_label='100ms',
                       Ybar=Vbar, Ybar_label='%imV' % Vbar, 
                       Ybar_label2='%.1fpA'%(1e3*I.max()), ycolor='tab:grey', ycolor2='k') 
    pt.annotate(ax, '      R$_{in}$=%.1fM$\Omega$ ' % results['Rin'], (0, 0), va='top')
    pt.annotate(ax, '\n      V$_{rest}$=%.1fmV ' % results['V0'], (0, 0), va='top')
    pt.annotate(ax, '\n\n      $\\tau_m$=%.2fms ' % res.x[0], (0, 0), va='top')

    # print(np.abs(res.x[1]/results['AMPS'][0])) # -> check the calculated input res
    
    # f-I curve inset
    inset = pt.inset(ax, [0, 0.6, 0.2, 0.4])
    inset.plot(1e3*results['AMPS'], results['RATES'], 'ko-', lw=0.5)
    pt.set_plot(inset, xlabel='amp. (pA)', ylabel='firing rate (Hz)')

    return fig

###############################################################################################

def resistance_profile(cell,
                       amp=-25e-3,
                       duration=150,
                       dt=0.025):

    h.dt = dt
    DISTANCE, RIN, RT = [], [], []
    for iB, branch in enumerate(cell.set_of_branches):

        Distance = []
        Rin, Rt = [], [] # input and transfer resistance
        for b in branch:

            x = cell.SEGMENTS['NEURON_segment'][b]/cell.SEGMENTS['NEURON_section'][b].nseg
            Distance.append(h.distance(cell.SEGMENTS['NEURON_section'][b](x),
                                       cell.soma[0](0.5)))

            ic = h.IClamp(cell.SEGMENTS['NEURON_section'][b](x))
            ic.amp, ic.dur = 0. , 1e3

            h.finitialize(-70)
            ic.amp = 0 # in relaxation period
            for i in range(int(duration/dt)):
                h.fadvance()
            V0soma = cell.soma[0](0.5).v # after relaxation
            V0in = cell.SEGMENTS['NEURON_section'][b](x).v
            ic.amp = amp # in evoked period
            for i in range(int(duration/dt)):
                h.fadvance()
            Rin.append((cell.SEGMENTS['NEURON_section'][b](x).v-V0in)/amp) # Mohm
            Rt.append((cell.soma[0](0.5).v-V0soma)/amp) # Mohm
        RIN.append(Rin)
        RT.append(Rt)
        DISTANCE.append(Distance)
    return {'distance':DISTANCE, 'Rin':RIN, 'Rt':RT}

