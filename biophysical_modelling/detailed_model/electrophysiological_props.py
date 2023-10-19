import sys, pathlib, os

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
    h.finitialize(cell.El)

    for i in range(int(50/dt)):
        h.fadvance()

    RATES = []
    for a, amp in enumerate(AMPS):

        ic.amp = amp
        apc.n = 0
        for i in range(int(duration/dt)):
            h.fadvance()
        if a==0:
            # calculate input res.
            Rin = (cell.soma[0](0.5).v-cell.El)/amp # Mohm

        RATES.append(apc.n*1e3/duration) # rates in Hz
        ic.amp = 0
        for i in range(int(duration/dt)):
            h.fadvance()
            
    return dict(Vm=np.array(Vm), dt=dt,
                Rin=Rin, AMPS=np.array(AMPS),
                RATES=RATES)


def FIcurve_plot(results):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(np.arange(len(results['Vm']))*results['dt'], results['Vm'], color='tab:grey')
    ax.axis('off')
    pt.draw_bar_scales(ax, loc='top-right',
                       Xbar=100, Xbar_label='100ms',
                       Ybar=10, Ybar_label='10mV')
    pt.annotate(ax, '      R$_{in}$=%.1fM$\Omega$ ' % results['Rin'], (0, 0), va='top')

    inset = pt.inset(ax, [0, 0.6, 0.2, 0.4])
    inset.plot(1e3*results['AMPS'], results['RATES'], 'ko-', lw=0.5)
    pt.set_plot(inset, xlabel='amp. (pA)', ylabel='firing rate (Hz)')
    return fig


###############################################################################################

def resistance_profile(cell,
                       amp=-25e-3,
                       duration=100,
                       dt=0.025):

    DISTANCE, RIN, RT = [], [], []
    for iB, branch in enumerate(cell.branches['branches']):

        Distance = []
        Rin, Rt = [], [] # input and transfer resistance
        for b in branch:

            x = cell.SEGMENTS['NEURON_segment'][b]/cell.SEGMENTS['NEURON_section'][b].nseg
            Distance.append(h.distance(cell.SEGMENTS['NEURON_section'][b](x),
                                       cell.soma[0](0.5)))

            ic = h.IClamp(cell.SEGMENTS['NEURON_section'][b](x))
            ic.amp, ic.dur = 0. , 1e3

            h.finitialize(cell.El)
            ic.amp = amp
            for i in range(int(duration/dt)):
                h.fadvance()

            Rin.append((cell.SEGMENTS['NEURON_section'][b](x).v-cell.El)/amp) # Mohm
            Rt.append((cell.soma[0](0.5).v-cell.El)/amp) # Mohm
        RIN.append(Rin)
        RT.append(Rt)
        DISTANCE.append(Distance)
    return {'distance':DISTANCE, 'Rin':RIN, 'Rt':RT}

