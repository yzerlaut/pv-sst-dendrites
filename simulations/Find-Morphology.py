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
import sys, os
sys.path.append('../neural_network_dynamics/')
import nrn
from nrn.plot import nrnvyz
from utils import plot_tools as pt

# %%
Valid_Morphologies = []
for x in os.listdir('morphologies'):
    if 'Pvalb' in x: # UPDATE for other cell types
        try:
            morpho = nrn.Morphology.from_swc_file(os.path.join('morphologies', x)) 
            SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)
            Valid_Morphologies.append(os.path.join('morphologies', x))
            print('"%s" valid !' % x)

        except BaseException:
            print('"%s" not valid...' % x)
            pass

# %%
fig, AX = pt.plt.subplots(len(Valid_Morphologies), 2, figsize=(4, 1.7*len(Valid_Morphologies)))
for i, f in enumerate(Valid_Morphologies):
    morpho = nrn.Morphology.from_swc_file(f) 
    SEGMENTS = nrn.morpho_analysis.compute_segments(morpho)
        
    vis = nrnvyz(SEGMENTS)
    vis.plot_segments(cond=(SEGMENTS['comp_type']!='axon'), ax=AX[i][1])

    vis.plot_segments(cond=(SEGMENTS['comp_type']=='axon'),
                            color='tab:blue',                                           
                            bar_scale_args=None, ax=AX[i][0])
    vis.plot_segments(ax=AX[i][0], cond=(SEGMENTS['comp_type']!='axon'))
    AX[i][0].set_title(str(i+1)+') '+f.split(os.path.sep)[-1].split('.')[0], fontsize=7)
    AX[i][1].set_title('dendrite', fontsize=6)


# %%
