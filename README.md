# Interneuron Specific Dendritic Computation in the Neocortex

*Analyze the dendritic integration properties of PV+ and SST+ interneurons in the mouse visual cortex*

## A) Dendritic Anatomy Analysis -- EM dataset

> relies on the MICrONS dataset, see: https://microns-explorer.org

### A.1) Dataset Presentation [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/anatomy/Dataset-Presentation.ipynb)

- details the set of Martinotti and Basket cells analyzed in the study
- plot all celular morhphologies 
- show the layer classification

### A.2) Analysis of Synaptic Locations [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/anatomy/Synaptic-Location-Analysis.ipynb)

- computes the positions of synapses along the dendritic tree
- computes a linear density estimates along the tree

### A.3) Diameter Analysis [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/anatomy/Diameter-Analysis.ipynb)

- computes the dendritic diameters along the different 

## B) Simplified Model of Dendritic Integration 

*Build the figure panels related to Figure X.*

### B.1) Model Presentation [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/reduced_model/Model-Presentation.ipynb)

- morphology drawing 
- input impedance characterization

### B.2) Analysis of Synaptic Integration [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/reduced_model/Analysis-of-Synaptic-Integration.ipynb)

- ...

## C) Biophysical Modelling of Morphologically-Detailed Reconstructions

/!\ Need to compile the NMODL mechanisms for NEURON with `nrnivmodl mechanisms`.

### C.1) Morphologies with Dendritic Branches [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/detailed_model/Find-Single-Dendritic-Branches.ipynb)

- show the different dendritic branches in the two models

### C.2) Electrophysiological Properties [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/detailed_model/Electrophysiological-Properties.ipynb)
 
- compute the input resistance and spiking responses (rheobase)
- compute the transfer resistance along the branch for each dendritic branch

### C.3) Integration of Proximal and Distal Input [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/detailed_model/Clustered-Input.ipynb)

- compute the response to stimulation in either proximal or distal segments
- compare with linear predictions

### C.4) Input-Output curves with Background Activity [(-Notebook-)](https://github.com/yzerlaut/pv-sst-dendrites/blob/main/biophysical_modelling/detailed_model/Stim-on-Background.ipynb)

- simulate backgrounds excitatory and inhibitory activity
- adds a stimulus input of increasing strength

## D) In Vivo Imaging of Neural Activity in Interneurons of the Mouse Visual Cortex

[...]

## Usage/Setup

- Part A) relies on the Allen Institute datasets
