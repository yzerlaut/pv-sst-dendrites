# Interneuron Specific Dendritic Computation in the Neocortex

*Analyze the dendritic integration properties of PV+ and SST+ interneurons in the mouse visual cortex*


## A) Dendritic Anatomy Analysis (Figure 3) -- EM dataset

> relies on the MICrONS dataset, see: https://microns-explorer.org

#### A.1) Dataset Presentation [(-Notebook-)](./anatomy/Dataset-Presentation.ipynb)

- details the set of Martinotti and Basket cells analyzed in the study
- plot all celular morhphologies 
- show the layer classification

#### A.2) Analysis of Synaptic Locations [(-Notebook-)](./anatomy/Synaptic-Location-Analysis.ipynb)

- computes the positions of synapses along the dendritic tree
- computes a linear density estimates along the tree

### A.3) Diameter Analysis [(-Notebook-)](./anatomy/Diameter-Analysis.ipynb)

- computes the dendritic diameters along the different 



## B) Simplified Model of Dendritic Integration (Figure 1)

#### B.1) Impedance Profile Characterization [(-Notebook-)](./reduced_model/Impedance-BRT.ipynb)

- morphology drawing 
- input and transfer impedance characterization

#### B.2) Analysis of Synaptic Integration [(-Notebook-)](./reduced_model/Multi-Input-Integration.ipynb)

- performs multi-input integration at different locations
- compares with linear predictions

## C) Biophysical Modelling of Morphologically-Detailed Reconstructions (Figure 4,5)

/!\ Need to compile the NMODL mechanisms for NEURON with `nrnivmodl mechanisms`.

#### C.1) Morphologies with Dendritic Branches [(-Notebook-)](./detailed_model/Find-Single-Dendritic-Branches.ipynb)

- show the different dendritic branches in the two models

#### C.2) Electrophysiological Properties [(-Notebook-)](./detailed_model/Electrophysiological-Properties.ipynb)
 
- compute the input resistance and spiking responses (rheobase)
- compute the transfer resistance along the branch for each dendritic branch

#### C.3) Integration of Proximal and Distal Input [(-Notebook-)](./detailed_model/Clustered-Input.ipynb)

- compute the response to stimulation in either proximal or distal segments
- compare with linear predictions

#### C.4) Input-Output curves with Background Activity [(-Notebook-)](./detailed_model/Stim-on-Background.ipynb)

- simulate backgrounds excitatory and inhibitory activity
- adds a stimulus input of increasing strength
- record stimulus-evoked spiking activity

#### C.5) Firing Dynamics following a Stochastic Process Stimulation [(-Notebook-)](./detailed_model/StochProc-input.ipynb)

- simulate excitatory and inhibitory activity driven by an Ornstein-Uhlenbeck process
- do multiple runs to compute population averages
- analyze the temporal transformation between input and output

## D) Spiking Activity of PV+ and SST+ Interneurons in the Mouse Visual Cortex (Figure 5)

Analysis for the *Visual Coding Neuropixels* dataset of [Siegle et al., 2021](https://www.nature.com/articles/s41586-020-03171-x)

Download the dataset with the script [./visual_coding/Download.py](./visual_coding/Download.py)

#### D.1) Optotagging [(-Notebook-)](./visual_coding/Optotagging.ipynb)

- finding the units in the visual cortex
- using the phototagging protocol to classify positive/negative units

#### D.2) Response to Natural Movie [(-Notebook-)](./visual_coding/Natural-Movie.ipynb)

- Computes the stimulus-evoked time-varying rate of positive and negative units
- Compute the cross-correlation function between positive and negative units

## E) In Vivo Calcium Imaging of Neural Activity in SST+ cells with and without the NDMAR (Figure 5)

#### E.1) Show Raw Data [(-Notebook-)](./in-vivo/Show-Raw-Data.ipynb)

- plot the raw data displayed on the paper

#### E.2) Analyze Temporal Dynamics [(-Notebook-)](./in-vivo/Final-Analysis.ipynb)

- perform the deconvolution of stimulus-evoked activity
- compares Wild-Type and SST:GluN1-KO mice

## Usage/Setup

Clone the repository with its submodules
```
git clone https://github.com/yzerlaut/pv-sst-dendrites --recurse-modules
```
