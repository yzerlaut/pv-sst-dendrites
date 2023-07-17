# Interneuron-Specific Dendritic Computations in the Neocortex

> _sketch of the poster presentation_
>
> _Gordon Conference 2023 "inhibition in the CNS"_

### Intro

There is a great diversity of interneurons (INs) in the mammalian cortex. Those different interneurons are thought to have different functional roles in the processing of information in cortical networks.
The mechanisms underlying the different recruitment processes associated to those different functional roles remains to be characterized.
So far, much work has been focused on the connectivity properties of interneurons (both in terms of input and output) to explain this functional diversity. However, the intrinsic cellular properties also have a great impact on the functional properties of a neuronal population. This aspect has been much less investigated. Notably, the properties of dendritic integration in interneurons of the neocortex remains unexplored. 
This is what we investigate in this study. We focus on two molecularly-defined interneuronal populations of the mouse cortex: SST+ and PV+ interneurons. 
We investigate what are the properties of dendritic integration in those interneurons ? How do they differ ? And whether such properties explain the diverse response of those interneurons ?

### Synaptic Physiology of Dendritic Integration

We first looked at the physiological properties of synaptic integration in those two population of interneurons. While there are some studies in the hippocampus[^1] and the cerebellum[^2], the integrative properties of interneurons in the neocortex were still to be characterized.

[^1]: **Hu et al. (2010)** "Dendritic Mechanisms Underlying Rapid Synaptic Activation of Fast-Spiking Hippocampal Interneurons" _Science_. **Cornford et al. (2019)** "Dendritic NMDA receptors in parvalbumin neurons enable strong and stable neuronal assemblies" _eLIFE_.
[^2]: **Vervaeke et al. (2012)** "Gap Junctions Compensate for Sublinear Dendritic Integration in an Inhibitory Network" _Science_. **Abrahamsson et al. (2012)** "Thin Dendrites of Cerebellar Interneurons Confer Sublinear Synaptic Integration and a Gradient of Short-Term Plasticity" _Neuron_.

We started by characterizing the NMDA-to-AMPA ratio at excitatory synapses in PV+ and SST+ interneurons. We performed intracellular recordings in Voltage-clamp coupled with extracellular stimulation[^3]. We found that SST+ positive had a strong NDMA component while this component was really weak in PV+ interneurons.

[^3]: AMPA/NMDA ratio were obtained by measuring response in SST-INs and PV-INs to extracellular electrical stimulation at -70 mV (mainly AMPA) and at +40 mV in the presence of NBQX (that blocks AMPAR)

We next asked how such diversity in synaptic transmission would impact the dendritic integration of multiple inputs. We probed local synaptic integration along the dendritic tree with glutamate-uncaging triggered by 2-photon stimulation[^4].We analyzed synaptic integration by recording the responses to both single site and multiple quasi-synchronous sequence of inputs. We computed the prediction from the sum of the individual PSP (the linear prediction) that we compared to the observed response of sequence stimulation. 

[^4]: Details on the glutamate-uncaging experiment: **1.**  MNI-glutamate (caged glutamate) was applied via local pipette perfusion. **2.** Uncaging location was positioned where light evoked EPSCs had a fastest rise time, suggesting presence of cluster of AMPA receptors and indicative of synaptic connections. This is important because, in contrast to Pyramidal neurons, Interneurons do not have spines.

We found that SST+ had a supra-linear behavior, with responses preserved or boosted with respect to the linear prediction case. This behavior is similar to the known behavior of pyramidal cells. Importantly, we found that this behavior disappeared with the removal of NMDA-mediated component of synaptic transmission (either pharmacologically with MK-DAP5 or genetically with a knockout of the GluN1 NMDAR subunit). This role of NMDARs is consistent with our previous measurement of the strong NMDAR component. 

On the other hand, we found an overall sublinear behavior for PV+ cells. Multiple inputs elicited a much weaker response than the sum of individual components, indicative of a strong synaptic saturation phenomenon in PV+ dendrites.

We further analyzed the dendritic saturation phenomenon in PV+ INs by analyzing the synaptic integration at different location of the dendritic tree. We performed quantal-adjusted glutamate uncaging experiments in the proximal and distal regions of the dendritic tree[^5]. The suppression phenomenon in PV+ INs was strong in distal regions while it was nearly absent in proximal regions. interestingly, even in distal regions the suppression phenomenon was absent in SST+ dendrites, synaptic integration was still supra-linear in those locations.

[^5]: For the proximal and distal stimulation, amplitude of evoked responses were adjusted to match previously obtained local synaptic responses. These local synaptic responses were recorded with local sucrose puff at the different locations along the dendrites (proximal and distal) in the presence of TTX. Sucrose evokes the fusion of synaptic vesicles only in the location close to the puff area. This allowed us to estimate the “real” amplitudes  of single synaptic vesicles at different distances along the dendrite that we matched during our uncaging experiments. 

Note that we could not explain this strong difference between PV+ and SST+ interneurons by a difference in dendritic diameter in the distal regions[^6].

[^6]: In our own measurements, non significant differences in distal diameter between PV+ and SST+ interneurons. In the Allen dataset reconstructed morphologies, even thinner distal dendrites in SST+ than in PV+, i.e. predicting more synaptic saturation than in PV+ cells. 

We conclude that dendritic integration of local synaptic inputs greatly varies between SST+ and PV+ interneurons. 

### Distribution of Synapses in Dendrites

The effect of those dendritic integration properties will strongly depend on the way synaptic inputs distribute on the dendritic tree. To further investigate dendritic properties in SST+ and PV+ interneurons, we next analyzed their synaptic distributions along the dendritic morphology.

To count the density of glutamatergic synapses we generated transgenic mice where PSD95 (a protein involved in the structure of glutamatergic synapses) is tagged with a fluorescent marker, mVENUS. This transgene was only expressed in SST+ INs or PV+ INs. Meaning that the fluorescent marker was only present in either SST+ INs or PV+ INs. We prepared brain sections from these animals and used antibodies (nanobodies) against mVenus to amplify the signal. At the same time, we infected neurons with td-tomato to label morphology of SST or PV-INs. This resulted in images where we could localize the pncta corresponding to synaptic inputs and td-tomato to identify the morphology of individual cells. Cells were imaged in confocal mode and at specific locations (proximal $\leq$ 30um) and distal (~100um from the soma). Then, the same dendritic location were imaged in STED mode to improve resolution of images. We quantified puncta in STED images at the proximal and distal locations. 

We found that the densities of synapses greatly varied between proximal and distal locations in PV+ INs. The proximal locations exhibited a high density of synapses while the distal locations had a much lower density of synapses along dendritic branches. On the other hand, the density of synapses did not seem to vary along the dendritc tree on SST+ INs, we did not observe a significant difference between the density of proximal and distal locations.

We further investigated this difference in an Electron Microscopy dataset publicly shared by the Allen Institute[^7]. This dataset corresponds to the full reconstruction of 1mm$^3$ of a sample of the mouse visual cortex. We combined their databases of (1) proof-read interneurons identifications and (2) synapses identifications to compute synaptic distributions along the dendritic tree. We were able to analyze dendritic synaptic distributions with great spatial resolutions. Confirming our previous observations, we could see the density of synapses gradually decreasing when moving away from the soma in PV+ interneurons. See the shown example of a ~200um-long dendritc branch together with the samples of proximal ($\leq$ 50um) and distal ($\geq$ 100um) locations. On the other hand, the distribution of synapses along the dendritic tree of SST+ INs seem nearly constant with distal regions displaying similar densities than proximal regions.

Here again, our analysis of dendritic diameters can not explain such a strong difference in synaptic densities in distal regions[^8].

[^7]: Schneider-Mizell et al. (2023) _biorXiv_ 
[^8]: In PV+ INs the reduction of dendritic diameters follows the reduction of synaptic densities. This reduction of diameters is also observed in SST+ INs while the density is preserved. Therefore, variations of branch diameters alone can not explain the dual observation. 

We conclude that synaptic distributions follow two drastically different strategies in PV+ and SST+ INs.

### Modelling Dendritic Integration

We now have those two sets of observations, in terms of (1) synaptic physiology and (2) dendritic distributions of synaptic inputs, that show a marked difference between PV+ and SST+ INs. To make sense of those observations, we implemented a reduced theoretical model of dendritic integration having the ability to vary those synaptic properties. 

We build up on the classical work of Wilfried Rall that combines cable theory in simple morphological settings to study the integration of synaptic inputs[^9]. Notably, this model displays some fundamental properties of synaptic integration observed experimentally (see model properties). The tapering phenomenon due to branching and diameter reduction in the morphology, has a strong impact on the input resistance along the dendritic tree (with also a moderate impact on transfer resistance to the soma). Because it will set the level of evoked depolarization, this input resistance profile will have a strong impact on synaptic integration at different distances from the soma. Indeed, the high depolarization observed in distal locations induce a strong saturation of synaptic efficacy for multiple inputs (because of the reduction of the synaptic driving force that comes with local depolarizations).

[^9]: Rall (1962) "Electrophysiology of a dendritic model" _Biophysical Journal_

We next design a way to analyze the impact of the different dendritic properties observed in PV+ and SST+ INs. First, in terms of synaptic distributions, we introduce a bias factor to vary the dendritic distribution of synapses along a single dendritic branch. Next, for the physiological properties, we implement a variable NMDA-to-AMPA ratio in glutamatergic transmission, varying from AMPA-only to NMDA+AMPA observed reproducing the levels observed in SST+ INs. We found that those two different strategies were able to effectively counteract the synaptic saturation phenomenon observed in dendritic morphologies with thin dendrites.

A particularly difficult task in dendritic arborizations is to faithfully transmit clustered input signals from distal locations (because of the synaptic saturation phenomenon). A first strategy, observed in the case of SST+ INs, is to counter-balance the synaptic saturation by a voltage-gated active mechanism (the NMDA receptor). For PV+ INs however, such active mechanisms were not observed. Instead, we found that the proximally-biased property of synaptic distributions lead to "less localized" distal clusters of synaptic activations and therefore reduced the synaptic saturation effect. Overall, this "proximally-biased distribution" property enabled to improve signal transmission for synaptic clusters located distally on the dendritic branches. This mechanism allow to equalize signal transmission in the presence of passive properties only, without the help of any active mechanism.

We conclude that those two strategies in terms of synaptic physiology and synaptic distributions enable to optimize signal transmission in thin dendritic arborizations.

### Role of Dendritic Mechanisms in Signal Processing In Vivo

We next aimed at testing the impact of some of those dendritic integration properties on the processing of information by interneurons in the cortex. 

One of the properties that we could test was the dependency of the NMDA receptor in SST+ processing thanks to the GluN1 Knockout mice. The prediction of the cellular model was that the NMDA receptor would control the response curve of SST+ interneurons. The NMDA receptor enable to shift the high gain part of the input-ouput curve toward lower stimulation level and saturation is reached earlier.

To test this prediction, we went to the visual cortex because the contrast and orientation of full-field gratings enable us to control the input level on SST+ cells. We tested the processing of those visual signals in SST+ INs using two-photon calcium imaging in awake behaving mice (see setup and raw data). 

In accordance with the model predictions, we found that the responsiveness of SST+ INs at low contrast (half contrast) were higher in Wild-Type mice than in GluN1-KO mice (see fractions of recruited cells). Half contrast stimulation seem to already reach the saturation levels in the WT mice as the proportion of recruited cells did not increase much. On the other hand, cells in the GluN1-KO mice showed a strong increase in recuitment for that same increase of contrast.

SST+ INs are overall weakly selective in terms of orientation. According to uor model, a possibility is that the NMDAR would set the response close to saturation, what would impede differences in spiking responses between different orientations. In accordance with this hypothesis, we found that orientation tuning was increased in GluN1-KO cells. Such higher tuning value was also found at half contrast in WT SST+ INs, consistent with the saturation hypothesis (at high input) underlying the low tuning of SST+ cells at full contrast.

We conclude that dendritic mechanism strongly control the processing of inputs by SST+ cells.
