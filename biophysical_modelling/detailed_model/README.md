# Biophysical Modelling

> starting from the modelling work of [Tzilivaki et al., *Nat. Comm.* (2019)](https://www.nature.com/articles/s41467-019-11537-7)

## Passive Properties

| **Parameter description**  | **location**      |  **Name**       |     **Value**        | **Unit**               |                                                         | 
|----------------------------|-------------------|-----------------|----------------------|------------------------|---------------------------------------------------------|
| **leak conductance**       |                   |                 |                      |                        | N.B. x2.7 to match ~80-100MOhm input resistance         |
|                            | soma              | `BC_soma_gPas`  | 3.55 10<sup>-4</sup> | S/cm<sup>2</sup>       |                                                          |
|                            | axon              | `BC_axon_gPas`  | 9.59 10<sup>-6</sup> | S/cm<sup>2</sup>       |                                                          |
|                            | proximal dendrite | `BC_prox_gPas`  | 3.55 10<sup>-4</sup> | S/cm<sup>2</sup>       |                                                          |
|                            | proximal dendrite | `BC_dist_gPas`  | 3.63 10<sup>-5</sup> | S/cm<sup>2</sup>       |                                                          |
| **leak reversal potential**|                   |                 |                      |                        |                                                         |
|                            | *all*             | `BC_ePas`       | -70.0                | mV                     |                                                         |
| **axial resistance**       |                   |                 |                      |                        |                                                         |
|                            | soma              | `BC_soma_Ra`    | 172                  | ohm.cm                 |                                                         |
|                            | axon              | `BC_axon_Ra`    | 172                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `BC_prox_Ra`    | 142                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `BC_dist_Ra`    | 142                  | ohm.cm                 |                                                         |
| **membrane capacitance**   |                   |                 |                      |                        |                                                         |
|                            | soma              | `BC_soma_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | axon              | `BC_axon_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `BC_prox_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `BC_dist_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |


## Active Mechanisms

### Library of Channels

| **Channel Type**  |  **Description/Comments**  | **Link to `mod` file** |
| --- | --- | --- |
| Passive current                         | classical leak current                                 |    |
| Sodium current                          | classical Na+ channel, with a fast inactivation        |  [`nafx.mod`](./mechanisms/nafx.mod)  |
| Delayed rectifier Potassium current     |                                                        |  [`kdrin.mod`](./mechanisms/kdrin.mod)  |
| Slowly inactivating Potassium current   |                                                        |  [`iksin.mod`](./mechanisms/iksin.mod)  |
| H-type cation current                   | E<sub>rev</sub>=-10mV,  implementation uses Na+ ions   |  [`hin.mod`](./mechanisms/hin.mod)  |
| A-type Potassium current (proximal)     | Klee et al. *J. Physiol.* (1995)[^K95]                 |  [`kaproxin.mod`](./mechanisms/kaproxin.mod)  |
| A-type Potassium current (distal)       | Klee et al. *J. Physiol.* (1995)[^K95]                 |  [`kadistin.mod`](./mechanisms/kadistin.mod)  |
| fast Ca2+ dependent Potassium current   | AP broadening, Shao et al., *J. Physiol.* (1999)[^S99] |  [`kctin.mod`](./mechanisms/kctin.mod)  |
| slow Ca2+ dependent Potassium current   | responsible for slow AHP dynamics                      |  [`kcain.mod`](./mechanisms/kcain.mod)  |
| T-type Ca2+ current (high threshold)    | for somatic and dendritic regions                      |  [`cat.mod`](./mechanisms/cat.mod)  |
| N-type Ca2+ current                     | for somatic and dendritic regions (ref. Borg ?)        |  [`can.mod`](./mechanisms/can.mod)  |
| L-type Ca2+ current (high threshold)    | for somatic and dendritic regions (ref. Borg ?)        |  [`cal.mod`](./mechanisms/cal.mod)  |
| + Calcium dynamics                      | simple first order model                               |  [`cadynin.mod`](./mechanisms/cadynin.mod)  |
| --- | --- | --- |


### Channel Types and Density in All Compartments

#### PV+ cells (based on PFC model[^T19])

|     | **Channel Type**  |  **Name**  |  **Density** (S/cm<sup>2</sup>) |           _comment_           |
| --- | --- | --- | --- | --- | 
| **soma** |     |     |     |          |
|          | Fast Sodium current                           | `BC_soma_gNa`      | 1.35 10<sup>-1</sup> |        |
|          | Delayed rectifier Potassium current           | `BC_soma_gKdrin`   | 3.60 10<sup>-2</sup> |        |
|          | Slowly inactivating Potassium current         | `BC_soma_gKslowin` | 7.25 10<sup>-4</sup> |
|          | H-type cation current                         | `BC_soma_gHin`     | 1.00 10<sup>-5</sup> |
|          | A-type Potassium current (proximal)           | `BC_soma_gKapin`   | 3.20 10<sup>-3</sup> |
|          | fast Ca2+ dependent Potassium current         | `BC_soma_gKctin`   | 1.00 10<sup>-4</sup> |
|          | slow Ca2+ dependent Potassium current         | `BC_soma_gKcain`   | 2.00 10<sup>-2</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`            |                      |
| **axon** |     |     |     |
|          | Passive current                               | `BC_axon_gpas`     | 9.59 10<sup>-6</sup> |
|          | Fast Sodium current ("with fast attenuation") | `BC_axon_gNafx`    | 6.75 10<sup>-1</sup> |
|          | Delayed rectifier Potassium current           | `BC_axon_gKdrin`   | 3.60 10<sup>-2</sup>  |
| **proximal dendrites** | ($\leq$ 100 um from soma) |     |     |     |
|          | Fast Sodium current                           | `BC_prox_gNafx`    | 1.80 10<sup>-2</sup> |
|          | Delayed rectifier Potassium current           | `BC_prox_gKdrin`   | 9.00 10<sup>-4</sup> |
|          | A-type Potassium current (proximal)           | `BC_prox_gKapin`   | 1.00 10<sup>-3</sup> |
|          | T-type Ca2+ current (high threshold)          | `BC_prox_gCat`     | 2.00 10<sup>-4</sup> | 
|          | N-type Ca2+ current                           | `BC_prox_gCan`     | 3.00 10<sup>-5</sup> |
|          | L-type Ca2+ current (high threshold)          | `BC_prox_gCal`     | 3.00 10<sup>-5</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`            |                      |
| **distal dendrites** | ($\leq$ 100 um from soma) |    |     |     |
|          | Fast Sodium current                           | `BC_dist_gNafx`    | 1.40 10<sup>-2</sup> |
|          | Delayed rectifier Potassium current           | `BC_dist_gKdrin`   | 9.00 10<sup>-3</sup> |
|          | A-type Potassium current (proximal)           | `BC_dist_gKapin`   | 9.00 10<sup>-4</sup> |
|          | A-type Potassium current (distal)             | `BC_dist_gKadin`   | 2.16 10<sup>-3</sup> |
|          | T-type Ca2+ current (high threshold)          | `BC_dist_gCat`     | 2.00 10<sup>-4</sup> | 
|          | N-type Ca2+ current                           | `BC_dist_gCan`     | 3.00 10<sup>-5</sup> |
|          | L-type Ca2+ current (high threshold)          | `BC_dist_gCal`     | 3.00 10<sup>-5</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`            |                      |
| --- | --- | --- | --- |

### SST+ cells


### References

[^T19]: https://www.nature.com/articles/s41467-019-11537-7
[^K95]: https://journals.physiology.org/doi/abs/10.1152/jn.1995.74.5.1982
[^S99]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269638/

## Synaptic properties

|                   | **Parameter description**  |  **Name**  |     **Value**      | **Unit**  |                                                         |
|-------------------|----------------------------|------------|--------------------|-----------|                                                         |
| **AMPA receptor** |                            |            |                    |           |                                                         |
|                   | conductance quantal        | `qAMPA`    | 5.0 10<sup>-4</sup>| uS        |                                                         |
|                   | decay time constant        |            | 2.0                | ms        |  changed directly in the [`ampain.mod`](./mechanisms/ampain.mod) file |
| **NMDA receptor** |                            |            |                    |           |                                                         |


