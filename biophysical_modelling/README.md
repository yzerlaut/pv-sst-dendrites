# Biophysical Modelling

> starting from the modelling work of [Tzilivaki et al., *Nat. Comm.* (2019)](https://www.nature.com/articles/s41467-019-11537-7)

## Library of Channels

| **Channel Type**  |  **Description/Comments**  | **Link to `mod` file** |
| --- | --- | --- |
| Passive current                         | classical leak current                                 |    |
| Fast Sodium current                     | classical Na+ channel, with a fast inactivation        |  [`nafx.mod`](./mechanisms/nafx.mod)  |
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


## Channel Types and Density in All Compartments

### PV+ cells (based on PFC model[^T19])

|     | **Channel Type**  |  **Name**  |  **Density** (S/cm<sup>2</sup>) |
| --- | --- | --- | --- |
| **soma** |     |     |     |
|          | Passive current                               | `pas`     | 1.32 10<sup>-4</sup> |
|          | Fast Sodium current                           | `Nafx`    | 1.35 10<sup>-1</sup> |
|          | Delayed rectifier pottassium current          | `Kdrin`   | 3.60 10<sup>-2</sup> |
|          | Slowly inactivating Potassium current         | `Kslowin` | 7.25 10<sup>-4</sup> |
|          | H-type cation current                         | `Hin`     | 1.00 10<sup>-5</sup> |
|          | A-type Potassium current (proximal)           | `Kapin`   | 3.20 10<sup>-3</sup> |
|          | fast Ca2+ dependent Potassium current         | `Kctin`   | 1.00 10<sup>-4</sup> |
|          | slow Ca2+ dependent Potassium current         | `Kcain`   | 2.00 10<sup>-2</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`   |                      |
| **axon** |     |     |     |
|          | Passive current                               | `pas`     | 3.55 10<sup>-6</sup> |
|          | Fast Sodium current ("with fast attenuation") | `Nafx`    | 0.6-1.5 | 
|          | Delayed rectifier pottassium current          | `Kdrin`   | 3.60 10<sup>-2</sup>  |
| **proximal dendrites** | ($\leq$ 100 um from soma) |     |     |     |
|          | Passive current                               | `pas`     | 1.32 10<sup>-4</sup> |
|          | Fast Sodium current                           | `Nafx`    | 1.80 10<sup>-2</sup> |
|          | Delayed rectifier pottassium current          | `Kdrin`   | 9.00 10<sup>-4</sup> |
|          | A-type Potassium current (proximal)           | `Kapin`   | 1.00 10<sup>-3</sup> |
|          | T-type Ca2+ current (high threshold)          | `Cat`     | 2.00 10<sup>-4</sup> | 
|          | N-type Ca2+ current                           | `Can`     | 3.00 10<sup>-5</sup> |
|          | L-type Ca2+ current (high threshold)          | `Cal`     | 3.00 10<sup>-5</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`   |                      |
| **distal dendrites** | ($\leq$ 100 um from soma) |     |     |     |
|          | Passive current                               | `pas`     | 1.32 10<sup>-5</sup> |
|          | Fast Sodium current                           | `Nafx`    | 1.40 10<sup>-2</sup> |
|          | Delayed rectifier pottassium current          | `Kdrin`   | 9.00 10<sup>-3</sup> |
|          | A-type Potassium current (proximal)           | `Kapin`   | 9.00 10<sup>-4</sup> |
|          | A-type Potassium current (distal)             | `Kadin`   | 2.16 10<sup>-3</sup> |
|          | T-type Ca2+ current (high threshold)          | `Cat`     | 2.00 10<sup>-4</sup> | 
|          | N-type Ca2+ current                           | `Can`     | 3.00 10<sup>-5</sup> |
|          | L-type Ca2+ current (high threshold)          | `Cal`     | 3.00 10<sup>-5</sup> |
|          | + Calcium buffering dynamics                  | `CaDyn`   |                      |
| --- | --- | --- | --- |

### SST+ cells


### References

[^T19]: https://www.nature.com/articles/s41467-019-11537-7
[^K95]: https://journals.physiology.org/doi/abs/10.1152/jn.1995.74.5.1982
[^S99]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269638/

